from ase import Atoms
from ase.io import read
import re
import numpy as np
import pandas as pd
import os
from ase.visualize import view
import csv
from collections import Counter
import glob
from copy import deepcopy


ELEMENT_PROPS = {
    'Fe': {'ele': 1.83, 'rad': 1.26, 'd_elec': 6},
    'Co': {'ele': 1.88, 'rad': 1.25, 'd_elec': 7},
    'Ni': {'ele': 1.91, 'rad': 1.24, 'd_elec': 8},
    'Mo': {'ele': 2.16, 'rad': 1.39, 'd_elec': 5}
}

TARGET_ELEMENTS = ['Fe', 'Co', 'Ni', 'Mo']

def inside_triangle(point, vertices):
    vertices = deepcopy(vertices)
    vertices = np.asarray(vertices)
    vertices[1] -= vertices[0]
    vertices[2] -= vertices[0]
    det1 = np.cross(point, vertices[1])
    det2 = np.cross(point, vertices[2])
    det01 = np.cross(vertices[0], vertices[1])
    det02 = np.cross(vertices[0], vertices[2])
    det12 = np.cross(vertices[1], vertices[2])
    a1 = (det2 - det02) / det12
    a2 = -(det1 - det01) / det12
    return 0 < a1 < 1 and 0 < a2 < 1

def expand_triangle(vertices, expansion):
    center = np.mean(vertices, axis=0)
    diffs = vertices - center
    return center + diffs*expansion

class Slab(Atoms):
    def __init__(self, atoms, ads, ads_atom):
        self.atoms_orig = atoms.copy()
        self.atoms_orig.set_constraint()
        
        atoms_no_constraint = atoms.copy()
        atoms_no_constraint.set_constraint()
        atoms_3x3 = atoms_no_constraint.repeat((3, 3, 1))
        
        super().__init__(atoms_3x3)
        self.ads = ads
        self.ads_atom = ads_atom
        self.ads_elems = re.findall('[A-Z][^A-Z]*', ads)
        
        self.get_ads_atom_idx()
        self.set_layer_tags()
        self.site = None

    def get_ads_atom_idx(self):
        ads_atom_ids = np.nonzero(self.symbols == self.ads_atom)[0]
        if len(ads_atom_ids) == 0:
            raise ValueError(f"No {self.ads_atom} atom found in the structure.")
        self.ads_atom_idx = ads_atom_ids[(len(ads_atom_ids) - 1) // 2]
        return self.ads_atom_idx
    
    def get_mean_location(self, atom_ids):
        return np.mean(self.positions[atom_ids], axis=0)
    
    def get_distances_from_index(self, atom_idx, layer=None):
        if layer is None:
            return np.sum((self.positions - self.positions[atom_idx]) ** 2, axis=1) ** 0.5
        else:
            positions = np.array([self.positions[atom.index] for atom in self if atom.tag == layer])
            return np.sum((positions - self.positions[atom_idx]) ** 2, axis=1) ** 0.5
    
    def get_distances_from_point(self, point, layer=None):
        if layer is None:
            return np.sum((self.positions - point) ** 2, axis=1) ** 0.5
        else:
            positions = np.array([self.positions[atom.index] for atom in self if atom.tag == layer])
            return np.sum((positions - point) ** 2, axis=1) ** 0.5

    def set_layer_tags(self):
        non_adsorbate_indices = np.array([i for i, symbol in enumerate(self.symbols) if symbol not in self.ads_elems])
        z_coords = self.positions[non_adsorbate_indices, 2]
        
        max_z = np.max(z_coords)
        min_z = np.min(z_coords)
        surface_cutoff = 1.50
        same_layer_tolerance = 0.8
        
        tags = np.zeros(len(self), dtype=int)
        
        four_layer_mask = (z_coords < (min_z + same_layer_tolerance)) & (z_coords >= min_z)
        tags[non_adsorbate_indices[four_layer_mask]] = 4
        
        three_layer_mask = (z_coords < (max_z - surface_cutoff)) & (z_coords >= min_z + same_layer_tolerance)
        tags[non_adsorbate_indices[three_layer_mask]] = 3
        
        subsurface_layer_mask = (z_coords < (max_z - surface_cutoff)) & (z_coords >= max_z - 2 * surface_cutoff)
        tags[non_adsorbate_indices[subsurface_layer_mask]] = 2
        
        surface_layer_mask = (z_coords <= max_z) & (z_coords >= (max_z - surface_cutoff))
        tags[non_adsorbate_indices[surface_layer_mask]] = 1
    
        adsorbate_indices = np.array([i for i, symbol in enumerate(self.symbols) if symbol in self.ads_elems])
        tags[adsorbate_indices] = 0
        self.set_tags(tags)
    
    def get_adsorption_site_ids(self, radius=2.2, hollow_radius=2.2):
        dists = self.get_distances_from_index(self.ads_atom_idx)
        return np.nonzero((dists < radius) * (self.positions[:, -1] < self.positions[self.ads_atom_idx, -1]))[0]
    
    def get_adsorption_site(self, radius=2.2, hollow_radius=2.2):
        site_ids = self.get_adsorption_site_ids(radius, hollow_radius)
        site_elems = [self.symbols[idx] for idx in site_ids if self.symbols[idx] not in self.ads_elems]
        site_elems.sort()
        n_atoms_site = len(site_elems)
        
        if n_atoms_site == 1:
            self.site = 'ontop'
            return ''.join(site_elems), 'ontop'
        elif n_atoms_site == 2:
            self.site = 'bridge'
            return ''.join(site_elems), 'bridge'
        elif n_atoms_site == 3:
            site_elems, site = self.get_hollow_site(radius, hollow_radius, site_ids)
            if None in [site_elems, site]: raise ValueError
            self.site = site
            return ''.join(site_elems), site
        else:
            raise ValueError(f'Number of atoms in site is {n_atoms_site} which is not recognized')

    def get_hollow_site(self, radius=2.2, hollow_radius=2.2, site_ids=None):
        if site_ids is None:
            dists = self.get_distances_from_index(self.ads_atom_idx)
            mask_below = self.positions[:, -1] < self.positions[self.ads_atom_idx, -1]
            ids_below = np.array(list(range(len(self))))[mask_below]
            sort_ids = np.argpartition(dists[mask_below], kth=3)[:3]
            site_ids = ids_below[sort_ids]
        
        vec1 = self.positions[site_ids[1], :2] - self.positions[site_ids[0], :2]
        len1 = sum(vec1 ** 2) ** 0.5
        vec2 = self.positions[site_ids[2], :2] - self.positions[site_ids[0], :2]
        len2 = sum(vec2 ** 2) ** 0.5
        dotp = np.dot(vec1, vec2)
        angle = np.arccos(dotp / (len1 * len2)) * 180. / np.pi
        
        if not (45. < angle < 75.):
            print(f'[WARNING] Hollow site with an angle of {angle} degrees far from 60-60-60 degrees detected')
            return None, None
        
        site_pos = self.get_mean_location(site_ids)
        site_pos[-1] = min(self.positions[site_ids, -1]) - 1.5
        dists = self.get_distances_from_point(site_pos)
        ids_below = np.nonzero((dists < hollow_radius) * (self.positions[:, -1] < site_pos[-1]))[0]
        
        site_elems = sorted([self.symbols[idx] for idx in site_ids if self.symbols[idx] not in self.ads_elems])
        n_atoms_below = len(ids_below)
        
        if n_atoms_below == 1: return site_elems, 'hcp'
        elif n_atoms_below == 3: return site_elems, 'fcc'
        else: raise ValueError(f'Number of atoms found under hollow site is {n_atoms_below} which is not recognized')
    
    def get_closest_ids(self, point, start, stop, layer):
        layer_atom_ids = np.array([atom.index for atom in self if atom.tag == layer])
        dists = self.get_distances_from_point(point, layer)
        if len(dists) == 0 or start >= len(dists): return np.array([])
        stop = min(stop, len(dists) - 1)
        ids_sort = np.argpartition(dists, kth=range(start, stop + 1))[start:stop + 1]
        return layer_atom_ids[ids_sort]
    
    def get_closest_distances(self, num_coll, layer):
        ads_pos = self.positions[self.ads_atom_idx]
        layer_atom_ids = np.array([atom.index for atom in self if atom.tag == layer])
        symbols = [self.symbols[i] for i in layer_atom_ids]
        dists = self.get_distances_from_point(ads_pos, layer)
        dists_with_ids = list(zip(dists, layer_atom_ids, symbols))
        dists_with_ids.sort(key=lambda x: x[0])
        return [(d, sym) for d, _, sym in dists_with_ids[:num_coll]]

    def _get_neighbors(self, layer, ranges):
        ads_pos = self.positions[self.ads_atom_idx]
        if self.site not in ranges:
            raise ValueError(f"Unrecognized site type: {self.site}")
            
        start, stop = ranges[self.site]
        ids = self.get_closest_ids(ads_pos, start, stop, layer=layer)
        return [self.symbols[i] for i in ids], ids

    def surface_active(self): 
        # T1: ontop(1), bridge(2), hcp(3), fcc(3)
        return self._get_neighbors(1, {'ontop':(0,0), 'bridge':(0,1), 'hcp':(0,2), 'fcc':(0,2)})
        
    def surface_neighbors(self): 
        # T2
        return self._get_neighbors(1, {'ontop':(1,6), 'bridge':(2,3), 'hcp':(3,5), 'fcc':(3,5)})
        
    def surface_neighbors2(self): 
        # T3
        return self._get_neighbors(1, {'ontop':(7,12), 'bridge':(4,9), 'hcp':(6,11), 'fcc':(6,11)})
        
    def subsurface_neighbors(self): 
        # S1
        return self._get_neighbors(2, {'ontop':(0,2), 'bridge':(0,0), 'hcp':(0,0), 'fcc':(0,2)})
        
    def subsurface_neighbors2(self): 
        # S2
        return self._get_neighbors(2, {'ontop':(3,5), 'bridge':(1,4), 'hcp':(1,6), 'fcc':(3,5)})
        
        
    def count_elements_around_site(self, indices_within_radius, metals):
        elements_count = {metal: 0 for metal in metals} 
        for idx in indices_within_radius:
            symbol = self.symbols[idx]
            if symbol not in self.ads_elems and symbol in elements_count:
                elements_count[symbol] += 1
        return [elements_count[metal] for metal in metals]

    def get_element_ratios_by_layers(self):
        tags = self.get_tags()
        non_adsorbate_elements = set(symbol for symbol, tag in zip(self.symbols, tags) if tag > 0)
        layer_elements_count = {layer: Counter() for layer in set(tags) if layer > 0}
        
        for tag, symbol in zip(tags, self.symbols):
            if tag > 0: layer_elements_count[tag][symbol] += 1
        
        layer_elements_ratios = {}
        for layer, count in layer_elements_count.items():
            total = sum(count.values())
            layer_elements_ratios[layer] = {element: round(count.get(element, 0) / total, 3) for element in non_adsorbate_elements}
        
        total_atoms = sum([sum(count.values()) for count in layer_elements_count.values()])
        layer_elements_ratios[0] = {element: round(sum([count.get(element, 0) for count in layer_elements_count.values()]) / total_atoms, 3) for element in non_adsorbate_elements}
        
        return layer_elements_ratios

    def get_zone_advanced_features(self, indices):
        
        if len(indices) == 0:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        
        props = [ELEMENT_PROPS[self.symbols[i]] for i in indices if self.symbols[i] in ELEMENT_PROPS]
        if not props:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
            
        rads = [p['rad'] for p in props]
        eles = [p['ele'] for p in props]
        decs = [p['d_elec'] for p in props]
        
        avg_rad = np.mean(rads)
        avg_ele = np.mean(eles)
        avg_dec = np.mean(decs)
        
        syms = [self.symbols[i] for i in indices if self.symbols[i] in ELEMENT_PROPS]
        counts = Counter(syms)
        tot = len(syms)
        entropy = -sum((c/tot) * np.log(c/tot) for c in counts.values())
        
        polarization = np.std(eles) if len(eles) > 1 else 0.0
        
        return [avg_rad, avg_ele, avg_dec, entropy, polarization]


def write_counts_to_csv(filename, counts_list, site_type, site_elements, metals, slab, 
                        distances, adv_features, output_csv='element_counts.csv'):
    file_exists = os.path.exists(output_csv)
    
    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        layers = range(1, 5)
        dist_list_collect = {1: 7, 2: 3}
        zones = ['T1', 'T2', 'S1', 'T3', 'S2']
        
        if not file_exists:
            header = ['File', 'Site_Type', 'Site_elements']
            for z in zones:
                header += [f"{m}_{z}" for m in metals]
            for layer in layers:
                header += [f"L{layer}_{metal}" for metal in metals]
            header += [f"Overall_{metal}" for metal in metals]
            header += [f"d_{z}" for z in zones]
            adv_names = ['mean_rad', 'mean_ele', 'mean_delec', 'Entropy', 'Polarization']
            for z in zones:
                header += [f"{z}_{feat}" for feat in adv_names]
            for index, num_cols in dist_list_collect.items():
                header += [f"L_{index}_A_{col}" for col in range(1, num_cols + 1)]
            
            writer.writerow(header)
        
        ratios_by_layers = slab.get_element_ratios_by_layers()
        layer_ratios = [ratios_by_layers.get(l, {}).get(m, 0.0) for l in layers for m in metals]
        overall_ratios = [ratios_by_layers.get(0, {}).get(m, 0.0) for m in metals]
        
        neighbor_info = []
        for index, num_cols in dist_list_collect.items():
            dist_list = slab.get_closest_distances(num_coll=num_cols, layer=index)
            neighbor_info.extend([d for d, _ in dist_list])
            
        row = ([filename, site_type, site_elements] + 
               counts_list + 
               layer_ratios + 
               overall_ratios + 
               distances + 
               adv_features + 
               neighbor_info)
        
        writer.writerow(row)


if __name__ == '__main__':
    output_csv = 'featured_data_HEI.csv'
    if os.path.exists(output_csv):
        os.remove(output_csv)
        
    metals = TARGET_ELEMENTS
    
    adsorbate = "H"
    adsorbing_atom = "H"

    folder = []
    if os.path.exists('collect_list.csv'):
        df = pd.read_csv('collect_list.csv')
        folder = [f"CONTCAR_{p}.vasp" for p in df.iloc[:, 0].tolist() if os.path.exists(f"CONTCAR_{p}.vasp")]
    else:
        folder = sorted(glob.glob('*.vasp'))
        
    for filename in folder:
        try:
            atoms = read(filename)
            slab = Slab(atoms, ads=adsorbate, ads_atom=adsorbing_atom)
            site_elements, site_type = slab.get_adsorption_site()
            
            _, t1_ids = slab.surface_active()
            _, t2_ids = slab.surface_neighbors()
            _, s1_ids = slab.subsurface_neighbors()
            _, t3_ids = slab.surface_neighbors2()
            _, s2_ids = slab.subsurface_neighbors2()
            
            zones_ids = [t1_ids, t2_ids, s1_ids, t3_ids, s2_ids]
            
            counts_list = []
            for ids in zones_ids:
                counts_list.extend(slab.count_elements_around_site(ids, metals))
            
            ads_pos = slab.positions[slab.ads_atom_idx]
            def mean_distance(ids):
                if len(ids) == 0: return 0.0
                return np.mean(np.linalg.norm(slab.positions[ids] - ads_pos, axis=1))
            
            distances = [mean_distance(ids) for ids in zones_ids]
            
            adv_features = []
            for ids in zones_ids:
                adv_features.extend(slab.get_zone_advanced_features(ids))
            
            write_counts_to_csv(filename, counts_list, site_type, site_elements, metals, slab, 
                                distances, adv_features, output_csv=output_csv)
            
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            
    print(f"finish {output_csv}")
