#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import time   
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import AdaBoostRegressor
from catboost import CatBoostRegressor
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = False
plt.rcParams['ytick.right'] = False

from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    learning_curve,
    cross_val_score,
    KFold,                   
    cross_validate           
)
from skopt import BayesSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump

FILE1 = "ML-21+140-element_counts-H.csv"
TARGET_COL = 4
FEATURE_START = 5
FEATURE_END = 42

TEST_SIZE = 0.2
RANDOM_STATE = 46 
                  
TRAIN_COLOR = '#4A90E2'
TEST_COLOR = '#E94E77'

print("Merging datasets...")
df = pd.read_csv(FILE1)

y = df.iloc[:, TARGET_COL]
X = df.iloc[:,  list(range(5,29))]
feature_names = X.columns.tolist()
print(f"Total samples: {len(X)}, Features: {X.shape[1]}")
print (X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
dump(scaler, "scaler.joblib")
print("Scaler saved -> scaler.joblib")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
)

models = {
    "XGBRegressor": (XGBRegressor(random_state=RANDOM_STATE), {
        "n_estimators": [100, 300, 500, 700, 900],
        "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1],
        "max_depth": [3, 4, 5, 6, 7],
        "min_child_weight": [1, 3, 5, 7, 9],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.5, 1, 2],
        "reg_alpha": [0, 0.1, 0.5, 1, 2],
        "reg_lambda":[0, 0.1, 0.5, 1, 2]
    }),
    
    "GradientBoostingRegressor": (GradientBoostingRegressor(random_state=RANDOM_STATE), {
        "n_estimators": [100, 300, 500, 700, 900],
        "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1],
        "max_depth": [3, 4, 5, 6, 7],
        "min_samples_leaf": [2, 4, 6, 8],
        "subsample":[0.6, 0.7, 0.8, 0.9, 1.0],
        "max_features": ["sqrt", "log2"]
    }),

    "RandomForestRegressor": (RandomForestRegressor(random_state=RANDOM_STATE), {
        "n_estimators":  [100, 300, 500, 700, 900],
        "max_depth": [3, 4, 5, 6, 7],
        "min_samples_leaf": [2, 4, 6, 8],
        "max_features": ["sqrt", "log2"]
    }),

    "KernelRidge": (KernelRidge(), {
        "alpha": [0.01, 0.1, 1, 5, 7,10],
        "kernel": ["linear", "rbf", "poly"],
        "gamma": [0.01, 0.05, 0.1, 0.5, 0.7, 1]
    }),

    "BayesianRidge": (BayesianRidge(), {
        "max_iter": [100, 300, 500],
        "alpha_1": [1e-6, 1e-5],
        "alpha_2": [1e-6, 1e-5],
        "lambda_1": [1e-6, 1e-5],
        "lambda_2": [1e-6, 1e-5]
    }),

    # ~ "Lasso": (Lasso(random_state=RANDOM_STATE), {
    #     "alpha": [0.0001, 0.001, 0.01, 0.1],
    #     "max_iter": [1000, 5000]
    # ~ }),

    "KNeighborsRegressor": (KNeighborsRegressor(), {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform", "distance"]
    }),

    # ~ "LinearRegression": (LinearRegression(), {})
    
    # ~ "AdaBoostRegressor": (AdaBoostRegressor(random_state=RANDOM_STATE), {
        # ~ "n_estimators": [50, 100, 200, 300],
        # ~ "learning_rate": [0.01, 0.05, 0.1, 0.5],
        # ~ "loss": ["linear", "square", "exponential"]
    # ~ }),

    "CatBoostRegressor": (CatBoostRegressor(random_state=RANDOM_STATE, verbose=0), {
        "iterations": [100, 300, 500, 700, 900],
        "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1],
        "depth": [3, 4, 5, 6, 7],
        "l2_leaf_reg": [2, 4, 6, 8],
        "border_count": [32, 64, 128]
    }),    
}

best_model = None
best_mae = float('inf')
best_model_name = ""
results = []
best_train_metrics = {}
best_test_metrics = {}

print("\n============ Model Training & Tuning ============")
for name, (model, param_dist) in models.items():
    print(f"Training: {name} (Searching best hyperparameters...)")
    
    bayes_search = BayesSearchCV(
        model, 
        search_spaces=param_dist, 
        n_iter=50,                 
        cv=5,
        scoring='neg_mean_absolute_error', 
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )
    
    # ---- 开始计时 ----
    start_time = time.time()
    bayes_search.fit(X_train, y_train)
    elapsed = time.time() - start_time     
    print(f" {name} training time: {elapsed:.1f} s")
    
    print(f"\n{bayes_search.best_params_}")
    
    model_best = bayes_search.best_estimator_

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_validate(
        model_best, X_train, y_train, cv=cv,
        scoring={'r2': 'r2', 'mse': 'neg_mean_squared_error', 'mae': 'neg_mean_absolute_error'},
        n_jobs=-1, return_train_score=False
    )
    fold_r2 = cv_scores['test_r2']                  
    fold_rmse = np.sqrt(-cv_scores['test_mse'])      
    fold_mae = -cv_scores['test_mae']               
    mean_r2, std_r2 = np.mean(fold_r2), np.std(fold_r2)
    mean_rmse, std_rmse = np.mean(fold_rmse), np.std(fold_rmse)
    mean_mae, std_mae = np.mean(fold_mae), np.std(fold_mae)

    y_pred_train = model_best.predict(X_train)
    y_pred_test = model_best.predict(X_test)

    mae_tr = mean_absolute_error(y_train, y_pred_train)
    mae_te = mean_absolute_error(y_test, y_pred_test)
    rmse_tr = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_te = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_tr = r2_score(y_train, y_pred_train)
    r2_te = r2_score(y_test, y_pred_test)

    model_result = {
        "Model": name,
        "Best_Params": bayes_search.best_params_,
        "MAE_train": mae_tr, "MAE_test": mae_te,
        "RMSE_train": rmse_tr, "RMSE_test": rmse_te,
        "R2_train": r2_tr, "R2_test": r2_te,
        "Training_Time_s": round(elapsed, 1)
    }
    for i in range(5):
        model_result[f"Fold{i+1}_R2"] = round(fold_r2[i], 4)
        model_result[f"Fold{i+1}_RMSE"] = round(fold_rmse[i], 4)
        model_result[f"Fold{i+1}_MAE"] = round(fold_mae[i], 4)
    model_result["CV_Mean_R2"] = round(mean_r2, 4)
    model_result["CV_Std_R2"] = round(std_r2, 4)
    model_result["CV_Mean_RMSE"] = round(mean_rmse, 4)
    model_result["CV_Std_RMSE"] = round(std_rmse, 4)
    model_result["CV_Mean_MAE"] = round(mean_mae, 4)
    model_result["CV_Std_MAE"] = round(std_mae, 4)

    results.append(model_result)

    if mae_te < best_mae:
        best_mae = mae_te
        best_model = model_best
        best_model_name = name
        best_train_metrics = {'r2': r2_tr, 'rmse': rmse_tr, 'mae': mae_tr, 'y_pred': y_pred_train}
        best_test_metrics = {'r2': r2_te, 'rmse': rmse_te, 'mae': mae_te, 'y_pred': y_pred_test}

    print(f"Generating plots for {name} ...")
    
    g = sns.JointGrid(height=7, ratio=6, space=0.1)
    min_val = min(y_train.min(), y_pred_train.min(), y_test.min(), y_pred_test.min())-0.05
    max_val = max(y_train.max(), y_pred_train.max(), y_test.max(), y_pred_test.max())+0.05
    g.ax_joint.fill_between([min_val, max_val],
                            [min_val - 0.1, max_val - 0.1],
                            [min_val + 0.1, max_val + 0.1],
                            color='gray', alpha=0.25, zorder=1, label='±0.1 eV')
    g.ax_joint.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, zorder=2)
    g.ax_joint.scatter(y_train, y_pred_train, c=TRAIN_COLOR, alpha=0.5, s=45,
                       edgecolors='white', linewidths=0.2, label=f'Train Set ({len(y_train)})', zorder=3)
    g.ax_joint.scatter(y_test, y_pred_test, c=TEST_COLOR, marker='s', alpha=0.65, s=55,
                       edgecolors='white', linewidths=0.2, label=f'Test Set ({len(y_test)})', zorder=4)
    g.ax_joint.set_xlim([min_val, max_val])
    g.ax_joint.set_ylim([min_val, max_val])
    g.ax_joint.set_aspect('equal')
    bins_num = 30
    g.ax_marg_x.hist(y_train, bins=bins_num, color=TRAIN_COLOR, alpha=0.6, density=True)
    g.ax_marg_x.hist(y_test, bins=bins_num, color=TEST_COLOR, alpha=0.6, density=True)
    g.ax_marg_x.axis('off')
    g.ax_marg_y.hist(y_pred_train, bins=bins_num, color=TRAIN_COLOR, alpha=0.6, density=True, orientation='horizontal')
    g.ax_marg_y.hist(y_pred_test, bins=bins_num, color=TEST_COLOR, alpha=0.6, density=True, orientation='horizontal')
    g.ax_marg_y.axis('off')
    g.ax_joint.set_xlabel("DFT Adsorption Energy (eV)", fontsize=14, fontweight='bold')
    g.ax_joint.set_ylabel("ML Predicted Energy (eV)", fontsize=14, fontweight='bold')
    g.ax_joint.tick_params(labelsize=12)
    g.ax_joint.legend(loc='upper left', frameon=True, edgecolor='black', fontsize=12)
    for spine in g.ax_joint.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
    metrics_data = [
        ("Train $R^2$ :", f"{r2_tr:.3f}", TRAIN_COLOR),
        ("Test $R^2$ :", f"{r2_te:.3f}", TEST_COLOR),
        ("Train RMSE :", f"{rmse_tr:.3f} eV", TRAIN_COLOR),
        ("Test RMSE :", f"{rmse_te:.3f} eV", TEST_COLOR),
        ("Train MAE :", f"{mae_tr:.3f} eV", TRAIN_COLOR),
        ("Test MAE :", f"{mae_te:.3f} eV", TEST_COLOR)
    ]
    value_x = 0.96
    label_x = 0.83
    start_y = 0.28
    line_spacing = 0.045
    y_pos = start_y
    for label, val, color in metrics_data:
        g.ax_joint.text(label_x, y_pos, label, transform=g.ax_joint.transAxes,
                        fontsize=12, color='black', ha='right', va='center', zorder=6)
        g.ax_joint.text(value_x, y_pos, val, transform=g.ax_joint.transAxes,
                        fontsize=12, color=color, ha='right', va='center', zorder=6)
        y_pos -= line_spacing
    plt.savefig(f"prediction_scatter_{name}.png", dpi=300, bbox_inches='tight')
    plt.close()

    res_train = y_pred_train - y_train
    res_test = y_pred_test - y_test
    plt.figure(figsize=(5, 4))
    sns.histplot(res_train, bins=30, stat='density', color=TRAIN_COLOR, edgecolor='none', alpha=0.5, label='Train Error Dist')
    sns.histplot(res_test, bins=30, stat='density', color=TEST_COLOR, edgecolor='none', alpha=0.6, label='Test Error Dist')
    x_range = np.linspace(min(res_train.min(), res_test.min()) - 0.1, max(res_train.max(), res_test.max()) + 0.1, 200)
    mu_tr, std_tr = norm.fit(res_train)
    plt.plot(x_range, norm.pdf(x_range, mu_tr, std_tr), color='#2980B9', lw=1.5, linestyle='-', label=f'Train Gauss Fit \n($\mu$={mu_tr:.3f}, $\sigma$={std_tr:.3f})')
    mu_te, std_te = norm.fit(res_test)
    plt.plot(x_range, norm.pdf(x_range, mu_te, std_te), color='#C0392B', lw=1.5, linestyle='-', label=f'Test Gauss Fit \n($\mu$={mu_te:.3f}, $\sigma$={std_te:.3f})')
    plt.axvline(x=0, color='black', linestyle='--', lw=1.5, zorder=0)
    plt.xlabel("Prediction Error (eV)", fontsize=12, fontweight='bold')
    plt.ylabel("Density", fontsize=12, fontweight='bold')
    plt.title(f"Error Distribution ({name})", fontsize=13, fontweight='bold')
    plt.legend(loc='upper left', frameon=True, edgecolor='black', fontsize=10)
    plt.grid(alpha=0.2, linestyle='--')
    plt.tight_layout()
    plt.savefig(f"error_distribution_{name}.png", dpi=300, bbox_inches='tight')
    plt.close()

    LEARNING_CURVE_METRICS = ['MAE', 'RMSE', '1-R2']
    for metric in LEARNING_CURVE_METRICS:
        if metric == 'MAE':
            scoring = 'neg_mean_absolute_error'
            ylabel = 'MAE (eV)'
            transform = lambda scores: -scores
            suffix = 'mae'
        elif metric == 'RMSE':
            try:
                scoring = 'neg_root_mean_squared_error'
                transform = lambda scores: -scores
            except:
                scoring = 'neg_mean_squared_error'
                transform = lambda scores: np.sqrt(-scores)
            ylabel = 'RMSE (eV)'
            suffix = 'rmse'
        elif metric == '1-R2':
            scoring = 'r2'
            transform = lambda scores: 1 - scores
            ylabel = '1 - R²'
            suffix = '1minusR2'

        train_sizes, train_scores, val_scores = learning_curve(
            model_best, X_scaled, y, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 0.95, 8),
            scoring=scoring
        )
        train_mean = transform(np.mean(train_scores, axis=1))
        train_std = np.std(train_scores, axis=1)
        val_mean = transform(np.mean(val_scores, axis=1))
        val_std = np.std(val_scores, axis=1)

        plt.figure(figsize=(5, 4))
        plt.plot(train_sizes, train_mean, 'o-', color=TRAIN_COLOR, lw=2, label="Training")
        plt.plot(train_sizes, val_mean, 's-', color=TEST_COLOR, lw=2, label="Validation")
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color=TRAIN_COLOR)
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color=TEST_COLOR)
        plt.xlabel("Training Samples", fontsize=12, fontweight='bold')
        plt.ylabel(ylabel, fontsize=12, fontweight='bold')
        plt.title(f"Learning Curve ({name}) - {metric}", fontsize=13, fontweight='bold')
        plt.legend(loc="best", frameon=True, edgecolor='black', fontsize=12)
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(f"learning_curve_{suffix}_{name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f" learning_curve_{suffix}_{name}.png")

    if hasattr(model_best, "feature_importances_"):
        imp = model_best.feature_importances_
        sorted_idx = np.argsort(imp)[::-1]
        top_n = min(20, len(imp))
        top_imp = imp[sorted_idx][:top_n]
        top_names = np.array(feature_names)[sorted_idx][:top_n]
        plt.figure(figsize=(5, 4))
        bars = plt.barh(range(top_n), top_imp, color='#4A90E2', edgecolor='black', linewidth=0.5)
        plt.yticks(range(top_n), top_names, fontsize=8)
        plt.xlabel("Feature Importance", fontsize=12, fontweight='bold')
        plt.title(f"Top {top_n} Features ({name})", fontsize=13, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.xlim(0, max(top_imp) * 1.1)
        for i, (bar, val) in enumerate(zip(bars, top_imp)):
            plt.text(val + 0.002, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va='center', fontsize=7)
        plt.tight_layout()
        plt.savefig(f"feature_importance_{name}.png", dpi=300, bbox_inches='tight')
        plt.close()

    X_df = pd.DataFrame(X_scaled, columns=feature_names)
    corr = X_df.corrwith(y).abs().sort_values(ascending=False)
    top_features = corr[:20]  
    plt.figure(figsize=(5, 4))
    ax = sns.barplot(x=top_features.values, y=top_features.index, palette="magma")
    plt.xlim(0, max(top_features.values) * 1.1)
    for i, (idx, val) in enumerate(top_features.items()):
        ax.text(val + 0.002, i, f"{val:.3f}", va='center', fontsize=7, color='black')
    plt.xlabel("Pearson Correlation", fontsize=12, fontweight='bold')
    plt.title(f"Top 20 Correlations ({name})", fontsize=13, fontweight='bold')
    plt.gca().tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()
    plt.savefig(f"pearson_corr_{name}.png", dpi=300, bbox_inches='tight')
    plt.close()

dump(best_model, f"best_model_{best_model_name}.joblib")
print(f"\n Best model saved: best_model_{best_model_name}.joblib")

def dataset_size_analysis_mae(model, X, y):
    sizes = np.linspace(0.2, 0.95, 8)
    mae_list = []
    for s in sizes:
        X_s, _, y_s, _ = train_test_split(X, y, train_size=s, random_state=RANDOM_STATE)
        score = -cross_val_score(model, X_s, y_s, cv=3, scoring="neg_mean_absolute_error").mean()
        mae_list.append(score)
        
    plt.figure(figsize=(7, 5))
    plt.plot((sizes * 100), mae_list, 'o-', color='#8E44AD', linewidth=2.5, markersize=8)
    plt.xlabel("Percentage of Training Data Used (%)", fontsize=19, fontweight='bold')
    plt.ylabel("Cross-Validation MAE (eV)", fontsize=19, fontweight='bold')
    plt.title("Effect of Dataset Size on Performance", fontsize=17, fontweight='bold')
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig("dataset_size_effect_mae.png", dpi=300, bbox_inches='tight')
    plt.close()

# ~ dataset_size_analysis_mae(best_model, X_scaled, y)

result_df = pd.DataFrame(results).round(4)
result_df = result_df.sort_values(by="MAE_test", ascending=True)
result_df.to_csv("model_comparison.csv", index=False)
print("\n Model comparison saved -> model_comparison.csv (including per-fold CV R2, RMSE, MAE)")
