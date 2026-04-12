#!/usr/bin/env python3
"""
Analyse correlation ET baseline vs XGB (raw + logabs + few ratios).
Generates OOF predictions, calculates corr(pred_et, pred_xgb) and corr(err_et, err_xgb).
XGB optimized via Optuna (5 trials max, bridged config to prevent overfitting).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
import xgboost as xgb
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from odor_competition.data import (
    prepare_feature_frames,
    load_competition_data,
    infer_target_schema,
    compress_targets,
)
from odor_competition.metrics import competition_rmse

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_et_params(json_path: str) -> dict:
    """Load ET best params from JSON."""
    with open(json_path) as f:
        data = json.load(f)
    
    # Try different nested structures
    if "optuna" in data and "best_params" in data["optuna"]:
        params = data["optuna"]["best_params"]
    elif "best_params" in data:
        params = data["best_params"]
    else:
        params = data
    
    # Extract only ExtraTreesRegressor valid params
    valid_keys = {
        "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
        "max_features", "bootstrap", "max_samples", "random_state", "n_jobs"
    }
    
    return {k: v for k, v in params.items() if k in valid_keys}


def build_xgb_features(X: pd.DataFrame, raw_cols: list[str]) -> pd.DataFrame:
    """Build: raw + logabs + few robust ratios."""
    out = X[raw_cols].copy()
    
    # Log absolute values (safe engineering)
    for col in raw_cols:
        if col not in ['Env']:
            out[f"{col}_logabs"] = np.log(np.abs(X[col]) + 1e-8)
    
    # Few robust ratios (not the strongest)
    out["Y1_over_Y2_log"] = np.log(np.abs(X["Y1"]) / (np.abs(X["Y2"]) + 1e-8) + 1e-8)
    out["Z_over_Y3_log"] = np.log(np.abs(X["Z"]) / (np.abs(X["Y3"]) + 1e-8) + 1e-8)
    out["X6_over_X4"] = np.log(np.abs(X["X6"]) / (np.abs(X["X4"]) + 1e-8) + 1e-8)
    
    return out


def fit_et_features(X_train: pd.DataFrame, y_train: pd.DataFrame,
                     corr_threshold: float = 0.99,
                     signal_quantile: float = 0.20,
                     max_features: int = 45) -> list[str]:
    """Select ET features via correlation pruning."""
    # Get features with low correlation to each other
    all_cols = X_train.columns.tolist()
    corr_matrix = X_train[all_cols].corr().abs()
    
    selected = []
    for col in all_cols:
        if not any(corr_matrix.loc[col, s] > corr_threshold for s in selected):
            selected.append(col)
    
    # Signal filtering
    stds = X_train[selected].std()
    signal_threshold = stds.quantile(signal_quantile)
    selected = [c for c in selected if stds[c] >= signal_threshold]
    
    # Cap
    selected = selected[:max_features]
    logger.info(f"ET: selected {len(selected)} features")
    
    return selected


def optimize_xgb_params(X_train: pd.DataFrame, X_valid: pd.DataFrame,
                        y_train: pd.DataFrame, y_valid: pd.DataFrame,
                        trials: int = 3) -> dict:
    """Optuna XGB optimization - ULTRA simplified with fixed narrow ranges."""
    
    def objective(trial):
        params = {
            "n_estimators": 600,
            "max_depth": trial.suggest_int("max_depth", 6, 8),
            "learning_rate": 0.11,
            "subsample": 0.88,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.75),
            "min_child_weight": 12,
            "reg_alpha": 0.01,
            "reg_lambda": 0.08,
            "gamma": 0.08,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train.values)
        
        pred = model.predict(X_valid)
        rmse = np.sqrt(np.mean((y_valid.values - pred) ** 2))
        return rmse
    
    sampler = TPESampler(seed=42)
    study = optuna.create_study(sampler=sampler, direction="minimize")
    study.optimize(objective, n_trials=trials, timeout=180, show_progress_bar=False)
    
    return study.best_params


def main():
    parser = argparse.ArgumentParser(description="ET vs XGB correlation analysis (OOF)")
    parser.add_argument("--cv-folds", type=int, default=6)
    parser.add_argument("--et-params-json",
                        default="artifacts_extratrees_corr_optuna/02_experiments_OPEN/q20_feat45_corr990_cv6_trials24/best_score_actuel.json")
    parser.add_argument("--output-dir", default="artifacts_extratrees_corr_optuna/011_et_xgb_correlation_analysis_cv6")
    
    args = parser.parse_args()
    
    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data_dir = ROOT / "src" / "odor_competition" / "data"
    comp_data = load_competition_data(data_dir)
    
    X_train, _, _ = prepare_feature_frames(comp_data.x_train)
    y_df = comp_data.y_train
    
    schema = infer_target_schema(y_df)
    y_train = compress_targets(y_df, schema)
    
    logger.info(f"Loaded X_train: {X_train.shape}, y_train: {y_train.shape}")
    
    # Load ET params
    et_params = load_et_params(args.et_params_json)
    logger.info(f"ET params: {et_params}")
    
    # CV loop - generate OOF
    kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    
    oof_et = np.zeros((len(X_train), len(schema.model_targets)))
    oof_xgb = np.zeros((len(X_train), len(schema.model_targets)))
    fold_reports = []
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train), 1):
        logger.info(f"\n=== Fold {fold}/{args.cv_folds} ===")
        
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]
        
        # ===== ET =====
        et_selected = fit_et_features(X_tr, y_tr)
        et_model = ExtraTreesRegressor(**et_params)
        et_model.fit(X_tr[et_selected], y_tr.values)
        pred_et = et_model.predict(X_val[et_selected])
        oof_et[valid_idx] = pred_et
        
        rmse_et = np.sqrt(np.mean((y_val.values - pred_et) ** 2))
        logger.info(f"ET RMSE: {rmse_et:.6f}")
        
        # ===== XGB =====
        X_xgb_tr = build_xgb_features(X_tr, ["X12", "X13", "X14", "X15", "X4", "X5", "X6", "X7", "Z", "Y1", "Y2", "Y3"])
        X_xgb_val = build_xgb_features(X_val, ["X12", "X13", "X14", "X15", "X4", "X5", "X6", "X7", "Z", "Y1", "Y2", "Y3"])
        
        # Optuna optimize XGB (5 trials)
        logger.info(f"Optimizing XGB (5 trials)...")
        best_xgb_params = optimize_xgb_params(X_xgb_tr, X_xgb_val, y_tr, y_val, trials=5)
        logger.info(f"XGB best params: {best_xgb_params}")
        
        xgb_model = xgb.XGBRegressor(**best_xgb_params, random_state=42, n_jobs=-1, verbosity=0)
        xgb_model.fit(X_xgb_tr, y_tr.values)
        pred_xgb = xgb_model.predict(X_xgb_val)
        oof_xgb[valid_idx] = pred_xgb
        
        rmse_xgb = np.sqrt(np.mean((y_val.values - pred_xgb) ** 2))
        logger.info(f"XGB RMSE: {rmse_xgb:.6f}")
        
        fold_reports.append({
            "fold": fold,
            "rmse_et": float(rmse_et),
            "rmse_xgb": float(rmse_xgb),
            "xgb_feature_count": X_xgb_tr.shape[1],
            "et_feature_count": len(et_selected),
        })
    
    # ===== Correlation Analysis =====
    logger.info(f"\n{'='*60}\nCORRELATION ANALYSIS\n{'='*60}")
    
    # Flatten for overall correlation
    oof_et_flat = oof_et.flatten()
    oof_xgb_flat = oof_xgb.flatten()
    y_train_flat = y_train.values.flatten()
    
    # Predictions correlation
    corr_pred = np.corrcoef(oof_et_flat, oof_xgb_flat)[0, 1]
    logger.info(f"corr(pred_et, pred_xgb) = {corr_pred:.6f}")
    
    # Errors
    err_et = y_train_flat - oof_et_flat
    err_xgb = y_train_flat - oof_xgb_flat
    
    corr_err = np.corrcoef(err_et, err_xgb)[0, 1]
    logger.info(f"corr(err_et, err_xgb) = {corr_err:.6f}")
    
    # Per-target correlations
    logger.info(f"\nPer-target correlations:")
    for i, target in enumerate(schema.model_targets):
        corr_pred_target = np.corrcoef(oof_et[:, i], oof_xgb[:, i])[0, 1]
        err_et_target = y_train.iloc[:, i].values - oof_et[:, i]
        err_xgb_target = y_train.iloc[:, i].values - oof_xgb[:, i]
        corr_err_target = np.corrcoef(err_et_target, err_xgb_target)[0, 1]
        logger.info(f"  {target}: corr_pred={corr_pred_target:.4f}, corr_err={corr_err_target:.4f}")
    
    # CV Summary
    logger.info(f"\n{'='*60}\nCV SUMMARY\n{'='*60}")
    rmses_et = [r["rmse_et"] for r in fold_reports]
    rmses_xgb = [r["rmse_xgb"] for r in fold_reports]
    logger.info(f"ET mean RMSE: {np.mean(rmses_et):.6f} ± {np.std(rmses_et):.6f}")
    logger.info(f"XGB mean RMSE: {np.mean(rmses_xgb):.6f} ± {np.std(rmses_xgb):.6f}")
    
    # Save report
    report = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "experiment": "et_xgb_correlation_analysis",
        "cv_folds": args.cv_folds,
        "correlation": {
            "pred_et_vs_pred_xgb": float(corr_pred),
            "err_et_vs_err_xgb": float(corr_err),
        },
        "fold_reports": fold_reports,
        "summary": {
            "et_mean_rmse": float(np.mean(rmses_et)),
            "et_std_rmse": float(np.std(rmses_et)),
            "xgb_mean_rmse": float(np.mean(rmses_xgb)),
            "xgb_std_rmse": float(np.std(rmses_xgb)),
        }
    }
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"et_xgb_correlation_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nReport saved: {json_path}")


if __name__ == "__main__":
    main()
