#!/usr/bin/env python3
"""
Model 15: ET (fixed, robust) + XGB (stable, anti-overfit) with OOF analysis.
- ET: unchanged from best (45 features, corr=0.99, signal=0.2)
- XGB: conservative parameters tuned once on a holdout split
- Features: raw + logabs + 1-2 stable ratios
- Single Optuna optimization for XGB, then fixed params for all CV folds
- OOF generation for correlation analysis
- Blend testing: 0.8/0.2 XGB/ET and 0.7/0.3 XGB/ET
- CV=6
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from odor_competition.data import (
    load_competition_data,
    infer_target_schema,
    compress_targets,
    build_submission_frame,
)
from odor_competition.metrics import competition_rmse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============ Feature Engineering ============

def build_stable_xgb_features(X: pd.DataFrame, raw_cols: List[str], include_logabs: bool = True) -> pd.DataFrame:
    """Build stable XGB features: raw + logabs + 1-2 robust ratios."""
    features = pd.DataFrame(index=X.index)
    
    # Raw features
    for col in raw_cols:
        if col in X.columns:
            features[col] = X[col]
    
    # Log absolute values (stable transformation)
    if include_logabs:
        for col in raw_cols:
            if col not in ['Env'] and col in X.columns:
                features[f"{col}_logabs"] = np.log(np.abs(X[col]) + 1e-8)
    
    # 1-2 stable, non-correlated ratios (robust to zeros via log)
    features["Y1_over_Y2_log"] = np.log(np.abs(X["Y1"]) / (np.abs(X["Y2"]) + 1e-8) + 1e-8)
    features["Z_over_Y1_log"] = np.log(np.abs(X["Z"]) / (np.abs(X["Y1"]) + 1e-8) + 1e-8)
    
    return features

def fit_et_preprocessor(X_train: pd.DataFrame, y_train: pd.DataFrame,
                         corr_threshold: float = 0.99,
                         signal_quantile: float = 0.2,
                         max_features: int = 45) -> List[str]:
    """Select ET features via correlation and signal pruning."""
    selected = X_train.columns.tolist()
    
    # Prune high-correlation pairs
    corr_matrix = X_train[selected].corr().abs()
    to_drop = set()
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            if corr_matrix.iloc[i, j] > corr_threshold:
                to_drop.add(selected[j])
    
    selected = [c for c in selected if c not in to_drop]
    
    # Signal filtering (low variance)
    stds = X_train[selected].std()
    signal_threshold = stds.quantile(signal_quantile)
    selected = [c for c in selected if stds[c] >= signal_threshold]
    
    # Cap at max_features
    selected = selected[:max_features]
    
    logger.info(f"ET: selected {len(selected)} features")
    return selected

# ============ Single Optuna Optimization for XGB ============

def optimize_xgb_params_once(X_train: pd.DataFrame, X_valid: pd.DataFrame,
                               y_train: pd.DataFrame, y_valid: pd.DataFrame,
                               schema,
                               n_trials: int = 15,
                               timeout_sec: int = 900) -> Dict:
    """Optimize XGB params once (not per fold) with anti-overfit focus."""
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 350, 750),
            "max_depth": trial.suggest_int("max_depth", 3, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.025, 0.08),
            "subsample": trial.suggest_float("subsample", 0.45, 0.75),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.45, 0.75),
            "min_child_weight": trial.suggest_float("min_child_weight", 15.0, 40.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.2, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.8, 15.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 0.2),
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train.values if isinstance(y_train, pd.DataFrame) else y_train)
        
        pred_valid = model.predict(X_valid)
        
        # Compute RMSE directly on modeled targets (avoid expand_predictions mismatch)
        rmse = np.sqrt(np.mean(np.square(y_valid.values - pred_valid)))
        
        return rmse
    
    sampler = TPESampler(seed=42)
    pruner = MedianPruner()
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec, show_progress_bar=True)
    
    best = study.best_params.copy()
    best["n_estimators"] = int(best["n_estimators"])
    best["max_depth"] = int(best["max_depth"])
    
    logger.info(f"XGB best params (Optuna): {best}")
    logger.info(f"Best RMSE on holdout: {study.best_value:.10f}")
    
    return best


def tune_xgb_once(
    X_train: pd.DataFrame,
    y_train_model: pd.DataFrame,
    y_train_full: pd.DataFrame,
    schema,
    raw_cols: List[str],
    *,
    n_trials: int,
    timeout_sec: int,
    holdout_fraction: float,
) -> Dict:
    X_fit, X_valid, y_fit_model, y_valid_model, y_fit_full, y_valid_full = train_test_split(
        X_train,
        y_train_model,
        y_train_full,
        test_size=holdout_fraction,
        random_state=42,
    )

    X_fit_xgb = build_stable_xgb_features(X_fit, raw_cols)
    X_valid_xgb = build_stable_xgb_features(X_valid, raw_cols)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 350, 750),
            "max_depth": trial.suggest_int("max_depth", 3, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.025, 0.08),
            "subsample": trial.suggest_float("subsample", 0.45, 0.75),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.45, 0.75),
            "min_child_weight": trial.suggest_float("min_child_weight", 15.0, 40.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.2, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.8, 15.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 0.2),
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }

        model = xgb.XGBRegressor(**params)
        model.fit(X_fit_xgb, y_fit_model.values)
        pred_valid = model.predict(X_valid_xgb)
        pred_valid_full = schema.expand_predictions(pd.DataFrame(pred_valid, columns=y_fit_model.columns, index=X_valid_xgb.index))
        score = float(competition_rmse(y_valid_full, pred_valid_full))
        return score

    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=max(3, n_trials // 3), n_warmup_steps=0)
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec, show_progress_bar=True)

    best = study.best_params.copy()
    best["n_estimators"] = int(best["n_estimators"])
    best["max_depth"] = int(best["max_depth"])

    logger.info(f"XGB best params (Optuna holdout): {best}")
    logger.info(f"Best RMSE on holdout: {study.best_value:.10f}")
    return best

# ============ OOF + Correlation Analysis ============

def run_cv_with_oof(X_train: pd.DataFrame, y_train_model: pd.DataFrame, y_train_full: pd.DataFrame,
                     schema,
                     et_params: Dict, xgb_params: Dict,
                     raw_cols: List[str],
                     cv_folds: int = 6,
                     blend_weights: List[Tuple[float, float]] = None) -> Dict:
    """CV with OOF generation and correlation analysis."""
    
    if blend_weights is None:
        blend_weights = [(0.2, 0.8), (0.3, 0.7)]  # ET weight, XGB weight
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Storage for OOF
    oof_et = np.zeros((len(X_train), len(y_train_model.columns)))
    oof_xgb = np.zeros((len(X_train), len(y_train_model.columns)))
    oof_blend = {f"w{w[0]}_{w[1]}": np.zeros((len(X_train), len(y_train_model.columns))) for w in blend_weights}
    
    fold_reports = []
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train), 1):
        logger.info(f"\n=== Fold {fold}/{cv_folds} ===")
        
        X_tr, X_val = X_train.iloc[train_idx].copy(), X_train.iloc[valid_idx].copy()
        y_tr = y_train_model.iloc[train_idx].copy()
        y_val_model = y_train_model.iloc[valid_idx].copy()
        y_val_full = y_train_full.iloc[valid_idx].copy()
        
        # ET: preprocess and fit
        et_selected = fit_et_preprocessor(X_tr, y_tr)
        
        # Filter out n_jobs and random_state if they exist (to avoid duplicates)
        et_params_clean = {k: v for k, v in et_params.items() if k not in ['n_jobs', 'random_state']}
        
        et_model = ExtraTreesRegressor(**et_params_clean, n_jobs=-1, random_state=42)
        et_model.fit(X_tr[et_selected], y_tr.values)
        pred_et_val = et_model.predict(X_val[et_selected])
        oof_et[valid_idx] = pred_et_val
        
        # XGB: build stable features and fit
        X_xgb_tr = build_stable_xgb_features(X_tr, raw_cols)
        X_xgb_val = build_stable_xgb_features(X_val, raw_cols)
        
        xgb_model = xgb.XGBRegressor(**xgb_params, random_state=42, n_jobs=-1, verbosity=0)
        xgb_model.fit(X_xgb_tr, y_tr.values)
        pred_xgb_val = xgb_model.predict(X_xgb_val)
        oof_xgb[valid_idx] = pred_xgb_val
        
        # Blend predictions
        for et_w, xgb_w in blend_weights:
            blend = et_w * pred_et_val + xgb_w * pred_xgb_val
            oof_blend[f"w{et_w}_{xgb_w}"][valid_idx] = blend
        
        # Compute metrics for this fold
        pred_et_expanded = schema.expand_predictions(pd.DataFrame(pred_et_val, columns=y_tr.columns, index=X_val.index))
        pred_xgb_expanded = schema.expand_predictions(pd.DataFrame(pred_xgb_val, columns=y_tr.columns, index=X_val.index))
        
        rmse_et = competition_rmse(y_val_full.values, pred_et_expanded.values)
        rmse_xgb = competition_rmse(y_val_full.values, pred_xgb_expanded.values)
        
        fold_report = {
            "fold": fold,
            "rmse_et": rmse_et,
            "rmse_xgb": rmse_xgb,
            "rmse_blend": {},
            "et_features": len(et_selected),
            "xgb_features": X_xgb_tr.shape[1],
        }
        
        for et_w, xgb_w in blend_weights:
            blend_expanded = schema.expand_predictions(pd.DataFrame(oof_blend[f"w{et_w}_{xgb_w}"][valid_idx], 
                                                                     columns=y_tr.columns, index=X_val.index))
            rmse = competition_rmse(y_val_full.values, blend_expanded.values)
            fold_report["rmse_blend"][f"w{et_w}_{xgb_w}"] = rmse
        
        fold_reports.append(fold_report)
        
        logger.info(f"  ET RMSE:   {rmse_et:.10f}")
        logger.info(f"  XGB RMSE:  {rmse_xgb:.10f}")
        for et_w, xgb_w in blend_weights:
            logger.info(f"  Blend ({et_w}/{xgb_w}) RMSE: {fold_report['rmse_blend'][f'w{et_w}_{xgb_w}']:.10f}")
    
    # Compute correlations
    oof_et_df = pd.DataFrame(oof_et, columns=y_train_model.columns)
    oof_xgb_df = pd.DataFrame(oof_xgb, columns=y_train_model.columns)
    
    # Errors
    err_et = y_train_model.values - oof_et
    err_xgb = y_train_model.values - oof_xgb
    err_et_df = pd.DataFrame(err_et, columns=y_train_model.columns)
    err_xgb_df = pd.DataFrame(err_xgb, columns=y_train_model.columns)
    
    # Correlation of predictions
    corr_pred = np.mean([oof_et_df[col].corr(oof_xgb_df[col]) for col in y_train_model.columns])
    corr_err = np.mean([err_et_df[col].corr(err_xgb_df[col]) for col in y_train_model.columns])
    
    logger.info(f"\n{'='*60}")
    logger.info(f"OOF Correlation Analysis:")
    logger.info(f"  corr(pred_et, pred_xgb) = {corr_pred:.4f}")
    logger.info(f"  corr(err_et, err_xgb)   = {corr_err:.4f}")
    logger.info(f"{'='*60}\n")
    
    # Summary
    summary = {
        "et_mean_rmse": np.mean([r["rmse_et"] for r in fold_reports]),
        "xgb_mean_rmse": np.mean([r["rmse_xgb"] for r in fold_reports]),
        "corr_pred_et_xgb": corr_pred,
        "corr_err_et_xgb": corr_err,
    }
    
    for et_w, xgb_w in blend_weights:
        key = f"blend_w{et_w}_{xgb_w}_mean_rmse"
        summary[key] = np.mean([r["rmse_blend"][f"w{et_w}_{xgb_w}"] for r in fold_reports])
    
    return {
        "fold_reports": fold_reports,
        "summary": summary,
        "oof_et": oof_et_df,
        "oof_xgb": oof_xgb_df,
        "err_et": err_et_df,
        "err_xgb": err_xgb_df,
    }

# ============ Main ============

def main():
    parser = argparse.ArgumentParser(description="Model 15: ET (fixed) + XGB (stable, anti-overfit) with OOF analysis")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--et-params-json",
                        default="artifacts_extratrees_corr_optuna/02_experiments_OPEN/q20_feat45_corr990_cv6_trials24/best_score_actuel.json")
    parser.add_argument("--xgb-params-json", default=None)
    parser.add_argument("--xgb-optuna-trials", type=int, default=12)
    parser.add_argument("--xgb-optuna-timeout-sec", type=int, default=900)
    parser.add_argument("--xgb-optuna-holdout", type=float, default=0.2)
    parser.add_argument("--output-dir", default="artifacts_extratrees_corr_optuna/015_model_et_xgb_stable_antioverfit_cv6")
    parser.add_argument("--report-prefix", default="model15_et_xgb_conservative")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    
    # Load data
    data_dir = Path(__file__).parent.parent / "src" / "odor_competition" / "data"
    comp_data = load_competition_data(data_dir)
    X_train = comp_data.x_train.drop(columns=["ID"])
    y_df = comp_data.y_train
    
    schema = infer_target_schema(y_df)
    y_train_model = compress_targets(y_df, schema)
    y_train_full = y_df.drop(columns=["ID"]) if "ID" in y_df.columns else y_df.copy()
    
    raw_cols = ["Env", "X12", "X13", "X14", "X15", "X4", "X5", "X6", "X7", "Z", "Y1", "Y2", "Y3"]
    
    logger.info(f"X_train: {X_train.shape}, y_train_model: {y_train_model.shape}, y_train_full: {y_train_full.shape}")
    
    # Load ET params
    with open(args.et_params_json) as f:
        data = json.load(f)
    
    # Extract best_params from nested structure
    if "optuna" in data and "best_params" in data["optuna"]:
        et_params = data["optuna"]["best_params"]
    elif "best_params" in data:
        et_params = data["best_params"]
    else:
        et_params = {}
    
    if not et_params:
        # Fallback to default ET params from best model
        et_params = {
            "n_estimators": 440,
            "max_depth": 20,
            "min_samples_split": 15,
            "min_samples_leaf": 2,
            "max_features": 0.518,
            "bootstrap": True,
            "max_samples": 0.626,
            "random_state": 42,
            "n_jobs": -1,
        }
    logger.info(f"ET params: {et_params}")
    
    # XGB params tuned once on a holdout split, then reused unchanged for every fold.
    if args.xgb_params_json:
        with open(args.xgb_params_json) as f:
            xgb_params = json.load(f)
    else:
        xgb_params = tune_xgb_once(
            X_train,
            y_train_model,
            y_train_full,
            schema,
            raw_cols,
            n_trials=args.xgb_optuna_trials,
            timeout_sec=args.xgb_optuna_timeout_sec,
            holdout_fraction=args.xgb_optuna_holdout,
        )

    logger.info(f"XGB fixed params: {xgb_params}")

    # Step 1: Run CV with OOF and correlations
    logger.info(f"\n{'='*60}\nStep 1: CV={args.cv_folds} with OOF analysis\n{'='*60}")
    cv_results = run_cv_with_oof(X_train, y_train_model, y_train_full, schema,
                                  et_params, xgb_params, raw_cols,
                                  cv_folds=args.cv_folds,
                                  blend_weights=[(0.2, 0.8), (0.3, 0.7)])
    
    # Save outputs
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    
    # JSON report
    report = {
        "generated_at_utc": timestamp,
        "model": "Model15_ET_fixed_XGB_stable_antioverfit",
        "cv_folds": args.cv_folds,
        "et_params": et_params,
        "xgb_params": {k: float(v) if isinstance(v, (np.floating, float)) else v
                       for k, v in xgb_params.items()},
        "fold_reports": cv_results["fold_reports"],
        "summary": cv_results["summary"],
    }
    
    json_path = output_dir / f"{args.report_prefix}_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved report: {json_path}")
    
    # CSV OOF
    oof_et_path = output_dir / f"oof_et_{timestamp}.csv"
    oof_xgb_path = output_dir / f"oof_xgb_{timestamp}.csv"
    err_et_path = output_dir / f"err_et_{timestamp}.csv"
    err_xgb_path = output_dir / f"err_xgb_{timestamp}.csv"
    
    cv_results["oof_et"].to_csv(oof_et_path, index=False)
    cv_results["oof_xgb"].to_csv(oof_xgb_path, index=False)
    cv_results["err_et"].to_csv(err_et_path, index=False)
    cv_results["err_xgb"].to_csv(err_xgb_path, index=False)
    
    logger.info(f"Saved OOF files:")
    logger.info(f"  {oof_et_path}")
    logger.info(f"  {oof_xgb_path}")
    logger.info(f"  {err_et_path}")
    logger.info(f"  {err_xgb_path}")
    
    # Summary
    logger.info(f"\n{'='*60}\nFinal Summary\n{'='*60}")
    for key, val in cv_results["summary"].items():
        logger.info(f"{key}: {val:.10f}" if isinstance(val, float) else f"{key}: {val}")

if __name__ == "__main__":
    main()
