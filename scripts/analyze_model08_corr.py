from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def _load_model08_module(repo_root: Path):
    script_path = repo_root / "scripts" / "train_blend_et_fixed_xgb_raw_optuna_0.1433.py"
    module_name = "model08_module"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze model 08 ET/XGB correlation and RMSE gain.")
    parser.add_argument(
        "--output-dir",
        default="artifacts_extratrees_corr_optuna/08_blend_et_xgb_raw_best(0.1434)",
        help="Directory where the JSON analysis file is written.",
    )
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    mod = _load_model08_module(repo_root)

    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    et_params_json = repo_root / "artifacts_extratrees_corr_optuna/02_experiments_OPEN/q20_feat45_corr990_cv6_trials24/best_score_actuel.json"
    xgb_params_json = output_dir / "xgb_trial11_best_params.json"
    data_dir = repo_root / "src/odor_competition/data"

    et_params = mod.load_best_params_from_json(et_params_json)
    et_params["random_state"] = args.random_state
    et_params["n_jobs"] = -1

    xgb_params = mod.load_xgb_params_from_json(xgb_params_json)
    _ = float(xgb_params.pop("et_weight", 0.6340876216153135))

    cfg = mod.BlendConfig(
        et_params=et_params,
        et_corr_threshold=0.99,
        et_ratio_eps=1e-3,
        et_signal_quantile=0.20,
        et_max_selected_features=45,
        xgb_max_raw_features=13,
        xgb_n_jobs=6,
        random_state=args.random_state,
    )

    data = mod.load_competition_data(data_dir)
    schema = mod.infer_target_schema(data.y_train)
    X_train_raw = mod.raw_features(data.x_train)
    y_true_model = data.y_train.drop(columns=["ID"]).copy() if "ID" in data.y_train.columns else data.y_train.copy()
    y_true_model = y_true_model[schema.model_targets].copy()

    oof_et = pd.DataFrame(index=X_train_raw.index, columns=y_true_model.columns, dtype=float)
    oof_xgb = pd.DataFrame(index=X_train_raw.index, columns=y_true_model.columns, dtype=float)

    kfold = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
    for fit_idx, valid_idx in kfold.split(X_train_raw):
        X_fit = X_train_raw.iloc[fit_idx]
        X_valid = X_train_raw.iloc[valid_idx]
        y_fit = y_true_model.iloc[fit_idx]

        et_pre = mod.fit_feature_preprocessor(
            X_fit,
            y_fit,
            corr_threshold=cfg.et_corr_threshold,
            ratio_eps=cfg.et_ratio_eps,
            signal_quantile=cfg.et_signal_quantile,
            max_selected_features=cfg.et_max_selected_features,
        )
        X_fit_et = et_pre.transform(X_fit)
        X_valid_et = et_pre.transform(X_valid)

        et_model = mod.make_et_model(cfg.et_params)
        et_model.fit(X_fit_et, y_fit)
        pred_et = pd.DataFrame(et_model.predict(X_valid_et), columns=y_fit.columns, index=X_valid.index)

        xgb_cols = mod.select_xgb_raw_columns(X_fit, y_fit, cfg.xgb_max_raw_features)
        X_fit_xgb = X_fit[xgb_cols].copy()
        X_valid_xgb = X_valid[xgb_cols].copy()

        xgb_model = mod.make_xgb_model(xgb_params, n_jobs=cfg.xgb_n_jobs, random_state=cfg.random_state)
        xgb_model.fit(X_fit_xgb, y_fit)
        pred_xgb = pd.DataFrame(xgb_model.predict(X_valid_xgb), columns=y_fit.columns, index=X_valid.index)

        oof_et.loc[X_valid.index, :] = pred_et
        oof_xgb.loc[X_valid.index, :] = pred_xgb

    pred_et_flat = oof_et.to_numpy(dtype=float).ravel()
    pred_xgb_flat = oof_xgb.to_numpy(dtype=float).ravel()
    y_true_flat = y_true_model.to_numpy(dtype=float).ravel()

    corr_pred = float(np.corrcoef(pred_et_flat, pred_xgb_flat)[0, 1])

    err_et_flat = y_true_flat - pred_et_flat
    err_xgb_flat = y_true_flat - pred_xgb_flat
    corr_err = float(np.corrcoef(err_et_flat, err_xgb_flat)[0, 1])

    blend_flat = 0.7 * pred_et_flat + 0.3 * pred_xgb_flat
    rmse_et = float(np.sqrt(np.mean((y_true_flat - pred_et_flat) ** 2)))
    rmse_blend_70_30 = float(np.sqrt(np.mean((y_true_flat - blend_flat) ** 2)))
    rmse_gain = float(rmse_blend_70_30 - rmse_et)

    result = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "model_folder": "artifacts_extratrees_corr_optuna/08_blend_et_xgb_raw_best(0.1434)",
        "analysis_scope": "OOF predictions reconstructed with model-08 CV settings",
        "cv": {
            "folds": int(args.cv_folds),
            "random_state": int(args.random_state),
        },
        "formulae": {
            "corr_pred": "np.corrcoef(pred_et.flatten(), pred_xgb.flatten())[0,1]",
            "corr_err": "np.corrcoef((y_true - pred_et).flatten(), (y_true - pred_xgb).flatten())[0,1]",
            "blend": "0.7 * pred_et + 0.3 * pred_xgb",
            "rmse_gain": "rmse(blend) - rmse(pred_et)",
        },
        "values": {
            "corr_pred": corr_pred,
            "corr_err": corr_err,
            "rmse_pred_et": rmse_et,
            "rmse_blend_0p7_0p3": rmse_blend_70_30,
            "rmse_gain": rmse_gain,
        },
        "shape": {
            "n_rows": int(y_true_model.shape[0]),
            "n_targets": int(y_true_model.shape[1]),
            "n_values_flattened": int(y_true_flat.size),
        },
    }

    out_file = output_dir / f"model08_corr_err_rmsegain_70_30_{result['generated_at_utc']}.json"
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps({"out_file": str(out_file), "values": result["values"]}, indent=2))


if __name__ == "__main__":
    main()
