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
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

try:
    import optuna
    from optuna.exceptions import TrialPruned
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:  # pragma: no cover
    OPTUNA_AVAILABLE = False


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from odor_competition.data import (  # noqa: E402
    build_submission_frame,
    feature_target_signal,
    infer_target_schema,
    load_competition_data,
    raw_features,
)
from odor_competition.metrics import competition_rmse  # noqa: E402


@dataclass(frozen=True)
class ETPreprocessor:
    selected_columns: list[str]

    def transform(self, features: pd.DataFrame, *, eps: float) -> pd.DataFrame:
        expanded = build_ratio_features(features, eps=eps)
        return expanded[self.selected_columns].copy()


@dataclass(frozen=True)
class XGBPreprocessor:
    selected_columns: list[str]
    dropped_correlated_columns: list[str]

    def transform(self, features: pd.DataFrame, *, eps: float) -> pd.DataFrame:
        expanded = build_xgb_feature_bank(features, eps=eps)
        return expanded[self.selected_columns].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simple stacking: fixed best ExtraTrees + constrained XGB (18 non-correlated features), "
            "Optuna on CV5, correlation analysis, and dynamic blending."
        )
    )
    parser.add_argument("--data-dir", default="src/odor_competition/data")
    parser.add_argument(
        "--et-params-json",
        default="artifacts_extratrees_corr_optuna/02_experiments_OPEN/q20_feat45_corr990_cv6_trials24/best_score_actuel.json",
        help="Path to the provided best ET json summary.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts_extratrees_corr_optuna/21_stack_simple_best_et_xgb18_dynamic",
    )
    parser.add_argument("--report-prefix", default="stack_simple_best_et_xgb18_dynamic")
    parser.add_argument("--submission-prefix", default="stack_simple_best_et_xgb18_dynamic")

    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--et-corr-threshold", type=float, default=0.99)
    parser.add_argument("--et-signal-quantile", type=float, default=0.20)
    parser.add_argument("--et-max-selected-features", type=int, default=45)

    parser.add_argument("--xgb-max-selected-features", type=int, default=18)
    parser.add_argument("--xgb-corr-threshold", type=float, default=0.985)
    parser.add_argument("--xgb-signal-quantile", type=float, default=0.20)
    parser.add_argument("--ratio-eps", type=float, default=1e-3)

    parser.add_argument("--xgb-optuna-trials", type=int, default=15)
    parser.add_argument("--xgb-rmse-cap", type=float, default=0.03)
    parser.add_argument("--skip-submission", action="store_true")

    args = parser.parse_args()

    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be >= 2")
    if args.xgb_max_selected_features != 18:
        raise ValueError("--xgb-max-selected-features must be exactly 18 per project constraint")
    if not 0.0 < args.et_corr_threshold < 1.0:
        raise ValueError("--et-corr-threshold must be in (0, 1)")
    if not 0.0 < args.xgb_corr_threshold < 1.0:
        raise ValueError("--xgb-corr-threshold must be in (0, 1)")
    if not 0.0 < args.xgb_rmse_cap < 1.0:
        raise ValueError("--xgb-rmse-cap must be in (0, 1)")
    if args.xgb_optuna_trials < 1:
        raise ValueError("--xgb-optuna-trials must be >= 1")

    return args


def load_best_params_from_json(json_path: Path) -> dict:
    payload = json.loads(json_path.read_text())
    if "optuna" in payload and isinstance(payload["optuna"], dict) and "best_params" in payload["optuna"]:
        return payload["optuna"]["best_params"]
    if "best_params" in payload:
        return payload["best_params"]
    raise KeyError(f"No best_params found in {json_path}")


def _safe_denominator(values: pd.Series, eps: float) -> np.ndarray:
    raw = values.to_numpy(dtype=float)
    sign = np.where(raw >= 0.0, 1.0, -1.0)
    adjusted = raw + (sign * eps)
    tiny = np.abs(raw) < eps
    adjusted[tiny] = np.where(raw[tiny] >= 0.0, eps, -eps)
    return adjusted


def _signed_log1p(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def build_ratio_features(features: pd.DataFrame, *, eps: float) -> pd.DataFrame:
    base = raw_features(features)
    expanded = base.copy()
    cols = list(base.columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            ratio = base[left].to_numpy(dtype=float) / _safe_denominator(base[right], eps)
            expanded[f"{left}_over_{right}"] = _signed_log1p(ratio)
    return expanded


def build_xgb_feature_bank(features: pd.DataFrame, *, eps: float) -> pd.DataFrame:
    base = raw_features(features)
    engineered = pd.DataFrame(index=base.index)

    engineered["mean_y"] = base[["Y1", "Y2", "Y3"]].mean(axis=1)
    engineered["mean_y_logabs"] = np.log1p(base[["Y1", "Y2", "Y3"]].abs().mean(axis=1))
    engineered["spread_y"] = base[["Y1", "Y2", "Y3"]].max(axis=1) - base[["Y1", "Y2", "Y3"]].min(axis=1)
    engineered["mean_x_core"] = base[["X4", "X5", "X6", "X7"]].mean(axis=1)
    engineered["mean_x_core_logabs"] = np.log1p(base[["X4", "X5", "X6", "X7"]].abs().mean(axis=1))
    engineered["spread_x_core"] = base[["X4", "X5", "X6", "X7"]].max(axis=1) - base[["X4", "X5", "X6", "X7"]].min(axis=1)
    engineered["mean_x_tail"] = base[["X12", "X13", "X14", "X15"]].mean(axis=1)
    engineered["mean_x_tail_logabs"] = np.log1p(base[["X12", "X13", "X14", "X15"]].abs().mean(axis=1))
    engineered["spread_x_tail"] = base[["X12", "X13", "X14", "X15"]].max(axis=1) - base[["X12", "X13", "X14", "X15"]].min(axis=1)
    engineered["logabs_y1"] = _signed_log1p(base["Y1"].to_numpy(dtype=float))
    engineered["logabs_z"] = _signed_log1p(base["Z"].to_numpy(dtype=float))

    ratio_pairs = [
        ("Y1", "Y2"),
        ("Z", "Y1"),
        ("Y1", "Y3"),
        ("Z", "Y2"),
        ("Z", "Y3"),
        ("X4", "Y3"),
        ("X4", "Y1"),
        ("X6", "Y3"),
        ("X7", "Y3"),
        ("X6", "Y2"),
        ("X5", "Y3"),
        ("X14", "Y3"),
    ]
    for left, right in ratio_pairs:
        engineered[f"log_ratio_{left}_over_{right}"] = _signed_log1p(
            base[left].to_numpy(dtype=float) / _safe_denominator(base[right], eps)
        )

    engineered["log_ratio_y1_over_y2"] = _signed_log1p(
        base["Y1"].to_numpy(dtype=float) / _safe_denominator(base["Y2"], eps)
    )
    engineered["log_ratio_z_over_y1"] = _signed_log1p(
        base["Z"].to_numpy(dtype=float) / _safe_denominator(base["Y1"], eps)
    )
    engineered["log_ratio_x6_over_x4"] = _signed_log1p(
        base["X6"].to_numpy(dtype=float) / _safe_denominator(base["X4"], eps)
    )
    engineered["log_ratio_y_mean_over_x_core_mean"] = _signed_log1p(
        engineered["mean_y"].to_numpy(dtype=float) / (np.abs(engineered["mean_x_core"].to_numpy(dtype=float)) + eps)
    )
    engineered["log_ratio_x_core_mean_over_x_tail_mean"] = _signed_log1p(
        engineered["mean_x_core"].to_numpy(dtype=float) / (np.abs(engineered["mean_x_tail"].to_numpy(dtype=float)) + eps)
    )

    # 13 raw + engineered candidates -> signal + correlation pruning to exactly 18.
    return pd.concat([base, engineered], axis=1)


def _prune_correlated_by_order(
    features: pd.DataFrame,
    ordered_columns: list[str],
    threshold: float,
) -> tuple[list[str], list[str]]:
    corr = features[ordered_columns].corr().abs().fillna(0.0)
    kept: list[str] = []
    dropped: list[str] = []
    for col in ordered_columns:
        if any(corr.loc[col, k] >= threshold for k in kept):
            dropped.append(col)
        else:
            kept.append(col)
    return kept, dropped


def fit_et_preprocessor(
    X_fit: pd.DataFrame,
    y_fit_model: pd.DataFrame,
    *,
    eps: float,
    signal_quantile: float,
    corr_threshold: float,
    max_selected_features: int,
) -> ETPreprocessor:
    expanded = build_ratio_features(X_fit, eps=eps)
    signal = feature_target_signal(expanded, y_fit_model)
    min_signal = float(signal.quantile(signal_quantile))
    signal = signal[signal >= min_signal]
    ordered = list(signal.sort_values(ascending=False).index)
    selected, _ = _prune_correlated_by_order(expanded, ordered, corr_threshold)
    selected = selected[:max_selected_features]
    return ETPreprocessor(selected_columns=selected)


def fit_xgb_preprocessor(
    X_fit: pd.DataFrame,
    y_fit_model: pd.DataFrame,
    *,
    eps: float,
    signal_quantile: float,
    corr_threshold: float,
    max_selected_features: int,
) -> XGBPreprocessor:
    candidates = build_xgb_feature_bank(X_fit, eps=eps)
    signal = feature_target_signal(candidates, y_fit_model)
    min_signal = float(signal.quantile(signal_quantile))
    signal = signal[signal >= min_signal]
    if signal.empty:
        signal = feature_target_signal(candidates, y_fit_model)
    ordered = list(signal.sort_values(ascending=False).index)
    selected, dropped = _prune_correlated_by_order(candidates, ordered, corr_threshold)

    # Guarantee exactly 18 selected features under correlation constraints.
    if len(selected) < max_selected_features:
        remaining = [c for c in ordered if c not in selected]
        for col in remaining:
            if len(selected) >= max_selected_features:
                break
            if all(abs(candidates[[col, k]].corr().iloc[0, 1]) < corr_threshold for k in selected):
                selected.append(col)
    selected = selected[:max_selected_features]

    if len(selected) != max_selected_features:
        # Last-resort deterministic fill to honor fixed 18-feature contract.
        remaining = [c for c in candidates.columns if c not in selected]
        selected.extend(remaining[: max_selected_features - len(selected)])
        selected = selected[:max_selected_features]

    return XGBPreprocessor(selected_columns=selected, dropped_correlated_columns=dropped)


def make_xgb_model(params: dict, *, random_state: int) -> MultiOutputRegressor:
    base = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
        **params,
    )
    return MultiOutputRegressor(base, n_jobs=1)


def sanitize_et_params(params: dict, random_state: int) -> dict:
    out = dict(params)
    out["random_state"] = random_state
    out["n_jobs"] = -1
    if "n_estimators" in out:
        out["n_estimators"] = int(out["n_estimators"])
    if "max_depth" in out:
        out["max_depth"] = int(out["max_depth"])
    if "min_samples_split" in out:
        out["min_samples_split"] = int(out["min_samples_split"])
    if "min_samples_leaf" in out:
        out["min_samples_leaf"] = int(out["min_samples_leaf"])
    return out


def optimize_xgb_params_cv(
    X_train_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    y_train_full: pd.DataFrame,
    schema,
    *,
    args: argparse.Namespace,
) -> tuple[dict, float]:
    if not OPTUNA_AVAILABLE:
        raise RuntimeError("Optuna is required for this pipeline.")

    splitter = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)

    def objective(trial: optuna.Trial) -> float:
        xgb_params = {
            "n_estimators": trial.suggest_int("n_estimators", 250, 850, step=25),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.10),
            "subsample": trial.suggest_float("subsample", 0.55, 0.90),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.45, 0.90),
            "min_child_weight": trial.suggest_float("min_child_weight", 4.0, 30.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 8.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 30.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 0.35),
        }

        fold_scores: list[float] = []
        for fold_idx, (fit_idx, valid_idx) in enumerate(splitter.split(X_train_raw), start=1):
            X_fit = X_train_raw.iloc[fit_idx]
            X_valid = X_train_raw.iloc[valid_idx]
            y_fit_model = y_train_model.iloc[fit_idx]
            y_valid_full = y_train_full.iloc[valid_idx]

            pre = fit_xgb_preprocessor(
                X_fit,
                y_fit_model,
                eps=args.ratio_eps,
                signal_quantile=args.xgb_signal_quantile,
                corr_threshold=args.xgb_corr_threshold,
                max_selected_features=args.xgb_max_selected_features,
            )
            X_fit_pre = pre.transform(X_fit, eps=args.ratio_eps)
            X_valid_pre = pre.transform(X_valid, eps=args.ratio_eps)

            model = make_xgb_model(xgb_params, random_state=args.random_state)
            model.fit(X_fit_pre, y_fit_model)

            pred_valid_model = pd.DataFrame(
                model.predict(X_valid_pre),
                columns=y_fit_model.columns,
                index=X_valid_pre.index,
            )
            pred_valid_full = schema.expand_predictions(pred_valid_model)
            rmse = float(competition_rmse(y_valid_full, pred_valid_full))

            fold_scores.append(rmse)
            trial.report(float(np.mean(fold_scores)), step=fold_idx)
            if trial.should_prune():
                raise TrialPruned("Pruned by median rule")

        mean_rmse = float(np.mean(fold_scores))
        over_cap = max(0.0, mean_rmse - float(args.xgb_rmse_cap))
        penalty = over_cap * 25.0
        trial.set_user_attr("mean_rmse", mean_rmse)
        trial.set_user_attr("cap_violation", over_cap)
        return float(mean_rmse + penalty)

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=args.random_state),
        pruner=MedianPruner(n_startup_trials=4, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=args.xgb_optuna_trials, show_progress_bar=True)

    best_params = dict(study.best_params)
    best_params["n_estimators"] = int(best_params["n_estimators"])
    best_params["max_depth"] = int(best_params["max_depth"])
    best_rmse = float(study.best_trial.user_attrs.get("mean_rmse", study.best_value))

    return best_params, best_rmse


def evaluate_cv_stack(
    X_train_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    y_train_full: pd.DataFrame,
    schema,
    *,
    et_params: dict,
    xgb_params: dict,
    args: argparse.Namespace,
) -> tuple[list[dict], dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splitter = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)

    oof_et_model = pd.DataFrame(index=X_train_raw.index, columns=y_train_model.columns, dtype=float)
    oof_xgb_model = pd.DataFrame(index=X_train_raw.index, columns=y_train_model.columns, dtype=float)
    oof_blend_model = pd.DataFrame(index=X_train_raw.index, columns=y_train_model.columns, dtype=float)

    fold_reports: list[dict] = []
    dynamic_weights: list[float] = []

    for fold_idx, (fit_idx, valid_idx) in enumerate(splitter.split(X_train_raw), start=1):
        X_fit = X_train_raw.iloc[fit_idx]
        X_valid = X_train_raw.iloc[valid_idx]
        y_fit_model = y_train_model.iloc[fit_idx]
        y_valid_full = y_train_full.iloc[valid_idx]

        et_pre = fit_et_preprocessor(
            X_fit,
            y_fit_model,
            eps=args.ratio_eps,
            signal_quantile=args.et_signal_quantile,
            corr_threshold=args.et_corr_threshold,
            max_selected_features=args.et_max_selected_features,
        )
        xgb_pre = fit_xgb_preprocessor(
            X_fit,
            y_fit_model,
            eps=args.ratio_eps,
            signal_quantile=args.xgb_signal_quantile,
            corr_threshold=args.xgb_corr_threshold,
            max_selected_features=args.xgb_max_selected_features,
        )

        X_fit_et = et_pre.transform(X_fit, eps=args.ratio_eps)
        X_valid_et = et_pre.transform(X_valid, eps=args.ratio_eps)
        X_fit_xgb = xgb_pre.transform(X_fit, eps=args.ratio_eps)
        X_valid_xgb = xgb_pre.transform(X_valid, eps=args.ratio_eps)

        et_model = ExtraTreesRegressor(**et_params)
        et_model.fit(X_fit_et, y_fit_model)
        pred_et_model = pd.DataFrame(et_model.predict(X_valid_et), columns=y_fit_model.columns, index=X_valid.index)

        xgb_model = make_xgb_model(xgb_params, random_state=args.random_state)
        xgb_model.fit(X_fit_xgb, y_fit_model)
        pred_xgb_model = pd.DataFrame(xgb_model.predict(X_valid_xgb), columns=y_fit_model.columns, index=X_valid.index)

        pred_et_full = schema.expand_predictions(pred_et_model)
        pred_xgb_full = schema.expand_predictions(pred_xgb_model)
        rmse_et = float(competition_rmse(y_valid_full, pred_et_full))
        rmse_xgb = float(competition_rmse(y_valid_full, pred_xgb_full))

        inv_et = 1.0 / max(rmse_et, 1e-12)
        inv_xgb = 1.0 / max(rmse_xgb, 1e-12)
        et_weight = float(inv_et / (inv_et + inv_xgb))
        blend_model = (et_weight * pred_et_model) + ((1.0 - et_weight) * pred_xgb_model)
        blend_full = schema.expand_predictions(blend_model)
        rmse_blend = float(competition_rmse(y_valid_full, blend_full))

        oof_et_model.loc[X_valid.index] = pred_et_model
        oof_xgb_model.loc[X_valid.index] = pred_xgb_model
        oof_blend_model.loc[X_valid.index] = blend_model
        dynamic_weights.append(et_weight)

        fold_reports.append(
            {
                "fold": fold_idx,
                "rmse_et": rmse_et,
                "rmse_xgb": rmse_xgb,
                "rmse_blend_dynamic": rmse_blend,
                "dynamic_et_weight": et_weight,
                "dynamic_xgb_weight": float(1.0 - et_weight),
                "et_feature_count": int(X_fit_et.shape[1]),
                "xgb_feature_count": int(X_fit_xgb.shape[1]),
                "xgb_selected_columns": xgb_pre.selected_columns,
                "xgb_dropped_correlated_count": int(len(xgb_pre.dropped_correlated_columns)),
            }
        )

    et_scores = np.array([r["rmse_et"] for r in fold_reports], dtype=float)
    xgb_scores = np.array([r["rmse_xgb"] for r in fold_reports], dtype=float)
    blend_scores = np.array([r["rmse_blend_dynamic"] for r in fold_reports], dtype=float)

    summary = {
        "et_mean_rmse": float(np.mean(et_scores)),
        "xgb_mean_rmse": float(np.mean(xgb_scores)),
        "blend_dynamic_mean_rmse": float(np.mean(blend_scores)),
        "blend_dynamic_std_rmse": float(np.std(blend_scores)),
        "blend_dynamic_min_rmse": float(np.min(blend_scores)),
        "blend_dynamic_max_rmse": float(np.max(blend_scores)),
        "xgb_rmse_cap": float(args.xgb_rmse_cap),
        "xgb_mean_rmse_under_cap": bool(float(np.mean(xgb_scores)) <= float(args.xgb_rmse_cap)),
        "dynamic_et_weight_mean": float(np.mean(dynamic_weights)),
        "dynamic_et_weight_median": float(np.median(dynamic_weights)),
    }

    return fold_reports, summary, oof_et_model, oof_xgb_model, oof_blend_model


def compute_correlation_analysis(
    y_train_model: pd.DataFrame,
    oof_et_model: pd.DataFrame,
    oof_xgb_model: pd.DataFrame,
) -> dict:
    yv = y_train_model.to_numpy(dtype=float)
    pet = oof_et_model.to_numpy(dtype=float)
    pxgb = oof_xgb_model.to_numpy(dtype=float)

    pred_corr_global = float(np.corrcoef(pet.ravel(), pxgb.ravel())[0, 1])
    err_et = yv - pet
    err_xgb = yv - pxgb
    err_corr_global = float(np.corrcoef(err_et.ravel(), err_xgb.ravel())[0, 1])

    per_target: dict[str, dict[str, float]] = {}
    for i, target in enumerate(y_train_model.columns):
        pred_corr = float(np.corrcoef(pet[:, i], pxgb[:, i])[0, 1])
        err_corr = float(np.corrcoef(err_et[:, i], err_xgb[:, i])[0, 1])
        per_target[target] = {
            "pred_corr_et_vs_xgb": pred_corr,
            "error_corr_et_vs_xgb": err_corr,
        }

    return {
        "global": {
            "pred_corr_et_vs_xgb": pred_corr_global,
            "error_corr_et_vs_xgb": err_corr_global,
        },
        "per_target": per_target,
    }


def fit_full_and_predict(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    y_train_model: pd.DataFrame,
    *,
    et_params: dict,
    xgb_params: dict,
    final_et_weight: float,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict]:
    et_pre = fit_et_preprocessor(
        X_train_raw,
        y_train_model,
        eps=args.ratio_eps,
        signal_quantile=args.et_signal_quantile,
        corr_threshold=args.et_corr_threshold,
        max_selected_features=args.et_max_selected_features,
    )
    xgb_pre = fit_xgb_preprocessor(
        X_train_raw,
        y_train_model,
        eps=args.ratio_eps,
        signal_quantile=args.xgb_signal_quantile,
        corr_threshold=args.xgb_corr_threshold,
        max_selected_features=args.xgb_max_selected_features,
    )

    X_train_et = et_pre.transform(X_train_raw, eps=args.ratio_eps)
    X_test_et = et_pre.transform(X_test_raw, eps=args.ratio_eps)
    X_train_xgb = xgb_pre.transform(X_train_raw, eps=args.ratio_eps)
    X_test_xgb = xgb_pre.transform(X_test_raw, eps=args.ratio_eps)

    et_model = ExtraTreesRegressor(**et_params)
    et_model.fit(X_train_et, y_train_model)
    pred_test_et = pd.DataFrame(et_model.predict(X_test_et), columns=y_train_model.columns, index=X_test_raw.index)

    xgb_model = make_xgb_model(xgb_params, random_state=args.random_state)
    xgb_model.fit(X_train_xgb, y_train_model)
    pred_test_xgb = pd.DataFrame(
        xgb_model.predict(X_test_xgb),
        columns=y_train_model.columns,
        index=X_test_raw.index,
    )

    pred_test_blend = (final_et_weight * pred_test_et) + ((1.0 - final_et_weight) * pred_test_xgb)
    meta = {
        "final_et_weight": float(final_et_weight),
        "final_xgb_weight": float(1.0 - final_et_weight),
        "et_selected_feature_count": int(len(et_pre.selected_columns)),
        "xgb_selected_feature_count": int(len(xgb_pre.selected_columns)),
        "xgb_selected_columns": xgb_pre.selected_columns,
    }
    return pred_test_blend, meta


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()

    et_params_json = Path(args.et_params_json)
    if not et_params_json.is_absolute():
        et_params_json = (ROOT / et_params_json).resolve()

    data = load_competition_data(data_dir)
    schema = infer_target_schema(data.y_train)

    X_train_raw = raw_features(data.x_train)
    X_test_raw = raw_features(data.x_test)
    xgb_candidate_count = int(build_xgb_feature_bank(X_train_raw, eps=args.ratio_eps).shape[1])
    y_train_full = data.y_train.drop(columns=["ID"]).copy() if "ID" in data.y_train.columns else data.y_train.copy()
    y_train_model = y_train_full[schema.model_targets].copy()

    et_params = sanitize_et_params(load_best_params_from_json(et_params_json), args.random_state)
    xgb_params, xgb_optuna_rmse = optimize_xgb_params_cv(
        X_train_raw,
        y_train_model,
        y_train_full,
        schema,
        args=args,
    )

    fold_reports, cv_summary, oof_et_model, oof_xgb_model, oof_blend_model = evaluate_cv_stack(
        X_train_raw,
        y_train_model,
        y_train_full,
        schema,
        et_params=et_params,
        xgb_params=xgb_params,
        args=args,
    )

    corr_analysis = compute_correlation_analysis(y_train_model, oof_et_model, oof_xgb_model)
    final_et_weight = float(cv_summary["dynamic_et_weight_median"])

    pred_test_blend_model, fit_meta = fit_full_and_predict(
        X_train_raw,
        X_test_raw,
        y_train_model,
        et_params=et_params,
        xgb_params=xgb_params,
        final_et_weight=final_et_weight,
        args=args,
    )

    pred_test_full = schema.expand_predictions(pred_test_blend_model)
    submission = build_submission_frame(data.x_test["ID"], pred_test_full)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    oof_et_path = output_dir / f"{args.report_prefix}_{timestamp}_oof_et_model.csv"
    oof_xgb_path = output_dir / f"{args.report_prefix}_{timestamp}_oof_xgb_model.csv"
    oof_blend_path = output_dir / f"{args.report_prefix}_{timestamp}_oof_blend_model.csv"
    oof_et_model.to_csv(oof_et_path, index=False)
    oof_xgb_model.to_csv(oof_xgb_path, index=False)
    oof_blend_model.to_csv(oof_blend_path, index=False)

    submission_path = "skipped"
    if not args.skip_submission:
        submission_file = output_dir / f"{args.submission_prefix}_{timestamp}.csv"
        submission.to_csv(submission_file, index=False)
        submission_path = str(submission_file.relative_to(ROOT))

    summary = {
        "generated_at_utc": timestamp,
        "experiment": "stack_simple_best_et_xgb18_dynamic",
        "model": "Blend(ExtraTrees fixed + constrained XGB)",
        "constraints": {
            "xgb_rmse_cap": float(args.xgb_rmse_cap),
            "xgb_selected_feature_count_required": 18,
            "xgb_cv_folds": int(args.cv_folds),
        },
        "et_source": {
            "params_json": str(et_params_json.relative_to(ROOT)),
            "params": et_params,
        },
        "xgb": {
            "optuna_trials": int(args.xgb_optuna_trials),
            "optuna_best_cv_rmse": float(xgb_optuna_rmse),
            "best_params": xgb_params,
            "feature_pipeline": {
                "candidate_feature_count": xgb_candidate_count,
                "selected_feature_count": int(args.xgb_max_selected_features),
                "corr_threshold": float(args.xgb_corr_threshold),
                "signal_quantile": float(args.xgb_signal_quantile),
            },
        },
        "cv": {
            "fold_reports": fold_reports,
            "summary": cv_summary,
        },
        "correlation_analysis": corr_analysis,
        "dynamic_blender": {
            "final_et_weight": float(final_et_weight),
            "final_xgb_weight": float(1.0 - final_et_weight),
        },
        "artifacts": {
            "oof_et_model": str(oof_et_path.relative_to(ROOT)),
            "oof_xgb_model": str(oof_xgb_path.relative_to(ROOT)),
            "oof_blend_model": str(oof_blend_path.relative_to(ROOT)),
            "submission_path": submission_path,
            "rows_predicted": int(len(submission)) if not args.skip_submission else 0,
        },
        "fit_meta": fit_meta,
    }

    report_path = output_dir / f"{args.report_prefix}_{timestamp}.json"
    report_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
