from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from odor_competition.data import standardize_feature_columns  # noqa: E402


@dataclass(frozen=True)
class AdversarialData:
    x: pd.DataFrame
    y: pd.Series
    origin: pd.Series
    row_id: pd.Series
    numeric_columns: list[str]
    categorical_columns: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adversarial validation train vs test: predict whether a row comes from train or test, "
            "report CV ROC-AUC, feature importance, logistic coefficients, and SVG visualizations."
        )
    )
    parser.add_argument("--data-dir", default="src/odor_competition/data")
    parser.add_argument("--output-dir", default="artifacts_extratrees_corr_optuna/adversarial_validation")
    parser.add_argument("--prefix", default="adversarial_validation")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--model",
        choices=["auto", "xgboost", "extra_trees", "random_forest"],
        default="auto",
        help="auto uses XGBoost if installed, otherwise ExtraTrees.",
    )
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--hist-bins", type=int, default=50)
    parser.add_argument("--pca-sample-size", type=int, default=12000)
    parser.add_argument("--max-rows-per-source", type=int, default=None, help="Optional smoke-test subsample per source.")
    parser.add_argument("--skip-logistic", action="store_true")
    parser.add_argument("--skip-pca", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be >= 2.")
    if args.hist_bins < 5:
        raise ValueError("--hist-bins must be >= 5.")
    return args


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (ROOT / path).resolve()


def safe_stem(raw_value: str) -> str:
    stem = Path(str(raw_value)).name.strip()
    for char in '<>:"/\\|?*':
        stem = stem.replace(char, "_")
    return stem or "adversarial_validation"


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_adversarial_data(
    data_dir: Path,
    *,
    max_rows_per_source: int | None,
    random_state: int,
) -> AdversarialData:
    x_train = standardize_feature_columns(pd.read_csv(data_dir / "X_train.csv"))
    x_test = standardize_feature_columns(pd.read_csv(data_dir / "X_test.csv"))

    if max_rows_per_source is not None:
        x_train = x_train.sample(n=min(max_rows_per_source, len(x_train)), random_state=random_state).sort_index()
        x_test = x_test.sample(n=min(max_rows_per_source, len(x_test)), random_state=random_state + 1).sort_index()

    train_ids = x_train["ID"].copy() if "ID" in x_train.columns else pd.Series(np.arange(len(x_train)), name="ID")
    test_ids = x_test["ID"].copy() if "ID" in x_test.columns else pd.Series(np.arange(len(x_test)), name="ID")

    x_train_features = x_train.drop(columns=["ID"], errors="ignore").copy()
    x_test_features = x_test.drop(columns=["ID"], errors="ignore").copy()

    train_columns = set(x_train_features.columns)
    test_columns = set(x_test_features.columns)
    if train_columns != test_columns:
        missing_train = sorted(test_columns - train_columns)
        missing_test = sorted(train_columns - test_columns)
        raise ValueError(f"Train/test feature columns differ. Missing train={missing_train}, missing test={missing_test}")

    feature_columns = list(x_train_features.columns)
    x_train_features = x_train_features[feature_columns]
    x_test_features = x_test_features[feature_columns]

    x = pd.concat([x_train_features, x_test_features], axis=0, ignore_index=True)
    y = pd.Series(
        np.r_[np.zeros(len(x_train_features), dtype=np.int8), np.ones(len(x_test_features), dtype=np.int8)],
        name="is_test",
    )
    origin = pd.Series(np.where(y.to_numpy() == 1, "test", "train"), name="origin")
    row_id = pd.concat([train_ids, test_ids], axis=0, ignore_index=True).rename("ID")

    categorical_columns = []
    for column in x.columns:
        dtype = x[column].dtype
        if (
            pd.api.types.is_object_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(dtype)
        ):
            categorical_columns.append(column)
    numeric_columns = [column for column in x.columns if column not in categorical_columns]

    return AdversarialData(
        x=x,
        y=y,
        origin=origin,
        row_id=row_id,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )


def make_preprocessor(numeric_columns: list[str], categorical_columns: list[str], *, scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        transformers.append(("num", Pipeline(numeric_steps), numeric_columns))
    if categorical_columns:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical_columns,
            )
        )
    if not transformers:
        raise ValueError("No usable feature columns found.")
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def make_nonlinear_classifier(
    model_name: str,
    *,
    n_estimators: int,
    max_depth: int | None,
    random_state: int,
    n_jobs: int,
    positive_count: int,
    negative_count: int,
):
    if model_name in {"auto", "xgboost"}:
        try:
            from xgboost import XGBClassifier  # type: ignore

            scale_pos_weight = negative_count / max(positive_count, 1)
            return (
                "xgboost",
                XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=6 if max_depth is None else max_depth,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    min_child_weight=3,
                    reg_lambda=1.0,
                    objective="binary:logistic",
                    eval_metric="auc",
                    tree_method="hist",
                    random_state=random_state,
                    n_jobs=n_jobs,
                    scale_pos_weight=scale_pos_weight,
                ),
            )
        except ModuleNotFoundError:
            if model_name == "xgboost":
                raise

    if model_name in {"auto", "extra_trees"}:
        return (
            "extra_trees",
            ExtraTreesClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                random_state=random_state,
                n_jobs=n_jobs,
            ),
        )

    return (
        "random_forest",
        RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=n_jobs,
        ),
    )


def make_pipeline(preprocessor: ColumnTransformer, classifier) -> Pipeline:
    return Pipeline([("preprocess", preprocessor), ("model", classifier)])


def predict_positive_probability(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        decision = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-decision))
    raise TypeError("Model has neither predict_proba nor decision_function.")


def cross_validate_oof(
    pipeline: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    *,
    cv_folds: int,
    random_state: int,
    verbose: bool,
) -> tuple[np.ndarray, pd.DataFrame]:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    oof = np.zeros(len(y), dtype=np.float64)
    fold_rows: list[dict[str, float | int]] = []

    for fold, (fit_idx, valid_idx) in enumerate(cv.split(x, y), start=1):
        fold_model = clone(pipeline)
        fold_model.fit(x.iloc[fit_idx], y.iloc[fit_idx])
        pred = predict_positive_probability(fold_model, x.iloc[valid_idx])
        oof[valid_idx] = pred
        auc = roc_auc_score(y.iloc[valid_idx], pred)
        fold_rows.append(
            {
                "fold": fold,
                "auc_roc": float(auc),
                "fit_rows": int(len(fit_idx)),
                "valid_rows": int(len(valid_idx)),
                "valid_test_rate": float(y.iloc[valid_idx].mean()),
            }
        )
        if verbose:
            log(f"Fold {fold}/{cv_folds}: AUC={auc:.6f}")

    return oof, pd.DataFrame(fold_rows)


def get_transformed_feature_names(pipeline: Pipeline, raw_columns: list[str]) -> list[str]:
    preprocessor = pipeline.named_steps["preprocess"]
    try:
        return list(preprocessor.get_feature_names_out(raw_columns))
    except Exception:
        return list(raw_columns)


def fit_feature_importance(
    pipeline: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
) -> tuple[Pipeline, pd.DataFrame]:
    fitted = clone(pipeline)
    fitted.fit(x, y)
    feature_names = get_transformed_feature_names(fitted, list(x.columns))
    model = fitted.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_, dtype=np.float64)
    elif hasattr(model, "coef_"):
        importance = np.abs(np.asarray(model.coef_, dtype=np.float64)).ravel()
    else:
        importance = np.full(len(feature_names), np.nan, dtype=np.float64)

    frame = pd.DataFrame({"feature": feature_names, "importance": importance})
    frame = frame.sort_values("importance", ascending=False, na_position="last").reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    return fitted, frame[["rank", "feature", "importance"]]


def fit_logistic_coefficients(
    data: AdversarialData,
    *,
    cv_folds: int,
    random_state: int,
    n_jobs: int,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    preprocessor = make_preprocessor(data.numeric_columns, data.categorical_columns, scale_numeric=True)
    classifier = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=3000,
        class_weight="balanced",
        random_state=random_state,
    )
    pipeline = make_pipeline(preprocessor, classifier)
    oof, folds = cross_validate_oof(
        pipeline,
        data.x,
        data.y,
        cv_folds=cv_folds,
        random_state=random_state,
        verbose=False,
    )
    fitted = clone(pipeline)
    fitted.fit(data.x, data.y)
    feature_names = get_transformed_feature_names(fitted, list(data.x.columns))
    coefs = fitted.named_steps["model"].coef_.ravel()
    coef_frame = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefs,
            "abs_coefficient": np.abs(coefs),
            "direction": np.where(coefs >= 0, "test", "train"),
        }
    ).sort_values("abs_coefficient", ascending=False)
    coef_frame.insert(0, "rank", np.arange(1, len(coef_frame) + 1))
    return oof, folds, coef_frame


def interpretation_for_auc(mean_auc: float) -> str:
    if mean_auc < 0.55:
        return "AUC proche de 0.5 : train et test semblent tres proches."
    if mean_auc < 0.70:
        return "AUC moderement au-dessus de 0.5 : leger shift train/test."
    if mean_auc < 0.85:
        return "AUC elevee : shift notable entre train et test."
    return "AUC tres elevee : fort shift entre train et test."


def svg_polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def write_probability_hist_svg(
    path: Path,
    probabilities: np.ndarray,
    y: pd.Series,
    *,
    bin_count: int,
    title: str,
) -> None:
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    train_hist, _ = np.histogram(probabilities[y.to_numpy() == 0], bins=edges, density=True)
    test_hist, _ = np.histogram(probabilities[y.to_numpy() == 1], bins=edges, density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0

    width, height = 1160, 640
    left, right, top, bottom = 84, 36, 72, 78
    plot_w, plot_h = width - left - right, height - top - bottom
    max_y = max(float(train_hist.max()), float(test_hist.max()), 1e-9) * 1.12

    def sx(value: float) -> float:
        return left + value * plot_w

    def sy(value: float) -> float:
        return top + plot_h - (value / max_y) * plot_h

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" font-size="24" font-family="Arial" fill="#111827">{title}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fbfbfb" stroke="#d1d5db"/>',
    ]
    for tick in np.linspace(0, 1, 6):
        x_tick = sx(float(tick))
        elements.append(f'<line x1="{x_tick:.1f}" y1="{top}" x2="{x_tick:.1f}" y2="{top + plot_h}" stroke="#e5e7eb"/>')
        elements.append(f'<text x="{x_tick:.1f}" y="{top + plot_h + 28}" text-anchor="middle" font-size="12" font-family="Arial" fill="#374151">{tick:.1f}</text>')
    for tick in np.linspace(0, max_y, 5):
        y_tick = sy(float(tick))
        elements.append(f'<line x1="{left}" y1="{y_tick:.1f}" x2="{left + plot_w}" y2="{y_tick:.1f}" stroke="#eef2f7"/>')
        elements.append(f'<text x="{left - 12}" y="{y_tick + 4:.1f}" text-anchor="end" font-size="12" font-family="Arial" fill="#374151">{tick:.2f}</text>')

    bin_w = plot_w / bin_count
    for center, value in zip(centers, train_hist):
        x = sx(float(center)) - bin_w * 0.42
        y0 = sy(float(value))
        elements.append(f'<rect x="{x:.1f}" y="{y0:.1f}" width="{bin_w * 0.38:.1f}" height="{top + plot_h - y0:.1f}" fill="#2563eb" opacity="0.38"/>')
    for center, value in zip(centers, test_hist):
        x = sx(float(center)) + bin_w * 0.04
        y0 = sy(float(value))
        elements.append(f'<rect x="{x:.1f}" y="{y0:.1f}" width="{bin_w * 0.38:.1f}" height="{top + plot_h - y0:.1f}" fill="#f97316" opacity="0.42"/>')

    elements.extend(
        [
            f'<text x="{width / 2:.1f}" y="{height - 24}" text-anchor="middle" font-size="15" font-family="Arial" fill="#111827">Predicted probability of test origin</text>',
            f'<text x="28" y="{top + plot_h / 2:.1f}" text-anchor="middle" font-size="15" font-family="Arial" fill="#111827" transform="rotate(-90 28 {top + plot_h / 2:.1f})">Density</text>',
            f'<rect x="{left + plot_w - 180}" y="{top + 20}" width="140" height="62" rx="8" fill="#ffffff" opacity="0.94" stroke="#e5e7eb"/>',
            f'<rect x="{left + plot_w - 162}" y="{top + 38}" width="22" height="12" fill="#2563eb" opacity="0.38"/>',
            f'<text x="{left + plot_w - 132}" y="{top + 49}" font-size="13" font-family="Arial">train rows</text>',
            f'<rect x="{left + plot_w - 162}" y="{top + 62}" width="22" height="12" fill="#f97316" opacity="0.42"/>',
            f'<text x="{left + plot_w - 132}" y="{top + 73}" font-size="13" font-family="Arial">test rows</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(elements), encoding="utf-8")


def write_top_features_svg(path: Path, feature_importance: pd.DataFrame, *, top_k: int, title: str) -> None:
    top = feature_importance.head(top_k).iloc[::-1].copy()
    if top.empty:
        path.write_text("<svg></svg>", encoding="utf-8")
        return

    width = 1180
    row_h = 26
    height = max(520, 120 + row_h * len(top))
    left, right, top_margin, bottom = 330, 42, 72, 38
    plot_w = width - left - right
    max_imp = max(float(top["importance"].max()), 1e-12)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" font-size="24" font-family="Arial" fill="#111827">{title}</text>',
    ]
    for i, (_, row) in enumerate(top.iterrows()):
        y = top_margin + i * row_h
        bar_w = (float(row["importance"]) / max_imp) * plot_w
        label = str(row["feature"])
        if len(label) > 42:
            label = label[:39] + "..."
        elements.append(f'<text x="{left - 14}" y="{y + 17}" text-anchor="end" font-size="12" font-family="Arial" fill="#374151">{label}</text>')
        elements.append(f'<rect x="{left}" y="{y + 3}" width="{bar_w:.1f}" height="18" rx="3" fill="#2563eb" opacity="0.78"/>')
        elements.append(f'<text x="{left + bar_w + 6:.1f}" y="{y + 17}" font-size="12" font-family="Arial" fill="#111827">{float(row["importance"]):.5f}</text>')
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def write_pca_svg(
    path: Path,
    data: AdversarialData,
    *,
    sample_size: int,
    random_state: int,
) -> dict[str, float | int | str]:
    if len(data.numeric_columns) < 2:
        path.write_text("<svg></svg>", encoding="utf-8")
        return {"status": "skipped", "reason": "fewer than 2 numeric columns"}

    rng = np.random.default_rng(random_state)
    train_idx = np.flatnonzero(data.y.to_numpy() == 0)
    test_idx = np.flatnonzero(data.y.to_numpy() == 1)
    per_class = max(1, sample_size // 2)
    train_sample = rng.choice(train_idx, size=min(per_class, len(train_idx)), replace=False)
    test_sample = rng.choice(test_idx, size=min(per_class, len(test_idx)), replace=False)
    idx = np.r_[train_sample, test_sample]
    rng.shuffle(idx)

    x_num = data.x.iloc[idx][data.numeric_columns].copy()
    x_num = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(x_num), columns=data.numeric_columns)
    x_scaled = StandardScaler().fit_transform(x_num)
    coords = PCA(n_components=2, random_state=random_state).fit_transform(x_scaled)
    labels = data.y.iloc[idx].to_numpy()

    width, height = 920, 680
    left, right, top, bottom = 78, 40, 70, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    x_min, x_max = np.percentile(coords[:, 0], [1, 99])
    y_min, y_max = np.percentile(coords[:, 1], [1, 99])
    if x_max <= x_min:
        x_max = x_min + 1e-6
    if y_max <= y_min:
        y_max = y_min + 1e-6

    def sx(value: float) -> float:
        clipped = min(max(value, x_min), x_max)
        return left + ((clipped - x_min) / (x_max - x_min)) * plot_w

    def sy(value: float) -> float:
        clipped = min(max(value, y_min), y_max)
        return top + plot_h - ((clipped - y_min) / (y_max - y_min)) * plot_h

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" font-size="24" font-family="Arial" fill="#111827">PCA exploratory projection: train vs test</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fbfbfb" stroke="#d1d5db"/>',
    ]
    for point, label in zip(coords, labels):
        color = "#f97316" if label == 1 else "#2563eb"
        elements.append(f'<circle cx="{sx(float(point[0])):.1f}" cy="{sy(float(point[1])):.1f}" r="2.2" fill="{color}" opacity="0.36"/>')
    elements.extend(
        [
            f'<text x="{width / 2:.1f}" y="{height - 24}" text-anchor="middle" font-size="15" font-family="Arial">PC1</text>',
            f'<text x="28" y="{top + plot_h / 2:.1f}" text-anchor="middle" font-size="15" font-family="Arial" transform="rotate(-90 28 {top + plot_h / 2:.1f})">PC2</text>',
            f'<rect x="{left + plot_w - 152}" y="{top + 18}" width="118" height="58" rx="8" fill="#ffffff" opacity="0.94" stroke="#e5e7eb"/>',
            f'<circle cx="{left + plot_w - 132}" cy="{top + 38}" r="5" fill="#2563eb" opacity="0.6"/>',
            f'<text x="{left + plot_w - 118}" y="{top + 43}" font-size="13" font-family="Arial">train</text>',
            f'<circle cx="{left + plot_w - 132}" cy="{top + 62}" r="5" fill="#f97316" opacity="0.6"/>',
            f'<text x="{left + plot_w - 118}" y="{top + 67}" font-size="13" font-family="Arial">test</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(elements), encoding="utf-8")
    return {"status": "written", "sample_size": int(len(idx))}


def main() -> None:
    args = parse_args()
    data_dir = resolve_path(args.data_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = safe_stem(args.prefix)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    log("Loading train/test features for adversarial validation")
    data = load_adversarial_data(data_dir, max_rows_per_source=args.max_rows_per_source, random_state=int(args.random_state))
    log(f"Rows: train={(data.y == 0).sum()}, test={(data.y == 1).sum()}, features={data.x.shape[1]}")

    positive_count = int((data.y == 1).sum())
    negative_count = int((data.y == 0).sum())
    model_used, classifier = make_nonlinear_classifier(
        args.model,
        n_estimators=int(args.n_estimators),
        max_depth=args.max_depth,
        random_state=int(args.random_state),
        n_jobs=int(args.n_jobs),
        positive_count=positive_count,
        negative_count=negative_count,
    )
    if args.model in {"auto", "xgboost"} and model_used != "xgboost":
        log("XGBoost is not installed; using ExtraTreesClassifier fallback.")
    log(f"Non-linear adversarial model: {model_used}")

    tree_preprocessor = make_preprocessor(data.numeric_columns, data.categorical_columns, scale_numeric=False)
    nonlinear_pipeline = make_pipeline(tree_preprocessor, classifier)
    oof_prob, fold_scores = cross_validate_oof(
        nonlinear_pipeline,
        data.x,
        data.y,
        cv_folds=int(args.cv_folds),
        random_state=int(args.random_state),
        verbose=bool(args.verbose),
    )
    mean_auc = float(fold_scores["auc_roc"].mean())
    std_auc = float(fold_scores["auc_roc"].std(ddof=1)) if len(fold_scores) > 1 else 0.0
    log(f"Adversarial CV AUC={mean_auc:.6f} +/- {std_auc:.6f}")

    _, feature_importance = fit_feature_importance(nonlinear_pipeline, data.x, data.y)

    logistic_summary: dict[str, Any] | None = None
    logistic_coefficients = pd.DataFrame()
    if not args.skip_logistic:
        log("Fitting logistic regression baseline and coefficients")
        logistic_oof, logistic_folds, logistic_coefficients = fit_logistic_coefficients(
            data,
            cv_folds=int(args.cv_folds),
            random_state=int(args.random_state),
            n_jobs=int(args.n_jobs),
        )
        logistic_summary = {
            "oof_auc": float(roc_auc_score(data.y, logistic_oof)),
            "fold_auc_mean": float(logistic_folds["auc_roc"].mean()),
            "fold_auc_std": float(logistic_folds["auc_roc"].std(ddof=1)) if len(logistic_folds) > 1 else 0.0,
        }

    paths = {
        "summary": output_dir / f"{prefix}_{timestamp}.json",
        "fold_scores": output_dir / f"{prefix}_{timestamp}_fold_scores.csv",
        "oof_predictions": output_dir / f"{prefix}_{timestamp}_oof_predictions.csv",
        "feature_importance": output_dir / f"{prefix}_{timestamp}_feature_importance.csv",
        "logistic_coefficients": output_dir / f"{prefix}_{timestamp}_logistic_coefficients.csv",
        "probability_histogram": output_dir / f"{prefix}_{timestamp}_probability_histogram.svg",
        "top_features_plot": output_dir / f"{prefix}_{timestamp}_top_features.svg",
        "pca_plot": output_dir / f"{prefix}_{timestamp}_pca.svg",
    }

    fold_scores.to_csv(paths["fold_scores"], index=False)
    pd.DataFrame(
        {
            "ID": data.row_id,
            "origin": data.origin,
            "is_test": data.y,
            "prob_test": oof_prob,
        }
    ).to_csv(paths["oof_predictions"], index=False)
    feature_importance.to_csv(paths["feature_importance"], index=False)
    if not logistic_coefficients.empty:
        logistic_coefficients.to_csv(paths["logistic_coefficients"], index=False)

    write_probability_hist_svg(
        paths["probability_histogram"],
        oof_prob,
        data.y,
        bin_count=int(args.hist_bins),
        title="Adversarial validation OOF probabilities",
    )
    write_top_features_svg(
        paths["top_features_plot"],
        feature_importance,
        top_k=int(args.top_k),
        title=f"Top {int(args.top_k)} features for train/test separation",
    )
    pca_info: dict[str, Any] = {"status": "skipped", "reason": "--skip-pca"}
    if not args.skip_pca:
        pca_info = write_pca_svg(
            paths["pca_plot"],
            data,
            sample_size=int(args.pca_sample_size),
            random_state=int(args.random_state),
        )

    top_features = feature_importance.head(int(args.top_k)).to_dict(orient="records")
    summary = {
        "generated_at_utc": timestamp,
        "task": "adversarial_validation_train_vs_test",
        "data_dir": str(data_dir),
        "rows": {
            "train": negative_count,
            "test": positive_count,
            "total": int(len(data.y)),
        },
        "features": {
            "total": int(data.x.shape[1]),
            "numeric": data.numeric_columns,
            "categorical": data.categorical_columns,
        },
        "target_definition": {"train": 0, "test": 1},
        "cv": {
            "type": "StratifiedKFold",
            "folds": int(args.cv_folds),
            "shuffle": True,
            "random_state": int(args.random_state),
        },
        "nonlinear_model": {
            "requested": args.model,
            "used": model_used,
            "n_estimators": int(args.n_estimators),
            "max_depth": args.max_depth,
        },
        "auc": {
            "mean": mean_auc,
            "std": std_auc,
            "oof": float(roc_auc_score(data.y, oof_prob)),
            "interpretation": interpretation_for_auc(mean_auc),
        },
        "logistic_regression": logistic_summary,
        "top_features": top_features,
        "pca": pca_info,
        "artifacts": {key: str(value.relative_to(ROOT)) for key, value in paths.items() if value.exists()},
    }
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log(interpretation_for_auc(mean_auc))
    log(f"Top feature: {feature_importance.iloc[0]['feature']} ({float(feature_importance.iloc[0]['importance']):.6f})")
    log(f"Summary written to {paths['summary']}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
