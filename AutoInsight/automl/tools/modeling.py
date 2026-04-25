"""
Model training and evaluation utilities used by the Modeling Agent.

Trains multiple sklearn estimators, evaluates via cross-validation,
and returns ranked results with the best model's feature importance.

Key improvements over v1
-------------------------
- SVC and LogisticRegression are wrapped in Pipeline(StandardScaler, model).
  Without scaling these models perform poorly (SVC especially degrades heavily).
- GradientBoosting / RandomForest are tree-based and scale-invariant — no scaler needed.
- LinearRegression / Ridge are wrapped with a scaler as a best practice.
- The scaler is applied inside the pipeline so it is fit only on training folds
  during cross-validation, preventing data leakage.
- Feature importance extraction handles Pipeline-wrapped models transparently.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger("autoinsight.tools.modeling")
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Model registries
# ---------------------------------------------------------------------------

def _classification_models() -> list[tuple[str, Any]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC

    return [
        (
            "LogisticRegression",
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)),
            ]),
        ),
        (
            "RandomForestClassifier",
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        ),
        (
            "GradientBoostingClassifier",
            GradientBoostingClassifier(n_estimators=100, random_state=42),
        ),
        (
            "SVC",
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVC(probability=True, random_state=42)),
            ]),
        ),
    ]


def _regression_models() -> list[tuple[str, Any]]:
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.svm import SVR

    return [
        (
            "LinearRegression",
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", LinearRegression(n_jobs=-1)),
            ]),
        ),
        (
            "Ridge",
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=42)),
            ]),
        ),
        (
            "RandomForestRegressor",
            RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        ),
        (
            "GradientBoostingRegressor",
            GradientBoostingRegressor(n_estimators=100, random_state=42),
        ),
        (
            "SVR",
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf")),
            ]),
        ),
    ]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

SKLEARN_METRIC_MAP = {
    # Classification
    "accuracy":  "accuracy",
    "f1":        "f1_weighted",
    "roc_auc":   "roc_auc_ovr_weighted",
    "precision": "precision_weighted",
    "recall":    "recall_weighted",
    # Regression
    "rmse": "neg_root_mean_squared_error",
    "mae":  "neg_mean_absolute_error",
    "r2":   "r2",
    "mape": "neg_mean_absolute_percentage_error",
}

# Metrics where a higher value is better (regression error metrics are excluded)
HIGHER_IS_BETTER = {"accuracy", "f1", "roc_auc", "precision", "recall", "r2"}


def _sklearn_scoring(metric: str) -> str:
    return SKLEARN_METRIC_MAP.get(metric, "r2")


def _adjust_score(raw: float, metric: str) -> float:
    """sklearn negates error metrics — flip back to positive."""
    return abs(raw) if metric in ("rmse", "mae", "mape") else raw


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    metric: str,
    cv: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[dict], dict | None]:
    """
    Train all candidate models and return ranked results.

    Parameters
    ----------
    X            : Feature matrix (numeric, no NaN)
    y            : Target series
    task_type    : "classification" | "regression"
    metric       : Evaluation metric key (see SKLEARN_METRIC_MAP)
    cv           : Cross-validation folds
    test_size    : Fraction held out for hold-out evaluation
    random_state : RNG seed

    Returns
    -------
    results : list of dicts — model_name, score, train_score, params
    best    : dict of the top model with feature_importance added
    """
    # Encode string targets for classification
    if task_type == "classification" and y.dtype == object:
        y = pd.Series(LabelEncoder().fit_transform(y), index=y.index)

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if task_type == "classification" else None,
    )

    models = (
        _classification_models() if task_type == "classification"
        else _regression_models()
    )

    scoring = _sklearn_scoring(metric)
    results: list[dict] = []

    for name, model in models:
        try:
            logger.info("Training %s …", name)

            cv_scores = cross_val_score(
                model, X_train, y_train,
                scoring=scoring, cv=cv, n_jobs=-1,
            )
            cv_score = _adjust_score(float(cv_scores.mean()), metric)
            reg_r2_score = None
            reg_r2_train = None

            model.fit(X_train, y_train)

            # In-sample train score — more stable than 2-fold CV on small datasets.
            # If n_samples >= 100, use 3-fold CV for a less optimistic estimate.
            if len(X_train) >= 100:
                train_cv = cross_val_score(
                    model, X_train, y_train, scoring=scoring, cv=3, n_jobs=-1,
                )
                train_score = _adjust_score(float(train_cv.mean()), metric)
            else:
                from sklearn.metrics import get_scorer
                scorer = get_scorer(scoring)
                train_score = _adjust_score(float(scorer(model, X_train, y_train)), metric)

            # Always compute R2 for regression visibility, even if primary metric is RMSE/MAE/MAPE.
            if task_type == "regression":
                try:
                    r2_cv_scores = cross_val_score(
                        model, X_train, y_train, scoring="r2", cv=cv, n_jobs=-1,
                    )
                    reg_r2_score = float(r2_cv_scores.mean())
                    if len(X_train) >= 100:
                        r2_train_cv = cross_val_score(
                            model, X_train, y_train, scoring="r2", cv=3, n_jobs=-1,
                        )
                        reg_r2_train = float(r2_train_cv.mean())
                    else:
                        from sklearn.metrics import r2_score
                        reg_r2_train = float(r2_score(y_train, model.predict(X_train)))
                except Exception as exc:
                    logger.warning("Could not compute auxiliary R2 for %s: %s", name, exc)

            row = {
                "model_name": name,
                "score": round(cv_score, 4),
                "train_score": round(train_score, 4),
                "params": _get_params(model),
                "_fitted_model": model,
                "_feature_names": list(X.columns),
                "_X_train": X_train,
                "_y_train": y_train,
            }
            if task_type == "regression" and reg_r2_score is not None:
                row["r2_score"] = round(reg_r2_score, 4)
            if task_type == "regression" and reg_r2_train is not None:
                row["r2_train_score"] = round(reg_r2_train, 4)

            results.append(row)
            logger.info("  %s -> %s: %.4f", name, metric, cv_score)

        except Exception as exc:
            logger.warning("Model %s failed: %s", name, exc)

    if not results:
        return [], None

    reverse = metric in HIGHER_IS_BETTER
    results_sorted = sorted(results, key=lambda r: r["score"], reverse=reverse)
    best_raw = results_sorted[0]

    feat_importance = _get_feature_importance(
        best_raw["_fitted_model"],
        best_raw["_feature_names"],
        X_train=best_raw.get("_X_train"),
        y_train=best_raw.get("_y_train"),
    )

    best = {
        "model_name": best_raw["model_name"],
        "score": best_raw["score"],
        "train_score": best_raw["train_score"],
        "params": best_raw["params"],
        "feature_importance": feat_importance,
        "_fitted_model": best_raw["_fitted_model"],
    }
    if "r2_score" in best_raw:
        best["r2_score"] = best_raw["r2_score"]
    if "r2_train_score" in best_raw:
        best["r2_train_score"] = best_raw["r2_train_score"]

    # Strip internal keys before returning
    clean_results = [
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in results_sorted
    ]

    return clean_results, best


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_params(model: Any) -> dict:
    """Get params from a model or the last step of a Pipeline."""
    if isinstance(model, Pipeline):
        return model.steps[-1][1].get_params()
    return model.get_params()


def _get_feature_importance(
    model: Any,
    feature_names: list[str],
    X_train: pd.DataFrame | None = None,
    y_train: pd.Series | None = None,
) -> dict:
    """
    Extract feature importances from a fitted model or Pipeline.

    Priority
    --------
    1. tree  : feature_importances_  (RandomForest, GradientBoosting, etc.)
    2. linear: normalized |coef_|    (LogisticRegression, Ridge, etc.)
    3. fallback: permutation importance on X_train/y_train
       Works for ANY model type including kernel SVM and KNN.
       Uses 5 repeats for stability; requires X_train and y_train.
    """
    estimator = model.steps[-1][1] if isinstance(model, Pipeline) else model

    try:
        if hasattr(estimator, "feature_importances_"):
            imps = estimator.feature_importances_
            return dict(zip(feature_names, [round(float(v), 4) for v in imps]))

        if hasattr(estimator, "coef_"):
            coef = estimator.coef_
            if coef.ndim > 1:
                coef = np.abs(coef).mean(axis=0)
            coef = np.abs(coef).astype(float)
            total = coef.sum()
            if total > 0:
                coef = coef / total
            return dict(zip(feature_names, [round(float(v), 4) for v in coef]))

    except Exception as exc:
        logger.warning("Native feature importance extraction failed: %s", exc)

    # Fallback: permutation importance — model-agnostic, works for any estimator
    if X_train is not None and y_train is not None:
        try:
            from sklearn.inspection import permutation_importance

            logger.info(
                "Computing permutation importance for %s (this may take a moment)...",
                type(estimator).__name__,
            )
            r = permutation_importance(
                model, X_train, y_train,
                n_repeats=5,
                random_state=42,
                n_jobs=-1,
            )
            imps = r.importances_mean
            # Clip negatives to 0 (can occur if feature is pure noise)
            imps = np.clip(imps, 0, None)
            total = imps.sum()
            if total > 0:
                imps = imps / total
            return dict(zip(feature_names, [round(float(v), 4) for v in imps]))
        except Exception as exc:
            logger.warning("Permutation importance failed: %s", exc)

    return {}