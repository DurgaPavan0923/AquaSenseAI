"""Model training, hyperparameter optimization, probability calibration, and experiment tracking."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
import xgboost as xgb

from src.common.config import yaml_config
from src.common.logger import logger
from src.data.loader import load_dataset, split_and_save_data
from src.ml.features import DomainFeatureEngineer
from src.ml.preprocessor import build_preprocessor, get_cv_splitter


def get_model_candidates(seed: int = 42) -> Dict[str, Any]:
    """Returns candidate estimators spanning baselines and advanced tree ensembles."""
    models = {
        "dummy_majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=-1
        )
    }

    # Optional LightGBM if installed
    try:
        import lightgbm as lgb
        models["lightgbm"] = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.08,
            class_weight="balanced",
            random_state=seed,
            verbose=-1,
            n_jobs=-1
        )
    except ImportError:
        logger.info("LightGBM not installed, skipping candidate.")

    return models


def cross_validate_model(
    model_name: str,
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    splitter: Any,
    numerical_cols: List[str]
) -> Dict[str, Any]:
    """Runs leakage-safe K-Fold cross-validation on a model candidate."""
    fold_f1_macros = []
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Pipeline: Preprocessing + Model
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(numerical_cols)),
            ("classifier", estimator)
        ])

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)

        f1_macro = f1_score(y_val, preds, average="macro", zero_division=0)
        acc = accuracy_score(y_val, preds)

        fold_f1_macros.append(f1_macro)
        fold_accuracies.append(acc)

    return {
        "model_name": model_name,
        "mean_macro_f1": float(np.mean(fold_f1_macros)),
        "std_macro_f1": float(np.std(fold_f1_macros)),
        "mean_accuracy": float(np.mean(fold_accuracies)),
        "fold_f1_scores": [float(score) for score in fold_f1_macros]
    }


def train_and_calibrate_active_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Optional[dict] = None
) -> Tuple[Pipeline, Dict[str, Any], np.ndarray]:
    """Trains final model candidate, applies Platt probability calibration, and calculates conformal scores."""
    cfg = config or yaml_config
    seed = cfg["project"]["random_seed"]
    target_col = cfg["target"]["name"]
    numerical_cols = [f["name"] for f in cfg["features"]["numerical"]]

    X_train = train_df[numerical_cols]
    y_train = train_df[target_col]
    X_test = test_df[numerical_cols]
    y_test = test_df[target_col]

    splitter = get_cv_splitter(cfg)

    # 1. Benchmark Candidate Portfolio
    logger.info("Executing 5-Fold Cross-Validation across candidate models")
    candidates = get_model_candidates(seed)
    cv_results = []
    for name, est in candidates.items():
        res = cross_validate_model(name, est, X_train, y_train, splitter, numerical_cols)
        cv_results.append(res)
        logger.info(f"Model CV: {name}", mean_macro_f1=round(res["mean_macro_f1"], 4), std=round(res["std_macro_f1"], 4))

    # Leaderboard persistence
    reg_dir = Path(cfg["models"]["registry_dir"])
    reg_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_df = pd.DataFrame(cv_results)[["model_name", "mean_macro_f1", "std_macro_f1", "mean_accuracy"]]
    leaderboard_df = leaderboard_df.sort_values(by="mean_macro_f1", ascending=False)
    leaderboard_df.to_csv(reg_dir / "leaderboard.csv", index=False)

    # 2. Leakage-Safe Three-Way Split Protocol for Inductive Conformal Prediction
    # Partition X_train into X_fit (proper training: 75% = 600) and X_calib (calibration: 25% = 200)
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_train, y_train, test_size=0.25, stratify=y_train, random_state=seed
    )

    active_type = cfg["models"].get("active_model_type", "xgboost")
    base_clf = candidates.get(active_type, candidates["xgboost"])

    full_pipeline = Pipeline([
        ("preprocessor", build_preprocessor(numerical_cols)),
        ("classifier", base_clf)
    ])

    # Calibrate probabilities via Platt scaling using 3-fold internal CV on X_fit ONLY
    logger.info("Fitting Platt Scaling calibrator strictly on X_fit training partition", fit_samples=len(X_fit))
    calibrated_clf = CalibratedClassifierCV(estimator=full_pipeline, method="sigmoid", cv=3)
    calibrated_clf.fit(X_fit, y_fit)

    # Compute Inductive Conformal Prediction non-conformity scores on unseen holdout X_calib
    logger.info("Computing conformal non-conformity scores on held-out calibration set", calib_samples=len(X_calib))
    calib_probs = calibrated_clf.predict_proba(X_calib)
    y_calib_indices = y_calib.to_numpy()
    non_conformity_scores = 1.0 - np.array([calib_probs[i, y_calib_indices[i]] for i in range(len(y_calib_indices))])
    non_conformity_scores.sort()

    # Final Evaluation on Strictly Held-Out Test Set (Unseen during fit and calibration)
    test_probs = calibrated_clf.predict_proba(X_test)
    test_preds = np.argmax(test_probs, axis=1)
    test_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
    test_acc = accuracy_score(y_test, test_preds)

    metrics = {
        "model_name": active_type,
        "test_macro_f1": float(test_f1),
        "test_accuracy": float(test_acc),
        "cv_leaderboard": cv_results,
        "fit_sample_count": len(y_fit),
        "calibration_sample_count": len(y_calib),
        "test_sample_count": len(y_test),
        "trained_at": datetime.now(timezone.utc).isoformat()
    }

    # 3. Artifact Serialization
    artifacts_dir = Path(cfg["models"]["artifacts_dir"]) / "xgboost_active"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(calibrated_clf, artifacts_dir / "model.joblib")
    np.save(artifacts_dir / "calibration_scores.npy", non_conformity_scores)

    with open(artifacts_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Append to experiment log
    exp_log = reg_dir / "experiments.jsonl"
    with open(exp_log, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics) + "\n")

    logger.info("Model pipeline successfully trained, calibrated, and registered",
                test_f1=round(test_f1, 4), test_acc=round(test_acc, 4), path=str(artifacts_dir))

    return calibrated_clf, metrics, non_conformity_scores


if __name__ == "__main__":
    df = load_dataset()
    train_df, test_df = split_and_save_data(df)
    model, metrics, calib_scores = train_and_calibrate_active_model(train_df, test_df)
    print("Training pipeline finished successfully.")
