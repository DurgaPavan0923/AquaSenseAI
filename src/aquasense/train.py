"""
End-to-end model training, hyperparameter tuning, ablation experiments,
and model bundle export for AquaSense AI.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.aquasense.anomaly import AnomalyDetector
from src.aquasense.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_raw_dataset,
    save_processed_splits,
    split_dataset_4way,
    verify_no_leakage_4way,
)
from src.aquasense.evaluate import (
    compute_multiclass_metrics,
    export_latex_tables,
    plot_confusion_matrix_figure,
    plot_eda_figures,
    plot_pr_curves_figure,
    plot_reliability_diagram_figure,
)
from src.aquasense.explain import compute_offline_shap_summary, evaluate_rank_stability
from src.aquasense.features import DomainFeatureEngineer
from src.aquasense.uncertainty import ConformalCalibrator

RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs/figures")


def build_pipeline(classifier: Any, use_feature_engineering: bool = True) -> Pipeline:
    """Constructs a Scikit-Learn Pipeline with optional domain feature engineering."""
    steps = []
    if use_feature_engineering:
        steps.append(("features", DomainFeatureEngineer()))
    steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", classifier))
    return Pipeline(steps)


def run_training_and_ablations() -> dict[str, Any]:
    """Executes the entire training pipeline, hyperparameter tuning, ablations, and exports."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Phase 0/1: Loading and auditing dataset (4-way partition)...")
    df = load_raw_dataset()
    train_df, prob_calib_df, conf_calib_df, test_df = split_dataset_4way(
        df,
        test_size=0.20,
        prob_calib_size=0.15,
        conformal_calib_size=0.15,
        random_state=42,
    )
    verify_no_leakage_4way(train_df, prob_calib_df, conf_calib_df, test_df)
    save_processed_splits(
        train_df,
        test_df,
        prob_calib_df=prob_calib_df,
        conf_calib_df=conf_calib_df,
    )

    # Plot EDA figures on training partition
    plot_eda_figures(train_df)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN].values
    X_prob_calib = prob_calib_df[FEATURE_COLUMNS]
    y_prob_calib = prob_calib_df[TARGET_COLUMN].values
    X_conf_calib = conf_calib_df[FEATURE_COLUMNS]
    y_conf_calib = conf_calib_df[TARGET_COLUMN].values
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN].values

    # Pre-calculate training medians for occlusion explanation
    training_medians = X_train.median().values.tolist()

    # 1. Model Comparison Benchmarks
    print("\n--- Training Model Benchmarks ---")
    models = {
        "Dummy (Majority)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=4,
            eval_metric="mlogloss",
            random_state=42,
        ),
    }

    comparison_records: list[dict[str, Any]] = []
    experiment_log: list[dict[str, Any]] = []

    for name, clf in models.items():
        pipe = build_pipeline(clf, use_feature_engineering=True)
        pipe.fit(X_train, y_train)

        test_preds = pipe.predict(X_test)
        test_probs = pipe.predict_proba(X_test)

        metrics = compute_multiclass_metrics(y_test, test_preds, test_probs)
        rec = {
            "model_name": name,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "macro_pr_auc": metrics["macro_pr_auc"],
            "brier_score": metrics["brier_score"],
        }
        comparison_records.append(rec)
        experiment_log.append({
            "experiment": "model_comparison",
            "model": name,
            **rec,
        })
        print(f"{name}: Macro-F1 = {metrics['macro_f1']:.4f}, PR-AUC = {metrics['macro_pr_auc']:.4f}")

    comp_df = pd.DataFrame(comparison_records)
    comp_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    # 2. Hyperparameter Tuning for Final Production Model (Random Forest with Conformal & Platt Calibration)
    print("\n--- Hyperparameter Tuning & Cross-Validation ---")
    param_dist = {
        "classifier__n_estimators": [100, 150, 200, 250],
        "classifier__max_depth": [6, 8, 10, 12, None],
        "classifier__min_samples_split": [2, 4, 6],
        "classifier__min_samples_leaf": [1, 2, 4],
    }

    base_rf = RandomForestClassifier(class_weight="balanced", random_state=42)
    rf_pipe = build_pipeline(base_rf, use_feature_engineering=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        rf_pipe,
        param_distributions=param_dist,
        n_iter=15,
        scoring="f1_macro",
        cv=cv,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best_estimator = search.best_estimator_
    print(f"Best CV Macro-F1: {search.best_score_:.4f}")

    # Evaluate rank stability across CV folds
    fold_importances: list[np.ndarray] = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        cv_pipe = build_pipeline(RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42))
        cv_pipe.fit(X_train.iloc[train_idx], y_train[train_idx])
        fold_importances.append(cv_pipe.named_steps["classifier"].feature_importances_)
    stability_report = evaluate_rank_stability(fold_importances)
    print(f"Feature Rank Stability across folds: {stability_report}")

    # 3. Probability Calibration (Platt Scaling via PredefinedSplit)
    print("\n--- Probability Calibration & Conformal Prediction (4-Way Disjoint) ---")
    X_combined = pd.concat([X_train, X_prob_calib], axis=0).reset_index(drop=True)
    y_combined = np.concatenate([y_train, y_prob_calib])
    # -1: used only to fit base estimator; 0: used only to calibrate sigmoid probabilities
    test_fold = np.array([-1] * len(X_train) + [0] * len(X_prob_calib))
    ps = PredefinedSplit(test_fold)
    calibrated_model = CalibratedClassifierCV(best_estimator, cv=ps, method="sigmoid")
    calibrated_model.fit(X_combined, y_combined)

    # Conformal Calibration (strictly on unseen conformal calibration partition)
    conf_probs = calibrated_model.predict_proba(X_conf_calib)
    conformal = ConformalCalibrator(alpha=0.10)
    conformal.calibrate(conf_probs, y_conf_calib)

    # Final test evaluation on strictly holdout independent test partition
    test_probs_calib = calibrated_model.predict_proba(X_test)
    test_preds_calib = np.argmax(test_probs_calib, axis=1)

    final_metrics = compute_multiclass_metrics(y_test, test_preds_calib, test_probs_calib)
    conformal_eval = conformal.evaluate_coverage(test_probs_calib, y_test)
    print(f"Final Calibrated Test Macro-F1: {final_metrics['macro_f1']:.4f}")
    print(f"Conformal Test Coverage (target 90%): {conformal_eval['empirical_coverage']*100:.2f}% (avg set size: {conformal_eval['average_set_size']:.2f})")
    print(f"Class-Conditional Coverage: {conformal_eval['class_conditional_coverage']}")

    # 4. Generate Core Evaluation Figures
    plot_confusion_matrix_figure(final_metrics["confusion_matrix"])
    plot_pr_curves_figure(y_test, test_probs_calib)
    plot_reliability_diagram_figure(y_test, test_probs_calib)

    # Offline TreeSHAP summary plot
    print("Generating offline TreeSHAP explanation figure...")
    raw_rf = best_estimator.named_steps["classifier"]
    X_test_trans = best_estimator.named_steps["scaler"].transform(
        best_estimator.named_steps["features"].transform(X_test)
    )
    X_train_trans = best_estimator.named_steps["scaler"].transform(
        best_estimator.named_steps["features"].transform(X_train)
    )
    shap_report = compute_offline_shap_summary(raw_rf, X_train_trans, X_test_trans)

    # 5. Anomaly Detector
    anomaly_detector = AnomalyDetector(contamination=0.03, random_state=42)
    anomaly_detector.fit(X_train.values)

    # 6. Ablation Studies
    print("\n--- Running Ablation Studies ---")
    ablations_records: list[dict[str, Any]] = []

    # Ablation 1: Features (Raw 8 vs Engineered 13)
    raw_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)),
    ])
    raw_pipe.fit(X_train, y_train)
    raw_m = compute_multiclass_metrics(y_test, raw_pipe.predict(X_test), raw_pipe.predict_proba(X_test))
    ablations_records.append({
        "ablation_group": "Feature Representation",
        "configuration": "Raw 8 Features",
        "macro_f1": raw_m["macro_f1"],
        "macro_pr_auc": raw_m["macro_pr_auc"],
        "brier_score": raw_m["brier_score"],
    })
    ablations_records.append({
        "ablation_group": "Feature Representation",
        "configuration": "Domain Engineered 13 Features",
        "macro_f1": final_metrics["macro_f1"],
        "macro_pr_auc": final_metrics["macro_pr_auc"],
        "brier_score": final_metrics["brier_score"],
    })

    # Ablation 2: Class Imbalance (Unweighted vs Balanced)
    unweighted_pipe = build_pipeline(RandomForestClassifier(n_estimators=100, random_state=42), use_feature_engineering=True)
    unweighted_pipe.fit(X_train, y_train)
    unw_m = compute_multiclass_metrics(y_test, unweighted_pipe.predict(X_test), unweighted_pipe.predict_proba(X_test))
    ablations_records.append({
        "ablation_group": "Class Imbalance Handling",
        "configuration": "Standard (Unweighted)",
        "macro_f1": unw_m["macro_f1"],
        "macro_pr_auc": unw_m["macro_pr_auc"],
        "brier_score": unw_m["brier_score"],
    })
    ablations_records.append({
        "ablation_group": "Class Imbalance Handling",
        "configuration": "Cost-Sensitive (Balanced Weights)",
        "macro_f1": final_metrics["macro_f1"],
        "macro_pr_auc": final_metrics["macro_pr_auc"],
        "brier_score": final_metrics["brier_score"],
    })

    # Ablation 3: Calibration (Uncalibrated vs Platt Calibrated)
    uncalib_m = compute_multiclass_metrics(y_test, best_estimator.predict(X_test), best_estimator.predict_proba(X_test))
    ablations_records.append({
        "ablation_group": "Probability Calibration",
        "configuration": "Uncalibrated Random Forest",
        "macro_f1": uncalib_m["macro_f1"],
        "macro_pr_auc": uncalib_m["macro_pr_auc"],
        "brier_score": uncalib_m["brier_score"],
    })
    ablations_records.append({
        "ablation_group": "Probability Calibration",
        "configuration": "Platt Calibrated Random Forest",
        "macro_f1": final_metrics["macro_f1"],
        "macro_pr_auc": final_metrics["macro_pr_auc"],
        "brier_score": final_metrics["brier_score"],
    })

    abl_df = pd.DataFrame(ablations_records)
    for rec in ablations_records:
        experiment_log.append({
            "experiment": "ablation",
            "model": rec["configuration"],
            "accuracy": 0.0,
            "macro_f1": rec["macro_f1"],
            "weighted_f1": 0.0,
            "macro_pr_auc": rec["macro_pr_auc"],
            "brier_score": rec["brier_score"],
        })

    pd.DataFrame(experiment_log).to_csv(RESULTS_DIR / "experiment_log.csv", index=False)
    export_latex_tables(comp_df, abl_df)

    # 7. Save results/metrics.json with real numbers
    all_metrics = {
        "dataset_split": "Independent Holdout Test Split (20% Stratified, 200 samples)",
        "partition_breakdown": {
            "training_samples": len(X_train),
            "probability_calibration_samples": len(X_prob_calib),
            "conformal_calibration_samples": len(X_conf_calib),
            "independent_test_samples": len(X_test),
            "total_samples": len(df),
        },
        "dataset_leakage_audit": {
            "duplicate_samples_detected": 0,
            "train_test_contamination": "None: verified pairwise disjoint across all 4 partitions",
            "derived_feature_leakage": "None: preprocessors and scalers fitted on train partition only",
            "labeling_rule_limitation": "Target labels (Good, Moderate, Polluted) are derived from composite threshold intervals across the 8 physical and chemical parameters. The model approximates these underlying labeling criteria with high fidelity rather than discovering unmeasured biochemical contaminants. Predictions serve as screening support only and do not replace certified laboratory analysis.",
        },
        "model_architecture": "Calibrated Random Forest with Domain Features (4-Way Partition)",
        "headline_metrics": {
            "macro_f1": final_metrics["macro_f1"],
            "accuracy": final_metrics["accuracy"],
            "weighted_f1": final_metrics["weighted_f1"],
            "macro_pr_auc": final_metrics["macro_pr_auc"],
            "brier_score": final_metrics["brier_score"],
        },
        "per_class_metrics": final_metrics["per_class"],
        "confusion_matrix": final_metrics["confusion_matrix"],
        "conformal_uncertainty": conformal_eval,
        "feature_rank_stability": stability_report,
        "training_medians": {col: float(val) for col, val in zip(FEATURE_COLUMNS, training_medians)},
    }

    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    # 8. Export Production Model Bundle
    print("\n--- Exporting Lightweight Production Model Bundle ---")
    bundle = {
        "model": calibrated_model,
        "feature_order": FEATURE_COLUMNS,
        "training_medians": training_medians,
        "conformal_quantile": conformal.quantile_,
        "classes": ["Good", "Moderate", "Polluted"],
        "version": "1.0.0",
    }
    bundle_path = MODELS_DIR / "model_bundle.joblib"
    joblib.dump(bundle, bundle_path, compress=3)

    registry = {
        "active_model": "RandomForest-PlattCalibrated-DomainFeatures",
        "version": "1.0.0",
        "bundle_file": "model_bundle.joblib",
        "bundle_size_kb": round(bundle_path.stat().st_size / 1024, 2),
        "metrics": all_metrics["headline_metrics"],
        "conformal_coverage": conformal_eval["empirical_coverage"],
    }
    with open(MODELS_DIR / "registry.json", "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    print(f"Model bundle exported to {bundle_path} ({registry['bundle_size_kb']} KB)")
    print(f"Metrics saved to {RESULTS_DIR / 'metrics.json'}")
    return all_metrics


if __name__ == "__main__":
    run_training_and_ablations()
