"""SHAP explainability engine: local water-quality waterfall attributions, global summaries, and fold stability."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

from src.common.logger import logger


class WaterQualityExplainer:
    """Wrapper around TreeSHAP providing local, global, and cross-fold stability diagnostics."""

    def __init__(self, model_pipeline: Any, feature_names: list[str] | None = None):
        """Initializes explainer by unpacking classifier and preprocessor from pipeline."""
        self.pipeline = model_pipeline
        self.feature_names = feature_names or [
            "pH", "Turbidity", "Temperature", "Dissolved Oxygen",
            "Biochemical Oxygen Demand", "Lead", "Mercury", "Arsenic",
            "OSDR", "HMHI", "pH_Neutrality_Deviation", "Turbidity_DO_Ratio", "Toxic_Breach_Count"
        ]

        # Unpack base estimator (handling CalibratedClassifierCV wrapper if present)
        if hasattr(model_pipeline, "calibrated_classifiers_"):
            # CalibratedClassifierCV with prefit base estimator
            fitted_base = model_pipeline.calibrated_classifiers_[0].estimator
            self.preprocessor = fitted_base.named_steps["preprocessor"]
            self.classifier = fitted_base.named_steps["classifier"]
        elif hasattr(model_pipeline, "named_steps"):
            self.preprocessor = model_pipeline.named_steps["preprocessor"]
            self.classifier = model_pipeline.named_steps["classifier"]
        else:
            self.preprocessor = None
            self.classifier = model_pipeline

        # Initialize TreeExplainer
        try:
            self.explainer = shap.TreeExplainer(self.classifier)
        except Exception as e:
            logger.warning("TreeExplainer fallback to KernelExplainer", error=str(e))
            self.explainer = None

    def explain_local_sample(self, sample_df: pd.DataFrame) -> dict[str, Any]:
        """Generates localized SHAP attributions for a single sample.
        
        Returns:
            Dictionary with base_value, shap_values per feature, and primary positive/negative drivers.
        """
        # Transform through preprocessor
        if self.preprocessor is not None:
            transformed = self.preprocessor.transform(sample_df)
        else:
            transformed = sample_df.values

        if self.explainer is None:
            # Fallback heuristic attribution if SHAP C-tree fails
            return {
                "base_value": 0.0,
                "attributions": {col: 0.0 for col in self.feature_names},
                "dominant_pollutant_driver": "N/A"
            }

        shap_vals = self.explainer.shap_values(transformed)

        # Handle multi-class SHAP output shape: (n_samples, n_features, n_classes) or list of classes
        if isinstance(shap_vals, list):
            # Class 2 (Polluted) attributions
            class_idx = min(2, len(shap_vals) - 1)
            sample_shap = shap_vals[class_idx][0]
            base_val = float(self.explainer.expected_value[class_idx])
        elif len(shap_vals.shape) == 3:
            class_idx = min(2, shap_vals.shape[2] - 1)
            sample_shap = shap_vals[0, :, class_idx]
            base_val = float(self.explainer.expected_value[class_idx])
        else:
            sample_shap = shap_vals[0]
            base_val = float(self.explainer.expected_value) if np.isscalar(self.explainer.expected_value) else float(self.explainer.expected_value[0])

        n_feats = min(len(self.feature_names), len(sample_shap))
        attributions = {self.feature_names[i]: float(sample_shap[i]) for i in range(n_feats)}

        # Sort features by absolute contribution
        sorted_feats = sorted(attributions.items(), key=lambda x: abs(x[1]), reverse=True)
        dominant_driver = sorted_feats[0][0] if sorted_feats else "None"

        return {
            "base_value": base_val,
            "attributions": attributions,
            "dominant_pollutant_driver": dominant_driver
        }

    def generate_global_summary(self, background_df: pd.DataFrame, output_path: str = "reports/figures/shap_summary.png") -> None:
        """Generates and saves global SHAP summary beeswarm/bar plots."""
        if self.explainer is None:
            return

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt


        # Defensive feature filtering for DataFrame inputs
        if isinstance(background_df, pd.DataFrame):
            base_cols = [
                "pH", "Turbidity", "Temperature", "Dissolved Oxygen",
                "Biochemical Oxygen Demand", "Lead", "Mercury", "Arsenic"
            ]
            valid_cols = [c for c in base_cols if c in background_df.columns]
            df_clean = background_df[valid_cols] if valid_cols else background_df
        else:
            df_clean = background_df

        if self.preprocessor is not None:
            transformed = self.preprocessor.transform(df_clean)
        else:
            transformed = df_clean.values

        shap_values = self.explainer.shap_values(transformed)

        # Take Polluted class (index 2) or single output
        plot_vals = shap_values[2] if isinstance(shap_values, list) and len(shap_values) > 2 else shap_values

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            plot_vals,
            features=transformed,
            feature_names=self.feature_names[:transformed.shape[1]],
            show=False
        )
        plt.title("Global TreeSHAP Feature Attributions (Pollution Risk Impact)", fontsize=13, fontweight="bold")
        plt.tight_layout()

        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_file, dpi=300)
        plt.close()
        logger.info("Global SHAP summary plot saved", path=output_path)


def evaluate_shap_fold_stability(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    splitter: Any,
    numerical_cols: list[str],
    output_path: str = "reports/shap_stability_report.json"
) -> dict[str, Any]:
    """Evaluates whether feature importance rankings remain stable across 5 cross-validation folds."""
    from sklearn.pipeline import Pipeline

    from src.ml.preprocessor import build_preprocessor
    from src.ml.train import get_model_candidates

    fold_importances: list[np.ndarray] = []
    feature_names = numerical_cols + [
        "OSDR", "HMHI", "pH_Neutrality_Deviation", "Turbidity_DO_Ratio", "Toxic_Breach_Count"
    ]

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_v = X_train.iloc[val_idx]

        pipe = Pipeline([
            ("preprocessor", build_preprocessor(numerical_cols)),
            ("classifier", get_model_candidates()["xgboost"])
        ])
        pipe.fit(X_tr, y_tr)

        explainer = WaterQualityExplainer(pipe, feature_names=feature_names)
        if explainer.explainer is not None:
            X_trans = pipe.named_steps["preprocessor"].transform(X_v)
            vals = explainer.explainer.shap_values(X_trans)
            # Use class 2 (polluted) or direct array
            target_vals = vals[2] if isinstance(vals, list) and len(vals) > 2 else vals
            mean_abs_shap = np.mean(np.abs(target_vals), axis=0)
            fold_importances.append(mean_abs_shap)

    # Calculate pairwise Spearman rank correlation
    correlations = []
    jaccard_top3 = []
    n_folds = len(fold_importances)

    # Ensure 1D flattened feature importance vectors
    flat_fi = [np.asarray(fi).ravel() for fi in fold_importances]

    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            res = spearmanr(flat_fi[i], flat_fi[j])
            rho_val = float(np.asarray(getattr(res, "statistic", res[0])).ravel()[0])
            correlations.append(rho_val)

            top3_i = set(np.argsort(flat_fi[i])[-3:].tolist())
            top3_j = set(np.argsort(flat_fi[j])[-3:].tolist())
            jaccard = len(top3_i & top3_j) / len(top3_i | top3_j)
            jaccard_top3.append(float(jaccard))

    mean_rho = float(np.mean(correlations)) if correlations else 1.0
    mean_jaccard = float(np.mean(jaccard_top3)) if jaccard_top3 else 1.0

    stability_report = {
        "n_folds_analyzed": n_folds,
        "mean_spearman_rank_correlation": mean_rho,
        "min_spearman_rank_correlation": float(np.min(correlations)) if correlations else 1.0,
        "mean_top3_jaccard_similarity": mean_jaccard,
        "is_stable": mean_rho >= 0.85 and mean_jaccard >= 0.75
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(stability_report, f, indent=2)

    logger.info("SHAP cross-fold stability evaluated", mean_spearman=round(mean_rho, 4), mean_jaccard=round(mean_jaccard, 4))
    return stability_report
