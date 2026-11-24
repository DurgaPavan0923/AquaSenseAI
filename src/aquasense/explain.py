"""
Explainability module for AquaSense AI.
Provides fast occlusion-based local feature attribution for runtime inference
and offline TreeSHAP generation with cross-fold stability auditing.
"""

from pathlib import Path
from typing import Any

import numpy as np

FEATURE_NAMES: list[str] = [
    "pH",
    "Turbidity",
    "Temperature",
    "Dissolved Oxygen",
    "Biochemical Oxygen Demand",
    "Lead",
    "Mercury",
    "Arsenic",
]

ALL_FEATURE_NAMES: list[str] = FEATURE_NAMES + [
    "OSDR",
    "HMHI",
    "pH_Neutrality_Deviation",
    "Turbidity_DO_Ratio",
    "Toxic_Breach_Count",
]


def explain_sample_occlusion(
    model: Any,
    sample: np.ndarray,
    training_medians: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Computes local feature attributions via input occlusion against baseline training medians.
    Label: occlusion-based, approximate.
    Fast, deterministic, and free of heavy dependencies (numpy only).
    """
    names = feature_names or FEATURE_NAMES
    sample = np.asarray(sample, dtype=float).flatten()
    medians = np.asarray(training_medians, dtype=float).flatten()

    if len(sample) != len(names) or len(medians) != len(names):
        raise ValueError(
            f"Sample length ({len(sample)}) and medians ({len(medians)}) must match features ({len(names)})"
        )

    # Baseline prediction probability
    base_probs = model.predict_proba(sample.reshape(1, -1))[0]
    predicted_class = int(np.argmax(base_probs))
    base_class_prob = float(base_probs[predicted_class])

    attributions: dict[str, float] = {}
    for i, name in enumerate(names):
        occluded = sample.copy()
        occluded[i] = medians[i]
        occluded_probs = model.predict_proba(occluded.reshape(1, -1))[0]
        occluded_prob = float(occluded_probs[predicted_class])
        delta = base_class_prob - occluded_prob
        attributions[name] = round(float(delta), 4)

    # Sort attributions by absolute impact
    sorted_attributions = dict(sorted(attributions.items(), key=lambda item: abs(item[1]), reverse=True))

    return {
        "method": "occlusion-based, approximate",
        "predicted_class": predicted_class,
        "base_probability": round(base_class_prob, 4),
        "attributions": sorted_attributions,
        "notice": "Local feature attribution is calculated by counterfactual occlusion against training medians.",
    }


def compute_offline_shap_summary(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str] | None = None,
    output_path: str | Path = "outputs/figures/shap_summary.png",
) -> dict[str, Any]:
    """
    Generates global TreeSHAP explanation beeswarm plot offline and saves to outputs/figures.
    Requires shap and matplotlib.
    """
    import matplotlib
    import shap

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    names = feature_names or (ALL_FEATURE_NAMES if X_test.shape[1] == 13 else FEATURE_NAMES)

    # Use TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Multiclass SHAP values is a list or 3D array
    plt.figure(figsize=(10, 6))
    if isinstance(shap_values, list):
        # Multi-class list of arrays: plot for class 2 (Polluted) or aggregate
        target_shap = shap_values[2] if len(shap_values) > 2 else shap_values[0]
        shap.summary_plot(target_shap, X_test, feature_names=names, show=False)
    elif shap_values.ndim == 3:
        target_shap = shap_values[:, :, 2] if shap_values.shape[2] > 2 else shap_values[:, :, 0]
        shap.summary_plot(target_shap, X_test, feature_names=names, show=False)
    else:
        target_shap = shap_values
        shap.summary_plot(target_shap, X_test, feature_names=names, show=False)

    plt.title("AquaSense AI Global Feature Importance (TreeSHAP Summary)")
    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()

    mean_abs_shap = np.mean(np.abs(target_shap), axis=0)
    top_features = [names[idx] for idx in np.argsort(mean_abs_shap)[::-1]]

    return {
        "output_path": str(out_file),
        "top_features_by_importance": top_features,
    }


def evaluate_rank_stability(fold_importances: list[np.ndarray]) -> dict[str, Any]:
    """
    Calculates feature ranking stability across cross-validation folds using Spearman rank correlation.
    """
    from scipy.stats import spearmanr

    n_folds = len(fold_importances)
    correlations: list[float] = []

    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            rho, _ = spearmanr(fold_importances[i], fold_importances[j])
            if not np.isnan(rho):
                correlations.append(float(rho))

    avg_rho = float(np.mean(correlations)) if correlations else 1.0
    return {
        "mean_spearman_rank_correlation": round(avg_rho, 4),
        "stability_level": "High" if avg_rho > 0.85 else "Moderate",
    }
