"""
Evaluation metrics, calibration analysis, and visualization generation for AquaSense AI.
All generated figures are saved strictly to outputs/figures/.
All metrics tables are saved to results/.
"""

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

FIGURES_DIR = Path("outputs/figures")
RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"

CLASS_NAMES = ["Good", "Moderate", "Polluted"]


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
) -> dict[str, Any]:
    """Computes comprehensive classification metrics across all classes."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # Per-class metrics
    per_class: dict[str, dict[str, float]] = {}
    pr_aucs: list[float] = []

    for idx, cname in enumerate(CLASS_NAMES):
        binary_true = (y_true == idx).astype(int)
        c_prob = y_probs[:, idx]

        p = float(precision_score(binary_true, (y_pred == idx).astype(int), zero_division=0))
        r = float(recall_score(binary_true, (y_pred == idx).astype(int), zero_division=0))
        f = float(f1_score(binary_true, (y_pred == idx).astype(int), zero_division=0))

        prec_arr, rec_arr, _ = precision_recall_curve(binary_true, c_prob)
        c_pr_auc = float(auc(rec_arr, prec_arr))
        pr_aucs.append(c_pr_auc)

        per_class[cname] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f, 4),
            "pr_auc": round(c_pr_auc, 4),
        }

    macro_pr_auc = float(np.mean(pr_aucs))

    # Multiclass Brier Score (mean squared error of predicted probabilities)
    y_true_ohe = np.zeros_like(y_probs)
    for i, t in enumerate(y_true):
        y_true_ohe[i, t] = 1.0
    brier = float(np.mean(np.sum((y_probs - y_true_ohe) ** 2, axis=1)))

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_pr_auc": round(macro_pr_auc, 4),
        "brier_score": round(brier, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def plot_confusion_matrix_figure(
    cm: list[list[int]],
    output_path: Path = FIGURES_DIR / "confusion_matrix.png",
) -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    matrix = np.array(cm)

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(matrix, cmap="Blues")
    fig.colorbar(cax)

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Confusion Matrix (AquaSense AI)")

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            val = matrix[i, j]
            color = "white" if val > matrix.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return str(output_path)


def plot_pr_curves_figure(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    output_path: Path = FIGURES_DIR / "pr_curves.png",
) -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))

    colors = ["#0284c7", "#0f766e", "#b91c1c"]
    for idx, (cname, color) in enumerate(zip(CLASS_NAMES, colors)):
        binary_true = (y_true == idx).astype(int)
        c_prob = y_probs[:, idx]
        prec, rec, _ = precision_recall_curve(binary_true, c_prob)
        c_pr_auc = auc(rec, prec)
        plt.plot(rec, prec, color=color, lw=2, label=f"{cname} (PR-AUC = {c_pr_auc:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves by Risk Class")
    plt.legend(loc="lower left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return str(output_path)


def plot_reliability_diagram_figure(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    output_path: Path = FIGURES_DIR / "reliability_diagram.png",
) -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")

    colors = ["#0284c7", "#0f766e", "#b91c1c"]
    for idx, (cname, color) in enumerate(zip(CLASS_NAMES, colors)):
        binary_true = (y_true == idx).astype(int)
        c_prob = y_probs[:, idx]
        prob_true, prob_pred = calibration_curve(binary_true, c_prob, n_bins=5, strategy="uniform")
        plt.plot(prob_pred, prob_true, marker="s", lw=2, color=color, label=f"{cname}")

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Reliability Diagram (Probability Calibration)")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return str(output_path)


def plot_eda_figures(df: pd.DataFrame) -> tuple[str, str]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    from src.aquasense.data import FEATURE_COLUMNS

    # 1. Feature distributions
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()

    for i, col in enumerate(FEATURE_COLUMNS):
        axes[i].hist(df[col], bins=25, color="#0284c7", edgecolor="#0f172a", alpha=0.8)
        axes[i].set_title(col, fontsize=10)
        axes[i].grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Physicochemical & Toxic Metal Feature Distributions", fontsize=12)
    plt.tight_layout()
    dist_path = FIGURES_DIR / "feature_distributions.png"
    plt.savefig(dist_path, dpi=200, bbox_inches="tight")
    plt.close()

    # 2. Correlation matrix
    corr = df[FEATURE_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    cax = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax)

    ax.set_xticks(range(len(FEATURE_COLUMNS)))
    ax.set_yticks(range(len(FEATURE_COLUMNS)))
    ax.set_xticklabels(FEATURE_COLUMNS, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(FEATURE_COLUMNS, fontsize=9)

    for i in range(len(FEATURE_COLUMNS)):
        for j in range(len(FEATURE_COLUMNS)):
            val = corr.iloc[i, j]
            ax.text(
                j, i, f"{val:.2f}", ha="center", va="center", color="black" if abs(val) < 0.6 else "white", fontsize=8
            )

    plt.title("Feature Correlation Matrix", pad=20)
    plt.tight_layout()
    corr_path = FIGURES_DIR / "correlation_matrix.png"
    plt.savefig(corr_path, dpi=200, bbox_inches="tight")
    plt.close()

    return str(dist_path), str(corr_path)


def export_latex_tables(comparison_df: pd.DataFrame, ablations_df: pd.DataFrame) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Performance table
    perf_tex = comparison_df.to_latex(index=False, float_format="%.4f")
    (TABLES_DIR / "performance.tex").write_text(perf_tex, encoding="utf-8")

    # Ablations table
    abl_tex = ablations_df.to_latex(index=False, float_format="%.4f")
    (TABLES_DIR / "ablations.tex").write_text(abl_tex, encoding="utf-8")
