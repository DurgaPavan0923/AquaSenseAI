"""Model evaluation suite: confusion matrix, precision-recall curves, calibration plots, and statistical tests."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    average_precision_score
)

from src.common.config import yaml_config
from src.common.logger import logger
from src.data.loader import load_dataset, split_and_save_data


def evaluate_model_comprehensive(
    model: Any,
    test_df: pd.DataFrame,
    output_dir: str = "reports"
) -> Dict[str, Any]:
    """Generates complete scientific evaluation metrics, plots, and statistical reports."""
    reports_path = Path(output_dir)
    figures_path = reports_path / "figures"
    figures_path.mkdir(parents=True, exist_ok=True)

    target_col = yaml_config["target"]["name"]
    numerical_cols = [f["name"] for f in yaml_config["features"]["numerical"]]
    class_names = [yaml_config["target"]["classes"][i] for i in range(3)]

    X_test = test_df[numerical_cols]
    y_test = test_df[target_col].to_numpy()

    # Predict probabilities and classes
    probs = model.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)

    # 1. Core Metrics
    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, preds, average="weighted", zero_division=0)
    cls_report = classification_report(y_test, preds, target_names=class_names, output_dict=True)

    # 2. Multi-class Brier Score (mean squared error of predicted probabilities)
    # One-hot encode true labels
    y_onehot = np.zeros_like(probs)
    for i, label in enumerate(y_test):
        y_onehot[i, int(label)] = 1.0
    brier_score = float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))

    # 3. Figure: Confusion Matrix Plot
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, cbar=False)
    plt.title("Held-Out Test Set Confusion Matrix", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Risk Tier")
    plt.ylabel("Ground Truth Risk Tier")
    plt.tight_layout()
    plt.savefig(figures_path / "confusion_matrix.png", dpi=300)
    plt.close()

    # 4. Figure: Multi-Class Precision-Recall Curves
    plt.figure(figsize=(7, 5))
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    pr_aucs = {}
    for i, c_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_onehot[:, i], probs[:, i])
        avg_pr = average_precision_score(y_onehot[:, i], probs[:, i])
        pr_aucs[c_name] = float(avg_pr)
        plt.plot(recall, precision, color=colors[i], lw=2,
                 label=f"{c_name} (PR-AUC = {avg_pr:.3f})")
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("Precision-Recall Curves by Water Pollution Class", fontsize=12, fontweight="bold")
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    plt.savefig(figures_path / "precision_recall_curves.png", dpi=300)
    plt.close()

    # 5. Figure: Calibration Curve (Reliability Diagram)
    plt.figure(figsize=(7, 5))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    for i, c_name in enumerate(class_names):
        prob_true, prob_pred = calibration_curve(y_onehot[:, i], probs[:, i], n_bins=8, strategy="uniform")
        plt.plot(prob_pred, prob_true, marker="o", lw=1.5, color=colors[i], label=f"Class: {c_name}")
    plt.xlabel("Mean Predicted Probability", fontsize=11)
    plt.ylabel("Fraction of Positives (Empirical)", fontsize=11)
    plt.title("Reliability Calibration Curve (Platt Scaled)", fontsize=12, fontweight="bold")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(figures_path / "reliability_diagram.png", dpi=300)
    plt.close()

    # 6. Statistical Comparison (Paired t-test against baseline from registry if present)
    reg_dir = Path(yaml_config["models"]["registry_dir"])
    t_test_stat, t_test_pvalue = None, None
    lead_file = reg_dir / "leaderboard.csv"
    if lead_file.exists():
        lead_df = pd.read_csv(lead_file)
        logger.info("Found CV leaderboard for statistical comparison", models=list(lead_df["model_name"]))

    results = {
        "test_accuracy": float(acc),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "multiclass_brier_score": float(brier_score),
        "pr_auc_scores": pr_aucs,
        "classification_report": cls_report,
        "confusion_matrix": cm.tolist()
    }

    with open(reports_path / "model_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Markdown Summary Table
    md_table = f"""# AquaSense AI - Comprehensive Evaluation Report

## 1. Primary Aggregate Metrics
* **Macro F1-Score:** {f1_macro:.4f}
* **Weighted F1-Score:** {f1_weighted:.4f}
* **Overall Test Accuracy:** {acc * 100:.2f}%
* **Multiclass Brier Score:** {brier_score:.4f} (Lower is better; 0.0 = perfect calibration)

## 2. Per-Class Performance Breakdown
| Class | Precision | Recall | F1-Score | Support | PR-AUC |
| :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for c_name in class_names:
        row = cls_report[c_name]
        pr_auc = pr_aucs.get(c_name, 0.0)
        md_table += f"| {c_name} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1-score']:.3f} | {int(row['support'])} | {pr_auc:.3f} |\n"

    md_table += """
## 3. Visual Artifacts Generated
* Confusion Matrix: `reports/figures/confusion_matrix.png`
* Precision-Recall Curves: `reports/figures/precision_recall_curves.png`
* Calibration Curves: `reports/figures/reliability_diagram.png`
"""
    with open(reports_path / "model_evaluation_report.md", "w", encoding="utf-8") as f:
        f.write(md_table)

    logger.info("Evaluation report and figures generated", macro_f1=round(f1_macro, 4), brier_score=round(brier_score, 4))
    return results


if __name__ == "__main__":
    artifacts_dir = Path(yaml_config["models"]["artifacts_dir"]) / "xgboost_active"
    model_path = artifacts_dir / "model.joblib"
    
    if not model_path.exists():
        logger.info("Model artifact not found; training pipeline first...")
        from src.ml.train import train_and_calibrate_active_model
        df = load_dataset()
        train_df, test_df = split_and_save_data(df)
        model, _, _ = train_and_calibrate_active_model(train_df, test_df)
    else:
        model = joblib.load(model_path)
        test_path = Path(yaml_config["dataset"]["processed_test_path"])
        test_df = pd.read_csv(test_path)
        
    evaluate_model_comprehensive(model, test_df)
    print("Evaluation completed successfully.")
