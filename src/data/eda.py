"""Exploratory Data Analysis and Automated Data Quality Reporting."""

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.common.config import yaml_config
from src.common.logger import logger
from src.data.loader import load_dataset


def run_eda(df: pd.DataFrame, output_dir: str = "reports") -> dict[str, Any]:
    """Generates figures and data quality metrics from the dataset."""
    reports_path = Path(output_dir)
    figures_path = reports_path / "figures"
    figures_path.mkdir(parents=True, exist_ok=True)

    target_col = yaml_config["target"]["name"]
    numerical_cols = [f["name"] for f in yaml_config["features"]["numerical"]]

    # 1. Data Quality Audit
    total_rows = len(df)
    missing_counts = df.isnull().sum().to_dict()
    duplicates_count = int(df.duplicated().sum())

    # 2. Class Balance
    class_counts = df[target_col].value_counts().to_dict() if target_col in df.columns else {}
    class_proportions = (df[target_col].value_counts(normalize=True)).to_dict() if target_col in df.columns else {}

    # 3. Descriptive Stats
    summary_stats = df[numerical_cols].describe().to_dict()

    # Visualizations (Set clean aesthetic)
    sns.set_theme(style="whitegrid", palette="muted")

    # A. Class Distribution Plot
    if target_col in df.columns:
        plt.figure(figsize=(7, 4))
        target_labels = [yaml_config["target"]["classes"].get(c, str(c)) for c in df[target_col].values]
        ax = sns.countplot(x=target_labels, order=["Good", "Moderate", "Polluted"])
        plt.title("Water Quality Pollution Risk Class Distribution", fontsize=12, fontweight="bold")
        plt.xlabel("Pollution Level")
        plt.ylabel("Sample Count")
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f"{int(height)} ({height/total_rows*100:.1f}%)",
                        (p.get_x() + p.get_width() / 2., height + 5),
                        ha="center", va="baseline", fontsize=10)
        plt.tight_layout()
        plt.savefig(figures_path / "class_distribution.png", dpi=300)
        plt.close()

    # B. Feature Correlation Matrix
    plt.figure(figsize=(10, 8))
    corr = df[numerical_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, square=True, linewidths=0.5)
    plt.title("Physicochemical & Toxicological Feature Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_path / "correlation_matrix.png", dpi=300)
    plt.close()

    # C. Feature Distribution Histograms
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
    axes = axes.flatten()
    for i, col in enumerate(numerical_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color="#1f77b4")
        axes[i].set_title(col, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("")
    plt.suptitle("Feature Distributions Across Continuous Parameters", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_path / "feature_distributions.png", dpi=300)
    plt.close()

    # Compile Structured Report
    report = {
        "dataset_name": yaml_config["project"]["name"],
        "total_observations": total_rows,
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "missing_values": missing_counts,
        "exact_duplicates": duplicates_count,
        "target_column": target_col,
        "class_distribution": {
            "counts": class_counts,
            "proportions": {str(k): round(v, 4) for k, v in class_proportions.items()}
        },
        "numerical_summary": summary_stats
    }

    with open(reports_path / "data_quality_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Markdown Summary
    md_report = f"""# AquaSense AI - Data Quality & Exploratory Analysis Report

## 1. Overview
* **Total Observations:** {total_rows}
* **Feature Columns ({len(numerical_cols)}):** {', '.join(numerical_cols)}
* **Missing Values Detected:** {sum(missing_counts.values())} (0.00%)
* **Duplicate Rows:** {duplicates_count}

## 2. Class Balance Topology
| Class Code | Label | Count | Percentage |
| :---: | :---: | :---: | :---: |
"""
    for code, count in class_counts.items():
        label = yaml_config["target"]["classes"].get(code, str(code))
        pct = class_proportions.get(code, 0.0) * 100
        md_report += f"| {code} | {label} | {count} | {pct:.1f}% |\n"

    md_report += """
## 3. Visual Artifacts
* Figure 1: `reports/figures/class_distribution.png`
* Figure 2: `reports/figures/correlation_matrix.png`
* Figure 3: `reports/figures/feature_distributions.png`
"""
    with open(reports_path / "data_quality_report.md", "w", encoding="utf-8") as f:
        f.write(md_report)

    logger.info("EDA completed successfully", reports_path=str(reports_path))
    return report


if __name__ == "__main__":
    df = load_dataset()
    report = run_eda(df)
    print("EDA completed. Generated figures and report in reports/")
