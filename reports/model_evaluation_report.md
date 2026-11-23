# AquaSense AI - Comprehensive Evaluation Report

## 1. Primary Aggregate Metrics
* **Macro F1-Score:** 0.9715
* **Weighted F1-Score:** 0.9800
* **Overall Test Accuracy:** 98.00%
* **Multiclass Brier Score:** 0.0359 (Lower is better; 0.0 = perfect calibration)

## 2. Per-Class Performance Breakdown
| Class | Precision | Recall | F1-Score | Support | PR-AUC |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Good | 0.991 | 1.000 | 0.996 | 112 | 0.999 |
| Moderate | 0.929 | 0.975 | 0.951 | 40 | 0.989 |
| Polluted | 1.000 | 0.938 | 0.968 | 48 | 1.000 |

## 3. Visual Artifacts Generated
* Confusion Matrix: `reports/figures/confusion_matrix.png`
* Precision-Recall Curves: `reports/figures/precision_recall_curves.png`
* Calibration Curves: `reports/figures/reliability_diagram.png`
