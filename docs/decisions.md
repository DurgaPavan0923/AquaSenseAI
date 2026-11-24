# Architectural and Methodological Decisions

This document records the core design and methodological decisions for AquaSense AI, developed for course URP4201 Research Methodology.

## 1. Problem Formulation: Multi-Class Classification vs Regression

### Decision
The target variable "Pollution Level" is formulated as a 3-class discrete classification task (0: Good, 1: Moderate, 2: Polluted) rather than a continuous pollution index regression.

### Justification
1. Discrete Decision Boundaries: Environmental and municipal water management protocols operate on categorical action tiers (potable without treatment, requiring standard treatment, or hazardous and unfit for human consumption).
2. Conformal Set Guarantees: Multi-class classification enables Inductive Conformal Prediction (Least Ambiguous Classifier) to produce distribution-free confidence sets with exact finite-sample coverage guarantees.
3. Class Imbalance Handling: The audited dataset contains 559 Good (55.9%), 198 Moderate (19.8%), and 243 Polluted (24.3%) samples. A classification setup allows explicit cost-sensitive weighting and focal penalties to prevent the minority "Moderate" risk tier from being submerged.

## 2. Evaluation Metric Selection

### Decision
Primary model ranking is evaluated using Macro-averaged F1-Score and Per-Class PR-AUC (Precision-Recall Area Under Curve), complemented by Confusion Matrices, Reliability Diagrams, and Brier Score.

### Justification
1. Macro-F1 prevents majority-class dominance: An unweighted accuracy metric would reward a classifier predicting "Good" while misclassifying contaminated samples. Macro-F1 treats all three pollution tiers equally.
2. PR-AUC over ROC-AUC: Given class imbalance, PR-AUC focuses on positive class precision and recall without being inflated by true negative counts.
3. Calibration Assessment: Brier score and expected calibration error ensure that predicted probabilities reflect empirical frequencies.

## 3. Uncertainty Quantification Architecture

### Decision
A two-layer uncertainty quantification framework is implemented:
1. Platt Scaling Probability Calibration (via Sigmoid CalibratedClassifierCV) fitted on training folds.
2. Inductive Conformal Prediction sets with non-conformity score s_i = 1 - P(Y_i = y_i | x_i).

### Decision on Low Confidence
When the conformal prediction set contains multiple classes (ambiguous) or the top class calibrated probability is below 0.60, the prediction output is flagged with:
"Screening support only. Needs laboratory test."

## 4. Vercel Serverless Deployment Architecture (< 150 MB Target)

### Decision
The production Vercel serverless function (api/index.py) separates runtime inference from offline research tooling.

### Justification
1. Lean Runtime Dependencies: Root requirements.txt contains strictly fastapi, pydantic, numpy, scikit-learn, and joblib. Bulky research dependencies (xgboost, shap, pandas, plotly, streamlit, optuna, matplotlib, jupyter) are moved to requirements-train.txt.
2. Approximate Local Attribution: Full TreeSHAP calculation requires the heavy shap and numba libraries. In production, local attribution is computed via input occlusion against training feature medians and clearly labeled as "occlusion-based, approximate".
3. Precomputed Global Artifacts: Offline-generated TreeSHAP summary beeswarm plots and ROC/PR curves are rendered into outputs/figures/ and served statically.
4. Stateless Lambda Design: Serverless instances avoid persistent local database writes and /tmp dependencies. Session history is managed client-side in browser localStorage.
