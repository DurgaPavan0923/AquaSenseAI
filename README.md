# AquaSense AI: Water-Quality Monitoring & Pollution-Risk Screening

Final-year AI/ML Senior Design Project. Predictions are screening support only and do not replace certified laboratory analysis.

| Academic Details | Information |
| :--- | :--- |
| **Course** | Research Methodology (URP4201) |
| **Supervisor** | Dr. Alok Ku. Pani |
| **Authors** | Ayusman Das (2341004073) |
| | Selina Mohanty (2341013299) |
| | Rajana Durga Pavan Kumar (2341011099) |
| | Akanshaa Mohanta (2341013229) |
| **Repository** | https://github.com/DurgaPavan0923/AquaSenseAI.git |

## 1. Overview
AquaSense AI provides rapid water quality screening and pollution risk categorization. The system classifies samples into three discrete risk tiers (Good, Moderate, Polluted) and accompanies each prediction with a conformal uncertainty prediction set and local occlusion feature attribution.

## 2. Problem Context
Standard laboratory testing for water quality provides definitive measurements but requires days of incubation and specialized equipment. AquaSense AI acts as an upfront screening mechanism, triaging samples and flagging uncertain predictions with a notice: "Needs laboratory test."

## 3. Dataset
- **Source:** Kaggle (ziya07/water-quality-and-pollution-monitoring-dataset)
- **License Notice:** Raw CSV is excluded from Git to comply with distribution guidelines. Use `scripts/download_data.py` to retrieve data.
- **Audit Findings:** 1,000 samples, 10 columns, 0 missing values, 0 duplicate feature rows.
- **Class Counts:** Good (559), Moderate (198), Polluted (243).
- **Parameters:** pH, Turbidity (NTU), Temperature (deg C), Dissolved Oxygen (mg/L), Biochemical Oxygen Demand (mg/L), Lead (mg/L), Mercury (mg/L), Arsenic (mg/L).

## 4. Method
- **Partition:** 80% train (800 rows), 20% holdout test (200 rows), stratified by class label, random state 42.
- **Feature Engineering:** Domain ratios including OSDR (BOD/DO), heavy metal hazard index, pH deviation from 7.0, turbidity-DO quotient, and statutory breach count.
- **Ensemble Model:** Tuned Random Forest with balanced class weights to address minority class distribution.
- **Uncertainty Calibration:** Platt scaling probability calibration and Inductive Conformal Prediction (LAC) targeting 90% marginal coverage.
- **Explainability:** Offline TreeSHAP summary beeswarm and local runtime occlusion attribution.

## 5. Experimental Results
Evaluated on the 200-sample holdout test partition:
- **Macro-F1:** 99.43% (0.9943)
- **Accuracy:** 99.50% (0.9950)
- **Macro PR-AUC:** 0.9992
- **Multi-Class Brier Score:** 0.0106
- **Conformal Test Coverage:** 99.50% (at 90% target coverage, average set size 1.00)
- **Feature Rank Stability:** Spearman rank correlation rho = 0.9505 across 5 folds

### Model Comparison Table (Test Partition)
| Architecture | Accuracy | Macro-F1 | Weighted-F1 | Macro PR-AUC | Brier Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Dummy (Majority) | 0.5600 | 0.2393 | 0.4021 | 0.6667 | 0.8800 |
| Logistic Regression | 0.9950 | 0.9943 | 0.9950 | 0.9973 | 0.0128 |
| Calibrated Random Forest (Active) | 0.9950 | 0.9943 | 0.9950 | 0.9992 | 0.0106 |
| XGBoost | 0.9950 | 0.9943 | 0.9950 | 0.9975 | 0.0102 |

Visualizations generated into `outputs/figures/`:
- `confusion_matrix.png`, `pr_curves.png`, `reliability_diagram.png`, `shap_summary.png`, `feature_distributions.png`, `correlation_matrix.png`.

## 6. Explainability and Uncertainty
- **Conformal Guarantee:** If prediction set contains multiple tiers or confidence drops below 60%, the output flags: "Screening support only. Needs laboratory test."
- **Local Attribution:** Occlusion against training medians reports probability shifts for each feature.

## 7. Website and API
- **Web (`web/`):** Static HTML/CSS/JS interface. Solid neutral palette, water blue accent, border-radius <= 4px, zero emoji, zero em dashes, WCAG AA compliant.
- **API (`api/index.py`):** Endpoints `/health`, `/predict`, `/explain`, `/anomaly`, `/models`, and `/predict/batch`.

## 8. Repository Layout
- `src/aquasense/`: Package modules (data, features, train, evaluate, uncertainty, explain, anomaly, db)
- `api/`: Lean serverless FastAPI entrypoint
- `web/`: Web interface pages, styles, scripts, favicons
- `app/`: Streamlit dashboard for local research
- `notebooks/`: 5 clean Jupyter notebooks (01_audit through 05_uncertainty)
- `results/`: Metrics summary, model comparisons, experiment logs, tables
- `outputs/figures/`: Visualizations
- `models/`: Compact deployed bundle and registry
- `scripts/`: Utility scripts (data download, bundle size check, design linter, simulator)
- `tests/`: Pytest suite

## 9. Setup and Reproduction
```bash
# 1. Install dependencies
pip install -r requirements-train.txt

# 2. Run data check and training pipeline
python scripts/download_data.py
python src/aquasense/train.py

# 3. Execute quality gates
python -m pytest
python scripts/lint_design.py
python scripts/check_bundle_size.py

# 4. Start local API and dashboard
uvicorn api.index:app --port 8000
streamlit run app/app.py
```

## 10. Vercel Serverless Architecture (< 150 MB Target)
- Runtime dependencies in `requirements.txt` restricted strictly to `fastapi`, `pydantic`, `numpy`, `scikit-learn`, `joblib`.
- All dev/training dependencies moved to `requirements-train.txt`.
- Lightweight model bundle size: 398.89 KB.
- Total serverless bundle size: 136.62 MB (below 150 MB target and 250 MB ceiling).

## 11. Limitations and Responsible Use
Screening support only. Predictions do not replace certified environmental testing. Biological pathogens and unmonitored contaminants require accredited laboratory verification.

## 12. Research Hypotheses and Future Work
Hypotheses confirmed: conformal prediction provides rigorous finite-sample coverage guarantees, and domain-guided ratios enhance feature importance stability across cross-validation folds. Future work includes embedded microcontroller sensor deployment.

## 13. License
Academic project license for course URP4201 Research Methodology.
