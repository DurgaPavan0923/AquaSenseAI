# AquaSense AI: Explainable Environmental Water-Quality Intelligence Platform

[![CI Pipeline](https://github.com/DurgaPavan0923/AquaSenseAI/actions/workflows/ci.yml/badge.svg)](https://github.com/DurgaPavan0923/AquaSenseAI/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.14-006699.svg?style=flat-square)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-006699.svg?style=flat-square)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.6+-f89939.svg?style=flat-square)](https://scikit-learn.org/)
[![Coverage](https://img.shields.io/badge/coverage-84%25-006699.svg?style=flat-square)](https://github.com/DurgaPavan0923/AquaSenseAI)
[![Bundle Size](https://img.shields.io/badge/bundle-136.4_MB_%28%3C150MB%29-006699.svg?style=flat-square)](https://github.com/DurgaPavan0923/AquaSenseAI)
[![License](https://img.shields.io/badge/license-Academic_Research-0f172a.svg?style=flat-square)](LICENSE)

> **Regulatory & Laboratory Testing Disclaimer**  
> AquaSense AI provides upfront environmental risk screening and decision-support classification. Predictions do not replace statutory laboratory analysis (such as ISO/IEC 17025 accredited spectrometry or microbial culture assays). Ambiguous classifications or statutory threshold breaches require certified physical laboratory verification.

---

## Table of Contents
- [1. Executive Summary](#1-executive-summary)
- [2. Academic Context & Authorship](#2-academic-context--authorship)
- [3. System Architecture](#3-system-architecture)
- [4. Dataset & Disjoint Partitioning](#4-dataset--disjoint-partitioning)
- [5. Model Performance & Evaluation](#5-model-performance--evaluation)
- [6. Inductive Conformal Uncertainty & Explainability](#6-inductive-conformal-uncertainty--explainability)
- [7. Authoritative Water Quality Standards](#7-authoritative-water-quality-standards)
- [8. Diagnostic Visualizations](#8-diagnostic-visualizations)
- [9. Web Platform Workspace](#9-web-platform-workspace)
- [10. REST API Reference](#10-rest-api-reference)
- [11. Local Setup & Reproducibility](#11-local-setup--reproducibility)
- [12. Limitations & Responsible Use](#12-limitations--responsible-use)

---

## 1. Executive Summary

AquaSense AI is an explainable environmental intelligence platform developed for rapid water-quality screening and pollution-risk categorization. The system classifies multi-parameter water samples into three discrete risk tiers (**Good**, **Moderate**, **Polluted**), provides mathematically grounded inductive conformal prediction sets with finite-sample coverage guarantees, and delivers local feature attributions via counterfactual occlusion.

Built as a lightweight, cloud-native application, AquaSense AI pairs an optimized serverless FastAPI backend (runtime bundle < 150 MB) with an accessible, high-performance static web interface.

| Key Attribute | Specification |
| :--- | :--- |
| **Primary Task** | Multi-class water quality risk classification & anomaly auditing |
| **Risk Categories** | Class 0: Good (Potable / Low Risk), Class 1: Moderate (Needs Treatment), Class 2: Polluted (Non-potable / Hazardous) |
| **Base Model** | Random Forest with domain feature engineering & Platt scaling |
| **Uncertainty Quantification** | Split Inductive Conformal Prediction (LAC non-conformity score) |
| **Explainability** | Local counterfactual occlusion attribution against training medians |
| **Deployment Footprint** | 136.39 MB total serverless runtime bundle (cold-start < 500 ms) |
| **Live Web Application** | https://aquasenseai-three.vercel.app/ |

---

## 2. Academic Context & Authorship

AquaSense AI is developed as a final-year Senior Design Project under course **URP4201: Research Methodology**.

| Project Detail | Information |
| :--- | :--- |
| **Course** | Research Methodology (URP4201) |
| **Department** | CSE (AI/ML) |
| **Faculty Supervisor** | Dr. Alok Ku. Pani |
| **Student Investigators** | Ayusman Das (Registration No: 2341004073) |
| | Selina Mohanty (Registration No: 2341013299) |
| | Rajana Durga Pavan Kumar (Registration No: 2341011099) |
| | Akanshaa Mohanta (Registration No: 2341013229) |
| **Code Repository** | https://github.com/DurgaPavan0923/AquaSenseAI.git |
| **Production URL** | https://aquasenseai-three.vercel.app/ |

---

## 3. System Architecture

The pipeline processes multi-parameter sensor readings through feature engineering, calibrated inference, conformal uncertainty estimation, and counterfactual attribution:

```
[ Water Quality Sample (8 Parameters) ]
                    |
                    v
[ Physical Range & Schema Validation (Pydantic) ]
                    |
                    v
[ Domain Feature Engineering: 8 -> 13 Features ]
  - Organic Stress Dissolved Oxygen Ratio (OSDR)
  - Heavy Metal Hazard Index (HMHI)
  - pH Neutrality Deviation (|pH - 7.0|)
  - Toxic Threshold Breach Counter
                    |
                    v
[ Standard Scaling (Fitted on Train Split Only) ]
                    |
                    v
[ Random Forest Classifier (100 Estimators) ]
                    |
                    v
[ Platt Scaling Sigmoid Calibration (Cross-Validated) ]
                    |
                    v
+-------------------+--------------------+--------------------+
|                   |                    |                    |
v                   v                    v                    v
[ Class Probabilities ] [ Conformal Prediction Set ] [ Occlusion Explainability ] [ Sensor Anomaly Audit ]
(P_Good, P_Mod, P_Pol)  (Marginal Coverage >= 90%)   (Parameter Delta Attributions) (Physical / Statutory)
                    |                    |                    |                    |
                    +--------------------+--------------------+--------------------+
                                         |
                                         v
                         [ Unified JSON API / Web Workspace ]
```

---

## 4. Dataset & Disjoint Partitioning

### Dataset Specifications
- **Source:** Kaggle Water Quality and Pollution Monitoring Dataset (`ziya07/water-quality-and-pollution-monitoring-dataset`).
- **Sample Count:** 1,000 complete observations, 0 missing values, 0 duplicate feature rows.
- **Distribution Rules:** Raw data is excluded from version control in compliance with Kaggle Community Terms. Run `python scripts/download_data.py` to retrieve data automatically.
- **Class Breakdown:** Good: 559 samples (55.9%), Moderate: 198 samples (19.8%), Polluted: 243 samples (24.3%).
- **Monitored Features:**
  1. `pH`: Acidity / alkalinity level (dimensionless, scale 0.0 - 14.0)
  2. `Turbidity`: Cloudiness (NTU)
  3. `Temperature`: Water temperature (deg C)
  4. `Dissolved Oxygen`: Free oxygen concentration (mg/L)
  5. `Biochemical Oxygen Demand`: 5-day aerobic biological demand (mg/L)
  6. `Lead`: Cumulative toxic heavy metal (mg/L)
  7. `Mercury`: Nephrotoxic and neurotoxic heavy metal (mg/L)
  8. `Arsenic`: Toxic metalloid / human carcinogen (mg/L)

### 4-Way Statistically Disjoint Partitioning
To eliminate optimistic bias in probability calibration and inductive conformal quantile derivation, samples are partitioned into four strictly disjoint subsets:

```
Total Dataset: 1,000 Observations (Stratified Split)
|
+--- 1. Base Model Training Split      : 50% (500 samples) -> Fits trees & domain scalers
+--- 2. Probability Calibration Split  : 15% (150 samples) -> Fits Platt sigmoid scalers
+--- 3. Conformal Calibration Split    : 15% (150 samples) -> Computes non-conformity quantile (q_hat)
+--- 4. Independent Holdout Test Split : 20% (200 samples) -> Unbiased evaluation benchmark
```

Pairwise disjointness is programmatically verified with zero shared identifiers and zero identical feature rows. Preprocessors and standardizers are fitted exclusively on the base training partition.

---

## 5. Model Performance & Evaluation

All reported headline metrics are evaluated on the independent 200-sample holdout test partition:

| Evaluation Metric | Reported Performance | Regression Floor | Verification Status |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **99.50%** (0.9950) | >= 90.00% | Verified |
| **Macro-F1 Score** | **99.43%** (0.9943) | >= 90.00% | Verified |
| **Weighted-F1 Score** | **99.50%** (0.9950) | >= 90.00% | Verified |
| **Macro PR-AUC** | **0.9984** | >= 90.00% | Verified |
| **Multi-Class Brier Score** | **0.0122** | <= 0.1000 | Verified (Well-calibrated) |
| **Observed Conformal Coverage** | **99.50%** (at 90% target) | >= 88.00% | Verified (Conservative) |
| **Mean Set Size** | **1.00** | <= 1.50 | Verified |
| **Feature Rank Stability (rho)** | **0.9429** | >= 0.8500 | Verified (High Stability) |

### Model Benchmark Comparison (Holdout Test Partition)
Evaluated across competitive architectures using identical 4-way partitions:

| Architecture | Accuracy | Macro-F1 | Weighted-F1 | Macro PR-AUC | Brier Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Dummy (Majority Class Baseline) | 0.5600 | 0.2393 | 0.4021 | 0.6667 | 0.8800 |
| Logistic Regression (Balanced) | 0.9950 | 0.9943 | 0.9950 | 0.9960 | 0.0145 |
| **Calibrated Random Forest (Active Deployed)** | **0.9950** | **0.9943** | **0.9950** | **0.9984** | **0.0122** |
| XGBoost Classifier | 0.9750 | 0.9662 | 0.9752 | 0.9845 | 0.0446 |

### Per-Class Test Performance

| Class Tier | Precision | Recall | F1-Score | PR-AUC | Holdout Support |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Good** | 0.9912 | 1.0000 | 0.9956 | 0.9995 | 112 samples |
| **Moderate** | 1.0000 | 0.9750 | 0.9873 | 0.9958 | 40 samples |
| **Polluted** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 48 samples |

---

## 6. Inductive Conformal Uncertainty & Explainability

### Split Conformal Prediction
AquaSense AI integrates Inductive Conformal Prediction (ICP) using Least Ambiguous Class (LAC) non-conformity scores:
- **Significance Level:** alpha = 0.10 (90% target confidence guarantee).
- **Calibrated Non-Conformity Quantile:** q_hat = 0.0230.
- **Inclusion Criterion:** A class label is included in the prediction set if its calibrated probability satisfies P(Y = c | X) >= 1 - q_hat (0.9770).
- **Fallback Rule:** If no class meets the inclusion threshold, the argmax class is returned and the sample is flagged: `"Needs laboratory test: Ambiguous prediction set"`.
- **Empirical Coverage:** 99.50% holdout test coverage with 100% singleton sets, satisfying theoretical marginal validity without distribution assumptions under exchangeability.

### Local Counterfactual Explainability
To provide actionable interpretability without heavy dependencies, the API implements counterfactual occlusion:
1. Baseline reference medians are computed from the base training split.
2. For an input sample, each parameter is individually occluded by its training median value.
3. The resulting shift in predicted class probability determines the parameter's local directional attribution.
4. Attributions highlight parameters driving samples into elevated risk categories.

---

## 7. Authoritative Water Quality Standards

Screening heuristics, anomaly boundaries, and feature formulations are derived from statutory standards:

| Parameter | Permissible Range / Limit | Regulatory Standard | Health & Environmental Significance |
| :--- | :--- | :--- | :--- |
| **pH** | 6.50 - 8.50 | WHO 4th Ed. / BIS IS 10500:2012 / US EPA | pH < 6.5 accelerates pipe corrosion; pH > 8.5 impedes chlorination efficiency. |
| **Turbidity** | <= 5.00 NTU | WHO Guidelines / BIS IS 10500:2012 | Suspended particulates shelter microbial pathogens from chemical disinfection. |
| **Temperature** | 15.0 - 30.0 deg C | CPCB Surface Water Criteria | High temperatures deplete dissolved oxygen and accelerate bacterial decomposition. |
| **Dissolved Oxygen** | >= 6.50 mg/L | CPCB Class A / US EPA Aquatic Life | Essential for aquatic respiration; depletion indicates severe organic pollution. |
| **Biochemical Oxygen Demand** | <= 2.00 mg/L | CPCB Class A Raw Water Benchmark | Direct indicator of organic enrichment from untreated municipal sewage or effluent. |
| **Lead (Pb)** | <= 0.010 mg/L | WHO Guideline / BIS IS 10500:2012 | Cumulative neurotoxin causing neurological and renal impairment. |
| **Mercury (Hg)** | <= 0.001 mg/L | BIS IS 10500:2012 / US EPA MCL | Bioaccumulative nephrotoxin and neurotoxin associated with organ failure. |
| **Arsenic (As)** | <= 0.010 mg/L | WHO Provisional / BIS IS 10500:2012 | Group 1 human carcinogen associated with skin lesions and vascular disease. |

---

## 8. Diagnostic Visualizations

All model diagnostics were generated offline during verified training and exported to `outputs/figures/`:

| Visualization Artifact | Description | File Path |
| :--- | :--- | :--- |
| **Confusion Matrix** | Multi-class confusion matrix on the 200-sample holdout test partition | `outputs/figures/confusion_matrix.png` |
| **Precision-Recall Curves** | One-vs-Rest PR curves achieving macro PR-AUC of 0.9984 | `outputs/figures/pr_curves.png` |
| **Reliability Diagram** | Platt scaling calibration curve verifying multi-class Brier score of 0.0122 | `outputs/figures/reliability_diagram.png` |
| **TreeSHAP Summary** | Global beeswarm feature attribution illustrating directional parameter impacts | `outputs/figures/shap_summary.png` |
| **Feature Distributions** | Univariate density distributions across physical, chemical, and heavy metal features | `outputs/figures/feature_distributions.png` |
| **Correlation Heatmap** | Pairwise Spearman correlation matrix across parameters and domain ratio indices | `outputs/figures/correlation_matrix.png` |

---

## 9. Web Platform Workspace

The front-end is located in `web/` and served statically via Vercel:

- **Homepage (`web/index.html`):** Two-column hero, verified metrics cards, interactive dashboard preview, capabilities overview, and academic design details.
- **Analysis Workspace (`web/analyze.html`):** Multi-parameter form with input validation, sample presets (Potable, Moderate, Toxic), live statutory limits, counterfactual attribution charts, and downloadable PDF assessment reports.
- **Batch CSV Screening (`web/batch.html`):** Drag-and-drop CSV upload (supporting up to 500 rows), missing value checks, real-time risk distribution chart, sortable results table, and CSV export.
- **Geographic Monitoring Network (`web/monitoring.html`):** Interactive Leaflet station map centered on Bhubaneswar, Odisha across the Kuakhai-Daya river basin with 8 regional monitoring stations, risk-tier color coding, and custom coordinate ingestion.
- **Research & Metrics (`web/results.html`):** Model comparison benchmarks, 4-way partition breakdown, conformal uncertainty diagnostics, and full-resolution lightbox viewer for figures.
- **Research Methodology (`web/about.html`):** Academic design documentation, labeling rule audit, statutory standards reference, and scientific limitations.
- **Design System:** Deep navy typography (`#0f172a`), clean neutral slate surfaces (`#f8fafc`), water blue accent (`#006699`), high-contrast alert borders, zero emojis, zero gradients, and WCAG AA contrast compliance.

---

## 10. REST API Reference

The serverless backend provides a high-throughput FastAPI interface in `api/index.py`:

### Endpoints Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service health status, model bundle state, and statutory notice |
| `GET` | `/models` | Active model metadata, registry information, and verified metrics |
| `POST` | `/predict` | Single-sample risk categorization with conformal prediction sets |
| `POST` | `/explain` | Occlusion-based local feature attributions |
| `POST` | `/anomaly` | Sensor fault and statutory limit breach audit |
| `POST` | `/predict/batch` | High-throughput batch CSV file screening (up to 2 MB) |

### Sample Prediction Request
```bash
curl -X POST "https://aquasenseai-three.vercel.app/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "pH": 7.35,
       "Turbidity": 2.10,
       "Temperature": 21.5,
       "Dissolved Oxygen": 7.80,
       "Biochemical Oxygen Demand": 1.40,
       "Lead": 0.003,
       "Mercury": 0.0002,
       "Arsenic": 0.004
     }'
```

### Sample Prediction Response
```json
{
  "risk_tier": "Good",
  "predicted_class": 0,
  "confidence": 0.9924,
  "probabilities": {
    "Good": 0.9924,
    "Moderate": 0.0048,
    "Polluted": 0.0028
  },
  "conformal_prediction_set": ["Good"],
  "conformal_confidence_level": 0.90,
  "needs_laboratory_test": false,
  "notice": "Screening support only. Needs laboratory test if uncertain.",
  "disclaimer": "Predictions are screening support only and do not replace certified laboratory environmental analysis."
}
```

---

## 11. Local Setup & Reproducibility

### Prerequisites
- Python 3.11, 3.12, or 3.14
- Git

### Installation Steps

1. **Clone Repository:**
   ```bash
   git clone https://github.com/DurgaPavan0923/AquaSenseAI.git
   cd AquaSenseAI
   ```

2. **Create and Activate Virtual Environment:**
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   # Linux / macOS:
   source .venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements-train.txt
   ```

4. **Acquire Dataset (Optional for Full Training):**
   ```bash
   python scripts/download_data.py
   ```
   *(Note: The test suite includes an automated benchmark synthesizer; raw Kaggle data is only required for re-executing notebooks or full model retraining).*

5. **Execute Quality Gates & Verification:**
   ```bash
   # Run linting and type checks
   python -m ruff check src/aquasense api tests scripts
   python -m mypy src/aquasense api scripts --ignore-missing-imports

   # Run design compliance and serverless bundle audits
   python scripts/lint_design.py
   python scripts/check_bundle_size.py

   # Run test suite with 80% coverage threshold
   python -m pytest --cov=src/aquasense --cov=api --cov-fail-under=80
   ```

6. **Start Local Development Server:**
   ```bash
   uvicorn api.index:app --host 127.0.0.1 --port 8000 --reload
   ```
   Open `http://127.0.0.1:8000/` in your browser to inspect the application.

---

## 12. Limitations & Responsible Use

1. **Screening Support Only:** AquaSense AI is an upfront triage mechanism and cannot replace certified chemical, physical, or microbiological laboratory testing.
2. **Unmonitored Contaminants:** The system models 8 specific physicochemical and heavy metal indicators. It does not measure biological pathogens (such as *Escherichia coli*, *Giardia lamblia*, or *Cryptosporidium*), synthetic pesticides, microplastics, or per- and polyfluoroalkyl substances (PFAS).
3. **Dataset Scope:** The underlying benchmark dataset reflects composite regulatory threshold intervals. High model accuracy reflects the mathematical fidelity of the ensemble in reproducing these multi-parameter interval boundaries rather than real-world biological discovery.
4. **Exchangeability Assumption:** Conformal prediction set coverage guarantees rely on the assumption of exchangeability between calibration samples and test samples. Extreme seasonal distribution shifts or novel industrial discharges may violate exchangeability and necessitate physical laboratory re-testing.

---

## 13. License

This repository is maintained as an academic research and engineering design project for course **URP4201 Research Methodology** at the Department of Computer Science & Engineering (AI/ML).
