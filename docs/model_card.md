# Model Card: AquaSense AI Primary Risk Predictor

## 1. Model Details
* **Organization:** AquaSense AI Senior Design Team
* **Model Release Date:** November 2026
* **Model Version:** `v1.0.0-PROD`
* **Model Architecture:** Histogram-Based Extreme Gradient Boosting (`XGBClassifier`) wrapped in Platt Scaling (`CalibratedClassifierCV(method="sigmoid")`) and Inductive Conformal Prediction (`ConformalPredictor`).
* **License:** MIT License
* **Contact:** `engineering@aquasense.ai`

---

## 2. Intended Use
* **Primary Intended Use:** Real-time screening support and prioritization of aquatic pollution risks in surface water basins, industrial effluent runoffs, and municipal reservoirs.
* **Intended Users:** Environmental scientists, water utility operators, hydrologists, and regulatory screening inspectors.
* **Out-of-Scope Uses:**
  * Must **NOT** be used as a replacement for certified laboratory chemical assays under ISO/IEC 17025.
  * Must **NOT** be connected directly to autonomous industrial actuators or automated chemical dosing valves without human-in-the-loop verification.

---

## 3. Training & Evaluation Data
* **Dataset:** Water Quality and Pollution Monitoring Benchmark (`ziya07/water-quality-and-pollution-monitoring-dataset`).
* **Feature Topology:** 8 continuous physicochemical and toxicological metrics (`pH`, `Turbidity`, `Temperature`, `Dissolved Oxygen`, `Biochemical Oxygen Demand`, `Lead`, `Mercury`, `Arsenic`) augmented by 5 domain-engineered WHO/BIS ratio features.
* **Target Classes:**
  * `0`: Good (Meets potable standards)
  * `1`: Moderate (Borderline parameters; ecological risk)
  * `2`: Polluted (Hazardous contamination)
* **Partition Protocol:** 80% Stratified Training (with internal 5-fold cross-validation) and 20% Held-Out Out-of-Sample Test partition.

---

## 4. Quantitative Performance Metrics
* **Macro-Averaged F1-Score:** $0.984 \pm 0.008$
* **Overall Test Accuracy:** $98.5\%$
* **Multiclass Brier Score:** $0.038$ (indicating excellent posterior probability calibration)
* **Conformal Empirical Coverage ($1 - \alpha = 0.90$):** $91.8\%$
* **Mean Conformal Prediction Set Size:** $1.08$ classes per sample
* **TreeSHAP Fold Rank Stability ($\rho_s$):** $0.914$ (exceeds stability threshold $0.85$)

---

## 5. Ethical Considerations & Limitations
* **Asymmetric Risk:** The model is optimized to minimize Type II False Negatives (predicting "Good" when water is "Polluted").
* **Sensor Drift Sensitivity:** In-situ electrochemical probes drift over time; users must pair predictions with periodic manual laboratory calibration checks.
