# Ethical Considerations, Responsible AI & System Limitations

## 1. Statutory Decision-Support Boundary
AquaSense AI is engineered specifically as a rapid screening, decision-support, and risk-prioritization tool for environmental operators. It is **NOT** a legal substitute for certified physical or chemical analytical laboratory testing conducted in accordance with ISO/IEC 17025, WHO drinking water guidelines, or statutory environmental protection agencies.

---

## 2. Asymmetric Error Impacts in Public Health
In water quality monitoring, algorithmic errors carry starkly asymmetric human and environmental consequences:
* **False Negative (Type II Error):** Predicting "Good" when water is dangerously contaminated. This risks public exposure to neurotoxic heavy metals (`Lead`, `Mercury`, `Arsenic`) or pathogens. AquaSense AI combats this via:
  1. Class-weight balancing penalizing misclassification of the "Polluted" class.
  2. Inductive Conformal Prediction sets: ambiguous borderline samples output multi-class prediction sets (e.g. `['Moderate', 'Polluted']`) rather than forced single-label optimism.
  3. Deterministic rule-based bounds checking that bypasses ML inference when statutory thresholds are breached.
* **False Positive (Type I Error):** Predicting "Polluted" when water is safe. While this triggers re-testing or temporary treatment verification costs, it does not jeopardize public health.

---

## 3. Sensor Drift & Epistemic Uncertainty
In-situ electrochemical sensors inevitably experience biological fouling, galvanic exhaustion, and calibration drift. The platform mitigates this via:
* Dual-tier anomaly screening isolating rapid sensor step jumps.
* Transparent confidence intervals and conformal coverage metrics ($90\%$ and $95\%$).
* Explainability audits allowing operators to verify whether an alert is driven by plausible physicochemical relationships or isolated probe failure.
