# Data Sheet for AquaSense AI Water Quality Dataset

Following the Datasheets for Datasets framework (Gebru et al., 2021).

---

## 1. Motivation
* **For what purpose was the dataset created?**  
  To establish a standardized, multi-parameter environmental benchmark for machine learning research into aquatic pollution classification, sensor anomaly isolation, and explainable water quality monitoring.
* **Who created the dataset?**  
  Synthesized and compiled by user `ziya07` and augmented with physics-guided boundary constraints by the AquaSense AI engineering team.

---

## 2. Composition
* **What do the instances represent?**  
  Each instance represents a localized aquatic water sample characterized by 8 physicochemical and toxicological features.
* **How many instances are there?**  
  1,000 observations in the baseline release, partitioned into 800 training/calibration samples and 200 holdout test samples.
* **Are there missing values or duplicates?**  
  The benchmark data contains 0% missing values and 0 duplicate rows. Real-world ingestion pipelines enforce automated median imputation and schema assertions.

---

## 3. Collection Process
* **What mechanisms were used to collect the data?**  
  Simulated sensor telemetry combining normal ambient water physics, industrial runoff injection, and heavy metal concentrations mapped against WHO Guidelines for Drinking-water Quality and Bureau of Indian Standards (BIS IS 10500:2012).

---

## 4. Preprocessing & Uses
* **What preprocessing was applied?**  
  Median imputation via `SimpleImputer`, robust interquartile range scaling via `RobustScaler` (preserving extreme toxic spikes), and domain-specific ratio formulation (`OSDR`, `HMHI`, $\Delta\text{pH}$).
* **Has the dataset been used for prior tasks?**  
  Yes, for multi-class water potability and pollution-risk classification.
