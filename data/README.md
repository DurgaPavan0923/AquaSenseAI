# AquaSense AI Dataset Documentation

## Dataset Source
- Title: Water Quality and Pollution Monitoring Dataset
- Source: Kaggle (ziya07/water-quality-and-pollution-monitoring-dataset)
- URL: https://www.kaggle.com/datasets/ziya07/water-quality-and-pollution-monitoring-dataset
- License: Subject to Kaggle Community Terms and original author licensing. Raw data is excluded from version control to respect distribution guidelines.

## Dataset Specifications
- Record Count: 1,000 samples
- Attribute Count: 10 columns (1 identifier, 8 physicochemical and heavy metal features, 1 target label)
- Missing Values: 0 (complete records)
- Duplicate Records: 0

## Columns and Measurement Units
1. Sample_ID: Unique sample identifier string (e.g., WQS_0001)
2. pH: Acidity or alkalinity level (dimensionless, scale 0.0 to 14.0)
3. Turbidity: Water cloudiness measured in Nephelometric Turbidity Units (NTU)
4. Temperature: Water temperature in degrees Celsius (deg C)
5. Dissolved Oxygen: Concentration of free oxygen in water (mg/L)
6. Biochemical Oxygen Demand: Dissolved oxygen needed by aerobic biological organisms to break down organic material (mg/L)
7. Lead: Toxic heavy metal concentration (mg/L)
8. Mercury: Toxic heavy metal concentration (mg/L)
9. Arsenic: Toxic metalloid concentration (mg/L)
10. Pollution Level: Discrete target risk category:
    - 0: Good (potable or low pollution risk)
    - 1: Moderate (needs treatment or close monitoring)
    - 2: Polluted (hazardous, non-potable)

## Download and Setup
To retrieve the raw dataset into `data/raw/water_quality.csv`, run:
```bash
python scripts/download_data.py
```
If Kaggle credentials (KAGGLE_USERNAME and KAGGLE_KEY) are configured in `.env`, the script downloads the dataset automatically via the Kaggle API. Otherwise, place the downloaded `water_quality.csv` directly into `data/raw/`.
