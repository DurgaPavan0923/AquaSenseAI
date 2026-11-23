# AquaSense AI: System Architecture Blueprint

## Architectural Overview
AquaSense AI is architected into 6 modular, decoupled layers:

```
[Layer 1: Telemetry & Ingestion] -> [Layer 2: Preprocessing & Anomaly] -> [Layer 3: Calibrated Ensemble] 
                                                                                   |
[Layer 6: Persistence & UI]      <- [Layer 5: FastAPI REST Service]     <- [Layer 4: TreeSHAP & Conformal]
```

1. **Ingestion Layer:** Pydantic v2 schema assertions enforcing physical boundaries on incoming sensor payloads.
2. **Preprocessing Layer:** Median imputation and RobustScaler to preserve heavy metal contamination spikes.
3. **ML Risk Classifier:** Platt-scaled XGBoost with domain ratio feature engineering (OSDR, HMHI).
4. **Uncertainty & XAI Layer:** Inductive Conformal Prediction (ICP) sets guaranteeing $90\%$ and $95\%$ statistical coverage paired with TreeSHAP local additive attributions.
5. **Persistence Layer:** SQLAlchemy 2.0 ORM supporting dual-dialect deployment (SQLite dev, PostgreSQL 15 prod).
6. **Consumption Layer:** 7-page Streamlit analytical dashboard and FastAPI REST endpoints.
