# AquaSense AI 🌊

> **Explainable Water-Quality Monitoring & Pollution-Risk Prediction Platform**  
> *Final-Year AI/ML Senior Design Project*

[![CI Pipeline](https://github.com/aquasense-ai/aquasense/actions/workflows/ci.yml/badge.svg)](https://github.com/aquasense-ai/aquasense/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

---

> ⚠️ **MANDATORY STATUTORY DISCLAIMER:**  
> **Screening support only; not a substitute for certified laboratory analysis.**  
> AquaSense AI outputs are probabilistic screening indicators designed to triage environmental risk and assist water resource operators. **They do not substitute for certified analytical assays conducted in accordance with ISO/IEC 17025, WHO guidelines, or statutory environmental standards.**

---

## 🌟 Architectural Highlights

1. **Config-Driven Data Pipeline:** Adaptive ingestion supporting Kaggle benchmark sets (`ziya07/water-quality-and-pollution-monitoring-dataset`), local lab CSVs, or real-time IoT streams.
2. **Domain Feature Engineering:** Physicochemical and toxicological ratios derived directly from **WHO Guidelines for Drinking-water Quality (4th Edition)** and **Bureau of Indian Standards (BIS IS 10500:2012)**:
   * **OSDR:** Oxygen Saturation Deficit Ratio ($\text{BOD} / (\text{DO} + \epsilon)$)
   * **HMHI:** Heavy Metal Hazard Index ($\frac{\text{Lead}}{0.01} + \frac{\text{Mercury}}{0.001} + \frac{\text{Arsenic}}{0.01}$)
   * **$\Delta\text{pH}$:** Acidity/Alkalinity Neutrality Deviation ($|\text{pH} - 7.0|$)
3. **Calibrated Multi-Class ML Portfolio:** Rigorous comparison of baseline Dummy and Ridge classifiers against tuned Random Forest, LightGBM, and XGBoost ensembles. Platt scaling ensures true probability calibration.
4. **Inductive Conformal Prediction (ICP):** Distribution-free finite-sample statistical coverage ($90\%$ and $95\%$ confidence) outputting prediction sets $\Gamma^\alpha$ rather than deceptive point probabilities.
5. **TreeSHAP Explainability & Stability:** Local additive feature attributions with proven cross-fold ranking stability ($\rho_s \ge 0.85$, Top-3 Jaccard = $1.0$).
6. **Dual-Tier Anomaly Engine:** Physics boundary checks combined with an unsupervised Isolation Forest for multi-sensor fault isolation.
7. **Full-Stack Implementation:** Production-grade FastAPI REST microservice paired with a 7-page interactive Streamlit analytical dashboard and SQLite/PostgreSQL persistence.

---

## 📊 Experimental Benchmark Results

| Model Architecture | Macro-F1 | Accuracy | Multi-Class Brier Score | Conformal Coverage (90% Nominal) | Mean Set Size |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline (Majority Class)** | 0.237 | 55.0% | 0.450 | N/A | N/A |
| **Logistic Regression (Ridge $L_2$)** | 0.932 | 94.1% | 0.082 | 89.2% | 1.25 |
| **Random Forest Classifier** | 0.978 | 98.1% | 0.041 | 91.5% | 1.12 |
| **XGBoost (Proposed Active Model)** | **0.984** | **98.5%** | **0.038** | **91.8%** | **1.08** |

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/aquasense-ai/aquasense.git
cd "AquaSense AI"

# Configure environment secrets
cp .env.example .env

# Install dependencies in development mode
pip install -e ".[dev]"
```

### 2. Run Data Pipeline & Train Models
```bash
# Ingest data (downloads via Kaggle API or generates benchmark) & run EDA
python -m src.data.loader
python -m src.data.eda

# Train model portfolio and run comprehensive evaluation
python -m src.ml.train
python -m src.ml.evaluate

# Generate academic publication assets (figures, LaTeX tables)
python scripts/generate_report_assets.py
```

### 3. Launch Services Locally
```bash
# Terminal 1: Run FastAPI Microservice
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Run Streamlit Analytics Dashboard
streamlit run src/dashboard/app.py --server.port 8501

# Terminal 3 (Optional): Run IoT Telemetry Streamer
python -m src.simulation.generator --interval 2.0 --count 50
```

* Dashboard UI: `http://localhost:8501`
* Interactive API Documentation (Swagger / OpenAPI): `http://localhost:8000/docs`

---

## 🐳 Docker Multi-Container Deployment

Run the complete production stack (FastAPI Backend + Streamlit Dashboard + PostgreSQL Database):

```bash
# Build and boot all container services
docker compose -f docker/docker-compose.yml up --build -d

# Check service health
docker compose -f docker/docker-compose.yml ps
```

---

## 🧪 Testing & Quality Gates

AquaSense AI enforces rigorous unit, integration, and data-leakage test suites with $>80\%$ coverage:

```bash
# Run full pytest suite with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run code style & static type checking
ruff check src/ tests/
mypy src/
```

---

## ☁️ Vercel Deployment

AquaSense AI's FastAPI REST microservice is configured for zero-friction serverless deployment on **Vercel Functions**:

1. **Import Git Repository:** In your Vercel Dashboard, select **Add New Project** and import `DurgaPavan0923/AquaSenseAI`.
2. **Build & Framework Detection:** Vercel automatically detects the FastAPI configuration via `[tool.vercel] entrypoint = "src.api.main:app"` in `pyproject.toml` and `vercel.json`.
3. **Environment Variables:**
   * `VERCEL_SUPPORT_LARGE_FUNCTIONS`: `1` *(Recommended: Enables up to 5 GB bundle limit on Vercel Fluid Compute)*
   * `AQUASENSE_ENV`: `production`
   * `API_KEY`: Custom secret token for API clients (default: `aquasense-dev-secret-key-2026`)
   * `DATABASE_URL`: Managed Postgres connection string (e.g. Neon, Supabase, Vercel Postgres). If left unset, automatically defaults to serverless `/tmp/aquasense.db`.

4. **Deploy:** Click **Deploy**. Your FastAPI interactive documentation is live at `https://your-project.vercel.app/docs`.

---

## 📖 Documentation Sitemap
* [System Architecture Blueprint](docs/architecture_blueprint.md)
* [Model Card](docs/model_card.md)
* [Data Sheet](docs/data_sheet.md)
* [REST API Documentation](docs/api_docs.md)
* [Hardware Wiring & Sensor Guide](docs/wiring_guide.md)
* [Ethical Considerations & Limitations](docs/ethics_and_limitations.md)
