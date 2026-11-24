# AquaSense AI Codebase File Audit

Generated as part of Phase 7 verification for course URP4201 Research Methodology.

| Relative Path | Primary Purpose | Status |
| :--- | :--- | :--- |
| `config.yaml` | Master project and ML pipeline configuration | Verified |
| `pyproject.toml` | Build metadata, ruff, mypy, and pytest coverage configuration | Verified |
| `requirements.txt` | Minimal runtime dependencies for Vercel serverless (< 150 MB target) | Verified |
| `requirements-train.txt` | Full research, dashboard, and testing dependencies | Verified |
| `vercel.json` | Vercel serverless function routing and exclusion rules | Verified |
| `.vercelignore` | Excludes non-runtime assets from production lambda deployment | Verified |
| `.gitignore` | Excludes raw dataset CSVs, bytecode, caches, and local databases | Verified |
| `.env.example` | Environment variable template | Verified |
| `README.md` | Comprehensive academic documentation, benchmarks, and reproduction steps | Verified |
| `LICENSE` | Non-commercial academic research project license | Verified |
| `data/README.md` | Dataset provenance, Kaggle source link, and download guide | Verified |
| `scripts/download_data.py` | Automated Kaggle API dataset downloader and verification script | Verified |
| `scripts/export_model.py` | Model bundle exporter for production deployment | Verified |
| `scripts/check_bundle_size.py` | CI audit tool enforcing serverless bundle size < 250 MB | Verified |
| `scripts/lint_design.py` | Design rule linter auditing typography, colors, and constraints | Verified |
| `scripts/iot_simulator.py` | Real-time IoT sensor telemetry streaming simulator | Verified |
| `src/aquasense/__init__.py` | Package initializer and version specification | Verified |
| `src/aquasense/data.py` | Schema validation, duplicate removal, and stratified splitting | Verified |
| `src/aquasense/features.py` | WHO/BIS domain ratio feature engineering (NumPy and Pandas) | Verified |
| `src/aquasense/train.py` | End-to-end model training, hyperparameter tuning, and ablations | Verified |
| `src/aquasense/evaluate.py` | Metric computation, LaTeX tables, and figure generation | Verified |
| `src/aquasense/uncertainty.py` | Platt probability calibration and Inductive Conformal Prediction | Verified |
| `src/aquasense/explain.py` | Offline TreeSHAP generation and runtime counterfactual occlusion | Verified |
| `src/aquasense/anomaly.py` | Sensor boundary checks and Isolation Forest anomaly detector | Verified |
| `src/aquasense/db.py` | SQLAlchemy ORM models for local testing and transactions | Verified |
| `api/__init__.py` | Serverless API package initializer | Verified |
| `api/index.py` | Lean FastAPI serverless handler (endpoints: /health, /predict, etc.) | Verified |
| `web/index.html` | Portal home page with project overview and methodology | Verified |
| `web/analyze.html` | Interactive water screening form and batch CSV uploader | Verified |
| `web/results.html` | Experimental evaluation tables and diagnostic figures | Verified |
| `web/about.html` | Research methodology, academic supervisors, and student team | Verified |
| `web/privacy.html` | Academic data privacy and retention policy | Verified |
| `web/terms.html` | Terms of service and screening support limitations | Verified |
| `web/css/style.css` | Accessible CSS stylesheet adhering to strict design rules | Verified |
| `web/js/main.js` | Vanilla JS client handling API calls and localStorage history | Verified |
| `web/favicon.svg` | Custom geometric water droplet SVG mark | Verified |
| `web/favicon-32x32.png` | 32x32 raster favicon | Verified |
| `web/apple-touch-icon.png` | 180x180 touch icon | Verified |
| `app/app.py` | Streamlit interactive research and exploration dashboard | Verified |
| `notebooks/01_dataset_audit.ipynb` | Jupyter notebook for dataset audit and summary stats | Verified |
| `notebooks/02_exploratory_data_analysis.ipynb` | Jupyter notebook for parameter distributions and correlation | Verified |
| `notebooks/03_model_training_and_ablations.ipynb` | Jupyter notebook for model benchmarks and ablation studies | Verified |
| `notebooks/04_explainability_analysis.ipynb` | Jupyter notebook for global and local feature attributions | Verified |
| `notebooks/05_uncertainty_quantification.ipynb` | Jupyter notebook for calibrated probabilities and conformal sets | Verified |
| `docs/decisions.md` | Methodological rationale for classification and architecture | Verified |
| `docs/domain.md` | DNS records and custom domain routing instructions | Verified |
| `docs/file_audit.md` | Full repository audit manifest | Verified |
| `results/metrics.json` | Real evaluation numbers on holdout test partition | Verified |
| `results/model_comparison.csv` | Benchmark comparisons across model architectures | Verified |
| `results/experiment_log.csv` | Log of all benchmark and ablation trials | Verified |
| `results/tables/performance.tex` | Exported LaTeX performance table | Verified |
| `results/tables/ablations.tex` | Exported LaTeX ablation table | Verified |
| `outputs/figures/confusion_matrix.png` | Confusion matrix heatmap | Verified |
| `outputs/figures/pr_curves.png` | Precision-recall curves across risk tiers | Verified |
| `outputs/figures/reliability_diagram.png` | Calibration reliability curves | Verified |
| `outputs/figures/shap_summary.png` | TreeSHAP summary beeswarm visualization | Verified |
| `outputs/figures/feature_distributions.png` | Physicochemical parameter histograms | Verified |
| `outputs/figures/correlation_matrix.png` | Pairwise feature correlation matrix | Verified |
| `models/model_bundle.joblib` | Compact deployed model artifact (398 KB) | Verified |
| `models/registry.json` | Metadata registry for deployed model bundle | Verified |
| `tests/conftest.py` | Pytest fixtures and test client configuration | Verified |
| `tests/unit/test_anomaly.py` | Unit tests for physical and statutory anomaly checks | Verified |
| `tests/unit/test_conformal.py` | Unit tests for conformal prediction and calibration | Verified |
| `tests/unit/test_features.py` | Unit tests for domain feature engineering mathematics | Verified |
| `tests/unit/test_preprocessor.py` | Unit tests for pipeline transformations | Verified |
| `tests/unit/test_property_validation.py` | Unit tests for Pydantic boundary fuzzing | Verified |
| `tests/unit/test_explain.py` | Unit tests for occlusion attribution and rank stability | Verified |
| `tests/unit/test_bundle_size.py` | Unit test verifying bundle size < 250 MB | Verified |
| `tests/unit/test_design_lint.py` | Unit test verifying zero design rule violations | Verified |
| `tests/integration/test_api.py` | Integration tests for all FastAPI endpoints | Verified |
| `tests/integration/test_leakage.py` | Integration tests for zero data leakage | Verified |
| `tests/integration/test_metric_regression.py` | Performance floor regression tests | Verified |
| `tests/integration/test_security_audit.py` | Security tests for CSV formula injection and upload caps | Verified |
| `tests/integration/test_database.py` | Integration tests for SQLAlchemy models | Verified |
| `tests/integration/test_smoke_e2e.py` | End-to-end integration smoke tests | Verified |
| `.github/workflows/ci.yml` | GitHub Actions workflow: lint, mypy, audit, and pytest | Verified |
