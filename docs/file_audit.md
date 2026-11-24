# AquaSense AI Codebase File Audit

Generated for course URP4201 Research Methodology.
Covers every tracked file across the repository, documenting path, purpose, status, and changes.

| Path | Purpose | Status | What Changed |
| :--- | :--- | :--- | :--- |
| `config.yaml` | Master project and ML pipeline configuration | kept | Verified schema compliance, random seeds, features, and statutory limits |
| `pyproject.toml` | Build metadata, ruff, mypy, and pytest coverage configuration | fixed | Added `python-multipart>=0.0.6` runtime dependency for form handling |
| `requirements.txt` | Minimal runtime dependencies for Vercel serverless (< 150 MB target) | fixed | Added `python-multipart>=0.0.6` to ensure serverless batch processing runs cleanly |
| `requirements-train.txt` | Full research, dashboard, and testing dependencies | kept | Verified complete scientific stack (xgboost, shap, optuna, streamlit) |
| `vercel.json` | Vercel serverless function routing and exclusion rules | fixed | Configured query-preserving `_route` parameter rewrites for clean API routing |
| `.vercelignore` | Excludes non-runtime assets from production lambda deployment | kept | Enforces lambda bundle size budget by omitting heavy training assets |
| `.gitignore` | Excludes raw dataset CSVs, bytecode, caches, and local databases | kept | Ensures raw CSV data and temporary caches are untracked |
| `.env.example` | Environment variable template | kept | Maintained placeholder configuration without credentials |
| `.python-version` | Python runtime version pin (3.11) | kept | Standardizes execution environment |
| `.tool-versions` | Multi-tool version manager configuration | kept | Version alignment |
| `README.md` | Comprehensive academic documentation and benchmarks | fixed | Updated department to CSE (AI/ML), updated runtime dependencies, embedded diagnostic figures |
| `LICENSE` | Non-commercial academic research project license | kept | Formal academic project licensing |
| `data/README.md` | Dataset provenance, Kaggle source link, and download guide | kept | Authoritative documentation of raw dataset origin and license |
| `data/processed/train.csv` | Preprocessed base training partition | untracked (kept local) | Removed from Git index per license and data distribution guidelines |
| `data/processed/test.csv` | Independent holdout test partition | untracked (kept local) | Removed from Git index per license and data distribution guidelines |
| `data/raw/water_quality.csv` | Raw downloaded benchmark dataset | untracked (kept local) | Removed from Git index per Kaggle distribution terms; download via scripts/download_data.py |
| `scripts/download_data.py` | Automated Kaggle API dataset downloader and verification script | kept | Validates columns and schema against config.yaml |
| `scripts/export_model.py` | Model bundle exporter for production deployment | kept | Exports compact joblib bundle with preprocessing and calibrator |
| `scripts/check_bundle_size.py` | CI audit tool enforcing serverless bundle size < 250 MB | fixed | Added `multipart` auditing to verify serverless deployment size |
| `scripts/lint_design.py` | Design rule linter auditing typography, colors, and constraints | kept | Enforces zero em dashes, zero emoji, solid palette, and border-radius rules |
| `scripts/iot_simulator.py` | Real-time IoT sensor telemetry streaming simulator | kept | Simulates multi-station water parameter telemetry |
| `scripts/audit_codebase.py` | Automated syntax, secrets, and path integrity audit script | added | Automated codebase-wide audit verification |
| `src/__init__.py` | Source root package initializer | kept | Clean namespace setup |
| `src/aquasense/__init__.py` | Package initializer and version specification | kept | Package metadata |
| `src/aquasense/data.py` | Schema validation, duplicate removal, and stratified splitting | kept | Robust dataset handling and 4-way disjoint partitioning |
| `src/aquasense/features.py` | WHO/BIS domain ratio feature engineering (NumPy and Pandas) | kept | Domain feature transformer implementing OSDR, HMHI, and deviation features |
| `src/aquasense/train.py` | End-to-end model training, hyperparameter tuning, and ablations | kept | 4-way disjoint model training pipeline |
| `src/aquasense/evaluate.py` | Metric computation, LaTeX tables, and figure generation | kept | Metric computation, Brier scores, and PR curves |
| `src/aquasense/uncertainty.py` | Platt probability calibration and Inductive Conformal Prediction | kept | Least Ambiguous Classifier implementation with coverage guarantee |
| `src/aquasense/explain.py` | Offline TreeSHAP generation and runtime counterfactual occlusion | kept | Occlusion attribution algorithm against baseline training medians |
| `src/aquasense/anomaly.py` | Sensor boundary checks and Isolation Forest anomaly detector | kept | Physical sensor bounds and WHO/BIS threshold checks |
| `src/aquasense/db.py` | SQLAlchemy ORM models for local testing and transactions | kept | Database schema for sample and prediction storage |
| `api/__init__.py` | Serverless API package initializer | kept | Clean package setup |
| `api/index.py` | Lean FastAPI serverless handler | fixed | Registered `VercelRouteMiddleware`, added HTML route handlers for all pages, and dual API routes |
| `web/index.html` | Portal home page with project overview and methodology | fixed | Added dark/light theme toggle, updated department to CSE (AI/ML), responsive navigation |
| `web/analyze.html` | Interactive water screening form and batch CSV uploader | fixed | Added inline live field validation, parameter bounds checking, accessible tooltips |
| `web/batch.html` | Batch CSV screening workspace | fixed | Added drag-and-drop zone, formula injection protection, sortable results table |
| `web/results.html` | Experimental evaluation tables and diagnostic figures | fixed | Added tabbed interface, accessible diagnostic lightbox modal with focus trap |
| `web/monitoring.html` | Geographic Monitoring Network map | fixed | Added theme toggle button, centered on Bhubaneswar, Odisha (Kuakhai-Daya basin) |
| `web/about.html` | Research methodology, academic supervisors, and student team | fixed | Added accessible accordion system, updated author department records to CSE (AI/ML) |
| `web/privacy.html` | Academic data privacy and retention policy | fixed | Added theme toggle button to navigation header |
| `web/terms.html` | Terms of service and screening support limitations | fixed | Added theme toggle button to navigation header |
| `web/css/style.css` | Production design system stylesheet | fixed | Added dark theme tokens (solid colors, zero gradients), reduced motion query, lightbox and accordion styles |
| `web/js/main.js` | Production client application logic | fixed | Added theme toggle, live form validation, CSV sanitization, lightbox modal, tab switching, and accordion handlers |
| `web/template.csv` | Sample batch CSV template for download | kept | Clean template with valid parameter ranges |
| `web/css/style.css` | Accessible CSS stylesheet adhering to strict design rules | kept | Solid palette, border-radius <= 4px, WCAG AA compliant |
| `web/js/main.js` | Vanilla JS client handling API calls and localStorage history | fixed | Configured Bhubaneswar monitoring stations, added fallback API wrapper |
| `web/favicon.svg` | Custom geometric water droplet SVG mark | kept | Vector brand icon |
| `web/favicon-32x32.png` | 32x32 raster favicon | kept | Standard browser favicon |
| `web/apple-touch-icon.png` | 180x180 touch icon | kept | Mobile home screen icon |
| `web/figures/confusion_matrix.png` | Confusion matrix heatmap | added | Precomputed holdout confusion matrix in web assets directory |
| `web/figures/pr_curves.png` | Precision-recall curves across risk tiers | added | Precomputed holdout PR curves in web assets directory |
| `web/figures/reliability_diagram.png` | Calibration reliability curves | added | Precomputed probability calibration diagram in web assets directory |
| `web/figures/shap_summary.png` | TreeSHAP summary beeswarm visualization | added | Precomputed feature attribution beeswarm plot in web assets directory |
| `web/figures/feature_distributions.png` | Physicochemical parameter histograms | added | Precomputed univariate density distribution plot in web assets directory |
| `web/figures/correlation_matrix.png` | Pairwise feature correlation matrix | added | Precomputed Spearman correlation heatmap in web assets directory |
| `app/app.py` | Streamlit interactive research and exploration dashboard | kept | Local research exploration dashboard |
| `notebooks/01_dataset_audit.ipynb` | Jupyter notebook for dataset audit and summary stats | kept | Executed and validated notebook |
| `notebooks/02_exploratory_data_analysis.ipynb` | Jupyter notebook for parameter distributions and correlation | kept | Executed and validated notebook |
| `notebooks/03_model_training_and_ablations.ipynb` | Jupyter notebook for model benchmarks and ablation studies | kept | Executed and validated notebook |
| `notebooks/04_explainability_analysis.ipynb` | Jupyter notebook for global and local feature attributions | kept | Executed and validated notebook |
| `notebooks/05_uncertainty_quantification.ipynb` | Jupyter notebook for calibrated probabilities and conformal sets | kept | Executed and validated notebook |
| `docs/decisions.md` | Methodological rationale for classification and architecture | kept | Decision logs for 4-way disjoint design and model choices |
| `docs/domain.md` | DNS records and custom domain routing instructions | kept | DNS routing documentation |
| `docs/file_audit.md` | Full repository audit manifest | fixed | Updated with 4-column schema: Path, Purpose, Status, What Changed |
| `results/metrics.json` | Real evaluation numbers on holdout test partition | kept | Verified metrics (Macro-F1: 99.43%, Accuracy: 99.50%) |
| `results/model_comparison.csv` | Benchmark comparisons across model architectures | kept | Verified baseline vs RF vs XGBoost comparison table |
| `results/experiment_log.csv` | Log of all benchmark and ablation trials | kept | Verified experimental trial tracking |
| `results/tables/performance.tex` | Exported LaTeX performance table | kept | Paper-ready LaTeX performance table |
| `results/tables/ablations.tex` | Exported LaTeX ablation table | kept | Paper-ready LaTeX ablation table |
| `outputs/figures/confusion_matrix.png` | Confusion matrix heatmap | kept | Diagnostic evaluation artifact |
| `outputs/figures/pr_curves.png` | Precision-recall curves across risk tiers | kept | Diagnostic evaluation artifact |
| `outputs/figures/reliability_diagram.png` | Calibration reliability curves | kept | Diagnostic evaluation artifact |
| `outputs/figures/shap_summary.png` | TreeSHAP summary beeswarm visualization | kept | Diagnostic evaluation artifact |
| `outputs/figures/feature_distributions.png` | Physicochemical parameter histograms | kept | Diagnostic evaluation artifact |
| `outputs/figures/correlation_matrix.png` | Pairwise feature correlation matrix | kept | Diagnostic evaluation artifact |
| `models/model_bundle.joblib` | Compact deployed model artifact (136 KB) | kept | Active CalibratedClassifierCV bundle |
| `models/registry.json` | Metadata registry for deployed model bundle | kept | Active model registry metadata |
| `tests/conftest.py` | Pytest fixtures and test client configuration | kept | Test client and dataset fixtures |
| `tests/unit/test_anomaly.py` | Unit tests for physical and statutory anomaly checks | kept | Validates anomaly detector logic |
| `tests/unit/test_conformal.py` | Unit tests for conformal prediction and calibration | kept | Validates inductive conformal prediction set coverage |
| `tests/unit/test_features.py` | Unit tests for domain feature engineering mathematics | kept | Validates ratio and toxic threshold calculations |
| `tests/unit/test_preprocessor.py` | Unit tests for pipeline transformations | kept | Validates scikit-learn transformers |
| `tests/unit/test_property_validation.py` | Unit tests for Pydantic boundary fuzzing | kept | Validates boundary value validation |
| `tests/unit/test_explain.py` | Unit tests for occlusion attribution and rank stability | kept | Validates counterfactual occlusion attribution |
| `tests/unit/test_bundle_size.py` | Unit test verifying bundle size < 250 MB | kept | Validates lambda bundle size ceiling |
| `tests/unit/test_design_lint.py` | Unit test verifying zero design rule violations | kept | Validates design rule compliance in pytest |
| `tests/integration/test_api.py` | Integration tests for all FastAPI endpoints | kept | Tests /health, /predict, /explain, /anomaly, /models |
| `tests/integration/test_leakage.py` | Integration tests for zero data leakage | kept | Confirms strictly disjoint partitions |
| `tests/integration/test_metric_regression.py` | Performance floor regression tests | kept | Asserts performance floors on holdout set |
| `tests/integration/test_security_audit.py` | Security tests for CSV formula injection and upload caps | kept | Asserts security constraints |
| `tests/integration/test_database.py` | Integration tests for SQLAlchemy models | kept | Validates SQLite persistence |
| `tests/integration/test_smoke_e2e.py` | End-to-end integration smoke tests | kept | Validates complete inference workflow |
| `scripts/rerun_notebooks.py` | Pipeline to rerun all research notebooks in sequence | added | Clears outputs and executes 01 through 05 via nbconvert |
| `scripts/generate_readme.py` | Synchronizes metrics in README.md with results/metrics.json | added | Programmatically audits documentation metric consistency |
| `tests/e2e/test_ui.py` | Playwright end-to-end browser tests | added | Validates all pages, theme switcher, forms, modals, tabs, accordions, and mobile navigation |
| `.github/workflows/ci.yml` | GitHub Actions workflow: lint, mypy, audit, and pytest | kept | Continuous integration pipeline |
