"""Configuration management using Pydantic Settings and PyYAML."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class NumericalFeatureConfig(BaseModel):
    name: str
    unit: str
    min_physical: float
    max_physical: float
    sensor_fault_delta_max: float


class FeaturesConfig(BaseModel):
    numerical: List[NumericalFeatureConfig]
    optional_identifiers: List[str] = Field(default_factory=list)


class SplitConfig(BaseModel):
    strategy: str = "stratified_kfold"
    test_size: float = 0.20
    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 42


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    aquasense_env: str = Field(default="development", alias="AQUASENSE_ENV")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_key: Optional[str] = Field(default="aquasense-dev-secret-key-2026", alias="API_KEY")
    database_url: str = Field(default="sqlite:///./aquasense.db", alias="DATABASE_URL")
    model_dir: str = Field(default="artifacts/models/xgboost_active", alias="MODEL_DIR")
    config_path: str = Field(default="config/config.yaml", alias="CONFIG_PATH")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @property
    def effective_database_url(self) -> str:
        """Returns database URL, redirecting default local SQLite to /tmp in serverless/Vercel environments."""
        is_serverless = bool(os.getenv("VERCEL") or os.getenv("VERCEL_ENV") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))
        if is_serverless and self.database_url.startswith("sqlite:///."):
            return "sqlite:////tmp/aquasense.db"
        return self.database_url



def load_yaml_config(path: str = "config/config.yaml") -> Dict[str, Any]:
    """Loads raw YAML configuration dictionary."""
    config_file = Path(path)
    if not config_file.exists():
        # Fallback to root or default location
        base_dir = Path(__file__).resolve().parent.parent.parent
        config_file = base_dir / path
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Singleton instances
settings = AppConfig()
yaml_config = load_yaml_config(settings.config_path)
