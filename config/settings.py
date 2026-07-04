"""
Central application configuration.

All configuration is environment-driven (12-factor style) via a `.env` file
at the project root, with sane defaults for local development. Every other
module in the project (backend, training scripts, RAG pipeline, frontend)
imports the single `settings` instance from here instead of reading
`os.environ` directly, so configuration stays in one auditable place.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Absolute path to the project root (two levels up from this file:
# config/settings.py -> config/ -> project root).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Typed, validated application settings.

    Values are loaded, in order of precedence, from:
    1. Actual environment variables (highest precedence)
    2. A `.env` file at the project root
    3. The defaults declared below
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- General -----------------------------------------------------
    APP_NAME: str = "NeuroVision AI"
    APP_ENV: str = Field(default="development", description="development|staging|production")
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"

    # ---- Server --------------------------------------------------------
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    FRONTEND_PORT: int = 8501
    # Stored as a raw comma-separated string because pydantic-settings tries
    # to JSON-decode list-typed env vars before validators run; the `.env`
    # file is meant to hold a plain `a,b,c` value, not JSON. Use the
    # `cors_origins` property below to get the parsed list.
    CORS_ORIGINS: str = "http://localhost:8501,http://localhost:3000"

    # ---- Database ------------------------------------------------------
    DATABASE_URL: str = Field(default=f"sqlite:///{PROJECT_ROOT / 'neurovision.db'}")

    # ---- Auth / security -------------------------------------------------
    SECRET_KEY: str = Field(default="CHANGE_ME_IN_PRODUCTION")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # ---- Paths -----------------------------------------------------------
    DATASETS_DIR: Path = PROJECT_ROOT / "datasets"
    MODELS_DIR: Path = PROJECT_ROOT / "models" / "saved_models"
    REPORTS_DIR: Path = PROJECT_ROOT / "reports" / "generated"
    RAG_DOCUMENTS_DIR: Path = PROJECT_ROOT / "rag" / "documents"
    RAG_VECTOR_STORE_DIR: Path = PROJECT_ROOT / "rag" / "vector_store"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"

    # ---- Model file names (relative to MODELS_DIR/<module>/) -------------
    CLASSIFICATION_MODEL_NAME: str = "efficientnet_b0_classifier.keras"
    SEGMENTATION_MODEL_NAME: str = "unet_segmentation.keras"
    MULTIMODAL_MODEL_NAME: str = "multimodal_fusion.keras"
    PROGNOSIS_MODEL_NAME: str = "xgboost_prognosis.pkl"

    # ---- Image / training defaults ----------------------------------------
    IMAGE_SIZE: int = 224
    BATCH_SIZE: int = 32
    RANDOM_SEED: int = 42
    CLASS_NAMES: str = "glioma,meningioma,notumor,pituitary"

    # ---- LLM / RAG provider keys (optional; feature-gated at call sites) --
    OPENAI_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    RAG_CHUNK_SIZE: int = 800
    RAG_CHUNK_OVERLAP: int = 120
    RAG_TOP_K: int = 4

    # ---- Logging -----------------------------------------------------------
    LOG_LEVEL: str = "INFO"

    @property
    def cors_origins_list(self) -> list[str]:
        """Parsed `CORS_ORIGINS` as a list, for use in CORS middleware."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @property
    def class_names_list(self) -> list[str]:
        """Parsed `CLASS_NAMES` as a list, for use by training/inference code."""
        return [name.strip() for name in self.CLASS_NAMES.split(",") if name.strip()]

    def ensure_directories(self) -> None:
        """Create runtime directories if they don't already exist.

        Called once at application startup so the app never fails on a
        missing folder (e.g. a fresh clone with no `logs/` yet).
        """
        for directory in (
            self.MODELS_DIR,
            self.REPORTS_DIR,
            self.RAG_DOCUMENTS_DIR,
            self.RAG_VECTOR_STORE_DIR,
            self.LOGS_DIR,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton `Settings` instance.

    `lru_cache` ensures the `.env` file is parsed once per process rather
    than on every import, while still allowing tests to override
    dependencies (e.g. via FastAPI's `dependency_overrides`).
    """
    settings = Settings()
    settings.ensure_directories()
    return settings


settings = get_settings()
