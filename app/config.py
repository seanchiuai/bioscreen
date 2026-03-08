"""Application configuration via pydantic-settings.

Values are read from environment variables or a .env file.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the bioscreen application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ───────────────────────────────────────────────────────────
    app_env: str = Field("development", description="Runtime environment")
    log_level: str = Field("INFO", description="Logging verbosity")
    api_host: str = Field("0.0.0.0", description="API bind host")
    api_port: int = Field(8000, description="API bind port")

    # ── NVIDIA NIM ────────────────────────────────────────────────────────────
    nvidia_api_key: str = Field("", description="NVIDIA NIM API key")
    esmfold_api_url: str = Field(
        "https://health.api.nvidia.com/v1/biology/nvidia/esmfold",
        description="ESMFold NIM endpoint URL",
    )

    # ── Models ────────────────────────────────────────────────────────────────
    esm2_model_name: str = Field(
        "facebook/esm2_t33_650M_UR50D",
        description="HuggingFace model ID for ESM-2 embeddings",
    )
    device: str = Field("cpu", description="Torch device: 'cpu' or 'cuda'")

    # ── Database paths ────────────────────────────────────────────────────────
    toxin_db_path: Path = Field(
        Path("data/toxin_db.faiss"), description="FAISS index file"
    )
    toxin_meta_path: Path = Field(
        Path("data/toxin_meta.json"), description="Toxin metadata JSON"
    )

    # ── Foldseek ──────────────────────────────────────────────────────────────
    foldseek_bin: str = Field("foldseek", description="Path to foldseek binary")
    foldseek_db_path: Path = Field(
        Path("data/foldseek_db"), description="Foldseek database directory"
    )

    # ── UniProt build ─────────────────────────────────────────────────────────
    uniprot_batch_size: int = Field(500, description="Records per UniProt API page")
    max_toxin_records: int = Field(5000, description="Max toxins to fetch from UniProt")

    # ── Screening thresholds ──────────────────────────────────────────────────
    embedding_sim_threshold: float = Field(
        0.85, description="Cosine similarity threshold for embedding hits"
    )
    structure_sim_threshold: float = Field(
        0.70, description="TM-score threshold for structural hits"
    )
    risk_high_threshold: float = Field(
        0.75, description="Score above which risk is HIGH"
    )
    risk_medium_threshold: float = Field(
        0.50, description="Score above which risk is MEDIUM"
    )

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"

    @property
    def nim_headers(self) -> dict[str, str]:
        """Authorization headers for NVIDIA NIM API calls."""
        return {
            "Authorization": f"Bearer {self.nvidia_api_key}",
            "Content-Type": "application/json",
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance (singleton)."""
    return Settings()
