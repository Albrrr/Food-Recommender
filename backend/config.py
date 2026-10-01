"""Application configuration.

Every setting is read from the environment (or a local .env) exactly once and
validated at import time, so a misconfigured deployment fails at startup with a
clear message instead of midway through a request.

Secrets and anything deployment-specific are required. Tuning knobs carry
sensible defaults so a developer only has to supply an API key to get running.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
EmbedBackend = Literal["sentence-transformers", "fastembed"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # tolerate unrelated vars (HF_TOKEN, PATH, ...) in .env
    )

    # --- required ------------------------------------------------------------
    openrouter_api_key: SecretStr

    # --- generation ----------------------------------------------------------
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "meta-llama/llama-3.3-70b-instruct"
    llm_timeout: float = Field(default=30.0, gt=0)
    llm_max_tokens: int = Field(default=700, ge=64)
    llm_max_retries: int = Field(default=2, ge=0)
    # OpenRouter provider routing. Llama 3.3 70B is served by providers whose
    # throughput differs by more than an order of magnitude; left to itself the
    # router picked one running at ~8 tok/s. "throughput" sorts for speed.
    llm_provider_sort: Literal["throughput", "price", "latency", "default"] = (
        "throughput"
    )

    # --- retrieval -----------------------------------------------------------
    embed_backend: EmbedBackend = "sentence-transformers"
    embed_model: str = "Qwen/Qwen3-Embedding-0.6B"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    retrieve_k: int = Field(default=30, ge=1)
    rerank_top_n: int = Field(default=6, ge=1)

    # --- api -----------------------------------------------------------------
    max_query_length: int = Field(default=500, ge=1)
    rate_limit: str = "10/minute"
    cors_origins: str = "*"
    log_level: LogLevel = "INFO"

    @model_validator(mode="before")
    @classmethod
    def _blank_means_unset(cls, values):
        """Treat `KEY=` in a .env as "not configured" rather than an empty string.

        pydantic-settings reads a blank entry as "", which silently overrides the
        field default - so a half-filled .env produces an empty model name and a
        confusing failure deep inside the model loader instead of a default.
        """
        if isinstance(values, dict):
            return {
                k: v
                for k, v in values.items()
                if not (isinstance(v, str) and not v.strip())
            }
        return values

    @property
    def cors_origin_list(self) -> list[str]:
        """CORS_ORIGINS as a list.

        Kept as a comma-separated string on the model rather than a `list[str]`:
        pydantic-settings JSON-decodes complex types from env vars, so a plain
        `a.com,b.com` would fail to parse with a confusing error.
        """
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @model_validator(mode="after")
    def _rerank_fits_within_retrieval(self) -> "Settings":
        if self.rerank_top_n > self.retrieve_k:
            raise ValueError(
                f"rerank_top_n ({self.rerank_top_n}) cannot exceed retrieve_k "
                f"({self.retrieve_k}) - the reranker can only narrow the candidate set."
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor so the environment is read and validated once per process."""
    return Settings()
