"""Application settings using Pydantic."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Ollama LLM Configuration
    llm_ollama_base_url: str = "http://localhost:11434"
    llm_model_name: str = "gpt-oss:20b"
    llm_temperature: float = 0.0
    llm_request_timeout: int = 120
    llm_num_ctx: int = 8192

    # Chunking Configuration
    chunk_target_tokens: int = 6000
    chunk_overlap_tokens: int = 500
    chunk_min_tokens: int = 1000
    chunk_max_tokens: int = 7500

    # Processing Configuration
    max_retries: int = 3
    retry_delay_seconds: float = 2.0

    # Logging
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
