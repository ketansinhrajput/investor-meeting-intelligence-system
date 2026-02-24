"""Ollama LLM client configuration."""

from functools import lru_cache

from langchain_ollama import OllamaLLM
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """LLM configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        extra="ignore",
    )

    ollama_base_url: str = "http://localhost:11434"
    # model_name: str = "gemma3:latest"
    model_name: str = "gpt-oss:20b"
    # fallback_model_name: str = "gemma3:latest"  # Fallback when primary returns empty
    temperature: float = 0.0
    request_timeout: int = 120
    num_ctx: int = 8192
    num_predict: int = 4096  # Max tokens to generate


class LLMResponse(BaseModel):
    """Wrapper for LLM response with metadata."""

    content: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@lru_cache
def get_llm_settings() -> LLMSettings:
    """Get cached LLM settings."""
    return LLMSettings()


def create_llm_client(settings: LLMSettings | None = None) -> OllamaLLM:
    """Create configured Ollama LLM client.

    Args:
        settings: Optional custom settings. Uses defaults if not provided.

    Returns:
        Configured OllamaLLM instance.
    """
    settings = settings or get_llm_settings()

    return OllamaLLM(
        model=settings.model_name,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
        timeout=settings.request_timeout,
        num_ctx=settings.num_ctx,
        num_predict=settings.num_predict,
        streaming=False,
    )


def create_json_llm_client(settings: LLMSettings | None = None, use_fallback: bool = False) -> OllamaLLM:
    """Create LLM client configured for JSON output.

    Args:
        settings: Optional custom settings.
        use_fallback: If True, use the fallback model instead of primary.

    Returns:
        OllamaLLM instance configured for JSON responses.
    """
    settings = settings or get_llm_settings()
    model = settings.fallback_model_name if use_fallback else settings.model_name

    return OllamaLLM(
        model=model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
        timeout=settings.request_timeout,
        num_ctx=settings.num_ctx,
        num_predict=settings.num_predict,
        # Note: format="json" removed as gpt-oss:20b doesn't respect it
        # and may cause truncated responses. JSON extraction is handled
        # in chains.py _parse_json_response instead.
        streaming=False,
    )


def get_fallback_model_name() -> str:
    """Get the fallback model name from settings."""
    return get_llm_settings().fallback_model_name


def get_primary_model_name() -> str:
    """Get the primary model name from settings."""
    return get_llm_settings().model_name
