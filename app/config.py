"""Configuration layer.

Single source of truth for env-driven settings. Every other layer (db, providers,
middleware) reads config through the `Settings` object injected via `get_settings()`
rather than reading `os.environ` directly, so behaviour stays testable and overridable.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    app_name: str = "AI Gateway Service"
    app_env: Literal["development", "production", "test"] = "development"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # --- Database ---
    # aiosqlite today; swapping to postgresql+asyncpg://... is the only change needed
    # to move onto Postgres (see app/db/repositories/interfaces.py for the rest of the story).
    database_url: str = "sqlite+aiosqlite:///./data/gateway.db"

    # --- Redis (rate limiting store) ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Provider credentials ---
    # A provider with no API key configured is simply not registered at startup
    # (see app/providers/registry.py) rather than erroring — lets the gateway run
    # with whichever subset of providers the deployer has keys for.
    openai_api_key: str | None = None
    openai_default_model: str = "gpt-4o-mini"
    # None -> the OpenAI SDK's own default (api.openai.com). Set to an OpenAI-compatible
    # endpoint (e.g. "https://openrouter.ai/api/v1") to route the "openai" provider slot
    # through a different backend without any code changes — just match `model` to
    # whatever model string that endpoint expects (OpenRouter uses e.g. "openai/gpt-4o-mini").
    openai_base_url: str | None = None

    anthropic_api_key: str | None = None
    anthropic_default_model: str = "claude-haiku-4-5"

    google_api_key: str | None = None
    gemini_default_model: str = "gemini-1.5-flash"

    # Ollama has no API key (local daemon) so it's always registered if its base URL responds.
    ollama_base_url: str = "http://localhost:11434"
    ollama_default_model: str = "llama3.1"

    # --- Fallback ---
    # Comma-separated provider names, tried in order for requests that don't specify
    # their own `fallback` list. Stored as a plain string (not a list) so it round-trips
    # through a .env value with zero custom parsing; split via `default_fallback_chain_list`.
    default_fallback_chain: str = "openai,anthropic,gemini"

    # --- Rate limiting ---
    rate_limit_requests_per_window: int = 60
    rate_limit_window_seconds: int = 60

    # --- Provider HTTP calls ---
    provider_request_timeout_seconds: float = 30.0

    # --- Auth ---
    api_key_header_name: str = "X-API-Key"

    # --- CORS ---
    # Comma-separated origins, same round-trip pattern as default_fallback_chain. "*" (default)
    # allows any origin — fine for a portfolio/demo deployment; lock this down to real origins
    # for anything production-facing, e.g. "https://myapp.com,https://staging.myapp.com".
    cors_allowed_origins: str = "*"

    @property
    def default_fallback_chain_list(self) -> list[str]:
        return [name.strip() for name in self.default_fallback_chain.split(",") if name.strip()]

    @property
    def cors_allowed_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allowed_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Process-wide cached Settings instance, used as a FastAPI dependency."""
    return Settings()
