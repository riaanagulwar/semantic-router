"""
Central settings object. Every value that used to be a hardcoded module
constant in router.py / gemini_client.py lives here instead, sourced from
environment variables (or a local .env file) with defaults that match the
project's original hardcoded values — so behavior is unchanged out of the
box, but every knob is now overridable per-environment without a code
edit.

Usage:
    from app.config.settings import settings
    settings.similarity_threshold
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # router.py — the local embedding model and its matching thresholds.
    routes_config_path: str = "app/config/routes.yaml"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.55
    ambiguity_margin: float = 0.05
    max_query_len: int = 500

    # gemini_client.py — the LLM fallback path.
    gemini_api_key: str | None = None
    gemini_model_name: str = "gemini-3.8-flash"
    max_rpm: int = 10
    max_retries: int = 2
    # Google's API rejects any deadline below 10s outright (400
    # INVALID_ARGUMENT) — gemini_client.py also clamps up to this floor
    # defensively, but the default itself needs to already clear it.
    timeout_s: int = 10

    # api.py — HTTP auth and rate limiting. api_key has no default on
    # purpose: every request is rejected until it's explicitly set.
    api_key: str | None = None
    rate_limit: str = "30/minute"

    log_level: str = "INFO"


settings = Settings()
