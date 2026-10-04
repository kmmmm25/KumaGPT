from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "KumaGPT API"
    api_prefix: str = "/api/v1"
    database_url: str = "sqlite:///./data/kumagpt.db"
    cors_origins: str = "http://localhost:5173"
    model_backend: str = "torch"
    model_path: Path = PROJECT_ROOT / "KumaGPT4.pt"
    tokenizer_path: Path = PROJECT_ROOT / "kumagpt_unigram.model"
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_k: int = 30
    inference_timeout_seconds: float = 120
    max_content_length: int = 4000
    context_message_limit: int = 10

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="KUMAGPT_", extra="ignore"
    )

    @property
    def allowed_origins(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
