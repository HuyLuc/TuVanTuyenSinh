"""Cấu hình toàn dự án, đọc từ biến môi trường hoặc file .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    gemini_api_key: str = ""
    gemini_model_answer: str = "gemini-2.5-flash"
    gemini_model_router: str = "gemini-2.5-flash-lite"

    llm_cache_enabled: bool = True
    llm_max_retries: int = 5
    llm_retry_max_wait: float = 60.0

    data_dir: Path = PROJECT_ROOT / "data"

    query_log_backend: Literal["local", "hf_dataset"] = "local"
    hf_log_repo_id: str = ""
    hf_token: str = ""

    @property
    def cache_path(self) -> Path:
        return self.data_dir / "cache" / "llm_cache.sqlite"

    @property
    def qdrant_path(self) -> Path:
        return self.data_dir / "qdrant"

    @property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"


@lru_cache
def get_settings() -> Settings:
    return Settings()
