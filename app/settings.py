from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://docextract:docextract@localhost:5432/docextract"
    redis_url: str = "redis://localhost:6379/0"
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "docextract"
    minio_secure: bool = False
    debug: bool = False
    llm_fallback_enabled: bool = False
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3"
    llm_confidence_threshold: float = 0.5
    cloud_llm_api_key: str = ""
    cloud_llm_provider: str = ""  # "openai" or "anthropic"
    log_dir: str = "/app/logs"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
