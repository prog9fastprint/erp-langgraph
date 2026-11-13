"""
Application Configuration settings loaded from environment variables and .env file.
"""
from typing import Optional
from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 0. Service Configuration
    APP_NAME: str = "erp-ai-orchestrator"
    APP_ENV: str = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8002
    LOG_LEVEL: str = "info"
    APP_TIMEZONE: str = Field(default="Asia/Jakarta", validation_alias="APP_TIMEZONE")
    REQUIRE_USER_TOKEN: bool = Field(default=False, validation_alias="REQUIRE_USER_TOKEN")
    REQUIRE_SERVICE_KEY: bool = Field(default=False, validation_alias="REQUIRE_SERVICE_KEY")
    INTERNAL_SERVICE_KEY: Optional[str] = Field(default=None, validation_alias="INTERNAL_SERVICE_KEY")
    MAX_CHECKPOINT_THREADS: int = Field(default=500, validation_alias="MAX_CHECKPOINT_THREADS")

    # 1. ERP Backend Connection
    ERP_BASE_URL: str = Field(
        default="http://127.0.0.1:8000",
        validation_alias=AliasChoices("ERP_BASE_URL", "DJANGO_INTERNAL_API_URL"),
    )
    DEFAULT_COMPANY_ID: int = 1
    ERP_API_TOKEN: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("ERP_API_TOKEN", "DJANGO_INTERNAL_SERVICE_KEY"),
    )
    ERP_USERNAME: Optional[str] = None
    ERP_PASSWORD: Optional[str] = None
    REQUEST_TIMEOUT_SEC: int = Field(
        default=15,
        validation_alias=AliasChoices("REQUEST_TIMEOUT_SEC", "REQUEST_TIMEOUT"),
    )

    # 2. LLM Provider Selection: 'openrouter' or 'openai'
    MODEL_PROVIDER: str = "openrouter"

    # OpenRouter Configuration
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_MODEL: str = "nvidia/nemotron-3.5-super-vl:free"

    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"

    # 3. LangGraph & Tools Settings
    USE_DYNAMIC_TOOLS: bool = False
    ENABLE_HITL_INTERRUPT: bool = False

    # 4. Context Window & Memory Management
    MAX_WINDOW_MESSAGES: int = Field(default=10, validation_alias="MAX_WINDOW_MESSAGES")
    ENABLE_AUTO_SUMMARIZATION: bool = Field(default=True, validation_alias="ENABLE_AUTO_SUMMARIZATION")
    SUMMARIZATION_THRESHOLD_MESSAGES: int = Field(default=10, validation_alias="SUMMARIZATION_THRESHOLD_MESSAGES")
    ENABLE_PAYLOAD_SANITIZATION: bool = Field(default=True, validation_alias="ENABLE_PAYLOAD_SANITIZATION")

    # 5. RAG & Knowledge Settings
    ENABLE_RAG: bool = Field(default=True, validation_alias="ENABLE_RAG")
    RAG_TOP_K: int = Field(default=3, validation_alias="RAG_TOP_K")
    RAG_SIMILARITY_THRESHOLD: float = Field(default=0.65, validation_alias="RAG_SIMILARITY_THRESHOLD")
    VECTOR_DB_URL: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("VECTOR_DB_URL", "POSTGRES_VECTOR_URL"),
    )
    GEMINI_API_KEY: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    )
    GEMINI_EMBEDDING_MODEL: str = Field(default="gemini-embedding-001", validation_alias="GEMINI_EMBEDDING_MODEL")


settings = Settings()
