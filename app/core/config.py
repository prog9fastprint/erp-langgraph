"""
Application Configuration settings loaded from environment variables and .env file.
"""
import os
import sys
from typing import Literal, Optional, List
from pydantic import Field, AliasChoices, field_validator
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
    SSE_HEARTBEAT_SECONDS: float = Field(default=15.0, validation_alias="SSE_HEARTBEAT_SECONDS")

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
    ERP_LOGIN_SECRET: Optional[str] = Field(default=None, validation_alias="ERP_LOGIN_SECRET")
    REQUEST_TIMEOUT_SEC: int = Field(
        default=15,
        validation_alias=AliasChoices("REQUEST_TIMEOUT_SEC", "REQUEST_TIMEOUT"),
    )

    # 2. LLM Provider Priority: 1. OpenAI -> 2. Gemini -> 3. OpenRouter
    MODEL_PROVIDER: str = "openai"

    # OpenAI Configuration (Priority 1)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_TEMPERATURE: Optional[float] = Field(
        default=None,
        validation_alias="OPENAI_TEMPERATURE",
    )
    OPENAI_REASONING_EFFORT: Optional[str] = Field(
        default=None,
        validation_alias="OPENAI_REASONING_EFFORT",
    )

    # Google Gemini Configuration (Priority 2)
    GOOGLE_AI_API_KEY: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("GOOGLE_AI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"),
    )
    GOOGLE_AI_API_KEY_2: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("GOOGLE_AI_API_KEY_2", "GEMINI_API_KEY_2"),
    )
    GOOGLE_AI_API_KEY_3: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("GOOGLE_AI_API_KEY_3", "GEMINI_API_KEY_3"),
    )
    GEMINI_MODEL: str = Field(default="gemini-2.5-flash", validation_alias="GEMINI_MODEL")

    # OpenRouter Configuration (Priority 3)
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_MODEL: str = "nvidia/nemotron-3.5-super-vl:free"

    def get_gemini_api_keys(self) -> List[str]:
        """Return all non-empty configured Google Gemini API keys in priority order."""
        keys: List[str] = []
        for k in [self.GOOGLE_AI_API_KEY, self.GOOGLE_AI_API_KEY_2, self.GOOGLE_AI_API_KEY_3]:
            if k and str(k).strip():
                keys.append(str(k).strip())
        return keys

    # 3. LangGraph & Tools Settings
    ENABLE_HITL_INTERRUPT: bool = False
    ERP_MULTI_COMPANY_CONCURRENCY: int = 16

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
        validation_alias=AliasChoices("GOOGLE_AI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"),
    )
    GEMINI_EMBEDDING_MODEL: str = Field(default="gemini-embedding-001", validation_alias="GEMINI_EMBEDDING_MODEL")

    # 6. Dynamic Semantic Tool Retrieval & RBAC
    DYNAMIC_TOOL_RETRIEVAL_ENABLED: bool = Field(default=True, validation_alias="DYNAMIC_TOOL_RETRIEVAL_ENABLED")
    TOOL_CATALOG_TABLE_NAME: str = Field(default="ai_tool_catalog", validation_alias="TOOL_CATALOG_TABLE_NAME")
    TOOL_RETRIEVAL_TOP_K: int = Field(default=8, validation_alias="TOOL_RETRIEVAL_TOP_K")
    TOOL_RETRIEVAL_DISTANCE_THRESHOLD: float = Field(default=0.72, validation_alias="TOOL_RETRIEVAL_DISTANCE_THRESHOLD")

    # 7. Hybrid Semantic NLU Router & Shadow Rollout
    NLU_ROUTER_MODE: Literal["legacy_regex", "semantic_llm", "shadow"] = Field(
        default="legacy_regex",
        validation_alias="NLU_ROUTER_MODE",
    )
    NLU_SHADOW_LOG_MISMATCH_ONLY: bool = Field(
        default=True,
        validation_alias="NLU_SHADOW_LOG_MISMATCH_ONLY",
    )
    NLU_ROUTER_PROVIDER: Optional[str] = Field(
        default=None,
        validation_alias="NLU_ROUTER_PROVIDER",
    )
    NLU_ROUTER_MODEL: Optional[str] = Field(
        default=None,
        validation_alias="NLU_ROUTER_MODEL",
    )
    NLU_ROUTER_REASONING_EFFORT: Optional[str] = Field(
        default=None,
        validation_alias="NLU_ROUTER_REASONING_EFFORT",
    )

    # 8. Synthesizer Latency Optimization
    SYNTHESIZER_FAST_DETERMINISTIC: bool = Field(
        default=False,
        validation_alias="SYNTHESIZER_FAST_DETERMINISTIC",
    )

    @field_validator("NLU_ROUTER_MODE", mode="before")
    @classmethod
    def _normalize_nlu_router_mode(cls, value):
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("OPENAI_TEMPERATURE", mode="before")
    @classmethod
    def _normalize_optional_float(cls, value):
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("OPENAI_REASONING_EFFORT", "NLU_ROUTER_REASONING_EFFORT", mode="before")
    @classmethod
    def _normalize_optional_effort(cls, value):
        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned or None
        return value


def _is_test_runtime() -> bool:
    """Return True when running inside unit tests so local .env secrets never leak."""
    return (
        "unittest" in sys.modules
        or os.environ.get("FASTPRINT_TEST_ENV") == "true"
        or os.environ.get("APP_ENV", "").strip().lower() == "test"
    )


settings = Settings(_env_file=None if _is_test_runtime() else ".env")
