"""
Chat schemas for FastPrint ERP LangGraph AI Orchestrator API.
Provides request and response validation models for the /api/v1/chat endpoint.
"""
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatMessageHistoryItem(BaseModel):
    """
    Representation of an individual conversation message for stateless session recovery.
    """
    model_config = ConfigDict(extra="ignore")

    role: str = Field(..., description="Role of the sender: 'user' or 'assistant'")
    content: str = Field(..., description="Text content of the message")


class ChatRequest(BaseModel):
    """
    Incoming request model for ERP AI Chat Endpoint.
    """
    model_config = ConfigDict(extra="ignore")

    query: str = Field(..., min_length=1, description="Natural language query from user")
    company_id: Optional[int] = Field(default=1, description="Target ERP company ID (defaults to 1)")
    user_token: Optional[str] = Field(default=None, description="Optional JWT authentication token")
    thread_id: Optional[str] = Field(default=None, description="Session thread ID for conversational memory")
    history: Optional[List[ChatMessageHistoryItem]] = Field(
        default=None,
        description="Optional list of recent conversation messages for stateless memory recovery"
    )

    @field_validator("query")
    @classmethod
    def validate_non_empty_query(cls, v: str) -> str:
        """Ensure query is not empty or pure whitespace."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query string cannot be empty or whitespace only.")
        return cleaned


class FollowupSuggestion(BaseModel):
    """
    Structured recommendation for contextual next-step user prompts.
    """
    model_config = ConfigDict(extra="ignore")

    label: str = Field(..., description="Short button label displayed on chip")
    query: str = Field(..., description="Full executable prompt triggered when clicked")
    icon: Optional[str] = Field(default="fa-arrow-right", description="FontAwesome icon class")


class ChatResponse(BaseModel):
    """
    Outgoing response model for ERP AI Chat Endpoint.
    """
    model_config = ConfigDict(extra="ignore")

    final_answer: Optional[str] = Field(default=None, description="Synthesized Markdown answer")
    detected_domain: Optional[str] = Field(default=None, description="Classified intent domain")
    verification_status: Optional[str] = Field(default=None, description="Verification status (VALID, EMPTY, RETRY, ERROR)")
    error_message: Optional[str] = Field(default=None, description="Error message if execution encountered failures")
    suggested_followups: Optional[List[FollowupSuggestion]] = Field(
        default_factory=list,
        description="Context-aware follow-up queries generated directly by AI provider"
    )
    resolved_company_id: Optional[int] = Field(
        default=None,
        description="Effective ERP company ID used for query execution (reflects branch switches)"
    )
    execution_time_ms: Optional[float] = Field(
        default=None,
        description="End-to-end workflow execution latency in milliseconds"
    )
