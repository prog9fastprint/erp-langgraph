"""
State definitions for LangGraph agent workflows in FastPrint ERP AI Orchestrator.
Provides typed state management, conversational memory reducers, and multi-company scoping.
"""
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

from app.core.config import settings
from app.services.llm import get_default_model_name

# Structured verification status literal
VerificationStatus = Literal["VALID", "EMPTY", "RETRY", "ERROR"]
QueryMode = Literal["live_erp", "vector_rag", "hybrid", "general", "out_of_scope"]


class ToolCallPlan(TypedDict, total=False):
    """
    Subagent tool execution plan item for multi-tool / multi-intent orchestration.

    Attributes:
        tool: Specific ERP tool name (e.g., 'get_stock_level', 'get_account_balance').
        domain: ERP domain classification ('inventory', 'accounting', etc.).
        sub_query: Focused user query fragment for this subagent execution.
        params: Extracted and validated parameter dictionary for the tool schema.
        status: Execution lifecycle state ('pending', 'success', 'error').
    """
    tool: str
    domain: str
    sub_query: Optional[str]
    params: Optional[Dict[str, Any]]
    status: Optional[str]


class AgentState(TypedDict, total=False):
    """
    Centralized LangGraph Agent State.

    Combines conversational message history with structured pipeline attributes
    for domain routing, parameter extraction, tool execution, and guardrail verification.
    """

    # 1. Conversational Memory (LangGraph checkpointer & memory integration)
    messages: Annotated[List[BaseMessage], add_messages]

    # 2. User Context & Multi-Company Scoping
    user_query: str
    user_token: Optional[str]
    company_id: Optional[int]

    # 3. Domain & Tool Routing (Single & Multi-Tool)
    detected_domain: Optional[str]        # Legacy/primary domain e.g., "accounting", "inventory"
    detected_tool: Optional[str]          # Legacy/primary tool e.g., "get_account_balance"
    extracted_params: Optional[Dict[str, Any]] # Legacy/primary tool parameters
    is_multi_tool: Optional[bool]         # True if query spans multiple ERP intents/tools
    detected_domains: Optional[List[str]] # List of all detected domains e.g. ["inventory", "accounting"]
    detected_tools: Optional[List[str]]   # List of all detected tool names
    active_tools: Optional[List[ToolCallPlan]] # Ordered list of subagent tool execution plans

    # 4. Tool Execution & Output Data
    tool_result: Optional[Dict[str, Any]] # Legacy/primary raw data & calculations returned from ERP
    multi_tool_results: Optional[Dict[str, Any]] # Tool execution results mapped by tool_name

    # 5. Guardrail & Verification Cycle
    verification_status: Optional[VerificationStatus] # Aggregate status: "VALID", "EMPTY", "RETRY", "ERROR"
    multi_verification_status: Optional[Dict[str, str]] # Per-tool verification status
    retry_count: int
    retry_instruction: Optional[str]

    # 6. Final Presentation & Error Reporting
    final_answer: Optional[str]
    error_message: Optional[str]
    suggested_followups: Optional[List[Dict[str, str]]]
    active_model: Optional[str]

    # 7. Context Management & Long-Term Memory
    conversation_summary: Optional[str]

    # 8. RAG Knowledge & Retrieval Context
    query_mode: Optional[QueryMode]
    rag_query: Optional[str]
    rag_context: Optional[List[Dict[str, Any]]]
    rag_sources: Optional[List[str]]


def create_initial_state(
    user_query: str,
    user_token: Optional[str] = None,
    company_id: Optional[int] = None,
    messages: Optional[List[BaseMessage]] = None,
    is_multi_tool: Optional[bool] = False,
    active_model: Optional[str] = None,
) -> AgentState:
    """
    Construct a clean initial AgentState dictionary with standard defaults.

    Args:
        user_query: The incoming natural language query string from the user.
        user_token: Optional user JWT access token.
        company_id: Optional target company ID (defaults to settings.DEFAULT_COMPANY_ID).
        messages: Optional pre-existing message list (if None, initializes with HumanMessage).
        is_multi_tool: Flag indicating if the state is initialized for multi-tool workflow.

    Returns:
        A fully initialized AgentState dictionary.
    """
    initial_messages: List[BaseMessage] = (
        list(messages) if messages is not None else [HumanMessage(content=user_query)]
    )

    return {
        "messages": initial_messages,
        "user_query": user_query,
        "user_token": user_token,
        "company_id": company_id if company_id is not None else settings.DEFAULT_COMPANY_ID,
        "detected_domain": None,
        "detected_tool": None,
        "extracted_params": None,
        "tool_result": None,
        "verification_status": None,
        "retry_count": 0,
        "retry_instruction": None,
        "final_answer": None,
        "error_message": None,
        "conversation_summary": None,
        "query_mode": "live_erp",
        "rag_query": None,
        "rag_context": None,
        "rag_sources": None,
        # Multi-tool orchestration state extensions
        "is_multi_tool": is_multi_tool if is_multi_tool is not None else False,
        "active_model": active_model or get_default_model_name(),
        "detected_domains": [],
        "detected_tools": [],
        "active_tools": [],
        "multi_tool_results": {},
        "multi_verification_status": {},
    }
