"""
LangGraph ReAct Agent with Conversational Memory and Human-in-the-Loop (HITL) Write Protection.
"""
from pathlib import Path
from typing import Any, List, Optional
import warnings
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage

from app.core.config import settings
from app.services.erp_client import ERPClient
from app.services.llm import get_chat_model
from app.tools.static import create_static_tools
from app.tools.dynamic import create_dynamic_tools


def load_system_prompt() -> str:
    """
    Read erp_system_prompt.md from disk using pathlib.
    Looks in project root prompts/ directory.
    """
    # app/graph/agent.py -> parents[2] is project root
    project_root = Path(__file__).resolve().parents[2]
    prompt_file = project_root / "prompts" / "erp_system_prompt.md"
    if not prompt_file.exists():
        raise FileNotFoundError(
            f"Required ERP system prompt file not found at: {prompt_file}. "
            "Please ensure prompts/erp_system_prompt.md is present."
        )
    return prompt_file.read_text(encoding="utf-8")


def create_erp_graph(
    client: Optional[ERPClient] = None,
    use_dynamic_tools: bool = False,
    model_provider: Optional[str] = None,
    checkpointer: Optional[Any] = None,
):
    """
    Build and compile the LangGraph agent graph.

    .. deprecated::
        Use app.graph.workflow.create_workflow_graph instead.

    Args:
        client: ERPClient instance. If None, initialized from settings.
        use_dynamic_tools: If True, fetches tools dynamically from ERP via GET /ai/tools/.
                           If False, uses static typed wrappers.
        model_provider: LLM provider override ('openai', 'gemini', 'openrouter').
        checkpointer: LangGraph checkpointer for state persistence. Defaults to MemorySaver().
    """
    warnings.warn(
        "create_erp_graph is deprecated. Use app.graph.workflow.create_workflow_graph instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    erp_client = client or ERPClient()

    # 1. Load Tools
    if use_dynamic_tools:
        tools = create_dynamic_tools(erp_client)
    else:
        tools = create_static_tools(erp_client)

    # 2. Initialize LLM Model
    model = get_chat_model(provider=model_provider)
    model_with_tools = model.bind_tools(tools)

    # 3. Load System Prompt
    system_prompt_text = load_system_prompt()

    # 4. Agent Node
    def call_model(state: MessagesState):
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt_text)] + list(messages)

        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    # 5. Routing Condition
    def should_continue(state: MessagesState):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    # 6. Tool Execution Node
    tool_node = ToolNode(tools)

    # 7. Assemble StateGraph
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "agent")

    # 8. Compile with Memory
    memory = checkpointer if checkpointer is not None else MemorySaver()
    compiled_graph = workflow.compile(
        checkpointer=memory,
        interrupt_before=["tools"] if settings.ENABLE_HITL_INTERRUPT else [],
    )

    return compiled_graph, tools
