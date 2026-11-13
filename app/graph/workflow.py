"""
StateGraph Workflow Assembly & Compilation for FastPrint ERP LangGraph Agent (TASK-4.6).
Orchestrates router, extractor, tool execution, verifier, and synthesizer nodes
into an end-to-end multi-node pipeline with conditional branching, retry cycles,
and multi-turn conversational memory.
"""
import atexit
import asyncio
from collections import OrderedDict
import concurrent.futures
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple
import uuid
import weakref
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

from app.core.config import settings
from app.graph.state import AgentState, create_initial_state
from app.graph.nodes.summarizer import summarizer_node
from app.graph.nodes.router import router_node
from app.graph.nodes.rag_node import rag_node
from app.graph.nodes.extractor import extractor_node
from app.graph.nodes.tool_node import tool_node
from app.graph.nodes.verifier import verifier_node, decide_after_verification
from app.graph.nodes.synthesizer import synthesizer_node

logger = logging.getLogger("workflow")


def decide_after_router(state: AgentState) -> str:
    """
    Conditional edge routing after the router node.

    Routing rules:
    - If query_mode is 'vector_rag' or domain is 'knowledge':
      Routes directly to 'rag_node' for semantic document retrieval.
    - If domain is 'general' or 'out_of_scope':
      Routes directly to 'synthesizer' to provide general guidance or explain guardrails.
    - If active_tools contains planned tool calls or legacy detected_tool is set
      for accounting/inventory:
      Routes to 'extractor' to parse query entities and dates (supports multi-tool & single-tool).
    - Otherwise (no tools matched or unrecognized domain):
      Routes directly to 'synthesizer'.

    Args:
        state: Current AgentState.

    Returns:
        Target node identifier ('rag_node', 'extractor', or 'synthesizer').
    """
    query_mode = state.get("query_mode")
    detected_tool = state.get("detected_tool")
    detected_domain = state.get("detected_domain")
    active_tools = state.get("active_tools") or []

    # 1. Vector RAG route for SOPs, specs, and qualitative knowledge
    if query_mode == "vector_rag" or detected_domain == "knowledge":
        return "rag_node"

    # 2. Guardrail: general & out_of_scope inquiries NEVER invoke tool extraction
    if detected_domain in ("general", "out_of_scope"):
        return "synthesizer"

    # 3. Active ERP tools check (supporting multi-tool active_tools & scalar detected_tool)
    has_active_tools = any(
        isinstance(tool_call, dict) and bool(tool_call.get("tool"))
        for tool_call in active_tools
    )
    has_scalar_tool = bool(detected_tool) and detected_domain in ("accounting", "inventory")

    if has_active_tools or has_scalar_tool:
        return "extractor"

    return "synthesizer"


def decide_after_tool(state: AgentState) -> str:
    """
    Routing after live tool execution:
    - Hybrid queries proceed to 'rag_node' to fetch complementary SOP / policy context.
    - Pure live_erp queries proceed directly to 'verifier'.
    """
    if state.get("query_mode") == "hybrid":
        return "rag_node"
    return "verifier"


def decide_after_rag(state: AgentState) -> str:
    """
    Routing after RAG retrieval:
    - Hybrid queries proceed to 'verifier' to validate the tool execution data.
    - Pure vector_rag queries proceed directly to 'synthesizer'.
    """
    if state.get("query_mode") == "hybrid":
        return "verifier"
    return "synthesizer"


def create_workflow_graph(
    checkpointer: Optional[Any] = None,
    interrupt_before: Optional[List[str]] = None,
):
    """
    Construct, connect, and compile the full StateGraph pipeline.

    Flow:
        START -> router -> (conditional: extractor / synthesizer)
        extractor -> tool_node -> verifier -> (conditional: extractor / synthesizer)
        synthesizer -> END

    Args:
        checkpointer: LangGraph checkpoint saver (defaults to MemorySaver()).
        interrupt_before: List of node names to pause for human approval.

    Returns:
        Compiled LangGraph StateGraph instance.
    """
    workflow = StateGraph(AgentState)

    # 1. Register Pipeline Nodes
    workflow.add_node("summarizer", summarizer_node)
    workflow.add_node("router", router_node)
    workflow.add_node("rag_node", rag_node)
    workflow.add_node("extractor", extractor_node)
    workflow.add_node("tool_node", tool_node)
    workflow.add_node("verifier", verifier_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # 2. Connect Edges
    workflow.add_edge(START, "summarizer")
    workflow.add_edge("summarizer", "router")
    workflow.add_conditional_edges(
        "router",
        decide_after_router,
        {
            "rag_node": "rag_node",
            "extractor": "extractor",
            "synthesizer": "synthesizer",
        },
    )
    workflow.add_edge("extractor", "tool_node")
    workflow.add_conditional_edges(
        "tool_node",
        decide_after_tool,
        {
            "rag_node": "rag_node",
            "verifier": "verifier",
        },
    )
    workflow.add_conditional_edges(
        "rag_node",
        decide_after_rag,
        {
            "verifier": "verifier",
            "synthesizer": "synthesizer",
        },
    )
    workflow.add_conditional_edges(
        "verifier",
        decide_after_verification,
        {
            "extractor": "extractor",
            "synthesizer": "synthesizer",
        },
    )
    workflow.add_edge("synthesizer", END)

    # 3. Checkpointer & HITL Interrupts
    memory = checkpointer if checkpointer is not None else MemorySaver()
    interrupts = (
        interrupt_before
        if interrupt_before is not None
        else (["tool_node"] if settings.ENABLE_HITL_INTERRUPT else [])
    )

    compiled_graph = workflow.compile(
        checkpointer=memory,
        interrupt_before=interrupts,
    )
    return compiled_graph


# Singleton default compiled graph
default_workflow_graph = create_workflow_graph()


def _hydrate_messages_from_history(
    history: Optional[List[Any]],
    user_query: str,
) -> Optional[List[BaseMessage]]:
    """
    Build LangChain message list from serialized history payload if available.
    Ensures recent conversational context is preserved when RAM checkpointer is empty.
    """
    if not history:
        return None
    hydrated: List[BaseMessage] = []
    for item in history:
        role = getattr(item, "role", None) or (item.get("role") if isinstance(item, dict) else "")
        content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else "")
        if not content or not isinstance(content, str) or not content.strip():
            continue
        role_lower = str(role).strip().lower()
        if role_lower in ("user", "human"):
            hydrated.append(HumanMessage(content=content.strip()))
        else:
            hydrated.append(AIMessage(content=content.strip()))
    if not hydrated:
        return None
    hydrated.append(HumanMessage(content=user_query))
    return hydrated


_CHECKPOINT_LRU_LOCK = threading.Lock()
_CHECKPOINT_LRU_REGISTRY: "weakref.WeakKeyDictionary[Any, OrderedDict[str, None]]" = weakref.WeakKeyDictionary()


def _get_or_create_lru(checkpointer: Any) -> "OrderedDict[str, None]":
    """Retrieve or initialize the OrderedDict tracking LRU thread order for a checkpointer."""
    try:
        lru = _CHECKPOINT_LRU_REGISTRY.get(checkpointer)
        if lru is None:
            lru = OrderedDict()
            _CHECKPOINT_LRU_REGISTRY[checkpointer] = lru
        return lru
    except TypeError:
        # Fallback for custom checkpointers that are not weak-referenceable
        lru = getattr(checkpointer, "_lru_order", None)
        if lru is None:
            lru = OrderedDict()
            setattr(checkpointer, "_lru_order", lru)
        return lru


def _delete_thread_from_checkpointer(checkpointer: Any, thread_id: str) -> None:
    """Remove all stored checkpoints, writes, and blobs for thread_id."""
    if checkpointer is None or not thread_id:
        return
    if hasattr(checkpointer, "delete_thread"):
        try:
            checkpointer.delete_thread(thread_id)
            return
        except Exception as exc:
            logger.debug("delete_thread failed for '%s': %s", thread_id, exc)
    storage = getattr(checkpointer, "storage", None)
    if isinstance(storage, dict):
        storage.pop(thread_id, None)


def _touch_checkpoint_lru(graph: Any, thread_id: str) -> None:
    """Mark an existing thread_id as most recently used in the checkpointer's LRU tracker."""
    checkpointer = getattr(graph, "checkpointer", None)
    storage = getattr(checkpointer, "storage", None)
    if checkpointer is None or not isinstance(storage, dict) or not thread_id:
        return
    if thread_id not in storage:
        return
    with _CHECKPOINT_LRU_LOCK:
        lru = _get_or_create_lru(checkpointer)
        lru[thread_id] = None
        lru.move_to_end(thread_id, last=True)


def _cleanup_ephemeral_thread(graph: Any, thread_id: str) -> None:
    """Immediately delete an ephemeral thread's state so stateless requests consume 0 RAM."""
    if not thread_id:
        return
    checkpointer = getattr(graph, "checkpointer", None)
    storage = getattr(checkpointer, "storage", None)
    if checkpointer is None or not isinstance(storage, dict):
        return
    with _CHECKPOINT_LRU_LOCK:
        _delete_thread_from_checkpointer(checkpointer, thread_id)
        try:
            lru = _CHECKPOINT_LRU_REGISTRY.get(checkpointer)
        except TypeError:
            lru = getattr(checkpointer, "_lru_order", None)
        if lru is not None:
            lru.pop(thread_id, None)


def _enforce_checkpoint_lru_bound(
    graph: Any,
    active_thread_id: str,
    max_threads: Optional[int] = None,
) -> int:
    """
    Enforce a bounded LRU limit on MemorySaver threads for the given compiled graph.
    Evicts the least-recently-used threads when len(storage) exceeds max_threads.

    Returns:
        Number of evicted threads.
    """
    checkpointer = getattr(graph, "checkpointer", None)
    storage = getattr(checkpointer, "storage", None)
    if checkpointer is None or not isinstance(storage, dict):
        return 0

    raw_limit = settings.MAX_CHECKPOINT_THREADS if max_threads is None else max_threads
    limit = max(0, int(raw_limit))
    evicted = 0

    with _CHECKPOINT_LRU_LOCK:
        lru = _get_or_create_lru(checkpointer)

        # Reconcile LRU keys with storage (drop externally removed, add externally inserted)
        for existing_tid in list(lru.keys()):
            if existing_tid not in storage:
                lru.pop(existing_tid, None)
        for tid in list(storage.keys()):
            if tid not in lru:
                lru[tid] = None

        if active_thread_id and active_thread_id in storage:
            lru[active_thread_id] = None
            lru.move_to_end(active_thread_id, last=True)

        while len(storage) > limit:
            if lru:
                oldest_tid, _ = lru.popitem(last=False)
            else:
                oldest_tid = next(iter(storage))
            _delete_thread_from_checkpointer(checkpointer, oldest_tid)
            evicted += 1

    return evicted


def _resolve_effective_company_id(
    user_query: str,
    company_id: Optional[int],
    company_id_explicit: Optional[bool],
    checkpoint_values: Optional[Dict[str, Any]],
) -> int:
    """
    Resolve effective company_id with strict 4-tier priority:
      1. Explicit branch name in current user_query text
      2. Explicit company_id provided by API payload / caller (including company_id=1)
      3. Sticky company_id from previous checkpoint turn
      4. Fallback to company_id argument or settings.DEFAULT_COMPANY_ID
    """
    from app.graph.nodes.extractor import resolve_branch_company_id

    # Priority 1: Explicit branch name mentioned in current query text
    branch_from_text = resolve_branch_company_id(user_query)
    if branch_from_text is not None:
        return int(branch_from_text)

    # Compatibility: if caller omitted company_id_explicit, treat non-default ID as explicit
    is_explicit = (
        bool(company_id_explicit)
        if company_id_explicit is not None
        else (company_id is not None and company_id != settings.DEFAULT_COMPANY_ID)
    )

    # Priority 2: Explicit company_id supplied in payload/caller (including company_id=1)
    if is_explicit and company_id is not None:
        return int(company_id)

    # Priority 3: Sticky company_id from previous checkpoint turn
    if checkpoint_values and checkpoint_values.get("company_id"):
        return int(checkpoint_values["company_id"])

    # Priority 4: Fallback to provided company_id or DEFAULT_COMPANY_ID
    return int(company_id) if company_id is not None else int(settings.DEFAULT_COMPANY_ID)


async def _prepare_workflow_execution(
    user_query: str,
    user_token: Optional[str] = None,
    company_id: Optional[int] = None,
    thread_id: Optional[str] = None,
    graph: Optional[Any] = None,
    history: Optional[List[Any]] = None,
    company_id_explicit: Optional[bool] = None,
) -> Tuple[Any, Dict[str, Any], AgentState, bool, str]:
    """
    Resolve thread isolation, checkpoint state, and hydrated initial state for workflow execution.
    Requests without thread_id use a unique ephemeral ID to avoid cross-user leakage.
    """
    active_graph = graph or default_workflow_graph
    normalized_thread_id = str(thread_id).strip() if thread_id is not None else ""
    is_ephemeral = not bool(normalized_thread_id)
    effective_thread_id = f"ephemeral_{uuid.uuid4().hex}" if is_ephemeral else normalized_thread_id
    config = {"configurable": {"thread_id": effective_thread_id}}

    checkpoint = None
    if not is_ephemeral:
        try:
            checkpoint = await active_graph.aget_state(config)
        except Exception:
            pass
        _touch_checkpoint_lru(active_graph, effective_thread_id)

    checkpoint_values = checkpoint.values if (checkpoint and checkpoint.values) else None
    effective_company_id = _resolve_effective_company_id(
        user_query=user_query,
        company_id=company_id,
        company_id_explicit=company_id_explicit,
        checkpoint_values=checkpoint_values,
    )

    # History hydration for stateless requests or recovery after checkpoint eviction/restart
    hydrated_messages = None
    existing_messages = checkpoint_values.get("messages", []) if checkpoint_values else []
    if not existing_messages and history:
        hydrated_messages = _hydrate_messages_from_history(history, user_query)

    initial_state = create_initial_state(
        user_query=user_query,
        user_token=user_token,
        company_id=effective_company_id,
        messages=hydrated_messages,
    )
    return active_graph, config, initial_state, is_ephemeral, effective_thread_id


async def arun_workflow(
    user_query: str,
    user_token: Optional[str] = None,
    company_id: Optional[int] = None,
    thread_id: Optional[str] = None,
    graph: Optional[Any] = None,
    history: Optional[List[Any]] = None,
    company_id_explicit: Optional[bool] = None,
) -> AgentState:
    """
    Asynchronously run the compiled workflow graph for a user query.

    Args:
        user_query: Incoming natural language query.
        user_token: Optional JWT access token.
        company_id: Optional target company ID.
        thread_id: Optional conversational session thread ID for MemorySaver.
        graph: Optional compiled StateGraph instance.
        history: Optional list of past conversation messages for stateless memory recovery.
        company_id_explicit: Whether company_id was explicitly provided by caller/payload.

    Returns:
        Final resulting AgentState dictionary.
    """
    active_graph, config, initial_state, is_ephemeral, effective_thread_id = (
        await _prepare_workflow_execution(
            user_query=user_query,
            user_token=user_token,
            company_id=company_id,
            thread_id=thread_id,
            graph=graph,
            history=history,
            company_id_explicit=company_id_explicit,
        )
    )

    try:
        final_state = await active_graph.ainvoke(initial_state, config=config)
        return final_state
    finally:
        if is_ephemeral:
            _cleanup_ephemeral_thread(active_graph, effective_thread_id)
        else:
            _enforce_checkpoint_lru_bound(
                active_graph,
                effective_thread_id,
                max_threads=settings.MAX_CHECKPOINT_THREADS,
            )


async def astream_workflow_steps(
    user_query: str,
    user_token: Optional[str] = None,
    company_id: Optional[int] = None,
    thread_id: Optional[str] = None,
    graph: Optional[Any] = None,
    history: Optional[List[Any]] = None,
    company_id_explicit: Optional[bool] = None,
):
    """
    Asynchronously stream individual node completion events from the workflow StateGraph.
    Yields tuples of (node_name, node_update_dict, accumulated_state_dict).
    """
    active_graph, config, initial_state, is_ephemeral, effective_thread_id = (
        await _prepare_workflow_execution(
            user_query=user_query,
            user_token=user_token,
            company_id=company_id,
            thread_id=thread_id,
            graph=graph,
            history=history,
            company_id_explicit=company_id_explicit,
        )
    )
    accumulated_state = dict(initial_state)

    try:
        async for chunk in active_graph.astream(initial_state, config=config, stream_mode="updates"):
            for node_name, update in chunk.items():
                if isinstance(update, dict):
                    accumulated_state.update(update)
                yield (node_name, update, accumulated_state)
    finally:
        if is_ephemeral:
            _cleanup_ephemeral_thread(active_graph, effective_thread_id)
        else:
            _enforce_checkpoint_lru_bound(
                active_graph,
                effective_thread_id,
                max_threads=settings.MAX_CHECKPOINT_THREADS,
            )


_WORKFLOW_EXECUTOR: Optional[concurrent.futures.ThreadPoolExecutor] = None
_EXECUTOR_LOCK = threading.Lock()


def get_workflow_executor(max_workers: Optional[int] = None) -> concurrent.futures.ThreadPoolExecutor:
    """
    Acquire or instantiate reusable ThreadPoolExecutor for run_workflow_sync.
    Thread-safe and lazily initialized.
    """
    global _WORKFLOW_EXECUTOR
    if _WORKFLOW_EXECUTOR is None or getattr(_WORKFLOW_EXECUTOR, "_shutdown", False):
        with _EXECUTOR_LOCK:
            if _WORKFLOW_EXECUTOR is None or getattr(_WORKFLOW_EXECUTOR, "_shutdown", False):
                _WORKFLOW_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers or 4,
                    thread_name_prefix="WorkflowSyncRunner",
                )
    return _WORKFLOW_EXECUTOR


def shutdown_workflow_executor(wait: bool = False) -> None:
    """
    Cleanly terminate the reusable workflow ThreadPoolExecutor.
    """
    global _WORKFLOW_EXECUTOR
    with _EXECUTOR_LOCK:
        if _WORKFLOW_EXECUTOR is not None:
            _WORKFLOW_EXECUTOR.shutdown(wait=wait)
            _WORKFLOW_EXECUTOR = None


atexit.register(shutdown_workflow_executor)


def run_workflow_sync(
    user_query: str,
    user_token: Optional[str] = None,
    company_id: Optional[int] = None,
    thread_id: Optional[str] = None,
    graph: Optional[Any] = None,
    on_node_finish: Optional[Any] = None,
    history: Optional[List[Any]] = None,
    company_id_explicit: Optional[bool] = None,
) -> AgentState:
    """
    Synchronously run the compiled workflow graph (safe for CLI and test runners).
    Handles environments with or without an active asyncio event loop.
    Reuses a cached module-level ThreadPoolExecutor when called within an active loop.
    Supports optional on_node_finish callback for live step-by-step progress tracking.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if on_node_finish:
        async def _run_streaming():
            acc: AgentState = {}
            async for node_name, update, acc_state in astream_workflow_steps(
                user_query=user_query,
                user_token=user_token,
                company_id=company_id,
                thread_id=thread_id,
                graph=graph,
                history=history,
                company_id_explicit=company_id_explicit,
            ):
                on_node_finish(node_name, update, acc_state)
                acc = acc_state
            return acc
        coro = _run_streaming()
    else:
        coro = arun_workflow(
            user_query=user_query,
            user_token=user_token,
            company_id=company_id,
            thread_id=thread_id,
            graph=graph,
            history=history,
            company_id_explicit=company_id_explicit,
        )

    if loop and loop.is_running():
        executor = get_workflow_executor()
        return executor.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)

