"""
RAG Node for FastPrint ERP LangGraph Agent Workflow.
Executes semantic retrieval for qualitative data (SOPs, product specifications, policies)
via DjangoCoreClient connecting to PGVector on the Django backend.
"""
import asyncio
import concurrent.futures
import logging
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.graph.state import AgentState
from app.services.django_client import (
    DjangoCoreClient,
    DjangoCoreClientError,
)
from app.services.vector_service import vector_service

logger = logging.getLogger("rag_node")


async def _search_single_rag_query(
    rag_query: str,
    company_id: int,
    user_token: Optional[str],
    client: Optional[DjangoCoreClient] = None,
) -> Dict[str, Any]:
    """Execute a single semantic RAG search via VectorService or Django Core fallback."""
    chunks: List[Dict[str, Any]] = []
    err_msg: Optional[str] = None
    is_client_error = False

    # Path 1: Check Dedicated Vector Service (PGVector + Gemini) first
    if settings.VECTOR_DB_URL and settings.GEMINI_API_KEY:
        try:
            chunks = await vector_service.search_knowledge(
                query=rag_query.strip(),
                company_id=company_id,
                top_k=settings.RAG_TOP_K,
                threshold=settings.RAG_SIMILARITY_THRESHOLD,
            )
        except Exception as vec_err:
            logger.warning(f"Dedicated vector service search failed: {vec_err}")
            chunks = []

    # Path 2: Fallback / standard path via Django Core API client
    if not chunks and (client is not None or not settings.VECTOR_DB_URL):
        own_client = client is None
        erp_client = client or DjangoCoreClient(require_user_token=settings.REQUIRE_USER_TOKEN)
        try:
            raw_res = await erp_client.fetch_knowledge(
                params={
                    "query": rag_query.strip(),
                    "top_k": settings.RAG_TOP_K,
                },
                user_token=user_token,
                company_id=company_id,
            )
            if isinstance(raw_res, dict):
                chunks = raw_res.get("chunks", []) or raw_res.get("data", {}).get("chunks", [])
            elif isinstance(raw_res, list):
                chunks = raw_res
        except DjangoCoreClientError as err:
            logger.warning(f"DjangoCoreClientError in rag_node: {err}")
            err_msg = str(err)
            is_client_error = True
        except Exception as exc:
            logger.error(f"Unexpected error in rag_node: {exc}", exc_info=True)
            return {"chunks": [], "sources": [], "error": None, "is_client_error": False}
        finally:
            if own_client:
                await erp_client.aclose()

    if is_client_error:
        return {"chunks": [], "sources": [], "error": err_msg, "is_client_error": True}

    threshold = float(settings.RAG_SIMILARITY_THRESHOLD)
    valid_chunks: List[Dict[str, Any]] = []
    sources: List[str] = []

    for c in chunks:
        if not isinstance(c, dict):
            continue
        score = float(c.get("similarity_score", 1.0))
        if score >= threshold:
            valid_chunks.append(c)
            title = c.get("title") or c.get("metadata", {}).get("source") or "Dokumen SOP ERP"
            if title not in sources:
                sources.append(title)

    return {"chunks": valid_chunks, "sources": sources, "error": None, "is_client_error": False}


async def rag_node(
    state: AgentState,
    client: Optional[DjangoCoreClient] = None,
) -> Dict[str, Any]:
    """
    Asynchronous LangGraph node that performs semantic knowledge search.

    Architecture Support:
    1. First checks if Dedicated Vector Service (PGVector + Gemini) is configured.
       If VECTOR_DB_URL and GEMINI_API_KEY are active, queries dedicated PGVector directly.
    2. Fallback to Django Core API endpoint POST /ai/tools/knowledge/search/ if client provided or configured.
    """
    if not settings.ENABLE_RAG:
        return {"rag_context": [], "rag_sources": [], "rag_results_by_query": {}}

    query_mode = state.get("query_mode") or "vector_rag"
    company_id = state.get("company_id") or settings.DEFAULT_COMPANY_ID
    user_token = state.get("user_token")

    raw_rag_queries = state.get("rag_queries") or []
    rag_queries: List[str] = [
        q.strip() for q in raw_rag_queries if isinstance(q, str) and q.strip()
    ]
    if not rag_queries:
        fallback_q = (state.get("rag_query") or state.get("user_query") or "").strip()
        if fallback_q:
            rag_queries = [fallback_q]

    if not rag_queries:
        return {"rag_context": [], "rag_sources": [], "rag_results_by_query": {}}

    results_list = await asyncio.gather(
        *[
            _search_single_rag_query(
                rag_query=q,
                company_id=company_id,
                user_token=user_token,
                client=client,
            )
            for q in rag_queries
        ]
    )

    if len(rag_queries) == 1 and results_list[0].get("is_client_error"):
        err_delta: Dict[str, Any] = {
            "rag_context": [],
            "rag_sources": [],
            "rag_results_by_query": {
                rag_queries[0]: {
                    "chunks": [],
                    "sources": [],
                    "error": results_list[0].get("error"),
                }
            },
        }
        if query_mode == "vector_rag":
            err_delta["verification_status"] = "ERROR"
            err_delta["error_message"] = results_list[0].get("error")
        return err_delta

    valid_chunks: List[Dict[str, Any]] = []
    sources: List[str] = []
    rag_results_by_query: Dict[str, Dict[str, Any]] = {}
    seen_chunk_sigs = set()
    all_client_errors = True

    for q, res in zip(rag_queries, results_list):
        q_chunks = res.get("chunks") or []
        q_sources = res.get("sources") or []
        q_err = res.get("error")
        if not res.get("is_client_error"):
            all_client_errors = False
        rag_results_by_query[q] = {
            "chunks": q_chunks,
            "sources": q_sources,
            "error": q_err,
        }
        for c in q_chunks:
            sig = (
                c.get("title") or c.get("metadata", {}).get("source") or "",
                c.get("content") or c.get("text") or "",
            )
            if sig not in seen_chunk_sigs:
                seen_chunk_sigs.add(sig)
                valid_chunks.append(c)
        for s in q_sources:
            if s not in sources:
                sources.append(s)

    result_delta: Dict[str, Any] = {
        "rag_context": valid_chunks,
        "rag_sources": sources,
        "rag_results_by_query": rag_results_by_query,
    }

    if query_mode == "vector_rag":
        if all_client_errors and not valid_chunks:
            first_err = next((r.get("error") for r in results_list if r.get("error")), None)
            result_delta["verification_status"] = "ERROR"
            if first_err:
                result_delta["error_message"] = first_err
        else:
            result_delta["verification_status"] = "VALID" if valid_chunks else "EMPTY"

    return result_delta


def rag_node_sync(
    state: AgentState,
    client: Optional[DjangoCoreClient] = None,
) -> Dict[str, Any]:
    """
    Synchronous wrapper for rag_node.
    Enables safe execution in synchronous CLI runners, testing suites, or scripts.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return executor.submit(asyncio.run, rag_node(state, client=client)).result()
    else:
        return asyncio.run(rag_node(state, client=client))
