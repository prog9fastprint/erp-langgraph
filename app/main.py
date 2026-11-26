"""
Main FastAPI application entry point for ERP AI Orchestrator.
"""
import asyncio
from contextlib import asynccontextmanager
import hmac
import json
import logging
import os
import re
import sys
import time
from typing import Annotated, Any, Dict, Optional, Set, Tuple
import uuid

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.config import settings
from app.graph.workflow import (
    arun_workflow,
    astream_workflow_events,
    astream_workflow_steps,
    shutdown_workflow_executor,
)
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.django_client import aclose_shared_django_client
from app.services.llm import afetch_free_models_catalog, get_default_model_name
from app.services.vector_service import vector_service

_LOG_LEVEL_NAME = (settings.LOG_LEVEL or "info").strip().upper()
logging.basicConfig(level=getattr(logging, _LOG_LEVEL_NAME, logging.INFO))
logger = logging.getLogger("api_main")

_BACKGROUND_TASKS: Set[asyncio.Task[Any]] = set()


def _on_background_task_done(task: asyncio.Task[Any]) -> None:
    """Detach finished lifespan background task and consume unexpected exceptions."""
    _BACKGROUND_TASKS.discard(task)
    if not task.cancelled():
        exc = task.exception()
        if exc is not None:
            logger.debug("Background lifespan task exited with error: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup pre-warm tasks and graceful resource cleanup on shutdown."""
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            has_openrouter_cfg = bool((settings.OPENROUTER_API_KEY or "").strip()) or (
                settings.MODEL_PROVIDER.strip().lower() == "openrouter"
            )
            if has_openrouter_cfg:
                prewarm_task = loop.create_task(afetch_free_models_catalog())
                _BACKGROUND_TASKS.add(prewarm_task)
                prewarm_task.add_done_callback(_on_background_task_done)

            is_test_env = "unittest" in sys.modules or os.environ.get("FASTPRINT_TEST_ENV") == "true" or settings.APP_ENV == "test"
            if not is_test_env and settings.VECTOR_DB_URL:
                init_vec_task = loop.create_task(vector_service.initialize_table())
                _BACKGROUND_TASKS.add(init_vec_task)
                init_vec_task.add_done_callback(_on_background_task_done)

                if settings.DYNAMIC_TOOL_RETRIEVAL_ENABLED:
                    from app.services.tool_registry_service import get_tool_registry_service
                    registry = get_tool_registry_service()
                    init_task = loop.create_task(registry.initialize_catalog(force_update=False))
                    _BACKGROUND_TASKS.add(init_task)
                    init_task.add_done_callback(_on_background_task_done)
    except Exception as exc:
        logger.debug("Background pre-warm or catalog sync skipped during lifespan initialization: %s", exc)

    try:
        yield
    finally:
        pending_tasks = [task for task in list(_BACKGROUND_TASKS) if not task.done()]
        for task in pending_tasks:
            task.cancel()
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
        _BACKGROUND_TASKS.clear()

        try:
            await vector_service.close_pool()
        except Exception as exc:
            logger.warning("Failed to close VectorService asyncpg pool during shutdown: %s", exc)

        try:
            await aclose_shared_django_client()
        except Exception as exc:
            logger.warning("Failed to close shared Django httpx client during shutdown: %s", exc)

        try:
            shutdown_workflow_executor(wait=False)
        except Exception as exc:
            logger.warning("Failed to shutdown workflow executor during shutdown: %s", exc)


_is_prod_env = settings.APP_ENV.strip().lower() == "production"
app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    description="FastPrint ERP Dedicated AI Orchestrator Service",
    docs_url=None if _is_prod_env else "/docs",
    redoc_url=None if _is_prod_env else "/redoc",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_observability_headers(request: Request, call_next):
    """Attach X-Request-ID and X-Process-Time-Ms headers to every HTTP response."""
    raw_request_id = request.headers.get("X-Request-ID") or ""
    sanitized_request_id = re.sub(r"[^A-Za-z0-9._-]", "", raw_request_id)[:64]
    request_id = sanitized_request_id or uuid.uuid4().hex
    request.state.request_id = request_id

    started_at = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


@app.get("/health", summary="Health check", tags=["System"])
async def health_check():
    """
    Health check endpoint returning system status.
    """
    return {"status": "healthy"}


@app.get("/health/ready", summary="Readiness check", tags=["System"])
async def readiness_check():
    """
    Report boolean readiness status for LLM provider, ERP connection config, and Vector DB
    without exposing URLs, credentials, or internal environment details.
    """
    provider = (settings.MODEL_PROVIDER or "openai").strip().lower()
    if provider == "openai":
        llm_configured = bool((settings.OPENAI_API_KEY or "").strip())
    elif provider == "gemini":
        llm_configured = bool(settings.get_gemini_api_keys())
    else:
        llm_configured = bool((settings.OPENROUTER_API_KEY or "").strip())

    erp_configured = bool((settings.ERP_BASE_URL or "").strip())
    vector_db_configured = bool((getattr(vector_service, "db_url", None) or settings.VECTOR_DB_URL or "").strip())
    is_ready = llm_configured and erp_configured

    payload = {
        "status": "ready" if is_ready else "degraded",
        "checks": {
            "llm_provider": provider,
            "llm_configured": llm_configured,
            "erp_configured": erp_configured,
            "rag_enabled": bool(settings.ENABLE_RAG),
            "vector_db_configured": vector_db_configured,
        },
    }
    return JSONResponse(
        content=payload,
        status_code=status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE,
    )


def _extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    """Extract JWT token from an Authorization: Bearer <token> header."""
    if not authorization or not isinstance(authorization, str):
        return None
    match = re.match(r"^\s*Bearer\s+(.+?)\s*$", authorization, re.IGNORECASE)
    if not match:
        return None
    token = match.group(1).strip()
    return token or None


def _resolve_and_validate_auth(
    payload: ChatRequest,
    authorization: Optional[str] = None,
    x_internal_service_key: Optional[str] = None,
) -> Tuple[Optional[str], bool]:
    """
    Resolve user JWT from body user_token or Authorization Bearer header,
    enforce X-Internal-Service-Key trust, and reject missing user tokens
    in production/staging or when REQUIRE_USER_TOKEN is enabled.
    """
    dedicated_key = (settings.INTERNAL_SERVICE_KEY or "").strip()
    fallback_key = (settings.ERP_API_TOKEN or "").strip()
    if settings.REQUIRE_SERVICE_KEY and not dedicated_key and fallback_key:
        logger.warning(
            "REQUIRE_SERVICE_KEY is using ERP_API_TOKEN fallback; configure INTERNAL_SERVICE_KEY instead."
        )
    expected_key = dedicated_key or fallback_key
    provided_key = (x_internal_service_key or "").strip()
    service_key_verified = bool(
        expected_key and provided_key and hmac.compare_digest(provided_key, expected_key)
    )

    if settings.REQUIRE_SERVICE_KEY and not service_key_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: invalid or missing X-Internal-Service-Key.",
        )

    body_token = (payload.user_token or payload.auth_token or "").strip() or None
    header_token = _extract_bearer_token(authorization)
    effective_token = body_token or header_token

    is_strict_mode = (
        settings.APP_ENV.strip().lower() in ("production", "staging")
        or bool(settings.REQUIRE_USER_TOKEN)
    )
    if is_strict_mode and not effective_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required: valid user_token or Authorization Bearer header must be provided.",
        )

    return effective_token, service_key_verified


def _build_workflow_kwargs(
    payload: ChatRequest,
    effective_token: Optional[str],
    service_key_verified: bool = False,
) -> Dict[str, Any]:
    """Construct keyword arguments forwarded to arun_workflow and astream_workflow_steps."""
    kwargs: Dict[str, Any] = {
        "user_query": payload.query,
        "user_token": effective_token,
        "company_id": payload.company_id,
        "thread_id": payload.thread_id,
        "company_id_explicit": ("company_id" in payload.model_fields_set),
    }
    if payload.company_ids is not None:
        kwargs["company_ids"] = payload.company_ids
    if payload.authorized_company_ids is not None:
        kwargs["authorized_company_ids"] = payload.authorized_company_ids
    if payload.history is not None:
        kwargs["history"] = payload.history
    if payload.auth_token is not None:
        kwargs["auth_token"] = payload.auth_token
    if payload.user_id is not None:
        kwargs["user_id"] = payload.user_id
    if payload.user_roles is not None:
        if service_key_verified:
            kwargs["user_roles"] = payload.user_roles
        else:
            logger.warning("Stripping unverified client-supplied user_roles without verified X-Internal-Service-Key.")
            kwargs["user_roles"] = []
    return kwargs


def _build_chat_response(final_state: Dict[str, Any], started_at: float) -> ChatResponse:
    """Serialize final AgentState into ChatResponse with resolved_company_id and timing."""
    raw_status = final_state.get("verification_status")
    if raw_status is not None:
        verification_status_str = str(raw_status.value) if hasattr(raw_status, "value") else str(raw_status)
    else:
        verification_status_str = None

    raw_company_id = final_state.get("company_id")
    resolved_company_id = int(raw_company_id) if raw_company_id is not None else None
    raw_company_ids = final_state.get("company_ids")
    resolved_company_ids = [int(c) for c in raw_company_ids] if raw_company_ids else None
    execution_time_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
    model_used = final_state.get("active_model") or get_default_model_name()

    return ChatResponse(
        final_answer=final_state.get("final_answer"),
        detected_domain=final_state.get("detected_domain"),
        verification_status=verification_status_str,
        error_message=final_state.get("error_message"),
        suggested_followups=final_state.get("suggested_followups") or [],
        resolved_company_id=resolved_company_id,
        resolved_company_ids=resolved_company_ids,
        execution_time_ms=execution_time_ms,
        model_used=model_used,
        table_query_meta=final_state.get("table_query_meta"),
    )


@app.post(
    "/api/v1/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute conversational AI workflow for FastPrint ERP",
    tags=["Chat"],
)
async def chat_endpoint(
    payload: ChatRequest,
    authorization: Annotated[Optional[str], Header()] = None,
    x_internal_service_key: Annotated[Optional[str], Header(alias="X-Internal-Service-Key")] = None,
) -> ChatResponse:
    """
    Handle natural language ERP inquiries via LangGraph StateGraph pipeline.
    """
    try:
        effective_token, service_key_verified = _resolve_and_validate_auth(
            payload=payload,
            authorization=authorization,
            x_internal_service_key=x_internal_service_key,
        )
        started_at = time.perf_counter()
        workflow_kwargs = _build_workflow_kwargs(
            payload,
            effective_token,
            service_key_verified=service_key_verified,
        )
        final_state: Dict[str, Any] = await arun_workflow(**workflow_kwargs)
        return _build_chat_response(final_state, started_at)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unhandled error processing chat endpoint request: %s", exc)
        detail = (
            "Chat workflow failed due to an internal server error."
            if settings.APP_ENV.strip().lower() == "production"
            else f"Chat workflow failed: {str(exc)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
        )


@app.post(
    "/api/v1/chat/stream",
    status_code=status.HTTP_200_OK,
    summary="Stream real-time LangGraph node progress, tool lifecycle, tokens, and final response via SSE",
    tags=["Chat"],
)
async def chat_stream_endpoint(
    payload: ChatRequest,
    authorization: Annotated[Optional[str], Header()] = None,
    x_internal_service_key: Annotated[Optional[str], Header(alias="X-Internal-Service-Key")] = None,
) -> StreamingResponse:
    """
    Execute LangGraph StateGraph pipeline and stream real-time events via Server-Sent Events (SSE):
      - event: step (node progress updates)
      - event: tool_start (sanitized ERP tool name and input filters)
      - event: tool_end (tool duration, execution status, and record counts)
      - event: token (synthesizer text chunks for typing effect)
      - event: complete (final ChatResponse payload)
      - : ping (heartbeat comment every 15s to keep proxy connections alive)
      - event: error (pipeline failure diagnostics)
    """
    effective_token, service_key_verified = _resolve_and_validate_auth(
        payload=payload,
        authorization=authorization,
        x_internal_service_key=x_internal_service_key,
    )
    workflow_kwargs = _build_workflow_kwargs(
        payload,
        effective_token,
        service_key_verified=service_key_verified,
    )
    started_at = time.perf_counter()

    is_test_env = (
        "unittest" in sys.modules
        or os.environ.get("FASTPRINT_TEST_ENV") == "true"
        or settings.APP_ENV.strip().lower() == "test"
    )
    token_delay_sec = 0.0 if is_test_env else 0.01

    async def _event_generator():
        q: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=256)
        accumulated_state: Dict[str, Any] = {}

        async def _producer():
            try:
                nonlocal accumulated_state
                async for event_type, payload_data in astream_workflow_events(
                    **workflow_kwargs,
                    token_delay_sec=token_delay_sec,
                    steps_generator=astream_workflow_steps,
                ):
                    if event_type == "step":
                        await q.put(f"event: step\ndata: {json.dumps(payload_data, ensure_ascii=False)}\n\n")
                    elif event_type == "tool_start":
                        await q.put(f"event: tool_start\ndata: {json.dumps(payload_data, ensure_ascii=False)}\n\n")
                    elif event_type == "tool_end":
                        await q.put(f"event: tool_end\ndata: {json.dumps(payload_data, ensure_ascii=False)}\n\n")
                    elif event_type == "token":
                        await q.put(f"event: token\ndata: {json.dumps(payload_data, ensure_ascii=False)}\n\n")
                    elif event_type == "complete":
                        accumulated_state = payload_data
                        chat_resp = _build_chat_response(accumulated_state, started_at)
                        await q.put(f"event: complete\ndata: {chat_resp.model_dump_json()}\n\n")
                    elif event_type == "error":
                        detail = (
                            "Chat workflow failed due to an internal server error."
                            if settings.APP_ENV.strip().lower() == "production"
                            else f"Chat workflow failed: {payload_data.get('detail', 'Unknown error')}"
                        )
                        await q.put(f"event: error\ndata: {json.dumps({'detail': detail}, ensure_ascii=False)}\n\n")
            except Exception as exc:
                logger.exception("Unhandled error during chat SSE stream producer: %s", exc)
                detail = (
                    "Chat workflow failed due to an internal server error."
                    if settings.APP_ENV.strip().lower() == "production"
                    else f"Chat workflow failed: {str(exc)}"
                )
                await q.put(f"event: error\ndata: {json.dumps({'detail': detail}, ensure_ascii=False)}\n\n")
            finally:
                await q.put(None)

        producer_task = asyncio.create_task(_producer())
        heartbeat_sec = float(getattr(settings, "SSE_HEARTBEAT_SECONDS", 15.0))

        try:
            while True:
                try:
                    frame = await asyncio.wait_for(q.get(), timeout=heartbeat_sec)
                    if frame is None:
                        break
                    yield frame
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            if not producer_task.done():
                producer_task.cancel()
                try:
                    await producer_task
                except (asyncio.CancelledError, Exception):
                    pass

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


__all__ = ["app", "ChatRequest", "ChatResponse"]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=(settings.APP_ENV.strip().lower() == "development"),
    )

