"""
LLM Model Factory supporting OpenRouter (with free model catalog & auto-fallback) and OpenAI.
"""
import asyncio
import contextvars
import hashlib
import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
import free_llm_models as flm

from app.core.config import settings

logger = logging.getLogger("llm_factory")

# In-memory cache for dynamic free models list to avoid redundant API queries
_CACHED_FREE_MODELS: List[str] = []
_LAST_CACHE_TIME: float = 0.0
_CACHE_TTL_SECONDS: float = 900.0  # 15 minutes
_CACHE_LOCK = threading.Lock()
_REFRESH_TASK: Optional[asyncio.Task] = None

# Curated fallback list when offline or when dynamic lookup is unavailable
DEFAULT_CURATED_FREE_MODELS = [
    "openrouter/free",
    "nvidia/nemotron-3.5-super-vl:free",
    "liquid/lfm-2.5-2.6b:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "liquid/lfm-3-1.7b:free",
    "nvidia/nemotron-3.5-lightning:free",
    "recursive/flint-v1.0:free",
]

# Provider outage tracker for fast circuit-breaker failover
_PROVIDER_OUTAGE_LOCK = threading.Lock()
_PROVIDER_OUTAGE_UNTIL: Dict[str, float] = {}
_PROVIDER_HEALTH_EPOCH: int = 0

# Warm BaseChatModel cache (reuses underlying HTTP/TLS connection pools across requests)
_CHAT_MODEL_CACHE_LOCK = threading.Lock()
_CHAT_MODEL_CACHE: Dict[Tuple[Any, ...], BaseChatModel] = {}
_CHAT_MODEL_CACHE_MAX_SIZE: int = 16

# Negative cache & one-shot retry state for models that reject 'temperature' at runtime
_UNSUPPORTED_TEMP_LOCK = threading.Lock()
_UNSUPPORTED_TEMP_MODELS: Set[Tuple[str, str]] = set()
_FORCE_OMIT_TEMP_CTX: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_force_omit_temp_ctx",
    default=False,
)
_META_ROUTER_MODELS = frozenset({"openrouter/free"})


def clear_unsupported_temperature_cache() -> None:
    """Clear runtime cache of models known to reject the 'temperature' parameter."""
    with _UNSUPPORTED_TEMP_LOCK:
        _UNSUPPORTED_TEMP_MODELS.clear()


def _is_cached_unsupported_temperature(provider_name: str, model_name: Optional[str]) -> bool:
    """Return True if (provider_name, model_name) has previously rejected 'temperature'."""
    norm_model = str(model_name or "").strip()
    if not norm_model:
        return False
    with _UNSUPPORTED_TEMP_LOCK:
        return (provider_name.lower().strip(), norm_model) in _UNSUPPORTED_TEMP_MODELS


def _is_400_bad_request(exc: Exception) -> bool:
    """Detect HTTP 400 / BadRequestError responses across OpenAI-compatible SDKs."""
    if getattr(exc, "status_code", None) == 400:
        return True
    err_str = str(exc).lower()
    return (
        "error code: 400" in err_str
        or "badrequesterror" in type(exc).__name__.lower()
        or "invalid_request_error" in err_str
    )


def _is_unsupported_temperature_error(exc: Exception) -> bool:
    """Return True when an HTTP 400 error specifically rejects the 'temperature' parameter."""
    if not _is_400_bad_request(exc):
        return False
    err_str = str(exc).lower()
    return "temperature" in err_str and (
        "unsupported" in err_str
        or "does not support" in err_str
        or "not supported" in err_str
    )


def clear_chat_model_cache() -> None:
    """Clear cached warm chat model instances."""
    with _CHAT_MODEL_CACHE_LOCK:
        _CHAT_MODEL_CACHE.clear()


def _config_keys_fingerprint() -> str:
    """Return a short hash of active provider API keys so test/env overrides never hit stale cached clients."""
    raw = "|".join([
        str(settings.OPENAI_API_KEY or ""),
        str(settings.OPENROUTER_API_KEY or ""),
        *settings.get_gemini_api_keys(),
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def is_provider_unhealthy(provider: str) -> bool:
    """Check if the specified provider is temporarily marked unhealthy (outage cache)."""
    with _PROVIDER_OUTAGE_LOCK:
        return time.time() < _PROVIDER_OUTAGE_UNTIL.get(provider.lower().strip(), 0.0)


def mark_provider_unhealthy(provider: str, duration_sec: float = 900.0) -> None:
    """Temporarily mark a provider as unhealthy to bypass futile network calls."""
    global _PROVIDER_HEALTH_EPOCH
    with _PROVIDER_OUTAGE_LOCK:
        _PROVIDER_OUTAGE_UNTIL[provider.lower().strip()] = time.time() + duration_sec
        _PROVIDER_HEALTH_EPOCH += 1
    clear_chat_model_cache()


def reset_provider_health(provider: Optional[str] = None) -> None:
    """Reset unhealthy status for a specific provider or all providers."""
    global _PROVIDER_HEALTH_EPOCH
    with _PROVIDER_OUTAGE_LOCK:
        if provider:
            _PROVIDER_OUTAGE_UNTIL.pop(provider.lower().strip(), None)
        else:
            _PROVIDER_OUTAGE_UNTIL.clear()
        _PROVIDER_HEALTH_EPOCH += 1
    clear_chat_model_cache()


class CircuitBreakerChatOpenAI(ChatOpenAI):
    """ChatOpenAI wrapper that flags auth/quota outages and self-heals temperature rejections."""
    provider_name: str = "openai"

    def _should_omit_temperature(self) -> bool:
        if _FORCE_OMIT_TEMP_CTX.get():
            return True
        return _is_cached_unsupported_temperature(self.provider_name, self.model_name)

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if self._should_omit_temperature():
            payload.pop("temperature", None)
        return payload

    def _record_unsupported_temperature(self, exc: Exception) -> bool:
        if not _is_unsupported_temperature_error(exc):
            return False
        norm_model = str(self.model_name or "").strip()
        if norm_model and norm_model.lower() not in _META_ROUTER_MODELS:
            with _UNSUPPORTED_TEMP_LOCK:
                _UNSUPPORTED_TEMP_MODELS.add((self.provider_name.lower().strip(), norm_model))
        logger.warning(
            "Model '%s' on provider '%s' rejected 'temperature'; retrying once without temperature.",
            norm_model or "unknown",
            self.provider_name,
        )
        return True

    def _handle_failure(self, exc: Exception) -> None:
        err_str = str(exc).lower()
        if (
            "expired" in err_str
            or "401" in err_str
            or "authenticationerror" in err_str
            or "invalid api key" in err_str
            or "quota" in err_str
            or "insufficient_quota" in err_str
            or "429" in err_str
            or "resource_exhausted" in err_str
        ):
            mark_provider_unhealthy(self.provider_name, duration_sec=900.0)
            logger.warning(
                f"Provider/key failure detected for '{self.provider_name}'. Marked unhealthy for 15m: {exc}"
            )
        elif _is_400_bad_request(exc):
            logger.warning(
                "HTTP 400 error on provider '%s' (model '%s'): %s",
                self.provider_name,
                self.model_name,
                exc,
            )

    def invoke(self, *args, **kwargs):
        try:
            return super().invoke(*args, **kwargs)
        except Exception as exc:
            if self._record_unsupported_temperature(exc):
                token = _FORCE_OMIT_TEMP_CTX.set(True)
                try:
                    return super().invoke(*args, **kwargs)
                except Exception as retry_exc:
                    self._handle_failure(retry_exc)
                    raise
                finally:
                    _FORCE_OMIT_TEMP_CTX.reset(token)
            self._handle_failure(exc)
            raise

    async def ainvoke(self, *args, **kwargs):
        try:
            return await super().ainvoke(*args, **kwargs)
        except Exception as exc:
            if self._record_unsupported_temperature(exc):
                token = _FORCE_OMIT_TEMP_CTX.set(True)
                try:
                    return await super().ainvoke(*args, **kwargs)
                except Exception as retry_exc:
                    self._handle_failure(retry_exc)
                    raise
                finally:
                    _FORCE_OMIT_TEMP_CTX.reset(token)
            self._handle_failure(exc)
            raise

    def stream(self, *args, **kwargs):
        yielded = False
        try:
            for chunk in super().stream(*args, **kwargs):
                yielded = True
                yield chunk
            return
        except Exception as exc:
            if not yielded and self._record_unsupported_temperature(exc):
                token = _FORCE_OMIT_TEMP_CTX.set(True)
                try:
                    for chunk in super().stream(*args, **kwargs):
                        yield chunk
                    return
                except Exception as retry_exc:
                    self._handle_failure(retry_exc)
                    raise
                finally:
                    _FORCE_OMIT_TEMP_CTX.reset(token)
            self._handle_failure(exc)
            raise

    async def astream(self, *args, **kwargs):
        yielded = False
        try:
            async for chunk in super().astream(*args, **kwargs):
                yielded = True
                yield chunk
            return
        except Exception as exc:
            if not yielded and self._record_unsupported_temperature(exc):
                token = _FORCE_OMIT_TEMP_CTX.set(True)
                try:
                    async for chunk in super().astream(*args, **kwargs):
                        yield chunk
                    return
                except Exception as retry_exc:
                    self._handle_failure(retry_exc)
                    raise
                finally:
                    _FORCE_OMIT_TEMP_CTX.reset(token)
            self._handle_failure(exc)
            raise


def _sync_fetch_free_models_catalog(
    api_key: Optional[str] = None,
    force_refresh: bool = False,
    prioritize_tools: bool = True,
) -> List[str]:
    """
    Synchronous worker that queries OpenRouter API and prioritizes models via free_llm_models.
    Updates in-memory cache and returns the model ID list.
    """
    global _CACHED_FREE_MODELS, _LAST_CACHE_TIME
    now = time.time()

    with _CACHE_LOCK:
        if not force_refresh and _CACHED_FREE_MODELS and (now - _LAST_CACHE_TIME < _CACHE_TTL_SECONDS):
            return list(_CACHED_FREE_MODELS)

    discovered_models: List[str] = []
    key = api_key or settings.OPENROUTER_API_KEY

    # 1. Live query to OpenRouter API /v1/models if key is available and not a test mock
    if key and not key.startswith("mock"):
        try:
            headers = {
                "Authorization": f"Bearer {key}",
                "HTTP-Referer": "https://fastprint.co.id",
                "X-Title": "FastPrint ERP LangGraph",
            }
            with httpx.Client(timeout=settings.REQUEST_TIMEOUT_SEC) as client:
                res = client.get("https://openrouter.ai/api/v1/models", headers=headers)
                if res.status_code == 200:
                    data = res.json().get("data", [])
                    for item in data:
                        mid = item.get("id", "")
                        pricing = item.get("pricing", {})
                        prompt_p = float(pricing.get("prompt", 1) or 1)
                        comp_p = float(pricing.get("completion", 1) or 1)
                        if mid.endswith(":free") or (prompt_p == 0 and comp_p == 0):
                            if mid not in discovered_models:
                                discovered_models.append(mid)
        except Exception as exc:
            logger.warning(f"Could not fetch dynamic free models from OpenRouter: {exc}")

    # 2. Merge with curated fallback models if discovery returned empty
    if not discovered_models:
        discovered_models = list(DEFAULT_CURATED_FREE_MODELS)

    # Always ensure 'openrouter/free' meta-router is present
    if "openrouter/free" not in discovered_models:
        discovered_models.insert(0, "openrouter/free")

    # 3. Sort / prioritize using `free_llm_models` metadata
    if prioritize_tools:
        def model_priority(model_id: str) -> int:
            if model_id == "openrouter/free":
                return 0  # Highest priority
            base_name = model_id.split("/")[-1].replace(":free", "")
            matches = flm.search(base_name)
            if matches:
                supports_tools = matches[0].get("supportsTools", False)
                supports_json = matches[0].get("supportsJson", False)
                if supports_tools and supports_json:
                    return 1
                if supports_tools or supports_json:
                    return 2
            return 3

        discovered_models.sort(key=model_priority)

    # 4. Update Cache
    with _CACHE_LOCK:
        _CACHED_FREE_MODELS = discovered_models
        _LAST_CACHE_TIME = now
        return list(_CACHED_FREE_MODELS)


async def afetch_free_models_catalog(
    api_key: Optional[str] = None,
    force_refresh: bool = False,
    prioritize_tools: bool = True,
) -> List[str]:
    """
    Asynchronously fetch free models catalog without blocking the event loop.
    Offloads synchronous HTTP request to asyncio.to_thread.
    """
    global _CACHED_FREE_MODELS, _LAST_CACHE_TIME
    now = time.time()

    with _CACHE_LOCK:
        if not force_refresh and _CACHED_FREE_MODELS and (now - _LAST_CACHE_TIME < _CACHE_TTL_SECONDS):
            return list(_CACHED_FREE_MODELS)

    return await asyncio.to_thread(
        _sync_fetch_free_models_catalog,
        api_key=api_key,
        force_refresh=force_refresh,
        prioritize_tools=prioritize_tools,
    )


def fetch_free_models_catalog(
    api_key: Optional[str] = None,
    force_refresh: bool = False,
    prioritize_tools: bool = True,
) -> List[str]:
    """
    Fetch and return an active list of free models on OpenRouter, cross-referenced
    with the `free_llm_models` library to prioritize models supporting tool calling & JSON output.

    When invoked from an active event loop, returns cached or curated models immediately
    and schedules a background non-blocking refresh via afetch_free_models_catalog to avoid
    blocking the loop. When invoked synchronously outside an event loop, runs directly.

    Args:
        api_key: Optional OpenRouter API key for live discovery.
        force_refresh: Bypass in-memory cache if True.
        prioritize_tools: If True, models known to support tools/JSON appear first.

    Returns:
        List of OpenRouter model identifier strings (e.g. ['openrouter/free', ...]).
    """
    global _CACHED_FREE_MODELS, _LAST_CACHE_TIME
    now = time.time()

    with _CACHE_LOCK:
        if not force_refresh and _CACHED_FREE_MODELS and (now - _LAST_CACHE_TIME < _CACHE_TTL_SECONDS):
            return list(_CACHED_FREE_MODELS)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Do not block the running event loop with synchronous network I/O.
        # Schedule one deduplicated background refresh via afetch_free_models_catalog.
        global _REFRESH_TASK
        if _REFRESH_TASK is None or _REFRESH_TASK.done():
            try:
                task = asyncio.create_task(
                    afetch_free_models_catalog(
                        api_key=api_key,
                        force_refresh=True,
                        prioritize_tools=prioritize_tools,
                    )
                )
                _REFRESH_TASK = task

                def _clear_refresh_task(done_task: asyncio.Task) -> None:
                    global _REFRESH_TASK
                    if _REFRESH_TASK is done_task:
                        _REFRESH_TASK = None
                    try:
                        done_task.exception()
                    except (asyncio.CancelledError, Exception):
                        pass

                task.add_done_callback(_clear_refresh_task)
            except Exception as exc:
                logger.debug(f"Could not schedule background catalog refresh: {exc}")

        with _CACHE_LOCK:
            if _CACHED_FREE_MODELS:
                return list(_CACHED_FREE_MODELS)
        return list(DEFAULT_CURATED_FREE_MODELS)

    return _sync_fetch_free_models_catalog(
        api_key=api_key,
        force_refresh=force_refresh,
        prioritize_tools=prioritize_tools,
    )


def is_gemini_unhealthy() -> bool:
    """Check if all configured Gemini keys are marked unhealthy in the circuit-breaker cache."""
    keys = settings.get_gemini_api_keys()
    if not keys:
        return True
    return all(is_provider_unhealthy(f"gemini:{k[-6:]}") for k in keys)


def _build_gemini_model(
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    enable_key_fallback: bool = True,
    reasoning_effort: Optional[str] = None,
) -> BaseChatModel:
    """
    Construct ChatOpenAI instance targeting Google Gemini's OpenAI-compatible endpoint.
    Configures multi-key failover across GOOGLE_AI_API_KEY, GOOGLE_AI_API_KEY_2, and GOOGLE_AI_API_KEY_3.
    """
    keys = settings.get_gemini_api_keys()
    if not keys:
        raise ValueError(
            "GOOGLE_AI_API_KEY (or GEMINI_API_KEY) environment variable is required when MODEL_PROVIDER=gemini"
        )

    target_model = model_name or settings.GEMINI_MODEL

    # Filter out keys that are currently marked unhealthy by circuit-breaker
    healthy_keys = [k for k in keys if not is_provider_unhealthy(f"gemini:{k[-6:]}")]
    effective_keys = healthy_keys if healthy_keys else keys

    extra_kwargs: Dict[str, Any] = {}
    if reasoning_effort and str(reasoning_effort).strip():
        extra_kwargs["reasoning_effort"] = str(reasoning_effort).strip()

    primary_key = effective_keys[0]
    primary_model = CircuitBreakerChatOpenAI(
        model=target_model,
        api_key=primary_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        temperature=temperature,
        request_timeout=settings.REQUEST_TIMEOUT_SEC,
        provider_name=f"gemini:{primary_key[-6:]}",
        **extra_kwargs,
    )

    if enable_key_fallback and len(effective_keys) > 1:
        fallbacks: List[BaseChatModel] = [
            CircuitBreakerChatOpenAI(
                model=target_model,
                api_key=k,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                temperature=temperature,
                request_timeout=settings.REQUEST_TIMEOUT_SEC,
                provider_name=f"gemini:{k[-6:]}",
                **extra_kwargs,
            )
            for k in effective_keys[1:]
        ]
        return primary_model.with_fallbacks(fallbacks)

    return primary_model


# OpenAI model families that enforce default sampling and reject temperature=0.0
_OPENAI_OMIT_TEMPERATURE_MODEL_RE = re.compile(
    r"^(?:openai/)?(?:o\d+(?:$|[-.])|gpt-(?:[5-9]|\d{2,})(?:$|[-.]))",
    re.IGNORECASE,
)


def _is_openai_reasoning_model(model_name: Optional[str]) -> bool:
    """Return True if model_name matches OpenAI o-series or GPT-5+ reasoning families."""
    return bool(model_name and _OPENAI_OMIT_TEMPERATURE_MODEL_RE.match(model_name.strip()))


def _resolve_openai_temperature(
    model_name: Optional[str],
    temperature: Optional[float],
    provider_name: str = "openai",
    explicit_override: Optional[float] = None,
) -> Optional[float]:
    """Resolve effective temperature, omitting it for reasoning or cached-unsupported models."""
    if _is_cached_unsupported_temperature(provider_name, model_name):
        return None
    check_reasoning = provider_name == "openai" or (
        provider_name == "openrouter"
        and bool(model_name and model_name.strip().lower().startswith("openai/"))
    )
    if check_reasoning and _is_openai_reasoning_model(model_name):
        return None
    if explicit_override is not None:
        return explicit_override
    return temperature


def get_chat_model(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = 0.0,
    fallback_models: Optional[List[str]] = None,
    enable_fallback: bool = True,
    reasoning_effort: Optional[str] = None,
) -> BaseChatModel:
    """
    Instantiate or return a cached warm ChatModel based on provider priority.
    Cache automatically invalidates on API key changes or circuit-breaker transitions.
    """
    selected_provider = (provider or settings.MODEL_PROVIDER).lower().strip()
    norm_effort = (reasoning_effort or "").strip() or None
    with _PROVIDER_OUTAGE_LOCK:
        health_epoch = _PROVIDER_HEALTH_EPOCH
    cache_key = (
        selected_provider,
        model_name,
        float(temperature) if temperature is not None else None,
        tuple(fallback_models) if fallback_models else (),
        bool(enable_fallback),
        norm_effort,
        settings.OPENAI_MODEL,
        settings.OPENAI_TEMPERATURE,
        settings.OPENAI_REASONING_EFFORT,
        settings.GEMINI_MODEL,
        settings.OPENROUTER_MODEL,
        settings.REQUEST_TIMEOUT_SEC,
        _config_keys_fingerprint(),
        health_epoch,
    )
    with _CHAT_MODEL_CACHE_LOCK:
        cached = _CHAT_MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

    model = _build_chat_model_uncached(
        selected_provider=selected_provider,
        model_name=model_name,
        temperature=temperature,
        fallback_models=fallback_models,
        enable_fallback=enable_fallback,
        reasoning_effort=norm_effort,
    )
    with _CHAT_MODEL_CACHE_LOCK:
        if len(_CHAT_MODEL_CACHE) >= _CHAT_MODEL_CACHE_MAX_SIZE:
            _CHAT_MODEL_CACHE.pop(next(iter(_CHAT_MODEL_CACHE)))
        _CHAT_MODEL_CACHE[cache_key] = model
    return model


def get_router_chat_model(
    temperature: Optional[float] = 0.0,
    enable_fallback: bool = True,
) -> BaseChatModel:
    """Return warm ChatModel configured for NLU routing, defaulting to get_chat_model()."""
    router_provider = (settings.NLU_ROUTER_PROVIDER or "").strip() or None
    router_model = (settings.NLU_ROUTER_MODEL or "").strip() or None
    router_effort = (settings.NLU_ROUTER_REASONING_EFFORT or "").strip() or None
    if router_provider or router_model or router_effort:
        return get_chat_model(
            provider=router_provider,
            model_name=router_model,
            temperature=temperature,
            enable_fallback=enable_fallback,
            reasoning_effort=router_effort,
        )
    return get_chat_model(temperature=temperature, enable_fallback=enable_fallback)


def _build_chat_model_uncached(
    selected_provider: str,
    model_name: Optional[str] = None,
    temperature: Optional[float] = 0.0,
    fallback_models: Optional[List[str]] = None,
    enable_fallback: bool = True,
    reasoning_effort: Optional[str] = None,
) -> BaseChatModel:
    """Internal builder for get_chat_model without warm cache lookup."""
    effort_kwargs: Dict[str, Any] = {}
    if reasoning_effort and str(reasoning_effort).strip():
        effort_kwargs["reasoning_effort"] = str(reasoning_effort).strip()

    if selected_provider == "openrouter":
        api_key = settings.OPENROUTER_API_KEY
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required when MODEL_PROVIDER=openrouter")

        target_model = model_name or settings.OPENROUTER_MODEL
        openrouter_temperature = _resolve_openai_temperature(
            target_model,
            temperature,
            provider_name="openrouter",
        )

        # Configure model fallback array for high-availability
        extra_body: Dict[str, Any] = {}
        if enable_fallback:
            if fallback_models:
                fallbacks = list(fallback_models)
            else:
                active_free = fetch_free_models_catalog(api_key=api_key)
                fallbacks = [m for m in active_free if m != target_model]

            # Construct OpenRouter fallback hierarchy: [primary, *fallbacks]
            # OpenRouter enforces that the 'models' parameter must contain 3 items or fewer.
            model_chain = [target_model] + [m for m in fallbacks if m != target_model]
            extra_body["models"] = model_chain[:3]  # Max 3 models allowed by OpenRouter API

        return CircuitBreakerChatOpenAI(
            model=target_model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=openrouter_temperature,
            request_timeout=settings.REQUEST_TIMEOUT_SEC,
            default_headers={
                "HTTP-Referer": "https://fastprint.co.id",
                "X-Title": "FastPrint ERP LangGraph",
            },
            extra_body=extra_body if extra_body else None,
            provider_name="openrouter",
            **effort_kwargs,
        )

    elif selected_provider == "gemini":
        # Fast circuit-breaker: if all Gemini keys are unhealthy and OpenRouter key is available, bypass directly to OpenRouter
        if enable_fallback and settings.OPENROUTER_API_KEY and is_gemini_unhealthy():
            logger.info("Provider 'gemini' is currently marked unhealthy in circuit-breaker cache. Bypassing directly to OpenRouter.")
            return get_chat_model(
                provider="openrouter",
                fallback_models=fallback_models,
                enable_fallback=True,
                temperature=temperature,
            )

        primary_model = _build_gemini_model(
            model_name=model_name,
            temperature=temperature,
            enable_key_fallback=enable_fallback,
            reasoning_effort=reasoning_effort,
        )

        if enable_fallback and settings.OPENROUTER_API_KEY:
            try:
                or_model = get_chat_model(
                    provider="openrouter",
                    fallback_models=fallback_models,
                    enable_fallback=True,
                    temperature=temperature,
                )
                return primary_model.with_fallbacks([or_model])
            except Exception as exc:
                logger.warning(f"Could not configure OpenRouter fallback for Gemini model: {exc}")

        return primary_model

    elif selected_provider == "openai":
        # Fast circuit-breaker: if OpenAI is unhealthy, bypass directly to next healthy tier
        if enable_fallback and is_provider_unhealthy("openai"):
            if settings.get_gemini_api_keys() and not is_gemini_unhealthy():
                logger.info("Provider 'openai' is currently marked unhealthy in circuit-breaker cache. Bypassing directly to Gemini.")
                return get_chat_model(
                    provider="gemini",
                    enable_fallback=True,
                    temperature=temperature,
                )
            elif settings.OPENROUTER_API_KEY:
                logger.info("Providers 'openai' and 'gemini' are marked unhealthy. Bypassing directly to OpenRouter.")
                return get_chat_model(
                    provider="openrouter",
                    fallback_models=fallback_models,
                    enable_fallback=True,
                    temperature=temperature,
                )

        api_key = settings.OPENAI_API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required when MODEL_PROVIDER=openai")

        target_model = model_name or settings.OPENAI_MODEL
        effective_temperature = _resolve_openai_temperature(
            target_model,
            temperature,
            provider_name="openai",
            explicit_override=settings.OPENAI_TEMPERATURE,
        )
        openai_effort_kwargs: Dict[str, Any] = {}
        effective_openai_effort = (
            (reasoning_effort or "").strip()
            or (settings.OPENAI_REASONING_EFFORT or "").strip()
            or None
        )
        if effective_openai_effort and _is_openai_reasoning_model(target_model):
            openai_effort_kwargs["reasoning_effort"] = effective_openai_effort

        primary_model = CircuitBreakerChatOpenAI(
            model=target_model,
            api_key=api_key,
            temperature=effective_temperature,
            request_timeout=settings.REQUEST_TIMEOUT_SEC,
            provider_name="openai",
            **openai_effort_kwargs,
        )

        if enable_fallback:
            tier_fallbacks: List[BaseChatModel] = []
            # Priority 2: Gemini (multi-key failover, skipped if all Gemini keys are unhealthy)
            if settings.get_gemini_api_keys() and not is_gemini_unhealthy():
                try:
                    gemini_model = _build_gemini_model(
                        temperature=temperature,
                        enable_key_fallback=True,
                    )
                    if hasattr(gemini_model, "runnable") and hasattr(gemini_model, "fallbacks"):
                        tier_fallbacks.append(gemini_model.runnable)
                        tier_fallbacks.extend(gemini_model.fallbacks)
                    else:
                        tier_fallbacks.append(gemini_model)
                except Exception as exc:
                    logger.warning(f"Could not configure Gemini fallback for OpenAI model: {exc}")

            # Priority 3: OpenRouter
            if settings.OPENROUTER_API_KEY:
                try:
                    or_model = get_chat_model(
                        provider="openrouter",
                        fallback_models=fallback_models,
                        enable_fallback=True,
                        temperature=temperature,
                    )
                    tier_fallbacks.append(or_model)
                except Exception as exc:
                    logger.warning(f"Could not configure OpenRouter fallback for OpenAI model: {exc}")

            if tier_fallbacks:
                return primary_model.with_fallbacks(tier_fallbacks)

        return primary_model

    else:
        raise ValueError(
            f"Unsupported MODEL_PROVIDER '{selected_provider}'. Supported values: 'openai', 'gemini', 'openrouter'."
        )


def get_default_model_name() -> str:
    """Return the active configured default model identifier based on settings."""
    provider = (settings.MODEL_PROVIDER or "openai").lower().strip()
    if provider == "openai":
        return settings.OPENAI_MODEL or "gpt-4o-mini"
    elif provider == "gemini":
        return settings.GEMINI_MODEL or "gemini-2.5-flash"
    elif provider == "openrouter":
        return settings.OPENROUTER_MODEL or "openrouter/free"
    return settings.OPENAI_MODEL or "gpt-4o-mini"


def clean_model_display_name(raw_name: Optional[str]) -> str:
    """
    Format a model string into a concise human-readable display identifier.
    Examples:
        'nvidia/nemotron-3.5-super-vl:free' -> 'nemotron-3.5-super-vl'
        'openai/gpt-4o-mini' -> 'gpt-4o-mini'
        'gpt-4o-mini' -> 'gpt-4o-mini'
    """
    if not raw_name:
        return "Unknown"
    cleaned = str(raw_name).strip()
    if cleaned.endswith(":free"):
        cleaned = cleaned[:-5]
    if "/" in cleaned:
        cleaned = cleaned.split("/")[-1]
    return cleaned or "Unknown"

