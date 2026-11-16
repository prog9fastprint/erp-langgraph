"""
LLM Model Factory supporting OpenRouter (with free model catalog & auto-fallback) and OpenAI.
"""
import asyncio
import logging
import threading
import time
from typing import Any, Dict, List, Optional
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


def is_provider_unhealthy(provider: str) -> bool:
    """Check if the specified provider is temporarily marked unhealthy (outage cache)."""
    with _PROVIDER_OUTAGE_LOCK:
        return time.time() < _PROVIDER_OUTAGE_UNTIL.get(provider.lower().strip(), 0.0)


def mark_provider_unhealthy(provider: str, duration_sec: float = 900.0) -> None:
    """Temporarily mark a provider as unhealthy to bypass futile network calls."""
    with _PROVIDER_OUTAGE_LOCK:
        _PROVIDER_OUTAGE_UNTIL[provider.lower().strip()] = time.time() + duration_sec


def reset_provider_health(provider: Optional[str] = None) -> None:
    """Reset unhealthy status for a specific provider or all providers."""
    with _PROVIDER_OUTAGE_LOCK:
        if provider:
            _PROVIDER_OUTAGE_UNTIL.pop(provider.lower().strip(), None)
        else:
            _PROVIDER_OUTAGE_UNTIL.clear()


class CircuitBreakerChatOpenAI(ChatOpenAI):
    """ChatOpenAI wrapper that flags a provider/key as unhealthy upon auth/quota failures."""
    provider_name: str = "openai"

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

    def invoke(self, *args, **kwargs):
        try:
            return super().invoke(*args, **kwargs)
        except Exception as exc:
            self._handle_failure(exc)
            raise

    async def ainvoke(self, *args, **kwargs):
        try:
            return await super().ainvoke(*args, **kwargs)
        except Exception as exc:
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
    key = api_key or settings.OPENROUTER_API_KEY or settings.OPENAI_API_KEY

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
        # Schedule background non-blocking refresh via afetch_free_models_catalog
        try:
            asyncio.create_task(
                afetch_free_models_catalog(
                    api_key=api_key,
                    force_refresh=True,
                    prioritize_tools=prioritize_tools,
                )
            )
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

    primary_key = effective_keys[0]
    primary_model = CircuitBreakerChatOpenAI(
        model=target_model,
        api_key=primary_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        temperature=temperature,
        request_timeout=settings.REQUEST_TIMEOUT_SEC,
        provider_name=f"gemini:{primary_key[-6:]}",
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
            )
            for k in effective_keys[1:]
        ]
        return primary_model.with_fallbacks(fallbacks)

    return primary_model


def get_chat_model(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    fallback_models: Optional[List[str]] = None,
    enable_fallback: bool = True,
) -> BaseChatModel:
    """
    Instantiate and return a ChatModel based on provider priority.

    Supported providers:
    - 'openai': Official OpenAI API (Priority 1).
    - 'gemini': Google Gemini API via OpenAI-compatible endpoint with multi-key failover (Priority 2).
    - 'openrouter': OpenRouter API with automatic free-model catalog fallback (Priority 3).

    Args:
        provider: 'openai', 'gemini', or 'openrouter' (defaults to settings.MODEL_PROVIDER).
        model_name: Target model identifier.
        temperature: Sampling temperature (default 0.0 for deterministic output).
        fallback_models: Explicit list of fallback model IDs for OpenRouter.
        enable_fallback: If True, attaches dynamic fallback models chain.

    Returns:
        Configured BaseChatModel instance.
    """
    selected_provider = (provider or settings.MODEL_PROVIDER).lower().strip()

    if selected_provider == "openrouter":
        api_key = settings.OPENROUTER_API_KEY or settings.OPENAI_API_KEY
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required when MODEL_PROVIDER=openrouter")

        target_model = model_name or settings.OPENROUTER_MODEL

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

        return ChatOpenAI(
            model=target_model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature,
            request_timeout=settings.REQUEST_TIMEOUT_SEC,
            default_headers={
                "HTTP-Referer": "https://fastprint.co.id",
                "X-Title": "FastPrint ERP LangGraph",
            },
            extra_body=extra_body if extra_body else None,
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
        primary_model = CircuitBreakerChatOpenAI(
            model=target_model,
            api_key=api_key,
            temperature=temperature,
            request_timeout=settings.REQUEST_TIMEOUT_SEC,
            provider_name="openai",
        )

        if enable_fallback:
            tier_fallbacks: List[BaseChatModel] = []
            # Priority 2: Gemini
            if settings.get_gemini_api_keys():
                try:
                    gemini_model = get_chat_model(
                        provider="gemini",
                        enable_fallback=False,
                        temperature=temperature,
                    )
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

