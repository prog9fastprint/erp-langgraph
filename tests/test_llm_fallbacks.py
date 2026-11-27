"""
Unit tests for Free LLM Models Catalog and Auto-Fallback (TASK: free-llm-models integration).
Validates dynamic discovery, priority sorting via free_llm_models metadata,
OpenRouter 3-item model array compliance, and fallback configuration.
"""
import unittest
from unittest.mock import MagicMock, patch
from app.core.config import settings
from app.services.llm import (
    fetch_free_models_catalog,
    afetch_free_models_catalog,
    get_chat_model,
    DEFAULT_CURATED_FREE_MODELS,
)


class TestFreeLLMModelsCatalog(unittest.TestCase):
    """Test suite for free models dynamic discovery & prioritization."""

    def test_default_curated_models_contains_openrouter_free(self):
        """Curated list must contain 'openrouter/free' and standard models."""
        self.assertIn("openrouter/free", DEFAULT_CURATED_FREE_MODELS)
        self.assertTrue(any("gemma" in m for m in DEFAULT_CURATED_FREE_MODELS))

    def test_fetch_catalog_offline_uses_curated_list(self):
        """When OpenRouter API is unreachable, catalog gracefully falls back to curated list."""
        with patch("httpx.Client.get", side_effect=Exception("Connection refused")):
            catalog = fetch_free_models_catalog(force_refresh=True)
            self.assertIn("openrouter/free", catalog)
            self.assertGreaterEqual(len(catalog), 3)

    def test_fetch_catalog_prioritizes_tools_support(self):
        """Models supporting tools/JSON from free_llm_models should be prioritized."""
        fake_api_response = MagicMock()
        fake_api_response.status_code = 200
        fake_api_response.json.return_value = {
            "data": [
                {"id": "unknown/random-model:free", "pricing": {"prompt": "0", "completion": "0"}},
                {"id": "google/gemma-4-31b-it:free", "pricing": {"prompt": "0", "completion": "0"}},
                {"id": "openrouter/free", "pricing": {"prompt": "0", "completion": "0"}},
            ]
        }

        with patch("httpx.Client.get", return_value=fake_api_response):
            catalog = fetch_free_models_catalog(api_key="mock-key", force_refresh=True, prioritize_tools=True)
            self.assertEqual(catalog[0], "openrouter/free")
            self.assertIn("google/gemma-4-31b-it:free", catalog)

    def test_fetch_catalog_in_running_event_loop_is_non_blocking(self):
        """When an event loop is running, fetch_free_models_catalog returns immediately without blocking."""
        import asyncio

        async def _test_inside_loop():
            models = fetch_free_models_catalog(force_refresh=True)
            self.assertIsInstance(models, list)
            self.assertIn("openrouter/free", models)

        asyncio.run(_test_inside_loop())


class TestAsyncFreeLLMModelsCatalog(unittest.IsolatedAsyncioTestCase):
    """Test suite for afetch_free_models_catalog non-blocking execution."""

    async def test_afetch_catalog_offline_uses_curated_list(self):
        """When OpenRouter API is unreachable, afetch_free_models_catalog gracefully falls back."""
        with patch("httpx.Client.get", side_effect=Exception("Connection refused")):
            catalog = await afetch_free_models_catalog(force_refresh=True)
            self.assertIn("openrouter/free", catalog)
            self.assertGreaterEqual(len(catalog), 3)

    async def test_afetch_catalog_prioritizes_tools_support(self):
        """afetch_free_models_catalog respects tool priority and runs offloaded to thread."""
        fake_api_response = MagicMock()
        fake_api_response.status_code = 200
        fake_api_response.json.return_value = {
            "data": [
                {"id": "unknown/random-model:free", "pricing": {"prompt": "0", "completion": "0"}},
                {"id": "google/gemma-4-31b-it:free", "pricing": {"prompt": "0", "completion": "0"}},
                {"id": "openrouter/free", "pricing": {"prompt": "0", "completion": "0"}},
            ]
        }

        with patch("httpx.Client.get", return_value=fake_api_response):
            catalog = await afetch_free_models_catalog(api_key="mock-key", force_refresh=True, prioritize_tools=True)
            self.assertEqual(catalog[0], "openrouter/free")
            self.assertIn("google/gemma-4-31b-it:free", catalog)

    async def test_afetch_catalog_cache_hit(self):
        """Subsequent afetch_free_models_catalog calls without force_refresh return cached results."""
        with patch("httpx.Client.get", side_effect=Exception("Should not be called on cache hit")):
            cat1 = await afetch_free_models_catalog(force_refresh=False)
            cat2 = await afetch_free_models_catalog(force_refresh=False)
            self.assertEqual(cat1, cat2)


class TestAutoFallbackModelFactory(unittest.TestCase):
    """Test suite for get_chat_model with auto-fallback configuration."""

    def test_openrouter_model_has_fallback_array(self):
        """OpenRouter ChatOpenAI should contain extra_body['models'] with max 3 items."""
        with patch.object(settings, "OPENROUTER_API_KEY", "mock-openrouter-key"):
            model = get_chat_model(
                provider="openrouter",
                model_name="google/gemma-4-31b-it:free",
                enable_fallback=True,
            )
            self.assertEqual(model.model_name, "google/gemma-4-31b-it:free")
            self.assertIsNotNone(model.extra_body)
            models_list = model.extra_body.get("models")
            self.assertIsInstance(models_list, list)
            # OpenRouter enforces max 3 models in fallback array
            self.assertLessEqual(len(models_list), 3)
            self.assertEqual(models_list[0], "google/gemma-4-31b-it:free")
            self.assertIn("openrouter/free", models_list)

    def test_openrouter_custom_fallback_override(self):
        """Explicit fallback list should be respected up to 3 items."""
        with patch.object(settings, "OPENROUTER_API_KEY", "mock-openrouter-key"):
            custom_fallbacks = ["nvidia/nemotron-3.5-lightning:free", "openrouter/free"]
            model = get_chat_model(
                provider="openrouter",
                model_name="primary/model:free",
                fallback_models=custom_fallbacks,
            )
            models_list = model.extra_body.get("models")
            self.assertEqual(models_list, ["primary/model:free", "nvidia/nemotron-3.5-lightning:free", "openrouter/free"])

    def test_openrouter_disable_fallback(self):
        """When enable_fallback is False, extra_body should be None or empty."""
        with patch.object(settings, "OPENROUTER_API_KEY", "mock-openrouter-key"):
            model = get_chat_model(
                provider="openrouter",
                model_name="primary/model:free",
                enable_fallback=False,
            )
            self.assertIsNone(model.extra_body)

    def test_openai_model_has_openrouter_fallback(self):
        """When provider='openai' and only openrouter is available, model should have openrouter fallback configured."""
        with patch.object(settings, "OPENAI_API_KEY", "mock-openai-key"), \
             patch.object(settings, "GOOGLE_AI_API_KEY", None), \
             patch.object(settings, "GOOGLE_AI_API_KEY_2", None), \
             patch.object(settings, "GOOGLE_AI_API_KEY_3", None), \
             patch.object(settings, "GEMINI_API_KEY", None), \
             patch.object(settings, "OPENROUTER_API_KEY", "mock-openrouter-key"):
            model = get_chat_model(
                provider="openai",
                model_name="gpt-4o-mini",
                enable_fallback=True,
            )
            self.assertEqual(model.model_name, "gpt-4o-mini")
            self.assertTrue(hasattr(model, "fallbacks"))
            self.assertEqual(len(model.fallbacks), 1)
            fb_model = model.fallbacks[0]
            self.assertIn("openrouter.ai", str(fb_model.openai_api_base))

    def test_openai_model_has_prioritized_gemini_and_openrouter_fallbacks(self):
        """When provider='openai' with all providers available, fallbacks should be 1. Gemini, 2. OpenRouter."""
        with patch.object(settings, "OPENAI_API_KEY", "mock-openai-key"), \
             patch.object(settings, "GOOGLE_AI_API_KEY", "mock-gemini-key"), \
             patch.object(settings, "GOOGLE_AI_API_KEY_2", None), \
             patch.object(settings, "GOOGLE_AI_API_KEY_3", None), \
             patch.object(settings, "OPENROUTER_API_KEY", "mock-openrouter-key"):
            model = get_chat_model(
                provider="openai",
                model_name="gpt-4o-mini",
                enable_fallback=True,
            )
            self.assertEqual(model.model_name, "gpt-4o-mini")
            self.assertTrue(hasattr(model, "fallbacks"))
            self.assertEqual(len(model.fallbacks), 2)
            # Tier 2: Gemini
            self.assertIn("generativelanguage.googleapis.com", str(model.fallbacks[0].openai_api_base))
            # Tier 3: OpenRouter
            self.assertIn("openrouter.ai", str(model.fallbacks[1].openai_api_base))

    def test_gemini_model_multi_key_fallback(self):
        """When multiple Gemini keys are configured, keys 2 and 3 should be attached as fallbacks."""
        with patch.object(settings, "GOOGLE_AI_API_KEY", "mock-key-1"), \
             patch.object(settings, "GOOGLE_AI_API_KEY_2", "mock-key-2"), \
             patch.object(settings, "GOOGLE_AI_API_KEY_3", "mock-key-3"), \
             patch.object(settings, "OPENROUTER_API_KEY", None):
            model = get_chat_model(
                provider="gemini",
                model_name="gemini-2.5-flash",
                enable_fallback=True,
            )
            self.assertEqual(model.model_name, "gemini-2.5-flash")
            self.assertTrue(hasattr(model, "fallbacks"))
            self.assertEqual(len(model.fallbacks), 2)
            # Verify secondary keys are wrapped in fallbacks
            self.assertEqual(model.fallbacks[0].openai_api_key.get_secret_value(), "mock-key-2")
            self.assertEqual(model.fallbacks[1].openai_api_key.get_secret_value(), "mock-key-3")

    def test_circuit_breaker_bypasses_unhealthy_openai_to_gemini(self):
        """When provider 'openai' is marked unhealthy, factory should bypass directly to Gemini."""
        from app.services.llm import mark_provider_unhealthy, reset_provider_health
        reset_provider_health()
        try:
            mark_provider_unhealthy("openai", duration_sec=300.0)
            with patch.object(settings, "OPENAI_API_KEY", "mock-openai-key"), \
                 patch.object(settings, "GOOGLE_AI_API_KEY", "mock-gemini-key"), \
                 patch.object(settings, "GOOGLE_AI_API_KEY_2", None), \
                 patch.object(settings, "GOOGLE_AI_API_KEY_3", None), \
                 patch.object(settings, "OPENROUTER_API_KEY", "mock-openrouter-key"):
                model = get_chat_model(
                    provider="openai",
                    model_name="gpt-4o-mini",
                    enable_fallback=True,
                )
                # Primary model should be Gemini, not OpenAI
                self.assertIn("generativelanguage.googleapis.com", str(model.openai_api_base))
                self.assertEqual(model.model_name, settings.GEMINI_MODEL)
        finally:
            reset_provider_health()

    def test_openai_model_disable_fallback(self):
        """When provider='openai' and enable_fallback=False, model should not have fallbacks."""
        with patch.object(settings, "OPENAI_API_KEY", "mock-openai-key"), \
             patch.object(settings, "OPENROUTER_API_KEY", "mock-openrouter-key"):
            model = get_chat_model(
                provider="openai",
                model_name="gpt-4o-mini",
                enable_fallback=False,
            )
            self.assertEqual(model.model_name, "gpt-4o-mini")
            self.assertFalse(hasattr(model, "fallbacks"))


class TestUnsupportedTemperatureSelfHealing(unittest.IsolatedAsyncioTestCase):
    """Test suite for CircuitBreakerChatOpenAI reactive 400 temperature self-healing."""

    @staticmethod
    def _make_temp_400_error() -> Exception:
        err = Exception(
            "Error code: 400 - Unsupported value: 'temperature' does not support 0.0 with this model."
        )
        err.status_code = 400
        return err

    @staticmethod
    def _make_other_400_error() -> Exception:
        err = Exception("Error code: 400 - Invalid schema definition.")
        err.status_code = 400
        return err

    def test_invoke_self_heals_unsupported_temperature_without_fallback(self):
        """Sync invoke retries once without temperature and caches concrete models without calling fallback."""
        from langchain_core.messages import AIMessage, HumanMessage
        from app.services.llm import (
            clear_chat_model_cache,
            clear_unsupported_temperature_cache,
            is_provider_unhealthy,
            reset_provider_health,
        )

        clear_chat_model_cache()
        clear_unsupported_temperature_cache()
        reset_provider_health()
        observed_payloads = []
        try:
            with patch.object(settings, "OPENAI_API_KEY", "mock-openai-key"), \
                 patch.object(settings, "OPENROUTER_API_KEY", "mock-or-key"), \
                 patch.object(settings, "GOOGLE_AI_API_KEY", None), \
                 patch.object(settings, "GOOGLE_AI_API_KEY_2", None), \
                 patch.object(settings, "GOOGLE_AI_API_KEY_3", None):
                model_with_fb = get_chat_model(
                    provider="openai",
                    model_name="chat-latest",
                    temperature=0.0,
                    enable_fallback=True,
                )
                primary = model_with_fb.runnable

                def _fake_invoke(bound_self, messages, *args, **kwargs):
                    payload = bound_self._get_request_payload(messages)
                    observed_payloads.append((bound_self.provider_name, dict(payload)))
                    if "temperature" in payload:
                        raise self._make_temp_400_error()
                    return AIMessage(content="ok-healed")

                with patch("langchain_openai.ChatOpenAI.invoke", autospec=True, side_effect=_fake_invoke):
                    res1 = model_with_fb.invoke([HumanMessage(content="ping")])
                    res2 = model_with_fb.invoke([HumanMessage(content="ping-2")])

                self.assertEqual(res1.content, "ok-healed")
                self.assertEqual(res2.content, "ok-healed")
                self.assertEqual(len(observed_payloads), 3)
                self.assertEqual(observed_payloads[0][0], "openai")
                self.assertIn("temperature", observed_payloads[0][1])
                self.assertEqual(observed_payloads[1][0], "openai")
                self.assertNotIn("temperature", observed_payloads[1][1])
                self.assertEqual(observed_payloads[2][0], "openai")
                self.assertNotIn("temperature", observed_payloads[2][1])
                self.assertFalse(is_provider_unhealthy("openai"))
                self.assertNotIn("temperature", primary._get_request_payload([HumanMessage(content="check")]))
        finally:
            reset_provider_health()
            clear_unsupported_temperature_cache()
            clear_chat_model_cache()

    async def test_ainvoke_and_astream_self_heal_and_meta_router_not_cached(self):
        """Async ainvoke/astream retry once without temperature; openrouter/free avoids permanent cache."""
        from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
        from app.services.llm import (
            _is_cached_unsupported_temperature,
            clear_chat_model_cache,
            clear_unsupported_temperature_cache,
            reset_provider_health,
        )

        clear_chat_model_cache()
        clear_unsupported_temperature_cache()
        reset_provider_health()
        try:
            with patch.object(settings, "OPENAI_API_KEY", "mock-openai-key"), \
                 patch.object(settings, "OPENROUTER_API_KEY", "mock-or-key"):
                or_meta = get_chat_model(
                    provider="openrouter",
                    model_name="openrouter/free",
                    temperature=0.0,
                    enable_fallback=False,
                )
                seen_ainvoke = []

                async def _fake_ainvoke(bound_self, messages, *args, **kwargs):
                    payload = bound_self._get_request_payload(messages)
                    seen_ainvoke.append("temperature" in payload)
                    if "temperature" in payload:
                        raise self._make_temp_400_error()
                    return AIMessage(content="ok-async")

                with patch("langchain_openai.ChatOpenAI.ainvoke", autospec=True, side_effect=_fake_ainvoke):
                    res = await or_meta.ainvoke([HumanMessage(content="hi")])

                self.assertEqual(res.content, "ok-async")
                self.assertEqual(seen_ainvoke, [True, False])
                self.assertFalse(_is_cached_unsupported_temperature("openrouter", "openrouter/free"))

                oa_model = get_chat_model(
                    provider="openai",
                    model_name="chat-latest-stream",
                    temperature=0.0,
                    enable_fallback=False,
                )
                seen_stream = []

                async def _fake_astream(bound_self, messages, *args, **kwargs):
                    payload = bound_self._get_request_payload(messages)
                    seen_stream.append("temperature" in payload)
                    if "temperature" in payload:
                        raise self._make_temp_400_error()
                    yield AIMessageChunk(content="chunk-1")

                with patch("langchain_openai.ChatOpenAI.astream", autospec=True, side_effect=_fake_astream):
                    chunks = [c.content async for c in oa_model.astream([HumanMessage(content="hi")])]

                self.assertEqual(chunks, ["chunk-1"])
                self.assertEqual(seen_stream, [True, False])
                self.assertTrue(_is_cached_unsupported_temperature("openai", "chat-latest-stream"))
        finally:
            reset_provider_health()
            clear_unsupported_temperature_cache()
            clear_chat_model_cache()

    async def test_mid_stream_400_does_not_retry_and_other_400_logs_and_falls_back(self):
        """Mid-stream errors never retry; non-temperature 400 logs warning, keeps breaker healthy, and uses fallback."""
        from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
        from app.services.llm import (
            clear_chat_model_cache,
            clear_unsupported_temperature_cache,
            is_provider_unhealthy,
            reset_provider_health,
        )

        clear_chat_model_cache()
        clear_unsupported_temperature_cache()
        reset_provider_health()
        try:
            with patch.object(settings, "OPENAI_API_KEY", "mock-openai-key"), \
                 patch.object(settings, "OPENROUTER_API_KEY", "mock-or-key"), \
                 patch.object(settings, "GOOGLE_AI_API_KEY", None), \
                 patch.object(settings, "GOOGLE_AI_API_KEY_2", None), \
                 patch.object(settings, "GOOGLE_AI_API_KEY_3", None):
                oa_only = get_chat_model(
                    provider="openai",
                    model_name="chat-latest-midstream",
                    temperature=0.0,
                    enable_fallback=False,
                )
                stream_calls = 0

                async def _mid_stream_fail(bound_self, messages, *args, **kwargs):
                    nonlocal stream_calls
                    stream_calls += 1
                    yield AIMessageChunk(content="partial")
                    raise self._make_temp_400_error()

                with patch("langchain_openai.ChatOpenAI.astream", autospec=True, side_effect=_mid_stream_fail):
                    with self.assertRaises(Exception):
                        async for _ in oa_only.astream([HumanMessage(content="hi")]):
                            pass
                self.assertEqual(stream_calls, 1)

                oa_with_fb = get_chat_model(
                    provider="openai",
                    model_name="gpt-4o-mini",
                    temperature=0.0,
                    enable_fallback=True,
                )

                def _other_400_then_or(bound_self, messages, *args, **kwargs):
                    if bound_self.provider_name == "openai":
                        raise self._make_other_400_error()
                    return AIMessage(content="from-openrouter-fallback")

                with patch("langchain_openai.ChatOpenAI.invoke", autospec=True, side_effect=_other_400_then_or), \
                     self.assertLogs("llm_factory", level="WARNING") as logs:
                    fb_res = oa_with_fb.invoke([HumanMessage(content="hi")])

                self.assertEqual(fb_res.content, "from-openrouter-fallback")
                self.assertFalse(is_provider_unhealthy("openai"))
                self.assertTrue(any("HTTP 400 error" in msg for msg in logs.output))
        finally:
            reset_provider_health()
            clear_unsupported_temperature_cache()
            clear_chat_model_cache()


if __name__ == "__main__":
    unittest.main()
