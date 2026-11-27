"""
Unit tests for Centralized Configuration and LLM Client (OpenRouter & OpenAI).
Validates TASK-2.2 requirements in the canonical architecture.
"""
import os
import unittest
from unittest.mock import patch
from pydantic import BaseModel, Field, ValidationError

from app.core.config import Settings, settings
from app.services.llm import get_chat_model


class DummyOutputSchema(BaseModel):
    summary: str = Field(description="Summary of result")
    score: int = Field(description="Quality score")


class TestCentralizedConfig(unittest.TestCase):
    """Test suite for app.core.config.Settings and alias resolution."""

    def test_canonical_import_and_defaults(self):
        """Verify canonical settings instance and default values."""
        self.assertEqual(settings.APP_NAME, "erp-ai-orchestrator")
        default_settings = Settings(_env_file=None)
        self.assertEqual(default_settings.DEFAULT_COMPANY_ID, 1)
        self.assertEqual(default_settings.APP_PORT, 8002)
        self.assertIsInstance(settings.DEFAULT_COMPANY_ID, int)
        self.assertEqual(settings.REQUEST_TIMEOUT_SEC, 15)
        self.assertIn(settings.MODEL_PROVIDER, ["openrouter", "openai", "gemini"])

    def test_settings_aliases_resolution(self):
        """Verify DJANGO_INTERNAL_* aliases and timeout resolution."""
        env_overrides = {
            "DJANGO_INTERNAL_API_URL": "http://django.internal:8000",
            "DJANGO_INTERNAL_SERVICE_KEY": "test-service-key-xyz",
            "REQUEST_TIMEOUT_SEC": "45",
        }
        # Isolate environment by clearing primary aliases to test fallback resolution
        env = {k: v for k, v in os.environ.items() if k not in ("ERP_BASE_URL", "ERP_API_TOKEN")}
        env.update(env_overrides)
        with patch.dict(os.environ, env, clear=True):
            custom_settings = Settings(_env_file=None)
            self.assertEqual(custom_settings.ERP_BASE_URL, "http://django.internal:8000")
            self.assertEqual(custom_settings.ERP_API_TOKEN, "test-service-key-xyz")
            self.assertEqual(custom_settings.REQUEST_TIMEOUT_SEC, 45)

        # Verify that explicit ERP_BASE_URL takes precedence over DJANGO_INTERNAL_API_URL
        env_precedence = {
            "ERP_BASE_URL": "http://primary.erp:8000",
            "DJANGO_INTERNAL_API_URL": "http://django.internal:8000",
        }
        with patch.dict(os.environ, env_precedence, clear=False):
            precedence_settings = Settings(_env_file=None)
            self.assertEqual(precedence_settings.ERP_BASE_URL, "http://primary.erp:8000")

    def test_nlu_router_and_retrieval_defaults_and_validation(self):
        """Verify NLU_ROUTER_MODE defaults, normalization, strict Literal validation, and retrieval defaults."""
        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k
            not in (
                "NLU_ROUTER_MODE",
                "NLU_SHADOW_LOG_MISMATCH_ONLY",
                "TOOL_RETRIEVAL_TOP_K",
                "TOOL_RETRIEVAL_DISTANCE_THRESHOLD",
            )
        }
        with patch.dict(os.environ, clean_env, clear=True):
            cfg = Settings(_env_file=None)
            self.assertEqual(cfg.NLU_ROUTER_MODE, "legacy_regex")
            self.assertTrue(cfg.NLU_SHADOW_LOG_MISMATCH_ONLY)
            self.assertEqual(cfg.TOOL_RETRIEVAL_TOP_K, 8)
            self.assertAlmostEqual(cfg.TOOL_RETRIEVAL_DISTANCE_THRESHOLD, 0.72)

        with patch.dict(os.environ, {"NLU_ROUTER_MODE": "  SEMANTIC_LLM "}, clear=False):
            cfg_sem = Settings(_env_file=None)
            self.assertEqual(cfg_sem.NLU_ROUTER_MODE, "semantic_llm")

        with patch.dict(os.environ, {"NLU_ROUTER_MODE": "SHADOW", "NLU_SHADOW_LOG_MISMATCH_ONLY": "false"}, clear=False):
            cfg_shadow = Settings(_env_file=None)
            self.assertEqual(cfg_shadow.NLU_ROUTER_MODE, "shadow")
            self.assertFalse(cfg_shadow.NLU_SHADOW_LOG_MISMATCH_ONLY)

        with patch.dict(os.environ, {"NLU_ROUTER_MODE": "invalid_mode"}, clear=False):
            with self.assertRaises(ValidationError):
                Settings(_env_file=None)

    def test_openai_sampling_and_reasoning_settings_normalization(self):
        """OPENAI_TEMPERATURE and OPENAI_REASONING_EFFORT default to None and normalize blank strings."""
        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("OPENAI_TEMPERATURE", "OPENAI_REASONING_EFFORT", "NLU_ROUTER_REASONING_EFFORT")
        }
        with patch.dict(os.environ, clean_env, clear=True):
            cfg_default = Settings(_env_file=None)
            self.assertIsNone(cfg_default.OPENAI_TEMPERATURE)
            self.assertIsNone(cfg_default.OPENAI_REASONING_EFFORT)
            self.assertIsNone(cfg_default.NLU_ROUTER_REASONING_EFFORT)

        with patch.dict(
            os.environ,
            {
                "OPENAI_TEMPERATURE": "   ",
                "OPENAI_REASONING_EFFORT": "  ",
                "NLU_ROUTER_REASONING_EFFORT": " ",
            },
            clear=False,
        ):
            cfg_blank = Settings(_env_file=None)
            self.assertIsNone(cfg_blank.OPENAI_TEMPERATURE)
            self.assertIsNone(cfg_blank.OPENAI_REASONING_EFFORT)
            self.assertIsNone(cfg_blank.NLU_ROUTER_REASONING_EFFORT)

        with patch.dict(
            os.environ,
            {
                "OPENAI_TEMPERATURE": "0.25",
                "OPENAI_REASONING_EFFORT": " low ",
            },
            clear=False,
        ):
            cfg_set = Settings(_env_file=None)
            self.assertAlmostEqual(cfg_set.OPENAI_TEMPERATURE, 0.25)
            self.assertEqual(cfg_set.OPENAI_REASONING_EFFORT, "low")


class TestLLMClientFactory(unittest.TestCase):
    """Test suite for LLM factory (OpenRouter & OpenAI) capabilities."""

    def test_openrouter_model_initialization(self):
        """Test ChatOpenAI instantiation for OpenRouter provider."""
        with patch.object(settings, "OPENROUTER_API_KEY", "mock-openrouter-key"):
            model = get_chat_model(provider="openrouter", model_name="meta-llama/llama-3.3-70b-instruct:free")
            self.assertEqual(str(model.openai_api_base).rstrip("/"), "https://openrouter.ai/api/v1")
            self.assertEqual(model.model_name, "meta-llama/llama-3.3-70b-instruct:free")
            self.assertEqual(model.request_timeout, float(settings.REQUEST_TIMEOUT_SEC))

            # Verify Tool Calling support
            self.assertTrue(hasattr(model, "bind_tools"))
            bound_model = model.bind_tools([])
            self.assertIsNotNone(bound_model)

            # Verify Structured Outputs support
            self.assertTrue(hasattr(model, "with_structured_output"))
            structured_model = model.with_structured_output(DummyOutputSchema)
            self.assertIsNotNone(structured_model)

    def test_openai_model_initialization(self):
        """Test ChatOpenAI instantiation for OpenAI provider."""
        with patch.object(settings, "OPENAI_API_KEY", "mock-openai-key"):
            model = get_chat_model(provider="openai", model_name="gpt-4o-mini")
            self.assertEqual(model.model_name, "gpt-4o-mini")
            self.assertEqual(model.request_timeout, float(settings.REQUEST_TIMEOUT_SEC))

            # Verify Tool Calling and Structured Output support
            self.assertTrue(hasattr(model, "bind_tools"))
            self.assertTrue(hasattr(model, "with_structured_output"))

    def test_gemini_model_initialization(self):
        """Test ChatOpenAI instantiation targeting Gemini OpenAI compatibility."""
        with patch.object(settings, "GOOGLE_AI_API_KEY", "mock-gemini-key"), \
             patch.object(settings, "GOOGLE_AI_API_KEY_2", None), \
             patch.object(settings, "GOOGLE_AI_API_KEY_3", None):
            model = get_chat_model(provider="gemini", model_name="gemini-2.5-flash")
            self.assertEqual(model.model_name, "gemini-2.5-flash")
            self.assertEqual(str(model.openai_api_base).rstrip("/"), "https://generativelanguage.googleapis.com/v1beta/openai")
            self.assertEqual(model.request_timeout, float(settings.REQUEST_TIMEOUT_SEC))

            # Verify Tool Calling and Structured Output support
            self.assertTrue(hasattr(model, "bind_tools"))
            self.assertTrue(hasattr(model, "with_structured_output"))
            structured_model = model.with_structured_output(DummyOutputSchema)
            self.assertIsNotNone(structured_model)

    def test_missing_api_key_raises_error(self):
        """Factory should raise ValueError if required API key is missing."""
        with patch.object(settings, "OPENROUTER_API_KEY", None), patch.object(settings, "OPENAI_API_KEY", None):
            with self.assertRaises(ValueError):
                get_chat_model(provider="openrouter")

        with patch.object(settings, "OPENAI_API_KEY", None):
            with self.assertRaises(ValueError):
                get_chat_model(provider="openai", enable_fallback=False)

        with patch.object(settings, "GOOGLE_AI_API_KEY", None), \
             patch.object(settings, "GOOGLE_AI_API_KEY_2", None), \
             patch.object(settings, "GOOGLE_AI_API_KEY_3", None), \
             patch.object(settings, "GEMINI_API_KEY", None):
            with self.assertRaises(ValueError):
                get_chat_model(provider="gemini", enable_fallback=False)

    def test_unsupported_provider_raises_error(self):
        """Factory should raise ValueError for unsupported provider (e.g. legacy 'ollama')."""
        with self.assertRaises(ValueError) as ctx:
            get_chat_model(provider="ollama")
        self.assertIn("Unsupported MODEL_PROVIDER", str(ctx.exception))
        self.assertIn("gemini", str(ctx.exception))

    def test_get_chat_model_reuses_warm_cache_and_invalidates_on_health_change(self):
        """Warm chat model cache reuses client for identical config and invalidates on circuit-breaker trip."""
        from app.services.llm import clear_chat_model_cache, mark_provider_unhealthy, reset_provider_health

        clear_chat_model_cache()
        reset_provider_health()
        try:
            with patch.object(settings, "OPENAI_API_KEY", "warm-cache-key-1"):
                m1 = get_chat_model(provider="openai", model_name="gpt-4o-mini", enable_fallback=False)
                m2 = get_chat_model(provider="openai", model_name="gpt-4o-mini", enable_fallback=False)
                self.assertIs(m1, m2)

            with patch.object(settings, "OPENAI_API_KEY", "warm-cache-key-2"):
                m3 = get_chat_model(provider="openai", model_name="gpt-4o-mini", enable_fallback=False)
                self.assertIsNot(m1, m3)

            with patch.object(settings, "OPENAI_API_KEY", "warm-cache-key-1"):
                mark_provider_unhealthy("openai")
                reset_provider_health()
                m4 = get_chat_model(provider="openai", model_name="gpt-4o-mini", enable_fallback=False)
                self.assertIsNot(m1, m4)
        finally:
            reset_provider_health()
            clear_chat_model_cache()

    def test_get_router_chat_model_honors_dedicated_router_overrides(self):
        """get_router_chat_model defaults to primary model and honors NLU_ROUTER_* overrides."""
        from app.services.llm import clear_chat_model_cache, get_router_chat_model, reset_provider_health

        clear_chat_model_cache()
        reset_provider_health()
        try:
            with patch.object(settings, "MODEL_PROVIDER", "openai"), \
                 patch.object(settings, "OPENAI_API_KEY", "router-key-1"), \
                 patch.object(settings, "NLU_ROUTER_PROVIDER", None), \
                 patch.object(settings, "NLU_ROUTER_MODEL", None), \
                 patch.object(settings, "NLU_ROUTER_REASONING_EFFORT", None):
                base_m = get_chat_model(enable_fallback=False)
                router_m = get_router_chat_model(enable_fallback=False)
                self.assertIs(base_m, router_m)

            with patch.object(settings, "MODEL_PROVIDER", "openai"), \
                 patch.object(settings, "OPENAI_API_KEY", "router-key-1"), \
                 patch.object(settings, "GOOGLE_AI_API_KEY", "gemini-router-key"), \
                 patch.object(settings, "NLU_ROUTER_PROVIDER", "gemini"), \
                 patch.object(settings, "NLU_ROUTER_MODEL", "gemini-2.5-flash-lite"), \
                 patch.object(settings, "NLU_ROUTER_REASONING_EFFORT", "none"):
                fast_m = get_router_chat_model(enable_fallback=False)
                self.assertEqual(fast_m.model_name, "gemini-2.5-flash-lite")
                self.assertEqual(getattr(fast_m, "reasoning_effort", None), "none")
        finally:
            reset_provider_health()
            clear_chat_model_cache()

    def test_openai_omits_temperature_for_reasoning_and_gpt5_plus_models(self):
        """GPT-5+, GPT-6+, future o*/gpt-7+ models, and OpenRouter openai/* reasoning IDs omit temperature=0.0."""
        from app.services.llm import (
            clear_chat_model_cache,
            clear_unsupported_temperature_cache,
            reset_provider_health,
        )

        clear_chat_model_cache()
        clear_unsupported_temperature_cache()
        reset_provider_health()
        try:
            with patch.object(settings, "OPENAI_API_KEY", "mock-openai-key"), \
                 patch.object(settings, "OPENROUTER_API_KEY", "mock-or-key"), \
                 patch.object(settings, "OPENAI_TEMPERATURE", None):
                for legacy_model in ("gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"):
                    m = get_chat_model(
                        provider="openai",
                        model_name=legacy_model,
                        temperature=0.0,
                        enable_fallback=False,
                    )
                    self.assertEqual(m.temperature, 0.0)
                    self.assertEqual(m._default_params.get("temperature"), 0.0)

                for reasoning_model in ("gpt-6-luna", "gpt-5-nano", "gpt-5.4-mini", "gpt-7", "o3", "o4-mini"):
                    m = get_chat_model(
                        provider="openai",
                        model_name=reasoning_model,
                        temperature=0.0,
                        enable_fallback=False,
                    )
                    self.assertIsNone(m.temperature)
                    self.assertNotIn("temperature", m._default_params)
                    self.assertIsNotNone(m.with_structured_output(DummyOutputSchema))

                or_reasoning = get_chat_model(
                    provider="openrouter",
                    model_name="openai/gpt-6-luna",
                    temperature=0.0,
                    enable_fallback=False,
                )
                self.assertIsNone(or_reasoning.temperature)
                self.assertNotIn("temperature", or_reasoning._default_params)

                or_standard = get_chat_model(
                    provider="openrouter",
                    model_name="meta-llama/llama-3.3-70b-instruct:free",
                    temperature=0.0,
                    enable_fallback=False,
                )
                self.assertEqual(or_standard.temperature, 0.0)
        finally:
            reset_provider_health()
            clear_unsupported_temperature_cache()
            clear_chat_model_cache()

    def test_openai_reasoning_effort_gating_and_cache_invalidation(self):
        """OPENAI_REASONING_EFFORT applies only to OpenAI reasoning models and invalidates warm cache."""
        from app.services.llm import (
            clear_chat_model_cache,
            clear_unsupported_temperature_cache,
            get_router_chat_model,
            reset_provider_health,
        )

        clear_chat_model_cache()
        clear_unsupported_temperature_cache()
        reset_provider_health()
        try:
            with patch.object(settings, "MODEL_PROVIDER", "openai"), \
                 patch.object(settings, "OPENAI_API_KEY", "mock-openai-key"), \
                 patch.object(settings, "GOOGLE_AI_API_KEY", "mock-gemini-key"), \
                 patch.object(settings, "OPENAI_TEMPERATURE", None), \
                 patch.object(settings, "OPENAI_REASONING_EFFORT", "low"), \
                 patch.object(settings, "NLU_ROUTER_PROVIDER", None), \
                 patch.object(settings, "NLU_ROUTER_MODEL", None), \
                 patch.object(settings, "NLU_ROUTER_REASONING_EFFORT", None):
                legacy_m = get_chat_model(provider="openai", model_name="gpt-4o-mini", enable_fallback=False)
                self.assertIsNone(getattr(legacy_m, "reasoning_effort", None))

                luna_low = get_chat_model(provider="openai", model_name="gpt-6-luna", enable_fallback=False)
                self.assertEqual(getattr(luna_low, "reasoning_effort", None), "low")

                o3_low = get_chat_model(provider="openai", model_name="o3", enable_fallback=False)
                self.assertEqual(getattr(o3_low, "reasoning_effort", None), "low")

                with patch.object(settings, "OPENAI_REASONING_EFFORT", "medium"):
                    luna_med = get_chat_model(provider="openai", model_name="gpt-6-luna", enable_fallback=False)
                    self.assertIsNot(luna_low, luna_med)
                    self.assertEqual(getattr(luna_med, "reasoning_effort", None), "medium")

                with patch.object(settings, "OPENAI_TEMPERATURE", 0.3):
                    legacy_override = get_chat_model(provider="openai", model_name="gpt-4o-mini", enable_fallback=False)
                    self.assertIsNot(legacy_m, legacy_override)
                    self.assertAlmostEqual(legacy_override.temperature, 0.3)

                with patch.object(settings, "NLU_ROUTER_MODEL", "gpt-6-luna"), \
                     patch.object(settings, "NLU_ROUTER_REASONING_EFFORT", "none"):
                    router_luna = get_router_chat_model(enable_fallback=False)
                    self.assertEqual(getattr(router_luna, "reasoning_effort", None), "none")

                with patch.object(settings, "NLU_ROUTER_PROVIDER", "gemini"), \
                     patch.object(settings, "NLU_ROUTER_MODEL", "gemini-2.5-flash-lite"), \
                     patch.object(settings, "NLU_ROUTER_REASONING_EFFORT", None):
                    router_gem = get_router_chat_model(enable_fallback=False)
                    self.assertEqual(router_gem.model_name, "gemini-2.5-flash-lite")
                    self.assertIsNone(getattr(router_gem, "reasoning_effort", None))
        finally:
            reset_provider_health()
            clear_unsupported_temperature_cache()
            clear_chat_model_cache()


if __name__ == "__main__":
    unittest.main()
