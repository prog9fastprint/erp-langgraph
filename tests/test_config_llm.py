"""
Unit tests for Centralized Configuration and LLM Client (OpenRouter & OpenAI).
Validates TASK-2.2 requirements in the canonical architecture.
"""
import os
import unittest
from unittest.mock import patch
from pydantic import BaseModel, Field

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


if __name__ == "__main__":
    unittest.main()
