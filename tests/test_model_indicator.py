"""
Unit tests for active AI model tracking, display name cleaning,
and API response serialization in FastPrint ERP AI Orchestrator.
"""
import json
import time
import unittest
from unittest.mock import patch, MagicMock

from app.core.config import settings
from app.graph.state import create_initial_state
from app.schemas.chat import ChatResponse
from app.services.llm import clean_model_display_name, get_default_model_name
from app.main import _build_chat_response


class TestModelIndicator(unittest.TestCase):
    """Test suite verifying active model resolution, display cleaning, and API payloads."""

    def test_get_default_model_name(self):
        """Verify get_default_model_name returns appropriate model based on provider."""
        with patch.object(settings, "MODEL_PROVIDER", "openrouter"):
            with patch.object(settings, "OPENROUTER_MODEL", "nvidia/nemotron-3.5-super-vl:free"):
                self.assertEqual(get_default_model_name(), "nvidia/nemotron-3.5-super-vl:free")

        with patch.object(settings, "MODEL_PROVIDER", "openai"):
            with patch.object(settings, "OPENAI_MODEL", "gpt-4o-mini"):
                self.assertEqual(get_default_model_name(), "gpt-4o-mini")

    def test_clean_model_display_name(self):
        """Verify model display name cleaning extracts readable identifier."""
        # Free-tier suffix stripping
        self.assertEqual(
            clean_model_display_name("nvidia/nemotron-3.5-super-vl:free"),
            "nemotron-3.5-super-vl",
        )
        # Vendor prefix stripping
        self.assertEqual(
            clean_model_display_name("meta-llama/llama-3.3-70b-instruct"),
            "llama-3.3-70b-instruct",
        )
        # Direct identifier
        self.assertEqual(
            clean_model_display_name("gpt-4o-mini"),
            "gpt-4o-mini",
        )
        # None and empty inputs
        self.assertEqual(clean_model_display_name(None), "Unknown")
        self.assertEqual(clean_model_display_name(""), "Unknown")
        self.assertEqual(clean_model_display_name("   "), "Unknown")

    def test_create_initial_state_contains_active_model(self):
        """Verify create_initial_state populates active_model."""
        state = create_initial_state("Cek stok kertas foto")
        self.assertIn("active_model", state)
        self.assertTrue(bool(state["active_model"]))

        # Explicit model override
        custom_state = create_initial_state(
            "Cek saldo kas",
            active_model="custom/test-model:free",
        )
        self.assertEqual(custom_state["active_model"], "custom/test-model:free")

    def test_chat_response_schema_serialization(self):
        """Verify ChatResponse model_used field serializes and deserializes properly."""
        resp = ChatResponse(
            final_answer="Saldo kas Rp 10.000.000",
            detected_domain="accounting",
            model_used="nvidia/nemotron-3.5-super-vl:free",
        )
        dumped = resp.model_dump()
        self.assertIn("model_used", dumped)
        self.assertEqual(dumped["model_used"], "nvidia/nemotron-3.5-super-vl:free")

        json_str = resp.model_dump_json()
        self.assertIn('"model_used":"nvidia/nemotron-3.5-super-vl:free"', json_str)

    def test_build_chat_response_includes_model_used(self):
        """Verify _build_chat_response populates model_used from final_state."""
        final_state = {
            "final_answer": "Test answer",
            "detected_domain": "inventory",
            "verification_status": "VALID",
            "company_id": 1,
            "active_model": "google/gemma-4-31b-it:free",
        }
        started_at = time.perf_counter()
        chat_resp = _build_chat_response(final_state, started_at)

        self.assertEqual(chat_resp.model_used, "google/gemma-4-31b-it:free")
        self.assertEqual(chat_resp.final_answer, "Test answer")

    def test_build_chat_response_fallback_model_used(self):
        """Verify _build_chat_response falls back to default model if active_model is missing."""
        final_state = {
            "final_answer": "Fallback answer",
            "detected_domain": "general",
            "verification_status": "VALID",
        }
        started_at = time.perf_counter()
        with patch.object(settings, "MODEL_PROVIDER", "openrouter"):
            with patch.object(settings, "OPENROUTER_MODEL", "nvidia/nemotron-3.5-super-vl:free"):
                chat_resp = _build_chat_response(final_state, started_at)
                self.assertEqual(chat_resp.model_used, "nvidia/nemotron-3.5-super-vl:free")


if __name__ == "__main__":
    unittest.main()
