"""
Unit tests for Context Window Management & Payload Sanitization in FastPrint ERP.
Tests message trimming, ERP payload sanitization, summarizer node, and workflow integration.
"""
import unittest
from langchain_core.messages import HumanMessage, AIMessage

from app.core.config import settings
from app.core.context import (
    trim_conversation_messages,
    sanitize_erp_payload,
    build_contextual_prompt,
)
from app.graph.state import create_initial_state
from app.graph.nodes.summarizer import summarizer_node
from app.graph.workflow import run_workflow_sync


class TestMessageTrimming(unittest.TestCase):
    """Test suite for centralized message sliding window trimming."""

    def test_trim_empty_and_short_messages(self):
        # Empty input
        self.assertEqual(trim_conversation_messages([]), [])
        self.assertEqual(trim_conversation_messages(None), [])

        # Short input under limit
        msgs = [
            HumanMessage(content="Halo"),
            AIMessage(content="Halo! Ada yang bisa dibantu?"),
        ]
        trimmed = trim_conversation_messages(msgs, max_messages=5)
        self.assertEqual(len(trimmed), 2)
        self.assertEqual(trimmed[0].content, "Halo")
        self.assertEqual(trimmed[1].content, "Halo! Ada yang bisa dibantu?")

    def test_trim_exceeding_max_messages(self):
        # 15 messages, max 5
        msgs = [HumanMessage(content=f"Message {i}") for i in range(15)]
        trimmed = trim_conversation_messages(msgs, max_messages=5)
        self.assertEqual(len(trimmed), 5)
        self.assertEqual(trimmed[0].content, "Message 10")
        self.assertEqual(trimmed[-1].content, "Message 14")

    def test_trim_filters_empty_content(self):
        msgs = [
            HumanMessage(content="Valid 1"),
            HumanMessage(content="   "),
            AIMessage(content=""),
            HumanMessage(content="Valid 2"),
        ]
        trimmed = trim_conversation_messages(msgs, max_messages=10)
        self.assertEqual(len(trimmed), 2)
        self.assertEqual(trimmed[0].content, "Valid 1")
        self.assertEqual(trimmed[1].content, "Valid 2")


class TestPayloadSanitization(unittest.TestCase):
    """Test suite for Django ERP payload pruning and sanitization."""

    def test_strips_django_technical_fields(self):
        raw_payload = {
            "id": 101,
            "invoice_number": "INV/2026/001",
            "amount_total": 5000000.0,
            "create_uid": 1,
            "write_uid": 2,
            "create_date": "2026-01-01 10:00:00",
            "write_date": "2026-01-02 11:00:00",
            "__last_update": "2026-01-02 11:00:00",
            "access_token": "secret_token_abc",
            "is_active": True,
            "_internal_debug": "strip_me",
        }
        sanitized = sanitize_erp_payload(raw_payload)

        self.assertIn("id", sanitized)
        self.assertIn("invoice_number", sanitized)
        self.assertIn("amount_total", sanitized)

        # Technical fields must be stripped
        self.assertNotIn("create_uid", sanitized)
        self.assertNotIn("write_uid", sanitized)
        self.assertNotIn("create_date", sanitized)
        self.assertNotIn("write_date", sanitized)
        self.assertNotIn("__last_update", sanitized)
        self.assertNotIn("access_token", sanitized)
        self.assertNotIn("is_active", sanitized)
        self.assertNotIn("_internal_debug", sanitized)

    def test_strips_null_and_empty_values(self):
        raw_payload = {
            "name": "Kertas Art Paper",
            "description": None,
            "empty_notes": "   ",
            "price": 45000.0,
        }
        sanitized = sanitize_erp_payload(raw_payload)
        self.assertEqual(sanitized, {"name": "Kertas Art Paper", "price": 45000.0})

    def test_handles_nested_records_and_calculations(self):
        raw_payload = {
            "invoices": [
                {
                    "id": 1,
                    "name": "INV-001",
                    "create_uid": 4,
                    "amount": 1000.0,
                },
                {
                    "id": 2,
                    "name": "INV-002",
                    "create_uid": 4,
                    "amount": 2000.0,
                },
            ],
            "calculations": {
                "total_records": 2,
                "total_amount": 3000.0,
                "write_date": "2026-01-01",
            },
            "records_count": 2,
        }
        sanitized = sanitize_erp_payload(raw_payload)
        self.assertEqual(len(sanitized["invoices"]), 2)
        self.assertNotIn("create_uid", sanitized["invoices"][0])
        self.assertEqual(sanitized["invoices"][0]["amount"], 1000.0)
        self.assertNotIn("write_date", sanitized["calculations"])
        self.assertEqual(sanitized["calculations"]["total_amount"], 3000.0)


class TestSummarizerNode(unittest.TestCase):
    """Test suite for summarizer node threshold checking and summary generation."""

    def test_passthrough_when_under_threshold(self):
        # 4 messages, default threshold is 10
        state = create_initial_state(
            user_query="Cek saldo kas",
            messages=[
                HumanMessage(content="Halo"),
                AIMessage(content="Halo"),
                HumanMessage(content="Cek kas"),
            ],
        )
        res = summarizer_node(state)
        # Should be an empty dict (0 ms pass-through)
        self.assertEqual(res, {})

    def test_triggers_deterministic_summary_when_over_threshold(self):
        # 12 messages with SKU and date entities
        messages = [
            HumanMessage(content=f"Cek barang ART-PAPER-210 turn {i}")
            for i in range(12)
        ]
        messages.append(HumanMessage(content="Bagaimana transaksi periode 2026-03-01?"))
        state = create_initial_state(user_query="Cek lagi", messages=messages)

        res = summarizer_node(state)
        self.assertIn("conversation_summary", res)
        summary = res["conversation_summary"]
        self.assertIn("Rangkuman Konteks Percakapan", summary)
        self.assertIn("ART-PAPER-210", summary)
        self.assertIn("2026-03-01", summary)

    def test_respects_enable_auto_summarization_setting(self):
        orig_setting = settings.ENABLE_AUTO_SUMMARIZATION
        try:
            settings.ENABLE_AUTO_SUMMARIZATION = False
            messages = [HumanMessage(content=f"Turn {i}") for i in range(15)]
            state = create_initial_state(user_query="Halo", messages=messages)
            res = summarizer_node(state)
            self.assertEqual(res, {})
        finally:
            settings.ENABLE_AUTO_SUMMARIZATION = orig_setting


class TestContextualPrompt(unittest.TestCase):
    """Test suite for prompt augmentation."""

    def test_build_contextual_prompt(self):
        base_prompt = "You are an ERP assistant."
        summary = "- Active Vendor: PT Maju Jaya\n- Active SKU: ART-150"

        augmented = build_contextual_prompt(base_prompt, summary)
        self.assertIn(base_prompt, augmented)
        self.assertIn("PT Maju Jaya", augmented)
        self.assertIn("ART-150", augmented)

        # Empty summary should return base prompt as is
        self.assertEqual(build_contextual_prompt(base_prompt, None), base_prompt)
        self.assertEqual(build_contextual_prompt(base_prompt, "   "), base_prompt)


class TestWorkflowContextIntegration(unittest.TestCase):
    """End-to-end integration test of workflow graph with summarizer and sanitization."""

    def test_workflow_execution_with_summarizer(self):
        # Test greeting workflow through the graph (START -> summarizer -> router -> synthesizer -> END)
        state = run_workflow_sync(user_query="Halo FastPrint!")
        self.assertIsNotNone(state.get("final_answer"))
        self.assertIn("Halo", state.get("final_answer"))
        self.assertEqual(state.get("detected_domain"), "general")


if __name__ == "__main__":
    unittest.main()
