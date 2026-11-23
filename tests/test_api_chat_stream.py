"""
Unit and integration test suite for Real-Time SSE Token-by-Token Streaming & API Gateway (Komponen 1).
Validates lifecycle events (step, tool_start, tool_end, token, complete, error), parameter sanitization,
typing effect chunking, and reverse-proxy keepalive heartbeat.
"""
import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from langgraph.checkpoint.memory import MemorySaver

from app.core.config import settings
from app.core.context import (
    chunk_text_for_streaming,
    resolve_tool_record_count,
    sanitize_tool_params,
)
from app.graph.workflow import create_workflow_graph
from app.main import app
from app.services.django_client import DjangoCoreClient


class TestApiChatStream(unittest.IsolatedAsyncioTestCase):
    """Test suite for POST /api/v1/chat/stream SSE endpoint and streaming event generators."""

    def _parse_sse_events(self, sse_text: str):
        """Parse raw SSE wire format into list of (event_name, data_or_comment)."""
        events = []
        blocks = [b for b in sse_text.strip().split("\n\n") if b.strip()]
        for block in blocks:
            lines = block.split("\n")
            ev_name = None
            ev_data = None
            is_comment = False
            for line in lines:
                line_stripped = line.strip()
                if line_stripped.startswith(":"):
                    is_comment = True
                    ev_name = "comment"
                    ev_data = line_stripped[1:].strip()
                elif line_stripped.startswith("event:"):
                    ev_name = line_stripped[len("event:"):].strip()
                elif line_stripped.startswith("data:"):
                    raw_data = line_stripped[len("data:"):].strip()
                    try:
                        ev_data = json.loads(raw_data)
                    except Exception:
                        ev_data = raw_data
            if ev_name is not None:
                events.append((ev_name, ev_data))
        return events

    async def test_stream_endpoint_named_events_lifecycle(self):
        """Verify full lifecycle: step -> tool_start -> tool_end -> token -> complete."""
        isolated_graph = create_workflow_graph(checkpointer=MemorySaver())
        mock_client = MagicMock(spec=DjangoCoreClient)
        mock_client.aclose = AsyncMock()
        mock_client.fetch_account_balance = AsyncMock(
            return_value={
                "accounts": [
                    {"code": "1110", "name": "Kas Bandung", "balance": 2500000.0, "company_id": 6}
                ]
            }
        )

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
            with patch("app.graph.workflow.default_workflow_graph", isolated_graph), \
                 patch("app.graph.nodes.tool_node.DjangoCoreClient", return_value=mock_client):
                resp = await http_client.post(
                    "/api/v1/chat/stream",
                    json={"query": "Cek saldo kas cabang Bandung", "thread_id": "test_stream_lifecycle"},
                )

        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.headers.get("content-type", "").startswith("text/event-stream"))
        self.assertEqual(resp.headers.get("cache-control"), "no-cache")
        self.assertEqual(resp.headers.get("x-accel-buffering"), "no")

        events = self._parse_sse_events(resp.text)
        event_types = [name for name, _ in events]

        self.assertIn("step", event_types)
        self.assertIn("tool_start", event_types)
        self.assertIn("tool_end", event_types)
        self.assertIn("token", event_types)
        self.assertIn("complete", event_types)

        # 1. Verify tool_start payload
        tool_start_events = [data for name, data in events if name == "tool_start"]
        self.assertGreaterEqual(len(tool_start_events), 1)
        self.assertEqual(tool_start_events[0]["tool"], "get_account_balance")
        self.assertIn("company_id", tool_start_events[0]["input"])
        self.assertEqual(tool_start_events[0]["input"]["company_id"], 6)

        # 2. Verify tool_end payload
        tool_end_events = [data for name, data in events if name == "tool_end"]
        self.assertGreaterEqual(len(tool_end_events), 1)
        self.assertEqual(tool_end_events[0]["tool"], "get_account_balance")
        self.assertEqual(tool_end_events[0]["status"], "success")
        self.assertGreaterEqual(tool_end_events[0]["records_found"], 1)
        self.assertIsInstance(tool_end_events[0]["elapsed_ms"], (int, float))

        # 3. Verify token chunks concatenation
        token_events = [data for name, data in events if name == "token"]
        self.assertGreaterEqual(len(token_events), 1)
        concatenated_text = "".join(item["content"] for item in token_events)

        # 4. Verify complete payload matches concatenated text
        complete_events = [data for name, data in events if name == "complete"]
        self.assertEqual(len(complete_events), 1)
        complete_payload = complete_events[0]
        self.assertEqual(complete_payload["detected_domain"], "accounting")
        self.assertEqual(complete_payload["verification_status"], "VALID")
        self.assertEqual(complete_payload["resolved_company_id"], 6)
        self.assertEqual(complete_payload["final_answer"], concatenated_text)

    async def test_stream_endpoint_tool_start_parameter_sanitization(self):
        """Sensitive parameters (token, secret, passwords) must be strictly pruned from tool_start."""
        dirty_params = {
            "code": "1110",
            "company_id": 1,
            "user_token": "secret_user_jwt_token_12345",
            "password": "supersecretpassword",
            "service_key": "internal-secret-service-key",
            "api_key": "openai-secret-key",
            "_internal_flag": True,
            "date_from": "2026-01-01",
            "empty_val": None,
            "blank_val": "   ",
        }
        sanitized = sanitize_tool_params(dirty_params)

        self.assertEqual(sanitized["code"], "1110")
        self.assertEqual(sanitized["company_id"], 1)
        self.assertEqual(sanitized["date_from"], "2026-01-01")
        self.assertNotIn("user_token", sanitized)
        self.assertNotIn("password", sanitized)
        self.assertNotIn("service_key", sanitized)
        self.assertNotIn("api_key", sanitized)
        self.assertNotIn("_internal_flag", sanitized)
        self.assertNotIn("empty_val", sanitized)
        self.assertNotIn("blank_val", sanitized)

    async def test_stream_endpoint_token_chunking_typing_effect(self):
        """chunk_text_for_streaming decomposes markdown without corrupting whitespace or text integrity."""
        sample_markdown = (
            "# Laporan Keuangan PT FastPrint\n\n"
            "Berikut adalah ringkasan saldo kas aktif:\n"
            "| No | Kode | Nama Akun | Saldo |\n"
            "|---|---|---|---|\n"
            "| 1 | 1110 | Kas Operasional | Rp 15.000.000,00 |\n"
            "| 2 | 1120 | Rekening Bank BCA | Rp 85.000.000,00 |\n\n"
            "*Semua angka telah divalidasi.*"
        )
        chunks = chunk_text_for_streaming(sample_markdown, max_chunk_len=40)
        self.assertGreater(len(chunks), 3)
        reconstructed = "".join(chunks)
        self.assertEqual(reconstructed, sample_markdown)

    async def test_stream_endpoint_heartbeat_ping(self):
        """When tool execution pauses longer than heartbeat interval, ': ping' comments are yielded."""
        async def slow_stream(**kwargs):
            yield ("step", {"node": "router", "status": "completed"})
            # Pause long enough to trigger a ping
            await asyncio.sleep(0.06)
            yield ("step", {"node": "synthesizer", "status": "completed"})
            yield ("complete", {"final_answer": "Selesai"})

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
            with patch("app.main.astream_workflow_events", side_effect=slow_stream), \
                 patch.object(settings, "SSE_HEARTBEAT_SECONDS", 0.02):
                resp = await http_client.post(
                    "/api/v1/chat/stream",
                    json={"query": "Halo", "user_token": "valid_jwt"},
                )

        self.assertEqual(resp.status_code, 200)
        self.assertIn(": ping", resp.text)
        events = self._parse_sse_events(resp.text)
        comment_events = [data for name, data in events if name == "comment"]
        self.assertGreaterEqual(len(comment_events), 1)
        self.assertIn("ping", comment_events)

    async def test_stream_endpoint_multi_tool_events(self):
        """Multi-tool queries emit dedicated tool_start and tool_end events for each planned tool."""
        isolated_graph = create_workflow_graph(checkpointer=MemorySaver())
        mock_client = MagicMock(spec=DjangoCoreClient)
        mock_client.aclose = AsyncMock()
        mock_client.fetch_account_balance = AsyncMock(
            return_value={"accounts": [{"code": "1110", "name": "Kas", "balance": 100000.0, "company_id": 1}]}
        )
        mock_client.fetch_stock_level = AsyncMock(
            return_value=[{"code": "AP-150", "name": "Art Paper 150g", "qty_on_hand": 20.0, "company_id": 1}]
        )

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
            with patch("app.graph.workflow.default_workflow_graph", isolated_graph), \
                 patch("app.graph.nodes.tool_node.DjangoCoreClient", return_value=mock_client):
                resp = await http_client.post(
                    "/api/v1/chat/stream",
                    json={"query": "Berapa saldo kas dan cek stok Art Paper 150g?", "thread_id": "test_multi_stream"},
                )

        self.assertEqual(resp.status_code, 200)
        events = self._parse_sse_events(resp.text)
        tool_starts = [data for name, data in events if name == "tool_start"]
        tool_ends = [data for name, data in events if name == "tool_end"]

        self.assertGreaterEqual(len(tool_starts), 2)
        self.assertGreaterEqual(len(tool_ends), 2)
        started_tools = [t["tool"] for t in tool_starts]
        ended_tools = [t["tool"] for t in tool_ends]
        self.assertIn("get_account_balance", started_tools)
        self.assertIn("get_stock_level", started_tools)
        self.assertIn("get_account_balance", ended_tools)
        self.assertIn("get_stock_level", ended_tools)

    async def test_stream_endpoint_general_query_bypasses_tool_events(self):
        """General non-tool inquiries emit steps and tokens without tool_start or tool_end."""
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
            resp = await http_client.post(
                "/api/v1/chat/stream",
                json={"query": "Halo selamat pagi", "thread_id": "general_stream_1"},
            )

        self.assertEqual(resp.status_code, 200)
        events = self._parse_sse_events(resp.text)
        event_types = [name for name, _ in events]

        self.assertIn("step", event_types)
        self.assertIn("token", event_types)
        self.assertIn("complete", event_types)
        self.assertNotIn("tool_start", event_types)
        self.assertNotIn("tool_end", event_types)

    async def test_resolve_tool_record_count_helper(self):
        """resolve_tool_record_count handles lists, paginated dicts, and empty responses accurately."""
        self.assertEqual(resolve_tool_record_count(None), 0)
        self.assertEqual(resolve_tool_record_count([]), 0)
        self.assertEqual(resolve_tool_record_count([{"id": 1}, {"id": 2}]), 2)
        self.assertEqual(resolve_tool_record_count({"total_count": 42}), 42)
        self.assertEqual(resolve_tool_record_count({"total_found": 88}), 88)
        self.assertEqual(resolve_tool_record_count({"entries": [1, 2, 3]}), 3)
        self.assertEqual(resolve_tool_record_count({"accounts": [{"code": "1"}]}), 1)

    async def test_synthesizer_graceful_fallback_on_stream_failure(self):
        """When LLM raises mid-stream failure, asynthesizer_node falls back to deterministic answer cleanly."""
        from app.graph.nodes.synthesizer import asynthesizer_node
        from app.graph.state import create_initial_state

        state = create_initial_state(user_query="Cek saldo kas operasional", company_id=1)
        state["verification_status"] = "VALID"
        state["detected_domain"] = "accounting"
        state["detected_tool"] = "get_account_balance"
        state["tool_result"] = {
            "accounts": [{
                "account_code": "1110",
                "account_name": "Kas Operasional",
                "ending_balance": 5000000.0,
                "period_debit": 0.0,
                "period_credit": 0.0,
                "company_id": 1,
            }]
        }

        failing_mock_llm = MagicMock()
        failing_mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("API 429 RateLimit Exceeded"))

        result = await asynthesizer_node(state, llm=failing_mock_llm)
        self.assertIn("Kas Operasional", result["final_answer"])
        self.assertIn("Rp 5.000.000", result["final_answer"])
        self.assertIsNotNone(result.get("deterministic_answer"))
        self.assertTrue(len(result["suggested_followups"]) > 0)
        # Verify no unhandled exception occurred
        self.assertTrue(result["final_answer"].startswith("### Informasi Saldo Chart of Account"))


if __name__ == "__main__":
    unittest.main()
