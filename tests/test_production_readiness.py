"""
Verification tests for production readiness improvements (D1, D2, D3, D4, D5, D6, Lifespan & Pool Cleanup).
"""
import asyncio
import json
import threading
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
import httpx
from langgraph.checkpoint.memory import MemorySaver

from app.core.config import settings
from app.core.context import sanitize_erp_payload
from app.graph.nodes.extractor import _fallback_extract_stock_level
from app.graph.nodes.synthesizer import _format_single_tool_presentation
from app.graph.workflow import (
    arun_workflow,
    astream_workflow_steps,
    create_workflow_graph,
)
import app.main as main_module
from app.main import app
import app.services.django_client as django_client_module
from app.services.django_client import (
    DjangoAuthError,
    DjangoCoreClient,
    aclose_shared_django_client,
)


class TestD1SessionStateAndBoundedCheckpointer(unittest.IsolatedAsyncioTestCase):
    """Verify D1: ephemeral stateless thread isolation, history hydration, and LRU eviction."""

    def test_max_checkpoint_threads_default_config(self):
        """settings.MAX_CHECKPOINT_THREADS defaults to 500."""
        self.assertEqual(settings.MAX_CHECKPOINT_THREADS, 500)

    async def test_stateless_requests_do_not_leak_company_id_or_messages(self):
        """
        Requests without thread_id use unique ephemeral thread IDs, clean up all RAM state
        afterwards, do not leak company_id/messages across requests, and always hydrate history.
        """
        checkpointer = MemorySaver()
        graph = create_workflow_graph(checkpointer=checkpointer)

        # Turn 1 (stateless, company_id=6)
        res_1 = await arun_workflow(
            user_query="Halo asisten",
            company_id=6,
            thread_id=None,
            graph=graph,
        )
        self.assertEqual(res_1.get("company_id"), 6)
        self.assertEqual(len(checkpointer.storage), 0)
        self.assertEqual(len(checkpointer.writes), 0)
        self.assertEqual(len(checkpointer.blobs), 0)

        # Turn 2 (stateless, no company_id, with history -> must hydrate history and default company_id=1)
        mock_client = MagicMock(spec=DjangoCoreClient)
        mock_client.aclose = AsyncMock()
        mock_client.fetch_locations = AsyncMock(return_value={"locations": []})
        mock_client.fetch_product_info = AsyncMock(
            return_value={
                "products": [
                    {
                        "id": 101,
                        "code": "SKU-EPHEMERAL-01",
                        "name": "Kertas Ephemeral",
                        "qty_available": 15.0,
                    }
                ]
            }
        )
        mock_client.fetch_stock_level = AsyncMock(
            return_value=[
                {
                    "code": "SKU-EPHEMERAL-01",
                    "name": "Kertas Ephemeral",
                    "company_id": 1,
                    "company_name": "FastPrint",
                    "location_name": "Gudang Utama",
                    "qty_on_hand": 15.0,
                }
            ]
        )

        history_payload = [
            {"role": "user", "content": "cek stok produk SKU-EPHEMERAL-01"},
            {"role": "assistant", "content": "Stok SKU-EPHEMERAL-01 belum ditemukan."},
        ]

        with patch("app.graph.nodes.tool_node.DjangoCoreClient", return_value=mock_client):
            res_2 = await arun_workflow(
                user_query="coba cek lagi",
                company_id=None,
                thread_id=None,
                history=history_payload,
                graph=graph,
            )

        self.assertEqual(res_2.get("company_id"), settings.DEFAULT_COMPANY_ID)
        contents = [getattr(m, "content", "") for m in res_2.get("messages", [])]
        self.assertTrue(any("SKU-EPHEMERAL-01" in c for c in contents))
        self.assertFalse(any("Halo asisten" in c for c in contents))
        self.assertEqual(len(checkpointer.storage), 0)
        self.assertEqual(len(checkpointer.writes), 0)
        self.assertEqual(len(checkpointer.blobs), 0)

    async def test_stateless_stream_cleans_up_ephemeral_thread_on_completion_and_early_close(self):
        """
        astream_workflow_steps with thread_id=None leaves 0 threads in MemorySaver,
        both on full stream completion and when closed early.
        """
        checkpointer = MemorySaver()
        graph = create_workflow_graph(checkpointer=checkpointer)

        # 1. Full completion
        steps = []
        async for step in astream_workflow_steps(
            user_query="Halo asisten",
            company_id=4,
            thread_id="",
            graph=graph,
        ):
            steps.append(step[0])
        self.assertIn("synthesizer", steps)
        self.assertEqual(len(checkpointer.storage), 0)

        # 2. Early generator close
        stream = astream_workflow_steps(
            user_query="Halo asisten",
            company_id=5,
            thread_id=None,
            graph=graph,
        )
        first_step = await stream.__anext__()
        self.assertIsNotNone(first_step)
        await stream.aclose()
        self.assertEqual(len(checkpointer.storage), 0)
        self.assertEqual(len(checkpointer.writes), 0)
        self.assertEqual(len(checkpointer.blobs), 0)

    async def test_bounded_checkpointer_lru_eviction(self):
        """
        MemorySaver never exceeds settings.MAX_CHECKPOINT_THREADS and evicts
        the least-recently-used thread across both arun_workflow and astream_workflow_steps.
        """
        checkpointer = MemorySaver()
        graph = create_workflow_graph(checkpointer=checkpointer)

        with patch.object(settings, "MAX_CHECKPOINT_THREADS", 3):
            await arun_workflow(user_query="Halo", thread_id="thread_a", graph=graph)
            await arun_workflow(user_query="Halo", thread_id="thread_b", graph=graph)
            await arun_workflow(user_query="Halo", thread_id="thread_c", graph=graph)
            self.assertEqual(set(checkpointer.storage.keys()), {"thread_a", "thread_b", "thread_c"})

            # Re-access thread_a so thread_b becomes the least-recently-used
            await arun_workflow(user_query="Apa kabar?", thread_id="thread_a", graph=graph)

            # Adding thread_d must evict thread_b
            await arun_workflow(user_query="Halo", thread_id="thread_d", graph=graph)
            self.assertEqual(len(checkpointer.storage), 3)
            self.assertEqual(set(checkpointer.storage.keys()), {"thread_a", "thread_c", "thread_d"})
            self.assertFalse(any(k[0] == "thread_b" for k in checkpointer.writes))
            self.assertFalse(any(k[0] == "thread_b" for k in checkpointer.blobs))

            # Streaming thread_e must evict thread_c (now oldest)
            async for _ in astream_workflow_steps(user_query="Halo", thread_id="thread_e", graph=graph):
                pass
            self.assertEqual(len(checkpointer.storage), 3)
            self.assertEqual(set(checkpointer.storage.keys()), {"thread_a", "thread_d", "thread_e"})
            self.assertFalse(any(k[0] == "thread_c" for k in checkpointer.writes))
            self.assertFalse(any(k[0] == "thread_c" for k in checkpointer.blobs))


class TestD2ApiAuthenticationPolicy(unittest.TestCase):
    """Verify D2: strict user JWT enforcement, Bearer header extraction, and service key guard."""

    def setUp(self):
        self.client = TestClient(app)

    def test_auth_policy_config_defaults(self):
        """REQUIRE_USER_TOKEN and REQUIRE_SERVICE_KEY default to False."""
        self.assertFalse(settings.REQUIRE_USER_TOKEN)
        self.assertFalse(settings.REQUIRE_SERVICE_KEY)

    @patch("app.main.arun_workflow", new_callable=AsyncMock)
    def test_production_strict_user_token_and_bearer_header(self, mock_workflow):
        """
        In production (or REQUIRE_USER_TOKEN=True), missing/blank token returns HTTP 401
        and Authorization: Bearer <jwt> or body user_token succeeds.
        """
        mock_workflow.return_value = {
            "final_answer": "OK",
            "detected_domain": "general",
            "verification_status": "VALID",
            "error_message": None,
        }

        with patch.object(settings, "APP_ENV", "production"):
            # Missing both body token and Bearer header -> 401
            resp_no_auth = self.client.post("/api/v1/chat", json={"query": "Cek saldo kas"})
            self.assertEqual(resp_no_auth.status_code, 401)
            mock_workflow.assert_not_awaited()

            # Blank user_token + non-Bearer Authorization -> 401
            resp_bad_scheme = self.client.post(
                "/api/v1/chat",
                json={"query": "Cek saldo kas", "user_token": "   "},
                headers={"Authorization": "Basic abc123"},
            )
            self.assertEqual(resp_bad_scheme.status_code, 401)
            mock_workflow.assert_not_awaited()

            # Valid Authorization: Bearer <jwt> -> 200 and forwards header token
            resp_bearer = self.client.post(
                "/api/v1/chat",
                json={"query": "Cek saldo kas"},
                headers={"Authorization": "Bearer jwt_from_bearer_header"},
            )
            self.assertEqual(resp_bearer.status_code, 200)
            self.assertEqual(mock_workflow.await_args.kwargs.get("user_token"), "jwt_from_bearer_header")

            # Body user_token takes precedence over Bearer header when both present
            resp_body = self.client.post(
                "/api/v1/chat",
                json={"query": "Cek saldo kas", "user_token": "jwt_from_body"},
                headers={"Authorization": "Bearer jwt_from_bearer_header"},
            )
            self.assertEqual(resp_body.status_code, 200)
            self.assertEqual(mock_workflow.await_args.kwargs.get("user_token"), "jwt_from_body")

        # REQUIRE_USER_TOKEN=True also enforces 401 even in development mode
        mock_workflow.reset_mock()
        with patch.object(settings, "APP_ENV", "development"), \
             patch.object(settings, "REQUIRE_USER_TOKEN", True):
            resp_flag = self.client.post("/api/v1/chat", json={"query": "Cek saldo kas"})
            self.assertEqual(resp_flag.status_code, 401)
            mock_workflow.assert_not_awaited()

    @patch("app.main.arun_workflow", new_callable=AsyncMock)
    def test_production_masks_internal_500_details(self, mock_workflow):
        """Production mode does not leak internal exception details on HTTP 500."""
        mock_workflow.side_effect = RuntimeError("postgresql://secret-internal-host/db failed")
        with patch.object(settings, "APP_ENV", "production"):
            resp = self.client.post(
                "/api/v1/chat",
                json={"query": "Cek saldo kas", "user_token": "valid_jwt"},
            )
        self.assertEqual(resp.status_code, 500)
        detail = resp.json().get("detail", "")
        self.assertEqual(detail, "Chat workflow failed due to an internal server error.")
        self.assertNotIn("secret-internal-host", detail)

    @patch("app.main.arun_workflow", new_callable=AsyncMock)
    def test_service_key_enforcement_when_enabled(self, mock_workflow):
        """When REQUIRE_SERVICE_KEY=True, X-Internal-Service-Key must match ERP_API_TOKEN."""
        mock_workflow.return_value = {
            "final_answer": "OK",
            "detected_domain": "general",
            "verification_status": "VALID",
            "error_message": None,
        }

        with patch.object(settings, "REQUIRE_SERVICE_KEY", True), \
             patch.object(settings, "ERP_API_TOKEN", "internal-secret-123"):
            resp_missing = self.client.post("/api/v1/chat", json={"query": "Halo"})
            self.assertEqual(resp_missing.status_code, 403)

            resp_wrong = self.client.post(
                "/api/v1/chat",
                json={"query": "Halo"},
                headers={"X-Internal-Service-Key": "wrong-key"},
            )
            self.assertEqual(resp_wrong.status_code, 403)

            resp_ok = self.client.post(
                "/api/v1/chat",
                json={"query": "Halo"},
                headers={"X-Internal-Service-Key": "internal-secret-123"},
            )
            self.assertEqual(resp_ok.status_code, 200)


class TestD2DjangoClientUserTokenGuard(unittest.IsolatedAsyncioTestCase):
    """Verify D2: DjangoCoreClient(require_user_token=True) blocks fallback to admin/system auth."""

    async def test_require_user_token_rejects_missing_token_without_fallback(self):
        """Missing or whitespace user_token raises DjangoAuthError(401) and never calls login()."""
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        client = DjangoCoreClient(
            base_url="http://erp.local",
            token="system_admin_token",
            username="admin",
            password="secret",
            client=mock_http,
            require_user_token=True,
        )
        client.login = AsyncMock()

        with self.assertRaises(DjangoAuthError) as ctx:
            await client.fetch_account_balance(params={}, user_token=None, company_id=1)
        self.assertEqual(ctx.exception.status_code, 401)
        client.login.assert_not_awaited()
        mock_http.request.assert_not_called()

        with self.assertRaises(DjangoAuthError):
            await client.fetch_account_balance(params={}, user_token="   ", company_id=1)
        client.login.assert_not_awaited()
        mock_http.request.assert_not_called()

    async def test_require_user_token_forwards_valid_jwt_and_never_falls_back_on_401(self):
        """Valid user_token is forwarded; 401 from Django does NOT trigger system login()."""
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 401
        mock_resp.text = "Token expired"
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.request = AsyncMock(return_value=mock_resp)

        client = DjangoCoreClient(
            base_url="http://erp.local",
            token="system_admin_token",
            username="admin",
            password="secret",
            client=mock_http,
            require_user_token=True,
        )
        client.login = AsyncMock()

        with self.assertRaises(DjangoAuthError):
            await client.fetch_account_balance(params={}, user_token="user_jwt_999", company_id=1)

        client.login.assert_not_awaited()
        called_headers = mock_http.request.call_args.kwargs["headers"]
        self.assertEqual(called_headers["Authorization"], "Bearer user_jwt_999")


class TestD3CompanyIdPriority(unittest.IsolatedAsyncioTestCase):
    """Verify D3: strict 4-tier company_id priority across arun_workflow, astream_workflow_steps, and FastAPI."""

    async def test_explicit_company_id_1_overrides_sticky_branch_checkpoint(self):
        """
        End-to-end API multi-turn priority check:
          Turn 1: 'cabang Bandung' -> company_id=6
          Turn 2: omitted company_id -> sticky company_id=6
          Turn 3: explicit company_id=1 -> overrides sticky back to company_id=1
          Turn 4: omitted company_id -> sticky company_id=1
          Turn 5: 'cabang Surabaya' + explicit company_id=1 -> text branch wins (company_id=7)
        """
        isolated_graph = create_workflow_graph(checkpointer=MemorySaver())
        observed_company_ids = []

        async def mock_fetch_balance(params=None, user_token=None, company_id=None):
            observed_company_ids.append(company_id)
            return {
                "accounts": [
                    {
                        "code": "1110",
                        "name": "Kas Besar",
                        "balance": 1000000.0,
                        "company_id": company_id,
                    }
                ]
            }

        mock_client = MagicMock(spec=DjangoCoreClient)
        mock_client.aclose = AsyncMock()
        mock_client.fetch_account_balance = AsyncMock(side_effect=mock_fetch_balance)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
            with patch("app.graph.workflow.default_workflow_graph", isolated_graph), \
                 patch("app.graph.nodes.tool_node.DjangoCoreClient", return_value=mock_client):
                r1 = await http_client.post(
                    "/api/v1/chat",
                    json={"query": "Cek saldo kas cabang Bandung", "thread_id": "d3_api_t1"},
                )
                r2 = await http_client.post(
                    "/api/v1/chat",
                    json={"query": "cek saldo kas", "thread_id": "d3_api_t1"},
                )
                r3 = await http_client.post(
                    "/api/v1/chat",
                    json={"query": "cek saldo kas", "company_id": 1, "thread_id": "d3_api_t1"},
                )
                r4 = await http_client.post(
                    "/api/v1/chat",
                    json={"query": "cek saldo kas", "thread_id": "d3_api_t1"},
                )
                r5 = await http_client.post(
                    "/api/v1/chat",
                    json={"query": "cek saldo kas cabang Surabaya", "company_id": 1, "thread_id": "d3_api_t1"},
                )

        for resp in (r1, r2, r3, r4, r5):
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json()["verification_status"], "VALID")

        self.assertEqual(observed_company_ids, [6, 6, 1, 1, 7])

    async def test_astream_workflow_steps_matches_arun_workflow_company_id_priority(self):
        """
        astream_workflow_steps applies identical 4-tier priority and supports
        Python compatibility where non-default company_id without flag is treated as explicit.
        """
        graph = create_workflow_graph(checkpointer=MemorySaver())

        # Turn 1: direct Python call with company_id=6 (no company_id_explicit arg) -> treated as explicit (6)
        acc_1 = {}
        async for _, _, acc in astream_workflow_steps(
            user_query="Halo",
            company_id=6,
            thread_id="d3_stream_t1",
            graph=graph,
        ):
            acc_1 = acc
        self.assertEqual(acc_1.get("company_id"), 6)

        # Turn 2: company_id=1 with company_id_explicit=False -> sticky checkpoint (6)
        acc_2 = {}
        async for _, _, acc in astream_workflow_steps(
            user_query="Halo lagi",
            company_id=1,
            thread_id="d3_stream_t1",
            graph=graph,
            company_id_explicit=False,
        ):
            acc_2 = acc
        self.assertEqual(acc_2.get("company_id"), 6)

        # Turn 3: company_id=1 with company_id_explicit=True -> overrides sticky back to 1
        acc_3 = {}
        async for _, _, acc in astream_workflow_steps(
            user_query="Kembali ke pusat",
            company_id=1,
            thread_id="d3_stream_t1",
            graph=graph,
            company_id_explicit=True,
        ):
            acc_3 = acc
        self.assertEqual(acc_3.get("company_id"), 1)

        # Turn 4: branch text ('Jakarta' -> 8) overrides explicit company_id=1
        acc_4 = {}
        async for _, _, acc in astream_workflow_steps(
            user_query="Info cabang Jakarta",
            company_id=1,
            thread_id="d3_stream_t1",
            graph=graph,
            company_id_explicit=True,
        ):
            acc_4 = acc
        self.assertEqual(acc_4.get("company_id"), 8)


class TestD4ApiContractAndSseStreaming(unittest.IsolatedAsyncioTestCase):
    """Verify D4: ChatResponse fields (resolved_company_id, execution_time_ms) & POST /api/v1/chat/stream SSE."""

    @staticmethod
    def _parse_sse_events(raw_body: str):
        events = []
        for block in raw_body.strip().split("\n\n"):
            if not block.strip():
                continue
            ev_name = None
            ev_data = None
            for line in block.splitlines():
                if line.startswith("event:"):
                    ev_name = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    ev_data = json.loads(line[len("data:"):].strip())
            events.append((ev_name, ev_data))
        return events

    async def test_chat_endpoint_returns_resolved_company_id_and_execution_time_ms(self):
        """POST /api/v1/chat populates resolved_company_id and execution_time_ms."""
        isolated_graph = create_workflow_graph(checkpointer=MemorySaver())
        mock_client = MagicMock(spec=DjangoCoreClient)
        mock_client.aclose = AsyncMock()
        mock_client.fetch_account_balance = AsyncMock(
            return_value={
                "accounts": [{"code": "1110", "name": "Kas Bandung", "balance": 5000000.0, "company_id": 6}]
            }
        )

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
            with patch("app.graph.workflow.default_workflow_graph", isolated_graph), \
                 patch("app.graph.nodes.tool_node.DjangoCoreClient", return_value=mock_client):
                resp = await http_client.post(
                    "/api/v1/chat",
                    json={"query": "Cek saldo kas cabang Bandung"},
                )

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["resolved_company_id"], 6)
        self.assertIsInstance(data["execution_time_ms"], float)
        self.assertGreaterEqual(data["execution_time_ms"], 0.0)

    async def test_chat_stream_sse_endpoint(self):
        """
        POST /api/v1/chat/stream emits event: step frames per node and final event: complete
        containing ChatResponse JSON with resolved_company_id and execution_time_ms.
        """
        isolated_graph = create_workflow_graph(checkpointer=MemorySaver())
        mock_client = MagicMock(spec=DjangoCoreClient)
        mock_client.aclose = AsyncMock()
        mock_client.fetch_account_balance = AsyncMock(
            return_value={
                "accounts": [{"code": "1110", "name": "Kas Bandung", "balance": 2500000.0, "company_id": 6}]
            }
        )

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
            with patch("app.graph.workflow.default_workflow_graph", isolated_graph), \
                 patch("app.graph.nodes.tool_node.DjangoCoreClient", return_value=mock_client):
                resp = await http_client.post(
                    "/api/v1/chat/stream",
                    json={"query": "Cek saldo kas cabang Bandung", "thread_id": "d4_sse_t1"},
                )

        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.headers.get("content-type", "").startswith("text/event-stream"))
        self.assertEqual(resp.headers.get("cache-control"), "no-cache")
        self.assertEqual(resp.headers.get("x-accel-buffering"), "no")

        events = self._parse_sse_events(resp.text)
        step_events = [data for name, data in events if name == "step"]
        complete_events = [data for name, data in events if name == "complete"]

        self.assertGreaterEqual(len(step_events), 4)
        node_names = [item["node"] for item in step_events]
        self.assertIn("router", node_names)
        self.assertIn("tool_node", node_names)
        self.assertIn("synthesizer", node_names)
        for item in step_events:
            self.assertEqual(item["status"], "completed")
            self.assertIsInstance(item["elapsed_ms"], (int, float))
            self.assertGreaterEqual(item["elapsed_ms"], 0.0)

        self.assertEqual(len(complete_events), 1)
        final_payload = complete_events[0]
        self.assertEqual(final_payload["detected_domain"], "accounting")
        self.assertEqual(final_payload["verification_status"], "VALID")
        self.assertEqual(final_payload["resolved_company_id"], 6)
        self.assertIsInstance(final_payload["execution_time_ms"], (int, float))
        self.assertGreaterEqual(final_payload["execution_time_ms"], 0.0)
        self.assertTrue(final_payload["final_answer"])

    async def test_chat_stream_enforces_production_auth_401_and_bearer_header(self):
        """POST /api/v1/chat/stream rejects missing token in production (401) and accepts Bearer header."""
        isolated_graph = create_workflow_graph(checkpointer=MemorySaver())
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
            with patch.object(settings, "APP_ENV", "production"), \
                 patch("app.graph.workflow.default_workflow_graph", isolated_graph):
                resp_unauth = await http_client.post(
                    "/api/v1/chat/stream",
                    json={"query": "Halo"},
                )
                self.assertEqual(resp_unauth.status_code, 401)

                resp_auth = await http_client.post(
                    "/api/v1/chat/stream",
                    json={"query": "Halo"},
                    headers={"Authorization": "Bearer valid_sse_jwt"},
                )
                self.assertEqual(resp_auth.status_code, 200)
                events = self._parse_sse_events(resp_auth.text)
                self.assertEqual(events[-1][0], "complete")

    async def test_chat_stream_emits_error_event_on_stream_exception(self):
        """Mid-stream exceptions emit event: error and mask details in production."""
        async def failing_stream(**kwargs):
            yield ("router", {"detected_domain": "general"}, {"company_id": 1})
            raise RuntimeError("internal-redis-secret-failure")

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
            with patch("app.main.astream_workflow_steps", side_effect=failing_stream), \
                 patch.object(settings, "APP_ENV", "production"):
                resp = await http_client.post(
                    "/api/v1/chat/stream",
                    json={"query": "Halo", "user_token": "valid_jwt"},
                )

        self.assertEqual(resp.status_code, 200)
        events = self._parse_sse_events(resp.text)
        self.assertEqual(events[0][0], "step")
        self.assertEqual(events[-1][0], "error")
        self.assertEqual(events[-1][1]["detail"], "Chat workflow failed due to an internal server error.")


class TestD5PreTruncationCalculationsAndPagination(unittest.IsolatedAsyncioTestCase):
    """Verify D5: pre-truncation aggregates (>50 rows) and Markdown pagination footers."""

    async def test_vendor_bills_totals_accurate_when_truncated_over_50(self):
        """75 vendor bills produce full 75-row AP totals (not 50-row sum) and a pagination footer."""
        bills_75 = [
            {
                "id": i,
                "number": f"BILL/2026/{i:03d}",
                "bill_date": "2026-04-01",
                "due_date": "2026-04-15",
                "vendor_name": f"Vendor {i}",
                "total_amount": 100000.0,
                "paid_amount": 25000.0 if i <= 50 else 0.0,
                "residual_balance": 75000.0 if i <= 50 else 100000.0,
                "payment_status": "partial" if i <= 50 else "unpaid",
                "due_status": "overdue" if i > 40 else "not_due",
            }
            for i in range(1, 76)
        ]
        # Expected across all 75 rows:
        # total_amount = 75 * 100_000 = 7_500_000
        # total_paid   = 50 * 25_000  = 1_250_000
        # total_residual = 50 * 75_000 + 25 * 100_000 = 3_750_000 + 2_500_000 = 6_250_000
        # overdue_count = 35 (items 41..75)
        raw_payload = {"bills": bills_75}

        sanitized = sanitize_erp_payload(raw_payload, max_records=50)
        self.assertEqual(len(sanitized["bills"]), 50)
        self.assertEqual(sanitized["total_count"], 75)
        self.assertEqual(
            sanitized["calculations"],
            {
                "total_amount": 7500000.0,
                "total_paid": 1250000.0,
                "total_residual": 6250000.0,
                "overdue_count": 35,
            },
        )

        rendered = _format_single_tool_presentation("get_vendor_bills", sanitized)
        self.assertIn("(75 Tagihan Ditemukan)", rendered)
        self.assertIn("Rp 7.500.000,00", rendered)
        self.assertIn("Rp 6.250.000,00", rendered)
        self.assertIn("35 faktur", rendered)
        self.assertIn("*Menampilkan 50 dari 75 data.", rendered)

        # Also verify end-to-end through arun_workflow + tool_node + synthesizer_node
        isolated_graph = create_workflow_graph(checkpointer=MemorySaver())
        with patch("app.services.django_client.DjangoCoreClient.fetch_vendor_bills", new_callable=AsyncMock) as mock_bills:
            mock_bills.return_value = raw_payload
            result = await arun_workflow(
                user_query="Tampilkan daftar vendor bills belum lunas",
                company_id=1,
                thread_id=None,
                graph=isolated_graph,
            )

        self.assertEqual(result["verification_status"], "VALID")
        answer = result["final_answer"]
        self.assertIn("Rp 7.500.000,00", answer)
        self.assertIn("Rp 6.250.000,00", answer)
        self.assertIn("35 faktur", answer)
        self.assertIn("*Menampilkan 50 dari 75 data.", answer)

    def test_sanitize_preserves_existing_calculations_and_total_count(self):
        """Existing server-side calculations and total_count/total_found are not overwritten."""
        bills_60 = [
            {
                "number": f"BILL/{i:03d}",
                "total_amount": 1000.0,
                "paid_amount": 0.0,
                "residual_balance": 1000.0,
                "due_status": "not_due",
            }
            for i in range(60)
        ]
        raw_payload = {
            "total_count": 120,
            "bills": bills_60,
            "calculations": {
                "total_amount": 120000.0,
                "total_paid": 20000.0,
                "total_residual": 100000.0,
                "overdue_count": 7,
            },
        }
        sanitized = sanitize_erp_payload(raw_payload, max_records=50)
        self.assertEqual(len(sanitized["bills"]), 50)
        self.assertEqual(sanitized["total_count"], 120)
        self.assertEqual(sanitized["calculations"]["total_amount"], 120000.0)
        self.assertEqual(sanitized["calculations"]["overdue_count"], 7)

    def test_pagination_footer_for_other_truncated_collections(self):
        """Truncated entries, alerts, and stock history payloads preserve total counts and show footer."""
        entries_payload = sanitize_erp_payload(
            {
                "entries": [
                    {"number": f"JE/2026/{i:03d}", "date": "2026-04-01", "vendor_name": "PT A", "total_debit": 5000.0}
                    for i in range(65)
                ]
            },
            max_records=50,
        )
        self.assertEqual(len(entries_payload["entries"]), 50)
        self.assertEqual(entries_payload["total_count"], 65)
        entries_md = _format_single_tool_presentation("get_journal_entries", entries_payload)
        self.assertIn("*Menampilkan 50 dari 65 data.", entries_md)

        alerts_payload = sanitize_erp_payload(
            {
                "alerts": [
                    {"code": f"SKU-{i:03d}", "name": f"Produk {i}", "warehouse_name": "Pusat", "qty_on_hand": 2, "threshold": 10}
                    for i in range(62)
                ]
            },
            max_records=50,
        )
        self.assertEqual(len(alerts_payload["alerts"]), 50)
        self.assertEqual(alerts_payload["total_found"], 62)
        alerts_md = _format_single_tool_presentation("get_low_stock_alerts", alerts_payload)
        self.assertIn("*Menampilkan 50 dari 62 data.", alerts_md)

        hist_payload = sanitize_erp_payload(
            {
                "history": [
                    {"date": "2026-04-01", "reference": f"WH/IN/{i:03d}", "code": "HVS-A4", "operation_type": "IN", "location_name": "WH", "qty_done": 5.0}
                    for i in range(55)
                ]
            },
            max_records=50,
        )
        self.assertEqual(len(hist_payload["history"]), 50)
        self.assertEqual(hist_payload["total_count"], 55)
        hist_md = _format_single_tool_presentation("get_stock_history", hist_payload)
        self.assertIn("*Menampilkan 50 dari 55 data.", hist_md)


class TestD6StockLocationAndBranchWarehouseAutoResolve(unittest.IsolatedAsyncioTestCase):
    """Verify D6: multi-word location regex, branch warehouse auto-resolve, and local fail-fast."""

    async def test_stock_query_multiword_location_and_branch_warehouse_autoresolve(self):
        """
        Tests:
        1. 'Tampilkan stok barang di Gudang Cabang Jakarta' -> location_name='Jakarta', company_id=8
        2. 'Cek stok barang di Cabang Bandung' & 'Cek stok barang Cabang Bandung' -> auto-resolve warehouse_id via fetch_warehouses
        3. 'Cek stok barang' -> local fail-fast MISSING_PRODUCT_IDENTIFIER with 0 HTTP calls to Django
        """
        isolated_graph = create_workflow_graph(checkpointer=MemorySaver())

        # 1. 'Tampilkan stok barang di Gudang Cabang Jakarta'
        jkt_extracted = _fallback_extract_stock_level("Tampilkan stok barang di Gudang Cabang Jakarta")
        self.assertEqual(jkt_extracted.get("location_name"), "Jakarta")
        self.assertIsNone(jkt_extracted.get("code"))

        with patch("app.services.django_client.DjangoCoreClient.fetch_locations", new_callable=AsyncMock) as mock_locs, \
             patch("app.services.django_client.DjangoCoreClient.fetch_warehouses", new_callable=AsyncMock) as mock_whs, \
             patch("app.services.django_client.DjangoCoreClient.fetch_stock_level", new_callable=AsyncMock) as mock_stock:
            mock_locs.return_value = {"locations": [{"id": 88, "name": "Gudang Jakarta"}]}
            mock_whs.return_value = {"warehouses": []}
            mock_stock.return_value = {
                "products": [
                    {
                        "code": "JKT-001",
                        "name": "Kertas Stiker Jakarta",
                        "company_id": 8,
                        "location_name": "Gudang Jakarta",
                        "qty_on_hand": 40.0,
                    }
                ]
            }

            res_jkt = await arun_workflow(
                user_query="Tampilkan stok barang di Gudang Cabang Jakarta",
                company_id=1,
                thread_id=None,
                graph=isolated_graph,
            )

        self.assertEqual(res_jkt["company_id"], 8)
        self.assertEqual(res_jkt["verification_status"], "VALID")
        mock_locs.assert_awaited_once()
        self.assertEqual(mock_locs.call_args.kwargs["params"]["search"], "Jakarta")
        self.assertEqual(mock_locs.call_args.kwargs["company_id"], 8)
        mock_stock.assert_awaited_once()
        self.assertEqual(mock_stock.call_args.kwargs["params"]["location_id"], 88)
        self.assertIn("Kertas Stiker Jakarta", res_jkt["final_answer"])

        # 2a. 'Cek stok barang di Cabang Bandung' when location/warehouse search by name yields empty,
        # auto-resolving branch warehouse via fetch_warehouses(params={})
        async def bandung_wh_side_effect(params=None, user_token=None, company_id=None):
            if params and params.get("search"):
                return {"warehouses": []}
            return {"warehouses": [{"id": 61, "name": "Gudang Utama Bandung"}]}

        with patch("app.services.django_client.DjangoCoreClient.fetch_locations", new_callable=AsyncMock) as mock_locs, \
             patch("app.services.django_client.DjangoCoreClient.fetch_warehouses", new_callable=AsyncMock, side_effect=bandung_wh_side_effect) as mock_whs, \
             patch("app.services.django_client.DjangoCoreClient.fetch_stock_level", new_callable=AsyncMock) as mock_stock:
            mock_locs.return_value = {"locations": []}
            mock_stock.return_value = {
                "products": [
                    {
                        "code": "BDG-001",
                        "name": "Tinta Bandung",
                        "company_id": 6,
                        "location_name": "Gudang Utama Bandung",
                        "qty_on_hand": 18.0,
                    }
                ]
            }

            res_bdg = await arun_workflow(
                user_query="Cek stok barang di Cabang Bandung",
                company_id=1,
                thread_id=None,
                graph=isolated_graph,
            )

        self.assertEqual(res_bdg["company_id"], 6)
        self.assertEqual(res_bdg["verification_status"], "VALID")
        mock_stock.assert_awaited_once()
        self.assertEqual(mock_stock.call_args.kwargs["params"]["warehouse_id"], 61)
        self.assertEqual(mock_stock.call_args.kwargs["company_id"], 6)
        self.assertIn("Tinta Bandung", res_bdg["final_answer"])

        # 2b. 'Cek stok barang Cabang Bandung' (no 'di', so no location_name extracted)
        bdg_no_di = _fallback_extract_stock_level("Cek stok barang Cabang Bandung")
        self.assertIsNone(bdg_no_di.get("code"))
        self.assertIsNone(bdg_no_di.get("location_name"))

        with patch("app.services.django_client.DjangoCoreClient.fetch_locations", new_callable=AsyncMock) as mock_locs, \
             patch("app.services.django_client.DjangoCoreClient.fetch_warehouses", new_callable=AsyncMock) as mock_whs, \
             patch("app.services.django_client.DjangoCoreClient.fetch_stock_level", new_callable=AsyncMock) as mock_stock:
            mock_whs.return_value = {"warehouses": [{"id": 62, "name": "Gudang Bandung"}]}
            mock_stock.return_value = {
                "products": [
                    {
                        "code": "BDG-002",
                        "name": "Art Paper Bandung",
                        "company_id": 6,
                        "qty_on_hand": 25.0,
                    }
                ]
            }

            res_bdg_no_di = await arun_workflow(
                user_query="Cek stok barang Cabang Bandung",
                company_id=1,
                thread_id=None,
                graph=isolated_graph,
            )

        self.assertEqual(res_bdg_no_di["company_id"], 6)
        self.assertEqual(res_bdg_no_di["verification_status"], "VALID")
        mock_locs.assert_not_called()
        mock_whs.assert_awaited_once()
        self.assertEqual(mock_whs.call_args.kwargs["params"], {})
        self.assertEqual(mock_whs.call_args.kwargs["company_id"], 6)
        mock_stock.assert_awaited_once()
        self.assertEqual(mock_stock.call_args.kwargs["params"]["warehouse_id"], 62)

        # 3. 'Cek stok barang' -> local fail-fast without hitting Django HTTP endpoints
        with patch("app.services.django_client.DjangoCoreClient.fetch_locations", new_callable=AsyncMock) as mock_locs, \
             patch("app.services.django_client.DjangoCoreClient.fetch_warehouses", new_callable=AsyncMock) as mock_whs, \
             patch("app.services.django_client.DjangoCoreClient.fetch_stock_level", new_callable=AsyncMock) as mock_stock:
            res_empty = await arun_workflow(
                user_query="Cek stok barang",
                company_id=1,
                thread_id=None,
                graph=isolated_graph,
            )

        self.assertEqual(res_empty["verification_status"], "ERROR")
        self.assertTrue((res_empty.get("error_message") or "").startswith("MISSING_PRODUCT_IDENTIFIER:"))
        mock_locs.assert_not_called()
        mock_whs.assert_not_called()
        mock_stock.assert_not_called()
        self.assertIn("Informasi Produk atau Gudang Belum Spesifik", res_empty["final_answer"])

    def test_multiword_location_regex_preserves_stopwords_and_two_word_locations(self):
        """Multi-word locations like 'Jakarta Selatan' and 'Main Store' are preserved; stopwords are dropped."""
        p1 = _fallback_extract_stock_level("Tampilkan stok barang di Gudang Jakarta Selatan")
        self.assertEqual(p1.get("location_name"), "Jakarta Selatan")

        p2 = _fallback_extract_stock_level("Cek stok di Gudang Main Store")
        self.assertEqual(p2.get("location_name"), "Main Store")

        p3 = _fallback_extract_stock_level("cek stok di gudang utama untuk AAEO-NPKG")
        self.assertEqual(p3.get("location_name"), "utama")
        self.assertEqual(p3.get("code"), "AAEO-NPKG")

        p4 = _fallback_extract_stock_level("cek stok di gudang untuk customer Andi")
        self.assertIsNone(p4.get("location_name"))


class TestLifespanCleanupAndSharedHttpPool(unittest.IsolatedAsyncioTestCase):
    """Verify FastAPI lifespan cleanup, background task tracking, shared httpx pool, and token cache."""

    async def asyncTearDown(self):
        await aclose_shared_django_client()
        main_module._BACKGROUND_TASKS.clear()

    async def test_shared_httpx_pool_system_token_cache_and_shutdown(self):
        """
        Internal DjangoCoreClient instances share one httpx.AsyncClient per active event loop,
        coalesce concurrent system logins into _CACHED_SYSTEM_TOKEN, invalidate cleanly on 401,
        keep pooled transport open across per-node aclose(), and close cleanly on
        aclose_shared_django_client().
        """
        await aclose_shared_django_client()

        c1 = DjangoCoreClient(base_url="http://test-erp:8000", username="admin", password="secret", token=None)
        c2 = DjangoCoreClient(base_url="http://test-erp:8000", username="admin", password="secret", token=None)
        self.assertIs(c1.client, c2.client)
        self.assertFalse(c1.client.is_closed)

        shared_http = c1.client
        login_calls = 0

        async def _handler(request: httpx.Request) -> httpx.Response:
            nonlocal login_calls
            if request.url.path == "/external_api/get_token/":
                login_calls += 1
                await asyncio.sleep(0.02)
                return httpx.Response(200, json={"access": f"sys-jwt-{login_calls}"})
            auth_header = request.headers.get("Authorization")
            if auth_header == "Bearer sys-jwt-1":
                return httpx.Response(401, text="Expired")
            return httpx.Response(200, json={"success": True, "data": {"ok": True, "token": auth_header}})

        shared_http._transport = httpx.MockTransport(_handler)

        # Concurrent login() calls coalesce into a single HTTP login request
        t1, t2 = await asyncio.gather(c1.login(), c2.login())
        self.assertEqual(t1, "sys-jwt-1")
        self.assertEqual(t2, "sys-jwt-1")
        self.assertEqual(login_calls, 1)
        self.assertEqual(django_client_module._CACHED_SYSTEM_TOKEN, "sys-jwt-1")

        # Per-instance aclose() does not kill the shared pooled httpx client
        await c1.aclose()
        await c2.aclose()
        self.assertFalse(shared_http.is_closed)

        # New DjangoCoreClient picks up _CACHED_SYSTEM_TOKEN immediately; 401 triggers refresh -> sys-jwt-2
        with patch.object(settings, "ERP_API_TOKEN", ""):
            c3 = DjangoCoreClient(base_url="http://test-erp:8000", username="admin", password="secret")
            self.assertEqual(c3.token, "sys-jwt-1")
            res = await c3.execute_tool(domain="accounting", tool_name="get_account_balance", params={})
            self.assertEqual(res["ok"], True)
            self.assertEqual(res["token"], "Bearer sys-jwt-2")
            self.assertEqual(login_calls, 2)
            self.assertEqual(django_client_module._CACHED_SYSTEM_TOKEN, "sys-jwt-2")

        # Shared pool shutdown closes httpx client and wipes cached system token
        await aclose_shared_django_client()
        self.assertTrue(shared_http.is_closed)
        self.assertIsNone(django_client_module._CACHED_SYSTEM_TOKEN)

    def test_fastapi_lifespan_tracks_background_tasks_and_closes_all_resources(self):
        """
        FastAPI lifespan tracks afetch_free_models_catalog in _BACKGROUND_TASKS during startup,
        closes real shared httpx client + VectorService asyncpg pool + workflow executor on exit,
        and remains resilient when one shutdown hook raises.
        """
        gate = asyncio.Event()
        prewarm_started = threading.Event()
        created_shared_clients = []

        async def _slow_prewarm(*args, **kwargs):
            # Capture the FastAPI portal loop's shared httpx client while inside lifespan
            portal_client = DjangoCoreClient()
            created_shared_clients.append(portal_client.client)
            prewarm_started.set()
            try:
                await gate.wait()
            except asyncio.CancelledError:
                raise

        with patch("app.main.afetch_free_models_catalog", side_effect=_slow_prewarm), \
             patch("app.main.vector_service.close_pool", new_callable=AsyncMock) as mock_close_pool, \
             patch("app.main.shutdown_workflow_executor") as mock_shutdown_exec:
            with TestClient(app) as test_client:
                self.assertTrue(prewarm_started.wait(timeout=2.0))
                resp = test_client.get("/health")
                self.assertEqual(resp.status_code, 200)
                self.assertGreaterEqual(len(main_module._BACKGROUND_TASKS), 1)

            # After TestClient exits lifespan: task set drained, pools & executor closed cleanly
            self.assertEqual(len(main_module._BACKGROUND_TASKS), 0)
            mock_close_pool.assert_awaited_once()
            mock_shutdown_exec.assert_called_once_with(wait=False)
            self.assertEqual(len(created_shared_clients), 1)
            self.assertTrue(created_shared_clients[0].is_closed)

        # Shutdown resilience: failure in close_pool still closes django client & workflow executor
        with patch("app.main.afetch_free_models_catalog", new_callable=AsyncMock, return_value=["openrouter/free"]), \
             patch("app.main.vector_service.close_pool", new_callable=AsyncMock, side_effect=RuntimeError("pg shutdown boom")), \
             patch("app.main.aclose_shared_django_client", new_callable=AsyncMock) as mock_aclose_http, \
             patch("app.main.shutdown_workflow_executor") as mock_shutdown_exec:
            with TestClient(app) as test_client:
                self.assertEqual(test_client.get("/health").status_code, 200)

            mock_aclose_http.assert_awaited_once()
            mock_shutdown_exec.assert_called_once_with(wait=False)

    def test_fastapi_lifespan_closes_real_asyncpg_pool_and_workflow_executor(self):
        """
        End-to-end lifespan shutdown closes an active VectorService asyncpg pool
        and shuts down the reusable workflow ThreadPoolExecutor on TestClient exit.
        """
        import app.graph.workflow as workflow_module

        mock_pg_pool = MagicMock()
        mock_pg_pool._closed = False
        mock_pg_pool.close = AsyncMock()
        initialized = threading.Event()

        async def _prewarm_with_resources(*args, **kwargs):
            await main_module.vector_service.get_pool()
            workflow_module.get_workflow_executor()
            initialized.set()
            return ["openrouter/free"]

        with patch("app.main.afetch_free_models_catalog", side_effect=_prewarm_with_resources), \
             patch.object(main_module.vector_service, "db_url", "postgresql://user:pass@localhost:5432/vec"), \
             patch("asyncpg.create_pool", new_callable=AsyncMock, return_value=mock_pg_pool):
            with TestClient(app) as test_client:
                self.assertTrue(initialized.wait(timeout=2.0))
                self.assertIsNotNone(main_module.vector_service._pool)
                self.assertIsNotNone(workflow_module._WORKFLOW_EXECUTOR)
                self.assertEqual(test_client.get("/health").status_code, 200)

        mock_pg_pool.close.assert_awaited_once()
        self.assertIsNone(main_module.vector_service._pool)
        self.assertIsNone(workflow_module._WORKFLOW_EXECUTOR)


class TestTimezoneWibConsistency(unittest.TestCase):
    """Timezone Drift UTC vs WIB (Asia/Jakarta) verification."""

    def test_app_timezone_default_config(self):
        self.assertEqual(settings.APP_TIMEZONE, "Asia/Jakarta")

    def test_get_current_erp_datetime_and_date_across_utc_midnight_boundary(self):
        from datetime import date, datetime, timezone
        from zoneinfo import ZoneInfoNotFoundError
        import app.core.context as context_module

        utc_instant = datetime(2026, 11, 13, 17, 30, 0, tzinfo=timezone.utc)

        class _FrozenDatetime:
            @classmethod
            def now(cls, tz=None):
                return utc_instant.astimezone(tz) if tz is not None else utc_instant.replace(tzinfo=None)

        with patch.object(context_module, "datetime", _FrozenDatetime):
            # Also verify fixed-offset fallback when ZoneInfo lacks tzdata (e.g. Windows/minimal container)
            with patch.object(context_module, "ZoneInfo", side_effect=ZoneInfoNotFoundError("no tzdata")):
                erp_dt = context_module.get_current_erp_datetime()
                erp_date = context_module.get_current_erp_date()

        self.assertEqual(erp_dt.strftime("%Y-%m-%d-%H:%M"), "2026-11-14-00:30")
        self.assertEqual(erp_date, date(2026, 11, 14))

    def test_extractor_resolve_relative_dates_uses_wib_date_at_utc_boundary(self):
        from datetime import datetime, timezone
        import app.core.context as context_module
        from app.graph.nodes.extractor import resolve_relative_dates

        # 2026-10-31 18:00 UTC == 2026-11-01 01:00 WIB (month & day boundary)
        utc_instant = datetime(2026, 10, 31, 18, 0, 0, tzinfo=timezone.utc)

        class _FrozenDatetime:
            @classmethod
            def now(cls, tz=None):
                return utc_instant.astimezone(tz) if tz is not None else utc_instant.replace(tzinfo=None)

        with patch.object(context_module, "datetime", _FrozenDatetime):
            today_range = resolve_relative_dates("transaksi hari ini")
            month_range = resolve_relative_dates("laporan bulan ini")

        self.assertEqual(today_range, ("2026-11-01", "2026-11-01"))
        self.assertEqual(month_range, ("2026-11-01", "2026-11-30"))

    def test_clients_generate_secret_key_uses_wib_timestamp(self):
        import hashlib
        from datetime import datetime, timezone
        import app.core.context as context_module
        from app.services.django_client import DjangoCoreClient
        from app.services.erp_client import ERPClient

        utc_instant = datetime(2026, 11, 13, 17, 30, 0, tzinfo=timezone.utc)

        class _FrozenDatetime:
            @classmethod
            def now(cls, tz=None):
                return utc_instant.astimezone(tz) if tz is not None else utc_instant.replace(tzinfo=None)

        expected_hash = hashlib.md5(b"erp_master_nexus-2026-11-14-00:30").hexdigest()

        with patch.object(context_module, "datetime", _FrozenDatetime):
            django_secret = DjangoCoreClient()._generate_secret_key()
            erp_secret = ERPClient()._generate_secret_key()

        self.assertEqual(django_secret, expected_hash)
        self.assertEqual(erp_secret, expected_hash)


class TestNonBlockingVectorEmbedding(unittest.TestCase):
    """Non-blocking Gemini embedding execution via asyncio.to_thread."""

    def test_get_embedding_runs_sync_sdk_in_worker_thread_without_blocking_event_loop(self):
        import threading
        import time
        from types import SimpleNamespace
        from app.services.vector_service import VectorService

        loop_thread_id = None
        embed_thread_id = None
        concurrent_ticks = 0

        def _blocking_embed_content(**kwargs):
            nonlocal embed_thread_id
            embed_thread_id = threading.get_ident()
            time.sleep(0.08)
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1, 0.2, 0.3])])

        fake_client = SimpleNamespace(
            models=SimpleNamespace(embed_content=_blocking_embed_content)
        )
        svc = VectorService(gemini_api_key="test-key", embedding_model="gemini-embedding-001")
        svc._genai_client = fake_client

        async def _run():
            nonlocal loop_thread_id, concurrent_ticks
            loop_thread_id = threading.get_ident()

            async def _ticker():
                nonlocal concurrent_ticks
                for _ in range(3):
                    await asyncio.sleep(0.01)
                    concurrent_ticks += 1

            vec, _ = await asyncio.gather(svc.get_embedding("cek stok kertas"), _ticker())
            return vec

        result = asyncio.run(_run())
        self.assertEqual(result, [0.1, 0.2, 0.3])
        self.assertIsNotNone(embed_thread_id)
        self.assertNotEqual(embed_thread_id, loop_thread_id)
        self.assertEqual(concurrent_ticks, 3)


class TestDedicatedInternalServiceKeyAndObservability(unittest.TestCase):
    """Verify INTERNAL_SERVICE_KEY precedence, X-Request-ID/X-Process-Time-Ms headers, and /health/ready."""

    def setUp(self):
        self.client = TestClient(app)

    @patch("app.main.arun_workflow", new_callable=AsyncMock)
    def test_internal_service_key_precedence_over_erp_api_token(self, mock_workflow):
        mock_workflow.return_value = {
            "final_answer": "OK",
            "detected_domain": "general",
            "verification_status": "VALID",
            "error_message": None,
        }

        with patch.object(settings, "REQUIRE_SERVICE_KEY", True), \
             patch.object(settings, "INTERNAL_SERVICE_KEY", "dedicated-service-secret"), \
             patch.object(settings, "ERP_API_TOKEN", "legacy-jwt-token"):
            resp_legacy = self.client.post(
                "/api/v1/chat",
                json={"query": "Halo"},
                headers={"X-Internal-Service-Key": "legacy-jwt-token"},
            )
            self.assertEqual(resp_legacy.status_code, 403)

            resp_dedicated = self.client.post(
                "/api/v1/chat",
                json={"query": "Halo"},
                headers={"X-Internal-Service-Key": "dedicated-service-secret"},
            )
            self.assertEqual(resp_dedicated.status_code, 200)

    def test_request_id_and_process_time_headers(self):
        resp_auto = self.client.get("/health")
        self.assertEqual(resp_auto.status_code, 200)
        self.assertTrue(resp_auto.headers.get("X-Request-ID"))
        self.assertIsNotNone(resp_auto.headers.get("X-Process-Time-Ms"))
        self.assertGreaterEqual(float(resp_auto.headers["X-Process-Time-Ms"]), 0.0)

        resp_echo = self.client.get("/health", headers={"X-Request-ID": "req-abc_123"})
        self.assertEqual(resp_echo.headers.get("X-Request-ID"), "req-abc_123")

    def test_health_ready_boolean_status_without_secret_leakage(self):
        with patch.object(settings, "MODEL_PROVIDER", "openrouter"), \
             patch.object(settings, "OPENROUTER_API_KEY", "sk-secret-key"), \
             patch.object(settings, "ERP_BASE_URL", "http://internal-erp.local:8000"), \
             patch.object(settings, "VECTOR_DB_URL", "postgresql://postgres:supersecret@db.internal:5432/vec"), \
             patch.object(main_module.vector_service, "db_url", "postgresql://postgres:supersecret@db.internal:5432/vec"):
            resp_ready = self.client.get("/health/ready")
            self.assertEqual(resp_ready.status_code, 200)
            body = resp_ready.json()
            self.assertEqual(body["status"], "ready")
            self.assertTrue(body["checks"]["llm_configured"])
            self.assertTrue(body["checks"]["erp_configured"])
            self.assertTrue(body["checks"]["vector_db_configured"])
            raw_text = resp_ready.text
            self.assertNotIn("supersecret", raw_text)
            self.assertNotIn("sk-secret-key", raw_text)
            self.assertNotIn("internal-erp.local", raw_text)
            self.assertNotIn("db.internal", raw_text)

        with patch.object(settings, "MODEL_PROVIDER", "openrouter"), \
             patch.object(settings, "OPENROUTER_API_KEY", ""), \
             patch.object(settings, "ERP_BASE_URL", "http://internal-erp.local:8000"):
            resp_degraded = self.client.get("/health/ready")
            self.assertEqual(resp_degraded.status_code, 503)
            self.assertEqual(resp_degraded.json()["status"], "degraded")


class TestPending4MultiToolSelectiveRetryAndBranchScoping(unittest.IsolatedAsyncioTestCase):
    """Verify multi-tool selective retry, strict doc identifier guard, and branch_co_id propagation."""

    async def test_multi_tool_selective_retry_skips_already_valid_tool(self):
        from app.graph.nodes.extractor import extractor_node
        from app.graph.nodes.tool_node import tool_node
        from app.graph.nodes.verifier import verifier_node

        valid_stock_payload = {
            "products": [{"sku": "INK-01", "name": "Tinta", "quantity": 12}],
            "total_count": 1,
        }
        empty_bills_payload = {"bills": [], "total_count": 0}
        found_bills_payload = {"bills": [{"id": 1, "number": "BILL/2026/001"}], "total_count": 1}

        mock_client = MagicMock()
        mock_client.fetch_stock_level = AsyncMock(return_value=valid_stock_payload)
        mock_client.fetch_vendor_bills = AsyncMock(side_effect=[empty_bills_payload, found_bills_payload])

        initial_state = {
            "user_query": "Cek stok tinta dan tagihan vendor minggu ini",
            "company_id": 1,
            "is_multi_tool": True,
            "retry_count": 0,
            "active_tools": [
                {
                    "tool": "get_stock_level",
                    "domain": "inventory",
                    "sub_query": "Cek stok tinta INK-01",
                    "params": {"code": "INK-01"},
                    "status": "pending",
                },
                {
                    "tool": "get_vendor_bills",
                    "domain": "accounting",
                    "sub_query": "Cek tagihan vendor minggu ini",
                    "params": {"date_from": "2026-11-09", "date_to": "2026-11-13"},
                    "status": "pending",
                },
            ],
        }

        with patch("app.graph.nodes.extractor._extract_single_tool_params") as mock_extract:
            mock_extract.side_effect = [
                {"code": "INK-01", "company_id": 1},
                {"date_from": "2026-11-09", "date_to": "2026-11-13", "company_id": 1},
            ]
            ext1 = extractor_node(initial_state)

        state_after_ext1 = {**initial_state, **ext1}
        t1 = await tool_node(state_after_ext1, client=mock_client)
        state_after_t1 = {**state_after_ext1, **t1}
        v1 = verifier_node(state_after_t1)

        self.assertEqual(v1["verification_status"], "RETRY")
        self.assertEqual(v1["multi_verification_status"]["get_stock_level"], "VALID")
        self.assertEqual(v1["multi_verification_status"]["get_vendor_bills"], "EMPTY")

        # Retry cycle: only get_vendor_bills should be re-extracted and re-executed
        state_retry = {**state_after_t1, **v1}
        with patch("app.graph.nodes.extractor._extract_single_tool_params") as mock_extract_retry:
            mock_extract_retry.return_value = {"company_id": 1}
            ext2 = extractor_node(state_retry)
            self.assertEqual(mock_extract_retry.call_count, 1)
            self.assertEqual(mock_extract_retry.call_args.kwargs["plan"]["tool"], "get_vendor_bills")

        self.assertEqual(ext2["active_tools"][0]["params"], {"code": "INK-01", "company_id": 1})
        state_after_ext2 = {**state_retry, **ext2}
        t2 = await tool_node(state_after_ext2, client=mock_client)

        self.assertEqual(mock_client.fetch_stock_level.await_count, 1)
        self.assertEqual(mock_client.fetch_vendor_bills.await_count, 2)
        self.assertEqual(t2["multi_tool_results"]["get_stock_level"], valid_stock_payload)
        self.assertEqual(t2["multi_tool_results"]["get_vendor_bills"]["bills"], found_bills_payload["bills"])
        self.assertEqual(t2["multi_tool_results"]["get_vendor_bills"]["total_count"], 1)

    async def test_multi_tool_strict_document_guard_skips_retry_and_preserves_params(self):
        from app.graph.nodes.extractor import extractor_node
        from app.graph.nodes.tool_node import tool_node
        from app.graph.nodes.verifier import verifier_node

        # Case 1: Only empty tool has strict document identifier -> verifier must return VALID (no retry)
        v_no_retry = verifier_node({
            "is_multi_tool": True,
            "retry_count": 0,
            "active_tools": [
                {"tool": "get_stock_level", "params": {"code": "INK-01"}, "status": "success", "result": {"products": [{"sku": "INK-01"}]}},
                {"tool": "get_journal_entries", "params": {"entry_number": "JE/2026/999"}, "status": "success", "result": {"entries": [], "total_count": 0}},
            ],
            "multi_tool_results": {
                "get_stock_level": {"products": [{"sku": "INK-01"}]},
                "get_journal_entries": {"entries": [], "total_count": 0},
            },
            "multi_verification_status": {"get_stock_level": "VALID", "get_journal_entries": "EMPTY"},
        })
        self.assertEqual(v_no_retry["verification_status"], "VALID")

        # Case 2: Non-strict tool triggers retry, strict-doc tool keeps params & result untouched
        mock_client = MagicMock()
        mock_client.fetch_journal_entries = AsyncMock()
        mock_client.fetch_vendor_bills = AsyncMock(return_value={"bills": [{"id": 1}], "total_count": 1})

        retry_state = {
            "user_query": "Cek jurnal JE/2026/999 dan tagihan vendor",
            "company_id": 1,
            "is_multi_tool": True,
            "retry_count": 1,
            "retry_instruction": "Longgarkan filter tanggal",
            "active_tools": [
                {
                    "tool": "get_journal_entries",
                    "domain": "accounting",
                    "sub_query": "Cek jurnal JE/2026/999",
                    "params": {"entry_number": "JE/2026/999", "company_id": 1},
                    "status": "success",
                    "result": {"entries": [], "total_count": 0},
                },
                {
                    "tool": "get_vendor_bills",
                    "domain": "accounting",
                    "sub_query": "Cek tagihan vendor",
                    "params": {"date_from": "2026-11-01", "company_id": 1},
                    "status": "success",
                    "result": {"bills": [], "total_count": 0},
                },
            ],
            "multi_verification_status": {"get_journal_entries": "EMPTY", "get_vendor_bills": "EMPTY"},
        }
        with patch("app.graph.nodes.extractor._extract_single_tool_params") as mock_extract:
            mock_extract.return_value = {"company_id": 1}
            ext_out = extractor_node(retry_state)
            self.assertEqual(mock_extract.call_count, 1)
            self.assertEqual(mock_extract.call_args.kwargs["plan"]["tool"], "get_vendor_bills")
        self.assertEqual(ext_out["active_tools"][0]["params"]["entry_number"], "JE/2026/999")

        tool_out = await tool_node({**retry_state, **ext_out}, client=mock_client)
        self.assertEqual(mock_client.fetch_journal_entries.await_count, 0)
        self.assertEqual(mock_client.fetch_vendor_bills.await_count, 1)
        self.assertEqual(tool_out["multi_tool_results"]["get_vendor_bills"]["total_count"], 1)

    async def test_multi_tool_branch_company_id_propagates_to_all_tools(self):
        from app.graph.nodes.extractor import extractor_node

        state = {
            "user_query": "Cek stok tinta dan posisi saldo kas di Cabang Bandung",
            "company_id": 1,
            "is_multi_tool": True,
            "retry_count": 0,
            "active_tools": [
                {"tool": "get_stock_level", "domain": "inventory", "sub_query": "Cek stok tinta", "params": {}, "status": "pending"},
                {"tool": "get_account_balance", "domain": "accounting", "sub_query": "Cek saldo kas", "params": {}, "status": "pending"},
            ],
        }
        with patch("app.graph.nodes.extractor._extract_single_tool_params", side_effect=[{"code": "INK-01"}, {"account_code": "1100"}]):
            ext_out = extractor_node(state)

        self.assertEqual(ext_out["company_id"], 6)
        self.assertEqual(ext_out["active_tools"][0]["params"]["company_id"], 6)
        self.assertEqual(ext_out["active_tools"][1]["params"]["company_id"], 6)


if __name__ == "__main__":
    unittest.main()
