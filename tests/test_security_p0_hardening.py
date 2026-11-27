"""
Security hardening unit & integration tests for P0-1:
  - Dedicated inbound INTERNAL_SERVICE_KEY separation (no ERP_API_TOKEN fallback)
  - authorized_company_ids gating on /api/v1/chat and /api/v1/chat/stream
  - Strict user_token alignment across staging/production in tool_node & rag_node
  - Multi-company fan-out clamping against verified authorized_company_ids
  - ChatRequest payload caps (HTTP 422) & safe history truncation (latest 30 messages)
  - In-process sliding-window rate limiter (per-user/token/IP keying, anti-bypass, Retry-After)
"""
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.rate_limit import (
    SlidingWindowRateLimiter,
    build_rate_limit_key,
    chat_rate_limiter,
)
from app.graph.nodes.rag_node import rag_node
from app.graph.nodes.router import router_node
from app.graph.nodes.tool_node import tool_node
from app.graph.state import create_initial_state
from app.graph.workflow import (
    MAX_HYDRATED_HISTORY_MESSAGES,
    _hydrate_messages_from_history,
    _resolve_effective_company_scope,
)
from app.main import app
from app.schemas.chat import (
    MAX_COMPANY_IDS_COUNT,
    MAX_HISTORY_ITEM_CONTENT_LENGTH,
    MAX_HISTORY_ITEMS,
    MAX_QUERY_LENGTH,
    MAX_THREAD_ID_LENGTH,
)


def _ok_workflow_state() -> dict:
    return {
        "final_answer": "OK",
        "detected_domain": "general",
        "verification_status": "VALID",
        "error_message": None,
    }


async def _ok_stream_events(**kwargs):
    yield ("token", {"content": "OK"})
    yield ("complete", _ok_workflow_state())


class TestP0InboundAuthAndCompanyGating(unittest.TestCase):
    """Verify inbound INTERNAL_SERVICE_KEY isolation and authorized_company_ids gating."""

    def setUp(self):
        self.client = TestClient(app)
        chat_rate_limiter.reset()

    def tearDown(self):
        chat_rate_limiter.reset()

    @patch("app.main.arun_workflow", new_callable=AsyncMock)
    def test_erp_api_token_rejected_as_inbound_service_key(self, mock_workflow):
        mock_workflow.return_value = _ok_workflow_state()
        with patch.object(settings, "INTERNAL_SERVICE_KEY", None), \
             patch.object(settings, "ERP_API_TOKEN", "erp-master-jwt"):
            # Even when REQUIRE_SERVICE_KEY=False, ERP_API_TOKEN cannot verify privileged fields
            resp = self.client.post(
                "/api/v1/chat",
                json={
                    "query": "Cek stok",
                    "user_roles": ["admin"],
                    "authorized_company_ids": [3, 4, 7],
                },
                headers={"X-Internal-Service-Key": "erp-master-jwt"},
            )
            self.assertEqual(resp.status_code, 200)
            kwargs = mock_workflow.await_args.kwargs
            self.assertEqual(kwargs["user_roles"], [])
            self.assertEqual(kwargs["authorized_company_ids"], [])

    @patch("app.main.arun_workflow", new_callable=AsyncMock)
    def test_require_service_key_fails_closed_when_internal_key_unset(self, mock_workflow):
        with patch.object(settings, "REQUIRE_SERVICE_KEY", True), \
             patch.object(settings, "INTERNAL_SERVICE_KEY", ""), \
             patch.object(settings, "ERP_API_TOKEN", "erp-master-jwt"):
            resp = self.client.post(
                "/api/v1/chat",
                json={"query": "Halo"},
                headers={"X-Internal-Service-Key": "erp-master-jwt"},
            )
            self.assertEqual(resp.status_code, 403)
            mock_workflow.assert_not_awaited()

    @patch("app.main.arun_workflow", new_callable=AsyncMock)
    def test_authorized_company_ids_gating_on_chat_endpoint(self, mock_workflow):
        mock_workflow.return_value = _ok_workflow_state()
        with patch.object(settings, "INTERNAL_SERVICE_KEY", "srv-secret"):
            resp_unverified = self.client.post(
                "/api/v1/chat",
                json={"query": "Cek omzet", "authorized_company_ids": [3, 4, 7]},
            )
            self.assertEqual(resp_unverified.status_code, 200)
            self.assertEqual(mock_workflow.await_args.kwargs["authorized_company_ids"], [])

            resp_verified = self.client.post(
                "/api/v1/chat",
                json={"query": "Cek omzet", "authorized_company_ids": [3, 4, 7]},
                headers={"X-Internal-Service-Key": "srv-secret"},
            )
            self.assertEqual(resp_verified.status_code, 200)
            self.assertEqual(mock_workflow.await_args.kwargs["authorized_company_ids"], [3, 4, 7])

    @patch("app.main.astream_workflow_events", side_effect=_ok_stream_events)
    def test_authorized_company_ids_gating_on_stream_endpoint(self, mock_stream):
        with patch.object(settings, "INTERNAL_SERVICE_KEY", "srv-secret"):
            resp_unverified = self.client.post(
                "/api/v1/chat/stream",
                json={"query": "Cek omzet", "authorized_company_ids": [3, 4, 7]},
            )
            self.assertEqual(resp_unverified.status_code, 200)
            self.assertEqual(mock_stream.call_args.kwargs["authorized_company_ids"], [])

            resp_verified = self.client.post(
                "/api/v1/chat/stream",
                json={"query": "Cek omzet", "authorized_company_ids": [3, 4, 7]},
                headers={"X-Internal-Service-Key": "srv-secret"},
            )
            self.assertEqual(resp_verified.status_code, 200)
            self.assertEqual(mock_stream.call_args.kwargs["authorized_company_ids"], [3, 4, 7])


class TestP0StrictUserTokenAlignment(unittest.IsolatedAsyncioTestCase):
    """Verify tool_node and rag_node enforce user_token in production/staging without fallback."""

    def test_settings_is_strict_user_token_required(self):
        with patch.object(settings, "APP_ENV", "development"), \
             patch.object(settings, "REQUIRE_USER_TOKEN", False):
            self.assertFalse(settings.is_strict_user_token_required())

        with patch.object(settings, "APP_ENV", "staging"), \
             patch.object(settings, "REQUIRE_USER_TOKEN", False):
            self.assertTrue(settings.is_strict_user_token_required())

        with patch.object(settings, "APP_ENV", "production"), \
             patch.object(settings, "REQUIRE_USER_TOKEN", False):
            self.assertTrue(settings.is_strict_user_token_required())

        with patch.object(settings, "APP_ENV", "development"), \
             patch.object(settings, "REQUIRE_USER_TOKEN", True):
            self.assertTrue(settings.is_strict_user_token_required())

    async def test_tool_node_rejects_missing_user_token_in_staging(self):
        state = create_initial_state("Cek saldo kas", company_id=1, user_token=None)
        state["detected_domain"] = "accounting"
        state["detected_tool"] = "get_account_balance"
        state["extracted_params"] = {}

        with patch.object(settings, "APP_ENV", "staging"), \
             patch.object(settings, "REQUIRE_USER_TOKEN", False), \
             patch("app.services.django_client.DjangoCoreClient.login", new_callable=AsyncMock) as mock_login:
            result = await tool_node(state=state)
            self.assertEqual(result["verification_status"], "ERROR")
            self.assertIn("user_token", str(result.get("error_message") or ""))
            mock_login.assert_not_awaited()

    async def test_rag_node_django_fallback_rejects_missing_user_token_in_production(self):
        state = create_initial_state("Apa SOP retur barang?", company_id=1, user_token=None)
        state["query_mode"] = "vector_rag"
        state["rag_query"] = "SOP retur barang"

        with patch.object(settings, "APP_ENV", "production"), \
             patch.object(settings, "REQUIRE_USER_TOKEN", False), \
             patch.object(settings, "VECTOR_DB_URL", None), \
             patch("app.services.django_client.DjangoCoreClient.login", new_callable=AsyncMock) as mock_login:
            result = await rag_node(state=state)
            self.assertEqual(result["rag_context"], [])
            mock_login.assert_not_awaited()


class TestP0MultiCompanyClamping(unittest.TestCase):
    """Verify verified authorized_company_ids clamps broad fan-outs while preserving explicit single branch."""

    def test_workflow_all_branches_clamped_to_authorized_company_ids(self):
        primary, cids, is_mc, _, src = _resolve_effective_company_scope(
            user_query="Total stok tinta di semua cabang",
            company_id=4,
            company_ids=None,
            company_id_explicit=True,
            checkpoint_values=None,
            authorized_company_ids=[3, 4, 7],
        )
        self.assertEqual(primary, 4)
        self.assertEqual(cids, [3, 4, 7])
        self.assertTrue(is_mc)
        self.assertEqual(src, "text")

    def test_workflow_all_branches_single_authorized_company_stays_single(self):
        primary, cids, is_mc, _, _ = _resolve_effective_company_scope(
            user_query="Total stok tinta di semua cabang",
            company_id=6,
            company_ids=None,
            company_id_explicit=True,
            checkpoint_values=None,
            authorized_company_ids=[6],
        )
        self.assertEqual(primary, 6)
        self.assertEqual(cids, [6])
        self.assertFalse(is_mc)

    def test_router_all_branches_and_ranking_clamped_to_authorized_company_ids(self):
        state_all = create_initial_state(
            "Cek stok tinta di semua cabang",
            company_id=3,
            authorized_company_ids=[3, 4, 7],
        )
        routed_all = router_node(state_all, llm=None)
        self.assertTrue(routed_all["is_multi_company"])
        self.assertEqual(routed_all["company_ids"], [3, 4, 7])

        state_rank = create_initial_state(
            "Cabang mana dengan omzet penjualan tertinggi bulan ini?",
            company_id=3,
            authorized_company_ids=[3, 4, 7],
        )
        routed_rank = router_node(state_rank, llm=None)
        self.assertTrue(routed_rank["is_multi_company"])
        self.assertEqual(routed_rank["company_ids"], [3, 4, 7])

    def test_router_hq_single_authorized_company_does_not_expand_to_all_17(self):
        state = create_initial_state(
            "Total omzet penjualan bulan ini",
            company_id=3,
            authorized_company_ids=[3],
        )
        routed = router_node(state, llm=None)
        self.assertFalse(routed.get("is_multi_company", False))
        self.assertEqual(routed["company_ids"], [3])

    def test_explicit_single_branch_still_passes_through_for_django_403(self):
        state = create_initial_state(
            "Cek stok tinta di cabang Bandung",
            company_id=4,
            authorized_company_ids=[4, 7],
        )
        routed = router_node(state, llm=None)
        self.assertFalse(routed.get("is_multi_company", False))
        self.assertEqual(routed["company_id"], 6)
        self.assertEqual(routed["company_ids"], [6])


class TestP0PayloadBoundsAndHistoryTruncation(unittest.TestCase):
    """Verify ChatRequest hard 422 caps and safe 30-message hydration truncation."""

    def setUp(self):
        self.client = TestClient(app)

    def test_query_over_max_length_rejected_422(self):
        resp = self.client.post("/api/v1/chat", json={"query": "a" * (MAX_QUERY_LENGTH + 1)})
        self.assertEqual(resp.status_code, 422)

    def test_thread_id_over_max_length_rejected_422(self):
        resp = self.client.post(
            "/api/v1/chat",
            json={"query": "Halo", "thread_id": "t" * (MAX_THREAD_ID_LENGTH + 1)},
        )
        self.assertEqual(resp.status_code, 422)

    def test_company_lists_over_max_count_rejected_422(self):
        resp_cids = self.client.post(
            "/api/v1/chat",
            json={"query": "Cek stok", "company_ids": list(range(1, MAX_COMPANY_IDS_COUNT + 2))},
        )
        self.assertEqual(resp_cids.status_code, 422)

        resp_auth = self.client.post(
            "/api/v1/chat",
            json={"query": "Cek stok", "authorized_company_ids": list(range(1, MAX_COMPANY_IDS_COUNT + 2))},
        )
        self.assertEqual(resp_auth.status_code, 422)

    def test_extreme_history_rejected_422(self):
        too_many = [{"role": "user", "content": "hi"} for _ in range(MAX_HISTORY_ITEMS + 1)]
        resp_count = self.client.post("/api/v1/chat", json={"query": "Halo", "history": too_many})
        self.assertEqual(resp_count.status_code, 422)

        too_long_msg = [{"role": "assistant", "content": "x" * (MAX_HISTORY_ITEM_CONTENT_LENGTH + 1)}]
        resp_content = self.client.post("/api/v1/chat", json={"query": "Halo", "history": too_long_msg})
        self.assertEqual(resp_content.status_code, 422)

    @patch("app.main.arun_workflow", new_callable=AsyncMock)
    def test_valid_long_history_accepted_and_truncated_to_30_on_hydration(self, mock_workflow):
        mock_workflow.return_value = _ok_workflow_state()
        history_45 = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg-{i}"}
            for i in range(45)
        ]
        resp = self.client.post("/api/v1/chat", json={"query": "pertanyaan terbaru", "history": history_45})
        self.assertEqual(resp.status_code, 200)

        hydrated = _hydrate_messages_from_history(history_45, "pertanyaan terbaru")
        self.assertIsNotNone(hydrated)
        # 30 most recent history messages + 1 current user query
        self.assertEqual(len(hydrated), MAX_HYDRATED_HISTORY_MESSAGES + 1)
        self.assertEqual(hydrated[0].content, "msg-15")
        self.assertEqual(hydrated[-2].content, "msg-44")
        self.assertEqual(hydrated[-1].content, "pertanyaan terbaru")


class TestP0RateLimiter(unittest.TestCase):
    """Verify sliding-window rate limiter behavior, user isolation, anti-bypass, and LRU eviction."""

    def setUp(self):
        self.client = TestClient(app)
        chat_rate_limiter.reset()

    def tearDown(self):
        chat_rate_limiter.reset()

    def test_sliding_window_unit_expiry_and_bounded_buckets(self):
        now = [100.0]
        limiter = SlidingWindowRateLimiter(max_buckets=3, clock=lambda: now[0])

        self.assertEqual(limiter.check("u1", limit=2, window_seconds=10), (True, 0))
        now[0] = 102.0
        self.assertEqual(limiter.check("u1", limit=2, window_seconds=10), (True, 0))
        allowed, retry_after = limiter.check("u1", limit=2, window_seconds=10)
        self.assertFalse(allowed)
        self.assertEqual(retry_after, 8)

        # Advance past first request expiry (100.0 + 10.0)
        now[0] = 110.5
        self.assertEqual(limiter.check("u1", limit=2, window_seconds=10), (True, 0))

        # Verify max_buckets LRU eviction prevents unbounded memory growth
        for key in ("u2", "u3", "u4", "u5"):
            limiter.check(key, limit=2, window_seconds=10)
        self.assertEqual(limiter.bucket_count(), 3)

    def test_build_rate_limit_key_priority_and_anti_spoofing(self):
        # Unverified user_id is ignored in favor of token hash / IP
        k_unverified_a = build_rate_limit_key(
            user_id="spoof-1",
            effective_token="jwt-same",
            client_host="10.0.0.1",
            service_key_verified=False,
        )
        k_unverified_b = build_rate_limit_key(
            user_id="spoof-2",
            effective_token="jwt-same",
            client_host="10.0.0.1",
            service_key_verified=False,
        )
        self.assertEqual(k_unverified_a, k_unverified_b)
        self.assertTrue(k_unverified_a.startswith("tok:"))

        # Verified user_id takes top precedence
        k_verified = build_rate_limit_key(
            user_id="user-42",
            effective_token="jwt-same",
            client_host="10.0.0.1",
            service_key_verified=True,
        )
        self.assertEqual(k_verified, "uid:user-42")

        # X-Forwarded-For ignored unless proxy trusted or service key verified
        k_untrusted_xff = build_rate_limit_key(
            client_host="127.0.0.1",
            forwarded_for="203.0.113.9, 10.0.0.1",
            service_key_verified=False,
            trust_proxy_headers=False,
        )
        self.assertEqual(k_untrusted_xff, "ip:127.0.0.1")

        k_trusted_xff = build_rate_limit_key(
            client_host="127.0.0.1",
            forwarded_for="203.0.113.9, 10.0.0.1",
            service_key_verified=False,
            trust_proxy_headers=True,
        )
        self.assertEqual(k_trusted_xff, "ip:203.0.113.9")

    @patch("app.main.arun_workflow", new_callable=AsyncMock)
    @patch("app.main.astream_workflow_events", side_effect=_ok_stream_events)
    def test_chat_and_stream_return_429_with_retry_after_and_health_bypassed(self, _mock_stream, mock_workflow):
        mock_workflow.return_value = _ok_workflow_state()
        with patch.object(settings, "RATE_LIMIT_ENABLED", True), \
             patch.object(settings, "RATE_LIMIT_MAX_REQUESTS", 2), \
             patch.object(settings, "RATE_LIMIT_WINDOW_SECONDS", 60):
            r1 = self.client.post("/api/v1/chat", json={"query": "Halo 1", "user_token": "tok-user-a"})
            r2 = self.client.post("/api/v1/chat/stream", json={"query": "Halo 2", "user_token": "tok-user-a"})
            r3 = self.client.post("/api/v1/chat", json={"query": "Halo 3", "user_token": "tok-user-a"})
            r4 = self.client.post("/api/v1/chat/stream", json={"query": "Halo 4", "user_token": "tok-user-a"})

            self.assertEqual(r1.status_code, 200)
            self.assertEqual(r2.status_code, 200)
            self.assertEqual(r3.status_code, 429)
            self.assertIn("Retry-After", r3.headers)
            self.assertGreaterEqual(int(r3.headers["Retry-After"]), 1)
            self.assertEqual(r4.status_code, 429)
            self.assertIn("Retry-After", r4.headers)

            # Different user token on same IP gets its own independent bucket
            r_other_user = self.client.post("/api/v1/chat", json={"query": "Halo", "user_token": "tok-user-b"})
            self.assertEqual(r_other_user.status_code, 200)

            # Unverified caller spoofing user_id cannot bypass quota on same IP/token
            r_spoof = self.client.post(
                "/api/v1/chat",
                json={"query": "Halo spoof", "user_token": "tok-user-a", "user_id": "random-id-999"},
            )
            self.assertEqual(r_spoof.status_code, 429)

            # Health endpoints are never rate-limited
            for _ in range(5):
                self.assertEqual(self.client.get("/health").status_code, 200)

    @patch("app.main.arun_workflow", new_callable=AsyncMock)
    def test_verified_user_id_isolates_users_behind_bff(self, mock_workflow):
        mock_workflow.return_value = _ok_workflow_state()
        with patch.object(settings, "INTERNAL_SERVICE_KEY", "srv-secret"), \
             patch.object(settings, "RATE_LIMIT_ENABLED", True), \
             patch.object(settings, "RATE_LIMIT_MAX_REQUESTS", 1), \
             patch.object(settings, "RATE_LIMIT_WINDOW_SECONDS", 60):
            r_u1_a = self.client.post(
                "/api/v1/chat",
                json={"query": "Halo", "user_id": "emp-1"},
                headers={"X-Internal-Service-Key": "srv-secret"},
            )
            r_u1_b = self.client.post(
                "/api/v1/chat",
                json={"query": "Halo lagi", "user_id": "emp-1"},
                headers={"X-Internal-Service-Key": "srv-secret"},
            )
            r_u2 = self.client.post(
                "/api/v1/chat",
                json={"query": "Halo", "user_id": "emp-2"},
                headers={"X-Internal-Service-Key": "srv-secret"},
            )
            self.assertEqual(r_u1_a.status_code, 200)
            self.assertEqual(r_u1_b.status_code, 429)
            self.assertEqual(r_u2.status_code, 200)


if __name__ == "__main__":
    unittest.main()
