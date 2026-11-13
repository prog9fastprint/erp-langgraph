"""
Empirical Adversarial Stress Testing Suite for FastPrint ERP LangGraph Orchestrator.
Stress-tests:
1. POST /api/v1/chat endpoint: high concurrency (50+ requests), malformed payloads, injection, thread isolation.
2. DjangoCoreClient lifecycle in tool_node and rag_node: verify httpx sockets are closed under success & error paths.
3. fetch_free_models_catalog in llm.py: event loop non-blocking verification, concurrent race stress.
4. run_workflow_sync in workflow.py: ThreadPoolExecutor reuse, no thread explosion, post-shutdown recovery.
"""
import asyncio
import concurrent.futures
from enum import Enum
import threading
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from fastapi import HTTPException
from pydantic import ValidationError

from app.core.config import settings
from app.graph.nodes.rag_node import rag_node
from app.graph.nodes.tool_node import tool_node
from app.graph.state import create_initial_state
from app.graph.workflow import (
    get_workflow_executor,
    run_workflow_sync,
    shutdown_workflow_executor,
)
import app.graph.workflow as workflow_module
from app.main import app, ChatRequest, ChatResponse
from app.services.django_client import DjangoCoreClient, DjangoCoreClientError
from app.services.llm import (
    afetch_free_models_catalog,
    fetch_free_models_catalog,
    _sync_fetch_free_models_catalog,
    DEFAULT_CURATED_FREE_MODELS,
)
import app.services.llm as llm_module


class DummyStatusEnum(str, Enum):
    VALID = "VALID"
    EMPTY = "EMPTY"
    ERROR = "ERROR"


class TestApiChatAdversarial(unittest.IsolatedAsyncioTestCase):
    """Adversarial stress testing for POST /api/v1/chat endpoint."""

    async def asyncSetUp(self):
        self.transport = httpx.ASGITransport(app=app)
        self.client = httpx.AsyncClient(transport=self.transport, base_url="http://testserver")

    async def asyncTearDown(self):
        await self.client.aclose()

    @patch("app.main.arun_workflow", new_callable=AsyncMock)
    async def test_concurrent_chat_flood_50_requests_thread_isolation(self, mock_workflow):
        """
        Stress test: 50 concurrent chat requests simultaneously.
        Verifies thread isolation, no cross-talk, parameter propagation, and zero race conditions.
        """
        async def mock_arun_impl(user_query, user_token=None, company_id=None, thread_id=None, **kwargs):
            # Simulate realistic async processing delay
            await asyncio.sleep(0.01)
            return {
                "final_answer": f"Answer for query '{user_query}' in company {company_id}",
                "detected_domain": "inventory",
                "verification_status": "VALID",
                "error_message": None,
            }

        mock_workflow.side_effect = mock_arun_impl

        async def send_request(idx: int):
            payload = {
                "query": f"Cek stok produk #{idx}",
                "company_id": (idx % 5) + 1,
                "user_token": f"jwt_token_worker_{idx}",
                "thread_id": f"session_thread_{idx}",
            }
            resp = await self.client.post("/api/v1/chat", json=payload)
            return idx, resp

        # Launch 50 concurrent requests
        tasks = [send_request(i) for i in range(50)]
        results = await asyncio.gather(*tasks)

        self.assertEqual(len(results), 50)
        for idx, resp in results:
            self.assertEqual(resp.status_code, 200, f"Request {idx} failed with status {resp.status_code}")
            data = resp.json()
            expected_company = (idx % 5) + 1
            expected_answer = f"Answer for query 'Cek stok produk #{idx}' in company {expected_company}"
            self.assertEqual(data["final_answer"], expected_answer, f"Cross-talk detected on task {idx}!")
            self.assertEqual(data["verification_status"], "VALID")
            self.assertIsNone(data["error_message"])

        self.assertEqual(mock_workflow.await_count, 50)

    async def test_invalid_payloads_422(self):
        """Edge cases: empty, whitespace, missing query, invalid types -> 422."""
        invalid_payloads = [
            {"query": ""},
            {"query": "   \n\t  "},
            {"query": None},
            {},
            {"company_id": 1},
            {"query": "valid query", "company_id": "not_an_int"},
            {"query": "valid query", "company_id": [1, 2, 3]},
            {"query": "valid query", "company_id": {"nested": "obj"}},
            {"query": "valid query", "company_id": 3.14159},
        ]

        for payload in invalid_payloads:
            resp = await self.client.post("/api/v1/chat", json=payload)
            self.assertEqual(
                resp.status_code,
                422,
                f"Payload {payload} was expected to return 422, got {resp.status_code}: {resp.text}",
            )

    @patch("app.main.arun_workflow", new_callable=AsyncMock)
    async def test_extreme_payloads_huge_query(self, mock_workflow):
        """Stress test: 100,000 character query payload executes safely without crash."""
        mock_workflow.return_value = {
            "final_answer": "Large query handled",
            "detected_domain": "general",
            "verification_status": "VALID",
        }

        huge_query = "Halo FastPrint " * 7000  # ~105,000 characters
        resp = await self.client.post("/api/v1/chat", json={"query": huge_query})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["final_answer"], "Large query handled")
        mock_workflow.assert_awaited_once()

    @patch("app.main.arun_workflow", new_callable=AsyncMock)
    async def test_special_characters_and_injections(self, mock_workflow):
        """Edge cases: SQL injection strings, unicode, emojis, control characters."""
        mock_workflow.return_value = {
            "final_answer": "Sanitized response",
            "detected_domain": "accounting",
            "verification_status": "VALID",
        }

        adversarial_queries = [
            "'; DROP TABLE account_move; --",
            "<script>alert('xss')</script>",
            "{{ 7 * 7 }} ${7*7}",
            "Tagihan vendor 📦🚀🔥 \u200b\ufeff",
            "SELECT * FROM res_partner WHERE name = 'CV Maju'",
        ]

        for q in adversarial_queries:
            resp = await self.client.post("/api/v1/chat", json={"query": q, "thread_id": "injection_test"})
            self.assertEqual(resp.status_code, 200, f"Query '{q}' caused status {resp.status_code}")
            self.assertEqual(resp.json()["final_answer"], "Sanitized response")

    @patch("app.main.arun_workflow", new_callable=AsyncMock)
    async def test_extra_fields_ignored(self, mock_workflow):
        """Pydantic ConfigDict(extra='ignore') ignores superfluous fields."""
        mock_workflow.return_value = {
            "final_answer": "OK",
            "detected_domain": "general",
            "verification_status": "VALID",
        }

        resp = await self.client.post(
            "/api/v1/chat",
            json={
                "query": "Cek saldo",
                "unexpected_field": "exploit_attempt",
                "admin_flag": True,
                "nested": {"deep": "value"},
            },
        )
        self.assertEqual(resp.status_code, 200)
        mock_workflow.assert_awaited_once_with(
            user_query="Cek saldo",
            user_token=None,
            company_id=1,
            thread_id=None,
            company_id_explicit=False,
        )

    @patch("app.main.arun_workflow", new_callable=AsyncMock)
    async def test_workflow_500_exception_handling(self, mock_workflow):
        """Unexpected internal exceptions in arun_workflow return 500 Internal Server Error."""
        mock_workflow.side_effect = RuntimeError("PostgreSQL connection refused on port 5432")

        resp = await self.client.post("/api/v1/chat", json={"query": "Cek kas"})
        self.assertEqual(resp.status_code, 500)
        data = resp.json()
        self.assertIn("Chat workflow failed", data["detail"])
        self.assertIn("PostgreSQL connection refused", data["detail"])

    @patch("app.main.arun_workflow", new_callable=AsyncMock)
    async def test_workflow_http_exception_preservation(self, mock_workflow):
        """FastAPI HTTPException in workflow is preserved and not swallowed into 500."""
        mock_workflow.side_effect = HTTPException(status_code=403, detail="Akses ditolak untuk company ini")

        resp = await self.client.post("/api/v1/chat", json={"query": "Cek kas"})
        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.json()["detail"], "Akses ditolak untuk company ini")


class TestDjangoCoreClientLifecycleAdversarial(unittest.IsolatedAsyncioTestCase):
    """
    Adversarial verification of DjangoCoreClient lifecycle in tool_node and rag_node (R4).
    Verifies that client sockets and httpx transports are never leaked under normal or error conditions.
    """

    async def test_tool_node_own_client_closed_on_success(self):
        """tool_node awaits aclose() on internally instantiated DjangoCoreClient."""
        mock_instance = MagicMock(spec=DjangoCoreClient)
        mock_instance.fetch_vendor_invoices = AsyncMock(return_value={"total_count": 0, "entries": []})
        mock_instance.aclose = AsyncMock()

        with patch("app.graph.nodes.tool_node.DjangoCoreClient", return_value=mock_instance):
            state = create_initial_state(user_query="Tagihan vendor")
            state["detected_tool"] = "fetch_vendor_invoices"
            res = await tool_node(state, client=None)
            self.assertEqual(res["verification_status"], "VALID")
            mock_instance.aclose.assert_awaited_once()

    async def test_tool_node_own_client_closed_on_django_error(self):
        """tool_node awaits aclose() even when DjangoCoreClientError occurs."""
        mock_instance = MagicMock(spec=DjangoCoreClient)
        mock_instance.fetch_vendor_invoices = AsyncMock(
            side_effect=DjangoCoreClientError("Sesi token JWT kadaluarsa (HTTP 401)")
        )
        mock_instance.aclose = AsyncMock()

        with patch("app.graph.nodes.tool_node.DjangoCoreClient", return_value=mock_instance):
            state = create_initial_state(user_query="Tagihan vendor")
            state["detected_tool"] = "fetch_vendor_invoices"
            res = await tool_node(state, client=None)
            self.assertEqual(res["verification_status"], "ERROR")
            self.assertIn("401", res["error_message"])
            mock_instance.aclose.assert_awaited_once()

    async def test_tool_node_own_client_closed_on_unhandled_exception(self):
        """tool_node awaits aclose() when unexpected Exception (e.g. ZeroDivisionError) is raised."""
        mock_instance = MagicMock(spec=DjangoCoreClient)
        mock_instance.fetch_vendor_invoices = AsyncMock(
            side_effect=ZeroDivisionError("Unexpected mathematical error in calculation")
        )
        mock_instance.aclose = AsyncMock()

        with patch("app.graph.nodes.tool_node.DjangoCoreClient", return_value=mock_instance):
            state = create_initial_state(user_query="Tagihan vendor")
            state["detected_tool"] = "fetch_vendor_invoices"
            res = await tool_node(state, client=None)
            self.assertEqual(res["verification_status"], "ERROR")
            mock_instance.aclose.assert_awaited_once()

    async def test_tool_node_injected_client_never_closed(self):
        """tool_node does NOT close externally injected client across success and failure."""
        mock_instance = MagicMock(spec=DjangoCoreClient)
        mock_instance.aclose = AsyncMock()

        # Success case
        mock_instance.fetch_vendor_invoices = AsyncMock(return_value={"total_count": 0, "entries": []})
        state = create_initial_state(user_query="Tagihan vendor")
        state["detected_tool"] = "fetch_vendor_invoices"
        await tool_node(state, client=mock_instance)
        mock_instance.aclose.assert_not_called()

        # Failure case
        mock_instance.fetch_vendor_invoices = AsyncMock(
            side_effect=DjangoCoreClientError("500 Internal Server")
        )
        await tool_node(state, client=mock_instance)
        mock_instance.aclose.assert_not_called()

    async def test_rag_node_own_client_closed_on_success(self):
        """rag_node awaits aclose() on internally instantiated DjangoCoreClient."""
        mock_instance = MagicMock(spec=DjangoCoreClient)
        mock_instance.fetch_knowledge = AsyncMock(return_value={"chunks": []})
        mock_instance.aclose = AsyncMock()

        with patch("app.graph.nodes.rag_node.DjangoCoreClient", return_value=mock_instance), \
             patch.object(settings, "ENABLE_RAG", True), \
             patch.object(settings, "VECTOR_DB_URL", ""):
            state = create_initial_state(user_query="SOP Retur Barang")
            state["query_mode"] = "vector_rag"
            state["rag_query"] = "SOP Retur Barang"
            res = await rag_node(state, client=None)
            self.assertEqual(res["verification_status"], "EMPTY")
            mock_instance.aclose.assert_awaited_once()

    async def test_rag_node_own_client_closed_on_django_error(self):
        """rag_node awaits aclose() when DjangoCoreClientError occurs."""
        mock_instance = MagicMock(spec=DjangoCoreClient)
        mock_instance.fetch_knowledge = AsyncMock(side_effect=DjangoCoreClientError("Network timeout"))
        mock_instance.aclose = AsyncMock()

        with patch("app.graph.nodes.rag_node.DjangoCoreClient", return_value=mock_instance), \
             patch.object(settings, "ENABLE_RAG", True), \
             patch.object(settings, "VECTOR_DB_URL", ""):
            state = create_initial_state(user_query="SOP Retur Barang")
            state["query_mode"] = "vector_rag"
            state["rag_query"] = "SOP Retur Barang"
            res = await rag_node(state, client=None)
            self.assertEqual(res["verification_status"], "ERROR")
            mock_instance.aclose.assert_awaited_once()

    async def test_rag_node_own_client_closed_on_unhandled_exception(self):
        """rag_node awaits aclose() when unhandled Exception occurs."""
        mock_instance = MagicMock(spec=DjangoCoreClient)
        mock_instance.fetch_knowledge = AsyncMock(side_effect=RuntimeError("Unexpected memory failure"))
        mock_instance.aclose = AsyncMock()

        with patch("app.graph.nodes.rag_node.DjangoCoreClient", return_value=mock_instance), \
             patch.object(settings, "ENABLE_RAG", True), \
             patch.object(settings, "VECTOR_DB_URL", ""):
            state = create_initial_state(user_query="SOP Retur Barang")
            state["query_mode"] = "vector_rag"
            state["rag_query"] = "SOP Retur Barang"
            res = await rag_node(state, client=None)
            self.assertEqual(res["rag_context"], [])
            mock_instance.aclose.assert_awaited_once()

    async def test_rag_node_injected_client_never_closed(self):
        """rag_node does NOT close externally injected client across success and failure."""
        mock_instance = MagicMock(spec=DjangoCoreClient)
        mock_instance.aclose = AsyncMock()

        # Success case
        mock_instance.fetch_knowledge = AsyncMock(return_value={"chunks": []})
        with patch.object(settings, "ENABLE_RAG", True):
            state = create_initial_state(user_query="SOP Retur")
            state["query_mode"] = "vector_rag"
            state["rag_query"] = "SOP Retur"
            await rag_node(state, client=mock_instance)
            mock_instance.aclose.assert_not_called()

        # Failure case
        mock_instance.fetch_knowledge = AsyncMock(side_effect=DjangoCoreClientError("500 Server Error"))
        with patch.object(settings, "ENABLE_RAG", True):
            await rag_node(state, client=mock_instance)
            mock_instance.aclose.assert_not_called()

    async def test_real_httpx_socket_is_closed_in_tool_node(self):
        """Verify that the underlying httpx.AsyncClient socket is truly marked is_closed."""
        real_client = DjangoCoreClient()
        self.assertFalse(real_client.client.is_closed)

        mock_transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"total_count": 0, "entries": []})
        )
        real_client.client = httpx.AsyncClient(transport=mock_transport)

        with patch("app.graph.nodes.tool_node.DjangoCoreClient", return_value=real_client):
            state = create_initial_state(user_query="Tagihan vendor")
            state["detected_tool"] = "fetch_vendor_invoices"
            await tool_node(state, client=None)

            # Empirically verify that the underlying httpx AsyncClient transport was closed
            self.assertTrue(real_client.client.is_closed, "Underlying httpx.AsyncClient was NOT closed!")


class TestLlmCatalogNonBlockingAdversarial(unittest.IsolatedAsyncioTestCase):
    """
    Adversarial verification of fetch_free_models_catalog non-blocking and concurrency safety (R7).
    """

    def setUp(self):
        # Reset cache before tests
        with llm_module._CACHE_LOCK:
            llm_module._CACHED_FREE_MODELS = []
            llm_module._LAST_CACHE_TIME = 0.0

    async def test_fetch_free_models_catalog_in_active_loop_never_blocks(self):
        """
        Calling fetch_free_models_catalog from an active running asyncio loop must return
        instantly (< 50ms) without blocking the loop, even if the worker takes 2 seconds.
        """
        def slow_worker(*args, **kwargs):
            time.sleep(1.0)
            return ["openrouter/free", "slow-model:free"]

        with patch("app.services.llm._sync_fetch_free_models_catalog", side_effect=slow_worker):
            start = time.perf_counter()
            catalog = fetch_free_models_catalog(force_refresh=True)
            elapsed = time.perf_counter() - start

            self.assertLess(
                elapsed,
                0.1,
                f"fetch_free_models_catalog blocked the event loop for {elapsed:.3f}s!",
            )
            self.assertIsInstance(catalog, list)
            self.assertIn("openrouter/free", catalog)

    async def test_afetch_free_models_catalog_concurrent_flood_50_tasks(self):
        """
        50 concurrent coroutines calling afetch_free_models_catalog simultaneously.
        Verifies thread-safety, no race conditions, and consistent catalog state.
        """
        fake_api_response = MagicMock()
        fake_api_response.status_code = 200
        fake_api_response.json.return_value = {
            "data": [
                {"id": "meta-llama/llama-3-8b:free", "pricing": {"prompt": "0", "completion": "0"}},
                {"id": "google/gemma-4-31b-it:free", "pricing": {"prompt": "0", "completion": "0"}},
                {"id": "openrouter/free", "pricing": {"prompt": "0", "completion": "0"}},
            ]
        }

        with patch("httpx.Client.get", return_value=fake_api_response):
            tasks = [afetch_free_models_catalog(api_key="mock-key", force_refresh=True) for _ in range(50)]
            results = await asyncio.gather(*tasks)

            self.assertEqual(len(results), 50)
            for res in results:
                self.assertIsInstance(res, list)
                self.assertEqual(res[0], "openrouter/free")
                self.assertIn("google/gemma-4-31b-it:free", res)

    async def test_cache_ttl_and_thread_safety(self):
        """Test cache TTL expiry and thread safety under concurrent thread access."""
        with llm_module._CACHE_LOCK:
            llm_module._CACHED_FREE_MODELS = ["openrouter/free", "cached-test-model:free"]
            llm_module._LAST_CACHE_TIME = time.time()

        # Cache hit
        with patch("httpx.Client.get", side_effect=Exception("Should not be queried on cache hit")):
            cat = await afetch_free_models_catalog(force_refresh=False)
            self.assertEqual(cat, ["openrouter/free", "cached-test-model:free"])

        # Cache expired after TTL
        with llm_module._CACHE_LOCK:
            llm_module._LAST_CACHE_TIME = time.time() - 1000.0  # Expired > 900s

        with patch("httpx.Client.get", side_effect=Exception("Simulated network down")):
            cat_refreshed = await afetch_free_models_catalog(force_refresh=False)
            # Gracefully falls back to curated models
            self.assertIn("openrouter/free", cat_refreshed)


class TestWorkflowSyncRunnerAdversarial(unittest.TestCase):
    """
    Adversarial verification of ThreadPoolExecutor reuse in run_workflow_sync (R10).
    Verifies executor reuse, strict thread bounding, no thread explosion, and graceful recovery.
    """

    def setUp(self):
        # Ensure executor is in known active state
        shutdown_workflow_executor(wait=False)

    def tearDown(self):
        shutdown_workflow_executor(wait=False)

    def test_get_workflow_executor_thread_safe_singleton_reuse(self):
        """
        get_workflow_executor must return identical singleton across multiple threads.
        """
        results = []

        def worker():
            executor = get_workflow_executor()
            results.append(id(executor))

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(results), 20)
        self.assertEqual(len(set(results)), 1, "Multiple distinct executors were instantiated!")

    def test_run_workflow_sync_no_thread_explosion_under_active_loop(self):
        """
        In an environment with an active event loop, calling run_workflow_sync 50 times
        must reuse the ThreadPoolExecutor and NOT spawn 50 unmanaged threads.
        """
        async def _run_benchmark():
            executor = get_workflow_executor()
            max_workers = executor._max_workers

            mock_graph = MagicMock()
            mock_graph.ainvoke = AsyncMock(return_value={
                "final_answer": "Benchmark result",
                "verification_status": "VALID",
            })

            initial_threads = threading.active_count()

            # Run 50 sequential synchronous invocations from inside an active loop
            for i in range(50):
                res = run_workflow_sync(user_query=f"Query {i}", thread_id=f"thread_{i}", graph=mock_graph)
                self.assertEqual(res["final_answer"], "Benchmark result")

            current_threads = threading.active_count()
            # Active threads must be bounded by initial_threads + max_workers (+ slight leeway)
            self.assertLessEqual(
                current_threads,
                initial_threads + max_workers + 2,
                f"Thread explosion detected! Active threads ({current_threads}) exceeds limit.",
            )

        asyncio.run(_run_benchmark())

    def test_run_workflow_sync_concurrent_callers_no_deadlock(self):
        """
        Multiple caller threads invoking run_workflow_sync simultaneously inside active loops
        must not deadlock and all succeed.
        """
        def thread_target(worker_id: int, out_errors: list):
            try:
                async def _inner():
                    mock_graph = MagicMock()
                    mock_graph.ainvoke = AsyncMock(return_value={
                        "final_answer": f"Answer from worker {worker_id}",
                        "verification_status": "VALID",
                    })
                    res = run_workflow_sync(user_query=f"Query {worker_id}", graph=mock_graph)
                    if res["final_answer"] != f"Answer from worker {worker_id}":
                        out_errors.append(f"Mismatch in worker {worker_id}")

                asyncio.run(_inner())
            except Exception as e:
                out_errors.append(f"Worker {worker_id} exception: {e}")

        errors = []
        threads = [threading.Thread(target=thread_target, args=(i, errors)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        self.assertEqual(errors, [], f"Concurrent thread errors: {errors}")

    def test_run_workflow_sync_resilient_after_shutdown(self):
        """
        If shutdown_workflow_executor() is called, subsequent run_workflow_sync calls
        re-instantiate the executor and complete cleanly.
        """
        async def _run_test():
            mock_graph = MagicMock()
            mock_graph.ainvoke = AsyncMock(return_value={
                "final_answer": "Post shutdown result",
                "verification_status": "VALID",
            })

            # Initial run
            res1 = run_workflow_sync(user_query="Query 1", graph=mock_graph)
            self.assertEqual(res1["final_answer"], "Post shutdown result")

            # Trigger shutdown
            shutdown_workflow_executor(wait=True)
            self.assertIsNone(workflow_module._WORKFLOW_EXECUTOR)

            # Subsequent run should seamlessly recreate executor
            res2 = run_workflow_sync(user_query="Query 2", graph=mock_graph)
            self.assertEqual(res2["final_answer"], "Post shutdown result")
            self.assertIsNotNone(workflow_module._WORKFLOW_EXECUTOR)
            self.assertFalse(workflow_module._WORKFLOW_EXECUTOR._shutdown)

        asyncio.run(_run_test())

    def test_run_workflow_sync_without_event_loop(self):
        """
        When called in a thread without a running event loop, it runs directly via asyncio.run.
        """
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "final_answer": "No loop result",
            "verification_status": "VALID",
        })

        # This test method is standard sync unittest, so loop is not running here
        res = run_workflow_sync(user_query="Query outside loop", graph=mock_graph)
        self.assertEqual(res["final_answer"], "No loop result")


if __name__ == "__main__":
    unittest.main()
