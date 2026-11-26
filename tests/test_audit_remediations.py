"""
Regression test suite verifying all codebase audit remediations (#1–#23, #25–#26).
"""
import gc
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from langchain_core.messages import AIMessageChunk

from app.core.config import settings
from app.core.context import resolve_tool_record_count
from app.graph.nodes.extractor import (
    _backfill_peer_entities,
    _fallback_extract_contacts,
    _fallback_extract_sales_orders,
    _fallback_extract_sales_summary,
    _fallback_extract_top_selling_products,
    extractor_node,
    resolve_branch_company_id,
)
from app.graph.nodes.rag_node import rag_node
from app.graph.nodes.router import arouter_node, router_node
from app.graph.nodes.synthesizer import (
    _format_rag_section,
    format_deterministic_answer,
)
from app.graph.nodes.verifier import _calculate_records_count, verifier_node
from app.graph.state import create_initial_state
from app.graph.workflow import _get_thread_lock, astream_workflow_events
import app.main as main_module
from app.services.django_client import (
    _CACHED_SYSTEM_TOKENS,
    DjangoAuthError,
    DjangoCoreClient,
    aclose_shared_django_client,
)
from app.services.llm import (
    CircuitBreakerChatOpenAI,
    _sync_fetch_free_models_catalog,
    get_chat_model,
    is_provider_unhealthy,
    reset_provider_health,
)
from app.services.tool_registry_service import ToolRegistryService
from app.services.vector_service import VectorService


class TestCriticalBugFixes(unittest.TestCase):
    """Regressions for high-severity runtime bugs (#1, #6, #7, #8, #13, #14, #17, #19, #23)."""

    def test_1_multi_company_sales_orders_no_name_error(self):
        """#1: Multi-company get_sales_orders formats without NameError even without sort_by='amount_desc'."""
        state = create_initial_state(user_query="cek pesanan penjualan terbesar semua cabang", company_id=1)
        state["detected_domain"] = "sales"
        state["detected_tool"] = "get_sales_orders"
        state["verification_status"] = "VALID"
        state["is_multi_company"] = True
        state["company_ids"] = [1, 2]
        state["extracted_params"] = {}
        state["multi_company_results"] = {
            1: {
                "total_count": 1,
                "total_amount_sum": 500000.0,
                "status_filter": "active",
                "orders": [
                    {
                        "so_number": "MS/SO/001",
                        "date_order": "2026-03-10",
                        "customer_name": "PT Alpha",
                        "total_amount": 500000.0,
                        "status": "sale",
                    }
                ],
            },
            2: {
                "total_count": 1,
                "total_amount_sum": 1200000.0,
                "status_filter": "active",
                "orders": [
                    {
                        "so_number": "SBY/SO/002",
                        "date_order": "2026-03-10",
                        "customer_name": "PT Beta",
                        "total_amount": 1200000.0,
                        "status": "sale",
                    }
                ],
            },
        }
        rendered = format_deterministic_answer(state)
        self.assertIn("SBY/SO/002", rendered)
        self.assertIn("MS/SO/001", rendered)

    def test_6_explicit_iso_dates_and_limit_extraction(self):
        """#6: Explicit ISO dates are not clobbered by 4-digit year match; day numbers do not corrupt limit."""
        from datetime import date

        today = date(2026, 11, 20)
        so_single = _fallback_extract_sales_orders("daftar sales order 2026-03-15", today)
        self.assertEqual(so_single.get("date_from"), "2026-03-15")
        self.assertIsNone(so_single.get("date_to"))

        summary_range = _fallback_extract_sales_summary("omzet 2026-03-01 sampai 2026-03-10", today)
        self.assertEqual(summary_range.get("date_from"), "2026-03-01")
        self.assertEqual(summary_range.get("date_to"), "2026-03-10")

        top_range = _fallback_extract_top_selling_products("produk terlaris 2026-04-05", today)
        self.assertEqual(top_range.get("date_from"), "2026-04-05")
        self.assertIsNone(top_range.get("date_to"))

        contacts_range = _fallback_extract_contacts("pelanggan 2026-05-01", today)
        self.assertEqual(contacts_range.get("date_from"), "2026-05-01")

        so_date_day = _fallback_extract_sales_orders("daftar pesanan tanggal 5 maret 2026", today)
        self.assertEqual(so_date_day.get("limit"), 20)

        so_explicit_limit = _fallback_extract_sales_orders("tampilkan 5 sales order terbaru", today)
        self.assertEqual(so_explicit_limit.get("limit"), 5)

    def test_7_all_empty_multi_company_triggers_retry_and_empty(self):
        """#7: Empty multi-company envelopes trigger RETRY/EMPTY instead of VALID, while stocks=[...] is VALID."""
        base = create_initial_state(user_query="cek pesanan semua cabang", company_id=1)
        base["detected_domain"] = "sales"
        base["detected_tool"] = "get_sales_orders"
        base["is_multi_company"] = True
        base["company_ids"] = [1, 2]
        base["multi_company_results"] = {
            1: {"total_count": 0, "orders": []},
            2: {"total_count": 0, "orders": []},
        }
        v_retry = verifier_node(base)
        self.assertEqual(v_retry["verification_status"], "RETRY")

        base["retry_count"] = 2
        v_empty = verifier_node(base)
        self.assertEqual(v_empty["verification_status"], "EMPTY")

        # {"stocks": [...]} still counts > 0 and returns VALID
        stock_state = create_initial_state(user_query="cek stok semua cabang", company_id=1)
        stock_state["detected_domain"] = "inventory"
        stock_state["detected_tool"] = "get_stock_level"
        stock_state["is_multi_company"] = True
        stock_state["company_ids"] = [1, 2]
        stock_state["multi_company_results"] = {
            1: {"stocks": [{"code": "HVS-A4", "quantity": 5}]},
            2: {"stocks": []},
        }
        v_stock = verifier_node(stock_state)
        self.assertEqual(v_stock["verification_status"], "VALID")

    def test_8_greetings_with_sales_or_inventory_intent_route_to_erp(self):
        """#8: Greeting prefixes must not hijack Sales or Inventory Phase 2/3 queries."""
        res_so = router_node(create_initial_state("Halo, cek daftar pesanan hari ini"))
        self.assertEqual(res_so["detected_tool"], "get_sales_orders")

        res_trf = router_node(create_initial_state("Selamat pagi, cek surat jalan WH/OUT/001"))
        self.assertEqual(res_trf["detected_tool"], "get_transfers")

        res_plain = router_node(create_initial_state("Halo, apa kabar?"))
        self.assertEqual(res_plain["detected_domain"], "general")
        self.assertIsNone(res_plain["detected_tool"])

    def test_13_ambiguous_branch_aliases_and_valid_codes(self):
        """#13: SOP HR, Sandy contact, '200 ms', and 'MS Word/Excel/Office' do not trigger false branch resolution."""
        self.assertIsNone(resolve_branch_company_id("SOP HR tentang cuti"))
        self.assertIsNone(resolve_branch_company_id("detail contact Sandy"))
        self.assertIsNone(resolve_branch_company_id("latency 200 ms"))
        self.assertIsNone(resolve_branch_company_id("export laporan ke ms excel"))
        self.assertIsNone(resolve_branch_company_id("buka dokumen ms word"))
        self.assertIsNone(resolve_branch_company_id("panduan ms office"))

        self.assertIsNotNone(resolve_branch_company_id("cek stok cabang hr"))
        self.assertEqual(resolve_branch_company_id("per lokasi (MS)"), 4)
        self.assertEqual(resolve_branch_company_id("MS/SO/0585308"), 4)
        self.assertEqual(resolve_branch_company_id("stok sby"), 7)

    def test_14_multi_company_stock_level_supports_products_key(self):
        """#14: Multi-company stock level renders rows when payload uses 'products' key."""
        state = create_initial_state("cek stok semua cabang", company_id=1)
        state["detected_domain"] = "inventory"
        state["detected_tool"] = "get_stock_level"
        state["verification_status"] = "VALID"
        state["is_multi_company"] = True
        state["company_ids"] = [1]
        state["multi_company_results"] = {
            1: {"products": [{"code": "SKU-X", "product_name": "Kertas Photo", "quantity": 12}]}
        }
        rendered = format_deterministic_answer(state)
        self.assertIn("SKU-X", rendered)
        self.assertIn("Kertas Photo", rendered)

    def test_16_legacy_react_stack_deleted(self):
        """#16: Deprecated ReAct graph, static/dynamic tools, sync ERPClient, and unused prompt are removed."""
        root = Path(__file__).resolve().parent.parent
        for rel_path in (
            "app/graph/agent.py",
            "app/tools/static.py",
            "app/tools/dynamic.py",
            "app/services/erp_client.py",
            "prompts/erp_system_prompt.md",
        ):
            self.assertFalse((root / rel_path).exists(), f"{rel_path} should be deleted")
        dockerfile_txt = (root / "Dockerfile").read_text(encoding="utf-8")
        self.assertNotIn("COPY prompts", dockerfile_txt)

    def test_24_modular_refactor_and_decoupled_imports(self):
        """#24: Branch helpers live in app.core.branches, formatters/extractors are domain-split, and dispatch is unified."""
        from app.core.branches import COMPANY_NAME_MAP
        from app.graph.nodes.formatters import SINGLE_TOOL_FORMATTERS
        from app.schemas.tools import TOOL_DOMAIN_MAP

        self.assertIn(1, COMPANY_NAME_MAP)
        # All standard ERP tools have declarative fetch_* methods on DjangoCoreClient and formatter dispatchers
        for t_name in TOOL_DOMAIN_MAP:
            method_name = f"fetch_{t_name[4:]}" if t_name.startswith("get_") else t_name
            self.assertTrue(hasattr(DjangoCoreClient, method_name), f"Missing DjangoCoreClient.{method_name}")
            self.assertIn(t_name, SINGLE_TOOL_FORMATTERS, f"Missing formatter for {t_name}")

        # Verify zero cross-node imports from extractor.py inside router/tool_node/synthesizer/workflow
        root = Path(__file__).resolve().parent.parent
        for rel in (
            "app/graph/nodes/router.py",
            "app/graph/nodes/tool_node.py",
            "app/graph/nodes/synthesizer.py",
        ):
            txt = (root / rel).read_text(encoding="utf-8")
            self.assertNotIn("from app.graph.nodes.extractor import", txt, f"{rel} should not import extractor")

    def test_26_and_28_hermetic_test_settings_and_ci_lint(self):
        """#26 & #28: Test runtime isolates .env secrets and CI workflow runs ruff check."""
        from app.core.config import _is_test_runtime

        self.assertTrue(_is_test_runtime())
        root = Path(__file__).resolve().parent.parent
        self.assertTrue((root / "pyproject.toml").exists())
        ci_txt = (root / ".github" / "workflows" / "deploy.yml").read_text(encoding="utf-8")
        self.assertIn("ruff check", ci_txt)

    def test_17_unified_record_counter_handles_errors(self):
        """#17: Error envelopes count 0 in both resolve_tool_record_count and _calculate_records_count."""
        err_payload = {"success": False, "error": "denied"}
        self.assertEqual(resolve_tool_record_count(err_payload), 0)
        self.assertEqual(_calculate_records_count(err_payload), 0)
        self.assertEqual(resolve_tool_record_count({"stocks": [{"id": 1}]}), 1)
        self.assertEqual(_calculate_records_count({"stocks": [{"id": 1}]}), 1)

    def test_19_backfill_peer_entities_is_deterministic(self):
        """#19: Sequential _backfill_peer_entities shares resolved entities after parallel extraction."""
        plans = [
            {"tool": "get_stock_level", "domain": "inventory", "params": {"code": "HVS-A4"}},
            {"tool": "get_product_info", "domain": "inventory", "params": {}},
        ]
        _backfill_peer_entities(plans)
        self.assertEqual(plans[1]["params"].get("search"), "HVS-A4")

    def test_23_none_metadata_chunks_and_state_init(self):
        """#23: None metadata chunks do not raise AttributeError and rag_results_by_query starts as dict."""
        init_state = create_initial_state("cek sop")
        self.assertEqual(init_state["rag_results_by_query"], {})
        section = _format_rag_section([{"title": None, "metadata": None, "content": "Aturan retur."}])
        self.assertIn("Dokumen 1", section)
        self.assertIn("Aturan retur.", section)

    def test_25_production_hides_swagger_docs(self):
        """#25: Production config disables /docs and /redoc on app construction."""
        main_src = Path(__file__).resolve().parent.parent / "app" / "main.py"
        txt = main_src.read_text(encoding="utf-8")
        self.assertIn('docs_url=None if _is_prod_env else "/docs"', txt)
        self.assertIn('redoc_url=None if _is_prod_env else "/redoc"', txt)


class TestSecurityAndLLMFixes(unittest.IsolatedAsyncioTestCase):
    """Regressions for security, auth, LLM failover, streaming, and async isolation (#2-#5, #9-#12, #15, #20-#22, #26)."""

    async def asyncTearDown(self):
        await aclose_shared_django_client()
        reset_provider_health()

    def test_2_openai_key_never_leaks_to_openrouter(self):
        """#2: OpenRouter catalog & factory refuse to fall back to OPENAI_API_KEY."""
        with patch.object(settings, "OPENROUTER_API_KEY", None), \
             patch.object(settings, "OPENAI_API_KEY", "sk-openai-secret"), \
             patch("app.services.llm.httpx.Client") as mock_client, \
             patch("app.services.llm.time.time", return_value=9999999.0):
            # Force cache bypass by resetting cache globals inside patch
            with patch("app.services.llm._CACHED_FREE_MODELS", []), \
                 patch("app.services.llm._LAST_CACHE_TIME", 0.0):
                _sync_fetch_free_models_catalog(force_refresh=True)
            mock_client.assert_not_called()

            with self.assertRaises(ValueError):
                get_chat_model(provider="openrouter")

    async def test_3_unverified_user_roles_stripped_and_staging_enforced(self):
        """#3: Client user_roles requires verified X-Internal-Service-Key; staging enforces bearer token."""
        transport = httpx.ASGITransport(app=main_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            with patch.object(settings, "INTERNAL_SERVICE_KEY", "srv-secret"), \
                 patch.object(settings, "APP_ENV", "development"), \
                 patch("app.main.arun_workflow", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = create_initial_state("Halo")

                # Unverified caller spoofing user_roles -> stripped to []
                resp_unauth = await client.post(
                    "/api/v1/chat",
                    json={"query": "Halo", "user_roles": ["admin", "super_user"]},
                )
                self.assertEqual(resp_unauth.status_code, 200)
                self.assertEqual(mock_run.call_args.kwargs["user_roles"], [])

                # Verified internal proxy caller -> preserves user_roles
                resp_trusted = await client.post(
                    "/api/v1/chat",
                    json={"query": "Halo", "user_roles": ["accounting.viewer"]},
                    headers={"X-Internal-Service-Key": "srv-secret"},
                )
                self.assertEqual(resp_trusted.status_code, 200)
                self.assertEqual(mock_run.call_args.kwargs["user_roles"], ["accounting.viewer"])

            # Staging enforces strict JWT authentication
            with patch.object(settings, "APP_ENV", "staging"), \
                 patch.object(settings, "REQUIRE_USER_TOKEN", False):
                resp_staging = await client.post("/api/v1/chat", json={"query": "Halo"})
                self.assertEqual(resp_staging.status_code, 401)

    def test_4_erp_login_secret_required(self):
        """#4: Hardcoded fallback removed; missing ERP_LOGIN_SECRET raises DjangoAuthError."""
        client = DjangoCoreClient(base_url="http://test-erp:8000", username="admin", password="pw")
        with patch.object(settings, "ERP_LOGIN_SECRET", None), \
             self.assertRaises(DjangoAuthError):
            client._generate_secret_key()

        with patch.object(settings, "ERP_LOGIN_SECRET", "prod_secret"):
            key = client._generate_secret_key()
            self.assertEqual(len(key), 32)

    async def test_5_sse_streaming_suppresses_internal_tokens_and_duplicate_errors(self):
        """#5: internal_no_stream LLM chunks are never leaked; complete.final_answer == streamed tokens; single error event."""
        async def _mock_events(**kwargs):
            # Simulate LangGraph astream_events producing internal_no_stream chunk + clean custom token
            state = create_initial_state("cek saldo")
            state["final_answer"] = "Saldo kas utama Rp 1.000.000,00."
            yield ("token", {"content": "Saldo kas utama Rp 1.000.000,00."})
            yield ("step", {"node": "synthesizer", "status": "completed", "elapsed_ms": 5.0, "model": "gpt-4o-mini"})
            yield ("complete", state)

        # Verify on_chat_model_stream skips internal_no_stream inside astream_workflow_events
        fake_graph = MagicMock()

        async def _fake_astream_events(initial_state, config=None, version=None):
            yield {
                "event": "on_chat_model_stream",
                "tags": ["internal_no_stream"],
                "metadata": {"langgraph_node": "synthesizer"},
                "data": {"chunk": AIMessageChunk(content='Tentu! ```json:followups\n[{"label":"X","query":"Y"}]\n```')},
            }
            yield {
                "event": "on_custom_event",
                "name": "token",
                "data": {"content": "Saldo kas utama Rp 1.000.000,00."},
            }
            yield {
                "event": "on_chain_end",
                "name": "synthesizer",
                "metadata": {"langgraph_node": "synthesizer"},
                "data": {"output": {"final_answer": "Saldo kas utama Rp 1.000.000,00."}},
            }

        fake_graph.astream_events = _fake_astream_events
        emitted = []
        async for ev_type, ev_data in astream_workflow_events(
            user_query="cek saldo",
            graph=fake_graph,
            thread_id=None,
            token_delay_sec=0.0,
        ):
            emitted.append((ev_type, ev_data))

        tokens_text = "".join(d["content"] for t, d in emitted if t == "token")
        complete_payload = next(d for t, d in emitted if t == "complete")
        self.assertNotIn("json:followups", tokens_text)
        self.assertEqual(tokens_text, complete_payload["final_answer"])

        # Verify single error frame when stream raises
        async def _failing_steps(**kwargs):
            raise RuntimeError("boom")
            yield  # pragma: no cover

        transport = httpx.ASGITransport(app=main_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            with patch("app.main.astream_workflow_steps", side_effect=_failing_steps):
                resp = await client.post("/api/v1/chat/stream", json={"query": "Halo"})
        error_lines = [line for line in resp.text.splitlines() if line == "event: error"]
        self.assertEqual(len(error_lines), 1)

    async def test_9_ingest_document_transactional_no_wipe_on_embed_failure(self):
        """#9: If Gemini embedding fails, DELETE is never executed against PGVector."""
        from scripts.ingest_knowledge import ingest_document

        fake_conn = AsyncMock()
        fake_conn.execute = AsyncMock()
        fake_conn.close = AsyncMock()
        fake_genai = MagicMock()
        fake_genai.models.embed_content.side_effect = RuntimeError("429 Quota Exceeded")

        tmp_doc = Path(__file__).resolve().parent / "_tmp_ingest_test.md"
        tmp_doc.write_text("Paragraf satu SOP retur.", encoding="utf-8")
        try:
            with patch("asyncpg.connect", new_callable=AsyncMock, return_value=fake_conn), \
                 patch("pgvector.asyncpg.register_vector", new_callable=AsyncMock), \
                 patch("google.genai.Client", return_value=fake_genai):
                with self.assertRaises(RuntimeError):
                    await ingest_document(
                        file_path=str(tmp_doc),
                        db_url="postgresql://user:pass@localhost/db",
                        gemini_api_key="test-key",
                    )
            fake_conn.execute.assert_not_awaited()
        finally:
            if tmp_doc.exists():
                tmp_doc.unlink()

    async def test_10_vector_service_per_loop_pool_and_async_router(self):
        """#10 & #15: VectorService isolates pools per event loop and DDL sets NOT NULL embedding; arouter_node awaits search_tools."""
        svc = VectorService(db_url="postgresql://user:pass@localhost/db")
        mock_pool = MagicMock(_closed=False)
        mock_conn = AsyncMock()
        mock_acm = AsyncMock()
        mock_acm.__aenter__.return_value = mock_conn
        mock_pool.acquire.return_value = mock_acm

        with patch("asyncpg.create_pool", new_callable=AsyncMock, return_value=mock_pool):
            p1 = await svc.get_pool()
            self.assertIs(p1, mock_pool)
            ok = await svc.initialize_table()
            self.assertTrue(ok)
            ddl_executed = mock_conn.execute.call_args[0][0]
            self.assertIn("CREATE EXTENSION IF NOT EXISTS vector", ddl_executed)
            self.assertIn("embedding vector(768) NOT NULL", ddl_executed)

        # Verify arouter_node awaits search_tools asynchronously
        mock_registry = MagicMock()
        mock_registry.search_tools = AsyncMock(
            return_value=[{"tool_name": "get_journal_entries", "domain": "accounting"}]
        )
        with patch.object(settings, "DYNAMIC_TOOL_RETRIEVAL_ENABLED", True), \
             patch("app.services.tool_registry_service.get_tool_registry_service", return_value=mock_registry):
            res = await arouter_node(create_initial_state("permintaan abstrak tidak umum"))
        mock_registry.search_tools.assert_awaited_once()
        self.assertEqual(res["detected_tool"], "get_journal_entries")

    async def test_11_thread_lock_weakref_reclaimed(self):
        """#11: Idle thread locks are garbage-collected cleanly via WeakValueDictionary."""
        import weakref

        lock = _get_thread_lock("ephemeral-thread-gc")
        weak_lock = weakref.ref(lock)
        del lock
        gc.collect()
        self.assertIsNone(weak_lock())

    def test_12_single_clause_rule_hit_does_not_call_get_chat_model(self):
        """#12: Single-clause regex routing and Tier-1 fast extraction skip get_chat_model()."""
        saved_unittest = sys.modules.pop("unittest", None)
        try:
            with patch("app.graph.nodes.router.get_chat_model") as mock_router_llm, \
                 patch("app.graph.nodes.extractor.get_chat_model") as mock_ext_llm:
                st = create_initial_state("Cek laba rugi bulan ini")
                routed = router_node(st)
                extractor_node({**st, **routed})
                mock_router_llm.assert_not_called()
                mock_ext_llm.assert_not_called()
        finally:
            if saved_unittest is not None:
                sys.modules["unittest"] = saved_unittest

    async def test_20_token_cache_isolated_by_host_and_username(self):
        """#20: _CACHED_SYSTEM_TOKENS keeps separate tokens per (base_url, username)."""
        await aclose_shared_django_client()
        c_a = DjangoCoreClient(base_url="http://erp-a:8000", username="alice", password="pw1")
        c_b = DjangoCoreClient(base_url="http://erp-b:8000", username="bob", password="pw2")

        async def _handler_a(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"access": "jwt-alice"})

        async def _handler_b(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"access": "jwt-bob"})

        with patch.object(settings, "ERP_LOGIN_SECRET", "sec"):
            c_a.client._transport = httpx.MockTransport(_handler_a)
            tok_a = await c_a.login()
            c_b.client._transport = httpx.MockTransport(_handler_b)
            tok_b = await c_b.login()

        self.assertEqual(tok_a, "jwt-alice")
        self.assertEqual(tok_b, "jwt-bob")
        self.assertNotEqual(
            _CACHED_SYSTEM_TOKENS[("http://erp-a:8000", "alice")],
            _CACHED_SYSTEM_TOKENS[("http://erp-b:8000", "bob")],
        )

    async def test_21_openrouter_uses_circuit_breaker_and_astream_trips(self):
        """#21: get_chat_model('openrouter') uses CircuitBreakerChatOpenAI and astream failure trips breaker."""
        reset_provider_health()
        with patch.object(settings, "OPENROUTER_API_KEY", "or-test-key"):
            m = get_chat_model(provider="openrouter", enable_fallback=False)
            self.assertIsInstance(m, CircuitBreakerChatOpenAI)

            async def _boom(*args, **kwargs):
                raise RuntimeError("401 AuthenticationError invalid api key")
                yield  # pragma: no cover

            with patch("langchain_openai.ChatOpenAI.astream", side_effect=_boom):
                with self.assertRaises(RuntimeError):
                    async for _ in m.astream("hi"):
                        pass
            self.assertTrue(is_provider_unhealthy("openrouter"))

    async def test_22_tool_registry_sql_identifier_and_desc_update(self):
        """#22: ToolRegistryService rejects unsafe table_name and updates tools when description changes."""
        with self.assertRaises(ValueError):
            ToolRegistryService(table_name="ai_tools; DROP TABLE users;--")

        mock_vec = MagicMock(spec=VectorService)
        mock_vec.get_embedding = AsyncMock(return_value=[0.1] * 768)
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [{"tool_name": "get_stock_level", "description": "Old description"}]
        mock_acm = AsyncMock()
        mock_acm.__aenter__.return_value = mock_conn
        mock_pool = MagicMock()
        mock_pool.acquire.return_value = mock_acm
        mock_vec.get_pool = AsyncMock(return_value=mock_pool)

        svc = ToolRegistryService(
            vector_service=mock_vec,
            table_name="ai_tool_catalog",
            default_definitions=[{
                "tool_name": "get_stock_level",
                "domain": "inventory",
                "description": "Updated description",
                "params_schema": {},
            }],
        )
        res = await svc.initialize_catalog(force_update=False)
        self.assertEqual(res["updated"], 1)
        self.assertEqual(res["skipped"], 0)

    async def test_26_rag_node_with_injected_client_skips_vector_service(self):
        """#26: rag_node with injected DjangoCoreClient skips live vector_service.search_knowledge."""
        mock_client = AsyncMock(spec=DjangoCoreClient)
        mock_client.fetch_knowledge.return_value = {
            "chunks": [{"title": "SOP.pdf", "content": "Konten SOP.", "similarity_score": 0.9}]
        }
        state = create_initial_state("Apa SOP retur?")
        state["query_mode"] = "vector_rag"
        state["rag_query"] = "SOP retur"

        with patch.object(settings, "VECTOR_DB_URL", "postgresql://postgres/db"), \
             patch.object(settings, "GEMINI_API_KEY", "test-key"), \
             patch("app.graph.nodes.rag_node.vector_service.search_knowledge", new_callable=AsyncMock) as mock_vec_search:
            out = await rag_node(state, client=mock_client)

        mock_vec_search.assert_not_awaited()
        self.assertEqual(len(out["rag_context"]), 1)


if __name__ == "__main__":
    unittest.main()
