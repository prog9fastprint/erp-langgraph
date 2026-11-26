"""
Unit tests for Hybrid Semantic NLU Router & Conservative Disambiguation (Task 2.2).
Covers MultiRouterDecision clarification validation, hard guardrails, explicit vs ambiguous
billing split, per-clause parallel hybrid retrieval, dynamic Top-K prompt construction,
multi-turn clarification resolution, and graceful failure fallback.
"""
import unittest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from app.graph.nodes.router import (
    ClarificationOption,
    MultiRouterDecision,
    PlannedToolCall,
    _try_route_contextual_analysis,
)
from app.graph.nodes.router_semantic import (
    aroute_semantic_nlu,
    aretrieve_candidate_tools,
    build_scoped_router_system_prompt,
    check_hard_guardrails,
    is_explicit_customer_billing_query,
    is_hard_write_block_query,
    match_strict_document_identifier,
    route_semantic_nlu_sync,
)
from app.graph.state import create_initial_state


def _make_mock_llm(decision: MultiRouterDecision) -> MagicMock:
    """Create a mock BaseChatModel whose with_structured_output returns decision."""
    structured = MagicMock()
    structured.ainvoke = AsyncMock(return_value=decision)
    structured.invoke = MagicMock(return_value=decision)
    model = MagicMock()
    model.with_structured_output = MagicMock(return_value=structured)
    return model


class TestMultiRouterDecisionClarification(unittest.TestCase):
    """Validate MultiRouterDecision clarification fields and Pydantic reconciliation."""

    def test_clarification_clears_tools_strips_emojis_and_caps_options(self):
        decision = MultiRouterDecision(
            domain="sales",
            query_mode="clarification",
            tool="get_sales_orders",
            planned_tools=[PlannedToolCall(tool="get_sales_orders", domain="sales")],
            needs_clarification=True,
            clarification_question="Order mana yang ingin Anda cek?",
            clarification_options=[
                ClarificationOption(label="📋 Sales Order ERP", query="Cek sales order minggu ini"),
                ClarificationOption(label="🛒 Order Marketplace", query="Cek order marketplace minggu ini"),
                ClarificationOption(label="Duplikat", query="cek sales order minggu ini!"),
                ClarificationOption(label="Pesanan Kilat", query="Cek pesanan kilat Shopee", icon="bad<tag>"),
                ClarificationOption(label="Opsi 4", query="Cek omzet minggu ini"),
                ClarificationOption(label="Opsi 5", query="Cek produk terlaris minggu ini"),
            ],
        )
        self.assertEqual(decision.domain, "clarification")
        self.assertEqual(decision.query_mode, "clarification")
        self.assertTrue(decision.needs_clarification)
        self.assertIsNone(decision.tool)
        self.assertEqual(decision.planned_tools, [])
        self.assertEqual(len(decision.clarification_options), 4)
        self.assertEqual(decision.clarification_options[0].label, "Sales Order ERP")
        self.assertEqual(decision.clarification_options[1].label, "Order Marketplace")
        self.assertEqual(decision.clarification_options[2].icon, "fa-arrow-right")

    def test_clarification_populates_default_question_when_blank(self):
        decision = MultiRouterDecision(
            domain="clarification",
            query_mode="clarification",
            needs_clarification=True,
            clarification_question="   ",
        )
        self.assertTrue(decision.clarification_question)
        self.assertIn("perjelas", decision.clarification_question.lower())


class TestSemanticHardGuardrails(unittest.TestCase):
    """Validate the 4 zero-risk deterministic hard guardrails + Branch 6/7 splits."""

    def test_greeting_help_meta_route_without_llm(self):
        for q in ("Halo!", "Selamat pagi admin", "Bisa bantu apa saja?", "yang kamu pakai perusahaan mana?"):
            with self.subTest(query=q):
                state = create_initial_state(q)
                res = check_hard_guardrails(q, state, [q])
                self.assertIsNotNone(res)
                self.assertEqual(res["detected_domain"], "general")
                self.assertEqual(res["query_mode"], "general")
                self.assertEqual(res["active_tools"], [])

    def test_write_block_vs_diagnostic_read_questions(self):
        self.assertTrue(is_hard_write_block_query("update stok ke shopee sekarang"))
        self.assertTrue(is_hard_write_block_query("merge order kilat shopee"))
        self.assertTrue(is_hard_write_block_query("buatkan jurnal memorial baru"))
        # Diagnostic read questions must NOT be blocked
        self.assertFalse(is_hard_write_block_query("kenapa log update stok shopee gagal?"))
        self.assertFalse(is_hard_write_block_query("simulasi push stok ke shopee SKU HVS-A4"))

    def test_explicit_vs_ambiguous_customer_billing_split(self):
        """Explicit customer billing stays hard-guardrailed; ambiguous 'cek tagihan <nama>' reaches LLM."""
        self.assertTrue(is_explicit_customer_billing_query("cek tagihan customer Budi"))
        self.assertTrue(is_explicit_customer_billing_query("faktur penjualan toko mainstore"))
        self.assertTrue(is_explicit_customer_billing_query("cek piutang pelanggan bulan ini"))
        self.assertTrue(is_explicit_customer_billing_query("kuitansi transaksi Andi"))

        # Ambiguous name without vendor/customer context -> False (reaches Scoped LLM for clarification)
        self.assertFalse(is_explicit_customer_billing_query("coba cek tagihan Frendy Krisnata di mainstore"))
        self.assertFalse(is_explicit_customer_billing_query("cek invoice Hendra"))
        # Explicit vendor -> False
        self.assertFalse(is_explicit_customer_billing_query("cek tagihan vendor PT Sumber Makmur"))

    def test_strict_document_identifier_fast_path(self):
        self.assertEqual(match_strict_document_identifier("cek surat jalan WH/OUT/00123"), ("get_transfers", "inventory"))
        self.assertEqual(match_strict_document_identifier("detail MS/SO/0585308"), ("get_sales_order_detail", "sales"))
        self.assertEqual(match_strict_document_identifier("lihat jurnal JE/2026/0009"), ("get_journal_entries", "accounting"))
        self.assertEqual(match_strict_document_identifier("cek voucher PC/2026/00002"), ("fetch_vendor_invoices", "accounting"))
        self.assertIsNone(match_strict_document_identifier("cek stok kertas A4"))

        # Multi-clause query with a doc ID must NOT short-circuit and drop the second intent
        multi_q = "cek detail MS/SO/0585308 dan cek stok tinta sublim"
        state = create_initial_state(multi_q)
        res = check_hard_guardrails(multi_q, state, ["cek detail MS/SO/0585308", "cek stok tinta sublim"])
        self.assertIsNone(res)


class TestSemanticNLURouterPipeline(unittest.IsolatedAsyncioTestCase):
    """End-to-end unit tests for aroute_semantic_nlu & route_semantic_nlu_sync with mocked LLM."""

    async def test_single_intent_slang_retrieves_scoped_candidates_and_routes(self):
        """Slang/paraphrase query retrieves Top-K candidates and routes via Scoped LLM with internal_no_stream tag."""
        q = "pesanan kilat shopee nyangkut hari ini"
        state = create_initial_state(q)
        decision = MultiRouterDecision(
            domain="marketplace",
            query_mode="live_erp",
            tool="get_kilat_order_diagnostics",
            planned_tools=[PlannedToolCall(tool="get_kilat_order_diagnostics", domain="marketplace", sub_query=q)],
        )
        mock_llm = _make_mock_llm(decision)

        res = await aroute_semantic_nlu(
            state=state,
            user_query=q,
            get_model_fn=lambda: mock_llm,
            contextual_analysis_fn=_try_route_contextual_analysis,
        )

        self.assertIsNotNone(res)
        self.assertEqual(res["detected_domain"], "marketplace")
        self.assertEqual(res["detected_tool"], "get_kilat_order_diagnostics")
        self.assertEqual(res["query_mode"], "live_erp")
        self.assertFalse(res["needs_clarification"])

        # Verify Scoped LLM prompt contained retrieved Top-K card & internal_no_stream tag
        structured = mock_llm.with_structured_output.return_value
        call_args = structured.ainvoke.await_args
        sent_messages = call_args.args[0]
        self.assertIn("get_kilat_order_diagnostics", sent_messages[0].content)
        self.assertEqual(call_args.kwargs.get("config"), {"tags": ["internal_no_stream"]})

    async def test_multi_intent_cross_domain_per_clause_retrieval(self):
        """Multi-intent query retrieves candidates from each clause in parallel and plans multiple tools."""
        q = "cek stok kertas A4 dan cek laporan laba rugi bulan ini"
        state = create_initial_state(q)
        decision = MultiRouterDecision(
            domain="inventory",
            query_mode="live_erp",
            planned_tools=[
                PlannedToolCall(tool="get_stock_level", domain="inventory", sub_query="cek stok kertas A4"),
                PlannedToolCall(tool="get_profit_and_loss", domain="accounting", sub_query="cek laporan laba rugi bulan ini"),
            ],
        )
        mock_llm = _make_mock_llm(decision)

        res = await aroute_semantic_nlu(
            state=state,
            user_query=q,
            get_model_fn=lambda: mock_llm,
        )

        self.assertTrue(res["is_multi_tool"])
        self.assertEqual(res["detected_tools"], ["get_stock_level", "get_profit_and_loss"])
        self.assertEqual(res["detected_domains"], ["inventory", "accounting"])
        self.assertEqual(len(res["active_tools"]), 2)
        sent_sys = mock_llm.with_structured_output.return_value.ainvoke.await_args.args[0][0].content
        self.assertIn("get_stock_level", sent_sys)
        self.assertIn("get_profit_and_loss", sent_sys)

    async def test_conservative_clarification_and_multiturn_resolution(self):
        """Turn 1 triggers clarification chips; Turn 2 short reply enriches retrieval and resolves directly."""
        # Turn 1: Ambiguous query -> clarification
        q1 = "cek order minggu ini"
        state1 = create_initial_state(q1)
        clarify_decision = MultiRouterDecision(
            domain="clarification",
            query_mode="clarification",
            needs_clarification=True,
            clarification_question="Order mana yang ingin Anda periksa untuk minggu ini?",
            clarification_options=[
                ClarificationOption(label="Sales Order ERP", query="Cek daftar sales order minggu ini"),
                ClarificationOption(label="Order Marketplace", query="Cek order marketplace minggu ini"),
            ],
        )
        mock_llm1 = _make_mock_llm(clarify_decision)
        res1 = await aroute_semantic_nlu(state=state1, user_query=q1, get_model_fn=lambda: mock_llm1)

        self.assertTrue(res1["needs_clarification"])
        self.assertEqual(res1["query_mode"], "clarification")
        self.assertEqual(res1["verification_status"], "CLARIFICATION")
        self.assertEqual(res1["active_tools"], [])
        self.assertEqual(len(res1["clarification_options"]), 2)

        # Turn 2: Short reply ("yang marketplace") enriches retrieval with prior query and resolves directly
        q2 = "yang marketplace"
        state2 = create_initial_state(
            user_query=q2,
            messages=[
                HumanMessage(content=q1),
                AIMessage(content="Order mana yang ingin Anda periksa untuk minggu ini?"),
                HumanMessage(content=q2),
            ],
        )
        cands2 = await aretrieve_candidate_tools(q2, [q2], state2)
        self.assertIn("get_marketplace_orders", [c["tool_name"] for c in cands2])

        resolved_decision = MultiRouterDecision(
            domain="marketplace",
            query_mode="live_erp",
            tool="get_marketplace_orders",
            planned_tools=[PlannedToolCall(tool="get_marketplace_orders", domain="marketplace", sub_query=q2)],
        )
        mock_llm2 = _make_mock_llm(resolved_decision)
        res2 = await aroute_semantic_nlu(state=state2, user_query=q2, get_model_fn=lambda: mock_llm2)

        self.assertFalse(res2["needs_clarification"])
        self.assertEqual(res2["detected_tool"], "get_marketplace_orders")
        self.assertEqual(res2["query_mode"], "live_erp")

    async def test_short_followup_injects_prior_detected_tool_into_candidates(self):
        """Branch follow-up ('di mainstore') injects prior_detected_tool so LLM always sees it."""
        q = "di mainstore"
        state = create_initial_state(
            user_query=q,
            messages=[
                HumanMessage(content="cek saldo kas hari ini"),
                AIMessage(content="Saldo kas pusat Rp 50.000.000"),
                HumanMessage(content=q),
            ],
        )
        state["prior_detected_tool"] = "get_account_balance"
        state["prior_detected_domain"] = "accounting"

        cands = await aretrieve_candidate_tools(q, [q], state)
        self.assertIn("get_account_balance", [c["tool_name"] for c in cands])

    async def test_post_llm_general_guardrail_coerces_offtopic_to_out_of_scope(self):
        """If LLM mistakenly outputs domain='general' for trivia, deterministic guardrail coerces it to out_of_scope."""
        q = "jelaskan teori relativitas Einstein"
        state = create_initial_state(q)
        bad_decision = MultiRouterDecision(domain="general", query_mode="general")
        mock_llm = _make_mock_llm(bad_decision)

        res = await aroute_semantic_nlu(state=state, user_query=q, get_model_fn=lambda: mock_llm)
        self.assertEqual(res["detected_domain"], "out_of_scope")
        self.assertEqual(res["query_mode"], "out_of_scope")
        self.assertEqual(res["active_tools"], [])

    async def test_returns_none_when_llm_unavailable_or_raises(self):
        """Missing or failing LLM returns None so caller degrades cleanly to legacy_regex."""
        q = "cek omzet bulan ini"
        state = create_initial_state(q)

        res_no_llm = await aroute_semantic_nlu(state=state, user_query=q, get_model_fn=lambda: None)
        self.assertIsNone(res_no_llm)

        failing_llm = MagicMock()
        failing_structured = MagicMock()
        failing_structured.ainvoke = AsyncMock(side_effect=RuntimeError("429 rate limited"))
        failing_llm.with_structured_output.return_value = failing_structured
        res_err = await aroute_semantic_nlu(state=state, user_query=q, get_model_fn=lambda: failing_llm)
        self.assertIsNone(res_err)

    def test_sync_semantic_router_and_dynamic_prompt_structure(self):
        """Sync counterpart routes via structured invoke and prompt enforces conservative disambiguation rules."""
        q = "cek SOP retur barang rusak"
        state = create_initial_state(q)
        decision = MultiRouterDecision(domain="knowledge", query_mode="vector_rag", rag_query="SOP retur barang rusak")
        mock_llm = _make_mock_llm(decision)

        res = route_semantic_nlu_sync(state=state, user_query=q, get_model_fn=lambda: mock_llm)
        self.assertEqual(res["detected_domain"], "knowledge")
        self.assertEqual(res["query_mode"], "vector_rag")
        self.assertEqual(res["rag_query"], "SOP retur barang rusak")
        self.assertEqual(res["rag_queries"], ["SOP retur barang rusak"])
        self.assertEqual(res["active_tools"], [])

        prompt = build_scoped_router_system_prompt([], state, [q])
        self.assertIn("ATURAN KLARIFIKASI KONSERVATIF", prompt)
        self.assertIn("DILARANG KERAS meminta klarifikasi hanya karena cabang/company atau rentang tanggal", prompt)

    def test_semantic_router_output_schema_is_compact_without_reasoning(self):
        """SemanticRouterOutput omits unused reasoning/scalar tool fields to minimize router token latency."""
        import json
        from app.graph.nodes.router_semantic import SemanticRouterOutput

        lean_schema = json.dumps(SemanticRouterOutput.model_json_schema())
        legacy_schema = json.dumps(MultiRouterDecision.model_json_schema())
        self.assertNotIn('"reasoning"', lean_schema)
        self.assertLess(len(lean_schema), len(legacy_schema) * 0.6)


class TestRouterModeDispatcherAndShadow(unittest.IsolatedAsyncioTestCase):
    """Test suite for Task 2.3 NLU_ROUTER_MODE dispatcher (legacy_regex, semantic_llm, shadow) and E2E workflow."""

    async def test_legacy_regex_mode_skips_semantic_router(self):
        from unittest.mock import patch
        from app.core.config import settings
        from app.graph.nodes.router import arouter_node, router_node

        state = create_initial_state("cek laporan laba rugi bulan ini")
        with patch.object(settings, "NLU_ROUTER_MODE", "legacy_regex"), \
             patch("app.graph.nodes.router_semantic.aroute_semantic_nlu", new_callable=AsyncMock) as mock_sem_async, \
             patch("app.graph.nodes.router_semantic.route_semantic_nlu_sync") as mock_sem_sync:
            res_async = await arouter_node(state)
            res_sync = router_node(state)

        mock_sem_async.assert_not_awaited()
        mock_sem_sync.assert_not_called()
        self.assertEqual(res_async["detected_tool"], "get_profit_and_loss")
        self.assertEqual(res_sync["detected_tool"], "get_profit_and_loss")

    async def test_semantic_llm_mode_dispatches_and_falls_back_on_llm_unavailable(self):
        from unittest.mock import patch
        from app.core.config import settings
        from app.graph.nodes.router import arouter_node, router_node

        q = "cek order minggu ini"
        clarify = MultiRouterDecision(
            domain="clarification",
            query_mode="clarification",
            needs_clarification=True,
            clarification_question="Order ERP atau Order Marketplace?",
            clarification_options=[
                ClarificationOption(label="Sales Order ERP", query="Cek sales order minggu ini"),
                ClarificationOption(label="Order Marketplace", query="Cek order marketplace minggu ini"),
            ],
        )
        mock_llm = _make_mock_llm(clarify)
        state = create_initial_state(q)

        with patch.object(settings, "NLU_ROUTER_MODE", "semantic_llm"):
            res_async = await arouter_node(state, llm=mock_llm)
            res_sync = router_node(state, llm=mock_llm)

        self.assertTrue(res_async["needs_clarification"])
        self.assertEqual(res_async["query_mode"], "clarification")
        self.assertTrue(res_sync["needs_clarification"])
        self.assertEqual(res_sync["query_mode"], "clarification")

        # When no LLM is available (returns None), semantic_llm falls back gracefully to legacy_regex
        fallback_state = create_initial_state("cek laba rugi bulan ini")
        with patch.object(settings, "NLU_ROUTER_MODE", "semantic_llm"):
            res_fb_async = await arouter_node(fallback_state, llm=None)
            res_fb_sync = router_node(fallback_state, llm=None)

        self.assertEqual(res_fb_async["detected_tool"], "get_profit_and_loss")
        self.assertEqual(res_fb_sync["detected_tool"], "get_profit_and_loss")

    async def test_shadow_mode_is_non_blocking_logs_mismatch_and_cleans_up_tasks(self):
        import asyncio
        from unittest.mock import patch
        from app.core.config import settings
        import app.graph.nodes.router as router_mod

        gate = asyncio.Event()

        async def _slow_semantic(**kwargs):
            await gate.wait()
            return {
                "detected_domain": "clarification",
                "detected_tool": None,
                "detected_tools": [],
                "query_mode": "clarification",
                "needs_clarification": True,
            }

        state = create_initial_state("cek daftar sales order hari ini")
        with patch.object(settings, "NLU_ROUTER_MODE", "shadow"), \
             patch.object(settings, "NLU_SHADOW_LOG_MISMATCH_ONLY", True), \
             patch("app.graph.nodes.router_semantic.aroute_semantic_nlu", side_effect=_slow_semantic), \
             self.assertLogs("router_node", level="INFO") as cm:
            # Must return legacy result immediately while shadow task is still blocked on gate
            res = await router_mod.arouter_node(state)
            self.assertEqual(res["detected_tool"], "get_sales_orders")
            self.assertFalse(res.get("needs_clarification", False))
            self.assertTrue(len(router_mod._SHADOW_ROUTER_TASKS) >= 1)

            # Release shadow task and wait for clean completion
            gate.set()
            pending = list(router_mod._SHADOW_ROUTER_TASKS)
            if pending:
                await asyncio.gather(*pending)

        self.assertEqual(len(router_mod._SHADOW_ROUTER_TASKS), 0)
        self.assertTrue(any("NLU_SHADOW_DIFF" in msg and '"mismatch": true' in msg for msg in cm.output))

    async def test_shadow_mode_handles_timeout_and_sync_skips_shadow(self):
        import asyncio
        from unittest.mock import patch
        from app.core.config import settings
        import app.graph.nodes.router as router_mod

        async def _hanging_semantic(**kwargs):
            await asyncio.sleep(2.0)
            return None

        state = create_initial_state("cek saldo kas hari ini")
        with patch.object(settings, "NLU_ROUTER_MODE", "shadow"), \
             patch.object(router_mod, "_SHADOW_ROUTER_TIMEOUT_SEC", 0.05), \
             patch("app.graph.nodes.router_semantic.aroute_semantic_nlu", side_effect=_hanging_semantic), \
             self.assertLogs("router_node", level="INFO") as cm:
            res = await router_mod.arouter_node(state)
            self.assertEqual(res["detected_tool"], "get_account_balance")
            pending = list(router_mod._SHADOW_ROUTER_TASKS)
            if pending:
                await asyncio.gather(*pending)

        self.assertEqual(len(router_mod._SHADOW_ROUTER_TASKS), 0)
        self.assertTrue(any("NLU_SHADOW_DIFF" in msg and '"status": "timeout"' in msg for msg in cm.output))

        # Sync router_node in shadow mode runs legacy_regex only and schedules zero tasks
        with patch.object(settings, "NLU_ROUTER_MODE", "shadow"), \
             patch("app.graph.nodes.router_semantic.route_semantic_nlu_sync") as mock_sync_sem:
            sync_res = router_mod.router_node(state)
        mock_sync_sem.assert_not_called()
        self.assertEqual(sync_res["detected_tool"], "get_account_balance")

    async def test_workflow_e2e_clarification_then_multiturn_resolution_in_semantic_mode(self):
        """Full StateGraph E2E: Turn 1 clarifies without hitting ERP; Turn 2 resets clarification and executes tool."""
        from unittest.mock import patch
        from langgraph.checkpoint.memory import MemorySaver
        from app.core.config import settings
        from app.graph.workflow import arun_workflow, create_workflow_graph
        from app.services.django_client import DjangoCoreClient

        clarify_decision = MultiRouterDecision(
            domain="clarification",
            query_mode="clarification",
            needs_clarification=True,
            clarification_question="Order mana yang ingin Anda cek untuk minggu ini?",
            clarification_options=[
                ClarificationOption(label="Sales Order ERP", query="Cek daftar sales order minggu ini"),
                ClarificationOption(label="Order Marketplace", query="Cek daftar order marketplace minggu ini"),
            ],
        )
        resolve_decision = MultiRouterDecision(
            domain="marketplace",
            query_mode="live_erp",
            tool="get_marketplace_orders",
            planned_tools=[
                PlannedToolCall(tool="get_marketplace_orders", domain="marketplace", sub_query="yang marketplace")
            ],
        )

        structured_mock = MagicMock()
        structured_mock.ainvoke = AsyncMock(side_effect=[clarify_decision, resolve_decision])
        router_llm = MagicMock()
        router_llm.with_structured_output.return_value = structured_mock

        mock_client = MagicMock(spec=DjangoCoreClient)
        mock_client.fetch_marketplace_orders = AsyncMock(
            return_value={
                "total_count": 1,
                "orders": [
                    {
                        "order_sn": "260320ABC123",
                        "marketplace": "shopee",
                        "status": "READY_TO_SHIP",
                        "sla_status": "due_soon",
                    }
                ],
            }
        )

        from app.graph.nodes.router import _prepare_router_context as real_prep

        def _prep_with_mock(state, llm, user_query):
            return real_prep(state=state, llm=llm or router_llm, user_query=user_query)

        graph = create_workflow_graph(checkpointer=MemorySaver())
        with patch.object(settings, "NLU_ROUTER_MODE", "semantic_llm"), \
             patch("app.graph.nodes.router._prepare_router_context", side_effect=_prep_with_mock), \
             patch("app.graph.nodes.tool_node.DjangoCoreClient", return_value=mock_client):
            res1 = await arun_workflow(
                user_query="cek order minggu ini",
                thread_id="semantic_clarify_e2e_thread",
                graph=graph,
            )
            self.assertTrue(res1["needs_clarification"])
            self.assertEqual(res1["query_mode"], "clarification")
            self.assertEqual(res1["verification_status"], "CLARIFICATION")
            self.assertEqual(res1["final_answer"], "Order mana yang ingin Anda cek untuk minggu ini?")
            self.assertEqual(len(res1["suggested_followups"]), 2)
            mock_client.fetch_marketplace_orders.assert_not_awaited()

            res2 = await arun_workflow(
                user_query="yang marketplace",
                thread_id="semantic_clarify_e2e_thread",
                graph=graph,
            )
            self.assertFalse(res2["needs_clarification"])
            self.assertEqual(res2["detected_tool"], "get_marketplace_orders")
            self.assertEqual(res2["verification_status"], "VALID")
            mock_client.fetch_marketplace_orders.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
