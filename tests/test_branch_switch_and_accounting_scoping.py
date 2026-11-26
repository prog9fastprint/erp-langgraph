"""
Regression test suite for thread 63f1a35a-986f-49af-87d8-ea49c2d6234e:
  1. Contrastive branch correction ('kenapa masih ho bukanya aku minta mainstore' -> Main Store).
  2. Thread-sticky in-chat branch override against unchanged Django session company_id=3
     plus stateless history recovery after server restart / checkpoint eviction.
  3. Prior-tool state continuity across clarification turns.
  4. Semantic NLU Router deterministic branch-switch fast-path & branch-aware prompt/retrieval.
  5. Accounting extractor branch-search cleanup, cross-branch account_code guard,
     category backfill ('akun bank mainstore'), and retry relaxation.
  6. General Ledger company-scoped top-5 active account pre-lookup in tool_node.
  7. General Ledger / EMPTY branch labeling and active-branch chip query embedding.
"""
import unittest
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.core.branches import (
    is_cross_branch_account_code,
    resolve_all_branch_company_ids,
    resolve_branch_company_id,
    resolve_sticky_branch_from_history,
)
from app.core.config import settings
from app.graph.nodes.extractor import extractor_node
from app.graph.nodes.extractors.accounting import (
    _extract_gl_account_hint,
    _fallback_extract_account_balance,
)
from app.graph.nodes.router import (
    ClarificationOption,
    MultiRouterDecision,
    arouter_node,
)
from app.graph.nodes.router_semantic import (
    _build_retrieval_queries,
    build_scoped_router_system_prompt,
)
from app.graph.nodes.synthesizer import (
    _build_clarification_synthesis_output,
    _finalize_followups,
    format_deterministic_answer,
)
from app.graph.nodes.tool_node import tool_node
from app.graph.state import create_initial_state
from app.graph.workflow import (
    _prepare_workflow_execution,
    arun_workflow,
    create_workflow_graph,
)
from app.schemas.tools import AccountBalanceFilterSchema, GeneralLedgerFilterSchema
from app.services.django_client import DjangoCoreClient


class TestContrastiveBranchAndCoaHelpers(unittest.TestCase):
    """Validate branch alias coverage, contrastive branch filtering, COA suffixes, and history recovery."""

    def test_headoffice_and_mainstore_variants(self):
        self.assertEqual(resolve_branch_company_id("cek saldo headoffice"), 3)
        self.assertEqual(resolve_branch_company_id("cek saldo main-store"), 4)
        self.assertEqual(resolve_branch_company_id("coba cek di mainstore"), 4)

    def test_contrastive_branch_correction_excludes_rejected_branch(self):
        cids, is_all = resolve_all_branch_company_ids("kenapa masih ho bukanya aku minta mainstore")
        self.assertFalse(is_all)
        self.assertEqual(cids, [4])
        self.assertEqual(resolve_branch_company_id("kenapa masih ho bukanya aku minta mainstore"), 4)

        cids2, _ = resolve_all_branch_company_ids("bukan di ho tapi mainstore")
        self.assertEqual(cids2, [4])

        cids3, _ = resolve_all_branch_company_ids("kok masih head office padahal aku minta surabaya")
        self.assertEqual(cids3, [7])

    def test_genuine_multi_branch_comparison_remains_multi_company(self):
        cids, is_all = resolve_all_branch_company_ids("bandingkan laba rugi ho dan mainstore")
        self.assertFalse(is_all)
        self.assertEqual(cids, [3, 4])

    def test_cross_branch_account_code_detection(self):
        self.assertTrue(is_cross_branch_account_code("11204HO", 4))
        self.assertTrue(is_cross_branch_account_code("11219HO-1CNY", 4))
        self.assertFalse(is_cross_branch_account_code("11204HO", 3))
        self.assertFalse(is_cross_branch_account_code("40000MS", 4))
        self.assertTrue(is_cross_branch_account_code("40000MS", 3))
        self.assertFalse(is_cross_branch_account_code("111100", 4))

    def test_resolve_sticky_branch_from_history(self):
        # In-chat switch followed by clarification -> recovers Main Store (4)
        msgs_clarify = [
            HumanMessage(content="Tampilkan buku besar akun kas untuk kuartal ini"),
            AIMessage(content="### Buku Besar Akun (General Ledger) (Cabang: Head Office) — Periode: 2026-10-01 s/d 2026-11-30"),
            HumanMessage(content="coba cek di mainstore"),
            AIMessage(content="Akun buku besar apa yang ingin Anda lihat?"),
        ]
        self.assertEqual(resolve_sticky_branch_from_history(msgs_clarify), 4)

        # Newer AI report on a different branch (Surabaya) -> returns None
        msgs_changed = [
            HumanMessage(content="coba cek di mainstore"),
            AIMessage(content="### Informasi Saldo Chart of Account (Accounting Balance) (Cabang: Surabaya)"),
        ]
        self.assertIsNone(resolve_sticky_branch_from_history(msgs_changed))

        # Newer user message expanded to all branches -> returns None
        msgs_all = [
            HumanMessage(content="coba cek di mainstore"),
            HumanMessage(content="bandingkan semua cabang"),
        ]
        self.assertIsNone(resolve_sticky_branch_from_history(msgs_all))


class TestThreadStickyInChatBranchOverride(unittest.IsolatedAsyncioTestCase):
    """Validate thread-sticky text branch override when Django resends unchanged company_id=3."""

    async def test_in_chat_mainstore_sticks_across_unchanged_django_payload_and_respects_dropdown_switch(self):
        observed_company_ids = []

        async def mock_fetch_pnl(params=None, user_token=None, company_id=None):
            observed_company_ids.append(("pnl", company_id))
            return {
                "date_from": "2026-01-01",
                "date_to": "2026-11-26",
                "company_id": company_id,
                "summary": {
                    "total_revenue": 1000000.0,
                    "cogs": 400000.0,
                    "gross_profit": 600000.0,
                    "gross_margin_percent": 60.0,
                    "operating_expenses": 100000.0,
                    "operating_profit": 500000.0,
                    "other_net": 0.0,
                    "net_profit": 500000.0,
                },
            }

        async def mock_fetch_balance(params=None, user_token=None, company_id=None):
            observed_company_ids.append(("bal", company_id))
            return {
                "company_id": company_id,
                "accounts": [
                    {
                        "account_code": f"11200_{company_id}",
                        "account_name": f"KAS-{company_id}",
                        "opening_balance": 0.0,
                        "period_debit": 500000.0,
                        "period_credit": 100000.0,
                        "ending_balance": 400000.0,
                    }
                ],
                "totals": {
                    "opening_balance": 0.0,
                    "period_debit": 500000.0,
                    "period_credit": 100000.0,
                    "ending_balance": 400000.0,
                },
            }

        mock_client = MagicMock(spec=DjangoCoreClient)
        mock_client.aclose = AsyncMock()
        mock_client.fetch_profit_and_loss = AsyncMock(side_effect=mock_fetch_pnl)
        mock_client.fetch_account_balance = AsyncMock(side_effect=mock_fetch_balance)

        isolated_graph = create_workflow_graph(checkpointer=MemorySaver())
        tid = "thread_sticky_mainstore_test"

        with patch("app.graph.nodes.tool_node.DjangoCoreClient", return_value=mock_client):
            # Turn 1: Initial query with Django session company_id=3 (Head Office)
            r1 = await arun_workflow(
                user_query="Laba rugi YTD",
                company_id=3,
                company_id_explicit=True,
                thread_id=tid,
                graph=isolated_graph,
            )
            self.assertEqual(r1["company_id"], 3)

            # Turn 2: User switches in chat to Mainstore while Django still sends company_id=3
            r2 = await arun_workflow(
                user_query="coba cek di mainstore",
                company_id=3,
                company_id_explicit=True,
                thread_id=tid,
                graph=isolated_graph,
            )
            self.assertEqual(r2["company_id"], 4)
            self.assertEqual(r2["company_scope_source"], "text")

            # Turn 3: Next top-level query without branch name, Django STILL sends company_id=3 -> stays Main Store (4)
            r3 = await arun_workflow(
                user_query="Tampilkan ringkasan saldo akun untuk kuartal ini",
                company_id=3,
                company_id_explicit=True,
                thread_id=tid,
                graph=isolated_graph,
            )
            self.assertEqual(r3["company_id"], 4)
            self.assertIn("Main Store", r3["final_answer"])

            # Turn 4: User switches ERP top-bar dropdown from 3 -> 7 (Surabaya) -> payload changed, switches to 7
            r4 = await arun_workflow(
                user_query="Berapa total saldo kas dan rekening bank aktif saat ini?",
                company_id=7,
                company_id_explicit=True,
                thread_id=tid,
                graph=isolated_graph,
            )
            self.assertEqual(r4["company_id"], 7)
            self.assertEqual(r4["company_scope_source"], "payload")
            self.assertIn("Surabaya", r4["final_answer"])

        self.assertEqual(observed_company_ids, [("pnl", 3), ("pnl", 4), ("bal", 4), ("bal", 7)])

    async def test_stateless_history_recovery_restores_mainstore_after_restart(self):
        isolated_graph = create_workflow_graph(checkpointer=MemorySaver())
        history_payload = [
            {"role": "user", "content": "Tampilkan buku besar akun kas untuk kuartal ini"},
            {"role": "assistant", "content": "### Buku Besar Akun (General Ledger) (Cabang: Head Office) — Periode: 2026-10-01 s/d 2026-11-30"},
            {"role": "user", "content": "coba cek di mainstore"},
            {"role": "assistant", "content": "Akun buku besar apa yang ingin Anda lihat?"},
        ]
        _, _, initial_state, _, _ = await _prepare_workflow_execution(
            user_query="Tampilkan ringkasan saldo akun untuk kuartal ini",
            company_id=3,
            company_id_explicit=True,
            thread_id="fresh_after_restart_tid",
            graph=isolated_graph,
            history=history_payload,
        )
        self.assertEqual(initial_state["company_id"], 4)
        self.assertEqual(initial_state["company_scope_source"], "text")
        self.assertEqual(initial_state["payload_company_id"], 3)

    async def test_prior_tool_preserved_across_clarification_turn(self):
        isolated_graph = create_workflow_graph(checkpointer=MemorySaver())
        cfg = {"configurable": {"thread_id": "clarification_continuity_tid"}}
        # Seed checkpoint mimicking Turn 490 (clarification right after get_general_ledger)
        isolated_graph.update_state(
            cfg,
            {
                "messages": [
                    HumanMessage(content="Tampilkan buku besar akun kas untuk kuartal ini"),
                    AIMessage(content="### Buku Besar Akun (General Ledger)"),
                    HumanMessage(content="coba cek di mainstore"),
                    AIMessage(content="Mohon perjelas permintaan Anda?"),
                ],
                "company_id": 4,
                "company_ids": [4],
                "payload_company_id": 3,
                "company_scope_source": "text",
                "detected_domain": "clarification",
                "detected_tool": None,
                "query_mode": "clarification",
                "tool_result": None,
                "prior_detected_domain": "accounting",
                "prior_detected_tool": "get_general_ledger",
                "prior_extracted_params": {"company_id": 3, "date_from": "2026-10-01", "date_to": "2026-11-30"},
            },
        )

        _, _, initial_state, _, _ = await _prepare_workflow_execution(
            user_query="mainstore",
            company_id=3,
            company_id_explicit=True,
            thread_id="clarification_continuity_tid",
            graph=isolated_graph,
        )
        self.assertEqual(initial_state["prior_detected_tool"], "get_general_ledger")
        self.assertEqual(initial_state["prior_detected_domain"], "accounting")
        self.assertEqual(initial_state["company_id"], 4)


class TestSemanticRouterBranchSwitchFastPath(unittest.IsolatedAsyncioTestCase):
    """Verify semantic_llm mode handles branch-switch follow-ups deterministically without warehouse clarification."""

    async def test_branch_switch_followups_bypass_llm_clarification(self):
        clarify_llm = MagicMock()
        clarify_structured = MagicMock()
        clarify_structured.ainvoke = AsyncMock(
            return_value=MultiRouterDecision(
                domain="clarification",
                query_mode="clarification",
                needs_clarification=True,
                clarification_question="Apakah gudang Mainstore?",
                clarification_options=[
                    ClarificationOption(label="Gudang", query="Tampilkan gudang Mainstore"),
                ],
            )
        )
        clarify_llm.with_structured_output = MagicMock(return_value=clarify_structured)

        base_history = [
            HumanMessage(content="Tampilkan buku besar akun kas untuk kuartal ini"),
            AIMessage(content="### Buku Besar Akun (General Ledger) (Cabang: Head Office) — Periode: 2026-10-01 s/d 2026-11-30"),
        ]

        for followup_q in (
            "coba cek di mainstore",
            "mainstore",
            "kenapa masih ho bukanya aku minta mainstore",
        ):
            with self.subTest(query=followup_q):
                clarify_structured.ainvoke.reset_mock()
                state = create_initial_state(
                    user_query=followup_q,
                    company_id=4,
                    messages=[*base_history, HumanMessage(content=followup_q)],
                )
                state["prior_detected_domain"] = "accounting"
                state["prior_detected_tool"] = "get_general_ledger"

                with patch.object(settings, "NLU_ROUTER_MODE", "semantic_llm"):
                    res = await arouter_node(state, llm=clarify_llm)

                clarify_structured.ainvoke.assert_not_awaited()
                self.assertEqual(res["detected_domain"], "accounting")
                self.assertEqual(res["detected_tool"], "get_general_ledger")
                self.assertEqual(res["query_mode"], "live_erp")
                self.assertFalse(res.get("needs_clarification"))
                self.assertEqual(res["company_id"], 4)
                self.assertEqual(res["company_ids"], [4])
                self.assertFalse(res["is_multi_company"])
                self.assertIn("Main Store", res["user_query"])
                self.assertIn("buku besar akun kas", res["user_query"].lower())

    def test_scoped_prompt_and_retrieval_strip_branch_bias(self):
        state = create_initial_state(
            user_query="akun bank mainstore",
            company_id=4,
            company_scope_source="text",
        )
        prompt = build_scoped_router_system_prompt([], state, ["akun bank mainstore"])
        self.assertIn("Main Store (company_id: 4)", prompt)
        self.assertIn("NAMA CABANG/PERUSAHAAN ERP", prompt)

        queries, _ = _build_retrieval_queries("akun bank mainstore", ["akun bank mainstore"], state)
        self.assertEqual(queries[0].lower(), "akun bank")


class TestAccountingExtractorSanitizationAndGLPreLookup(unittest.IsolatedAsyncioTestCase):
    """Verify branch noise stripping, cross-branch account_code guard, retry relaxation, and GL pre-lookup."""

    def test_akun_bank_mainstore_strips_branch_search_drops_ho_code_and_sets_bank_and_cash(self):
        bad_llm_output = AccountBalanceFilterSchema(
            company_id=3,
            account_code="11204",
            search="Mainstore",
        )
        structured = MagicMock()
        structured.invoke = MagicMock(return_value=bad_llm_output)
        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(return_value=structured)

        state = create_initial_state(
            user_query="akun bank mainstore",
            company_id=4,
            messages=[
                HumanMessage(content="Tampilkan buku besar akun bank"),
                AIMessage(content="#### Akun: `11204HO` — BANK MAYBANK-HO"),
                HumanMessage(content="akun bank mainstore"),
            ],
        )
        state["detected_domain"] = "accounting"
        state["detected_tool"] = "get_account_balance"
        state["prior_extracted_params"] = {"company_id": 3, "account_code": "11204HO"}

        out = extractor_node(state, llm=mock_llm, ref_date=date(2026, 11, 26))
        params = out["extracted_params"]
        self.assertEqual(params["company_id"], 4)
        self.assertNotIn("search", params)
        self.assertNotIn("account_code", params)
        self.assertEqual(params.get("account_type"), "Bank and Cash")

    def test_generic_gl_bank_drops_hallucinated_ho_account_code_but_keeps_explicit_code(self):
        # 1. Generic 'akun bank' must drop hallucinated '11204HO' from history
        bad_gl_output = GeneralLedgerFilterSchema(
            company_id=4,
            account_code="11204HO",
            date_from="2026-10-01",
            date_to="2026-11-30",
        )
        structured = MagicMock()
        structured.invoke = MagicMock(return_value=bad_gl_output)
        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(return_value=structured)

        state = create_initial_state(
            user_query="Tampilkan buku besar akun bank untuk periode 2026-10-01 s/d 2026-11-30 di Main Store",
            company_id=4,
        )
        state["detected_domain"] = "accounting"
        state["detected_tool"] = "get_general_ledger"

        out = extractor_node(state, llm=mock_llm, ref_date=date(2026, 11, 26))
        self.assertNotIn("account_code", out["extracted_params"])
        self.assertEqual(out["extracted_params"]["company_id"], 4)

        # 2. Explicit account code typed by user must be preserved
        state_explicit = create_initial_state(
            user_query="Tampilkan buku besar akun 11204HO untuk periode 2026-10-01 s/d 2026-11-30",
            company_id=3,
        )
        state_explicit["detected_domain"] = "accounting"
        state_explicit["detected_tool"] = "get_general_ledger"

        out_explicit = extractor_node(state_explicit, llm=mock_llm, ref_date=date(2026, 11, 26))
        self.assertEqual(out_explicit["extracted_params"].get("account_code"), "11204HO")

    def test_retry_relaxation_drops_non_explicit_account_balance_filters(self):
        llm_out = AccountBalanceFilterSchema(
            company_id=4,
            account_code="11204",
            search="Maybank",
            non_zero_only=True,
        )
        structured = MagicMock()
        structured.invoke = MagicMock(return_value=llm_out)
        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(return_value=structured)

        state = create_initial_state(user_query="cek saldo mainstore", company_id=4)
        state["detected_domain"] = "accounting"
        state["detected_tool"] = "get_account_balance"
        state["retry_count"] = 2

        out = extractor_node(state, llm=mock_llm, ref_date=date(2026, 11, 26))
        self.assertNotIn("search", out["extracted_params"])
        self.assertNotIn("account_code", out["extracted_params"])
        self.assertNotIn("non_zero_only", out["extracted_params"])

    def test_gl_and_balance_category_hints(self):
        self.assertEqual(_fallback_extract_account_balance("akun bank mainstore").get("account_type"), "Bank and Cash")
        self.assertEqual(_extract_gl_account_hint("Tampilkan buku besar akun kas untuk kuartal ini")["mode"], "kas")
        self.assertEqual(_extract_gl_account_hint("Tampilkan buku besar akun bank periode ini")["mode"], "bank")
        self.assertEqual(_extract_gl_account_hint("Buku besar bank BCA bulan ini")["search"], "BCA")

    async def test_tool_node_gl_category_pre_lookup_resolves_top_5_active_branch_accounts(self):
        mock_client = MagicMock(spec=DjangoCoreClient)
        mock_client.fetch_account_balance = AsyncMock(
            return_value={
                "accounts": [
                    {"account_id": 10, "account_code": "11200MS", "account_name": "KAS-MS", "period_debit": 900.0, "period_credit": 100.0, "ending_balance": 800.0},
                    {"account_id": 11, "account_code": "11201MS", "account_name": "BANK BCA-MS", "period_debit": 5000.0, "period_credit": 1000.0, "ending_balance": 4000.0},
                    {"account_id": 12, "account_code": "11202MS", "account_name": "BANK MANDIRI-MS", "period_debit": 4000.0, "period_credit": 500.0, "ending_balance": 3500.0},
                    {"account_id": 13, "account_code": "11203MS", "account_name": "BANK BRI-MS", "period_debit": 3000.0, "period_credit": 200.0, "ending_balance": 2800.0},
                    {"account_id": 14, "account_code": "11204MS", "account_name": "BANK BNI-MS", "period_debit": 2000.0, "period_credit": 100.0, "ending_balance": 1900.0},
                    {"account_id": 15, "account_code": "11205MS", "account_name": "BANK CIMB-MS", "period_debit": 1000.0, "period_credit": 50.0, "ending_balance": 950.0},
                    {"account_id": 16, "account_code": "11206MS", "account_name": "BANK DANAMON-MS", "period_debit": 100.0, "period_credit": 0.0, "ending_balance": 100.0},
                ]
            }
        )

        fetched_codes = []

        async def mock_fetch_gl(params=None, user_token=None, company_id=None):
            code = (params or {}).get("account_code")
            fetched_codes.append((company_id, code))
            return {
                "date_from": "2026-10-01",
                "date_to": "2026-11-30",
                "accounts": [
                    {
                        "account_code": code,
                        "account_name": f"NAME-{code}",
                        "opening_balance": 0.0,
                        "period_debit": 100.0,
                        "period_credit": 0.0,
                        "closing_balance": 100.0,
                        "items": [],
                    }
                ],
            }

        mock_client.fetch_general_ledger = AsyncMock(side_effect=mock_fetch_gl)

        state = create_initial_state(
            user_query="Tampilkan buku besar akun bank untuk periode 2026-10-01 s/d 2026-11-30 di Main Store",
            company_id=4,
        )
        state["detected_domain"] = "accounting"
        state["detected_tool"] = "get_general_ledger"
        state["extracted_params"] = {"company_id": 4, "date_from": "2026-10-01", "date_to": "2026-11-30"}

        res = await tool_node(state, client=mock_client)
        self.assertEqual(res["verification_status"], "VALID")
        # Excludes KAS-MS and 6th least-active bank (11206MS); fetches top 5 Main Store banks
        self.assertEqual(
            [c for _, c in fetched_codes],
            ["11201MS", "11202MS", "11203MS", "11204MS", "11205MS"],
        )
        self.assertTrue(all(cid == 4 for cid, _ in fetched_codes))
        self.assertEqual(len(res["tool_result"]["accounts"]), 5)
        self.assertEqual(res["tool_result"]["company_name"], "Main Store")


class TestBranchFormattingAndChipEmbedding(unittest.TestCase):
    """Verify GL heading branch label, EMPTY branch hint, and active-branch chip query embedding."""

    def test_general_ledger_heading_and_empty_notice_show_branch(self):
        gl_state = create_initial_state(
            user_query="Tampilkan buku besar akun kas di Main Store",
            company_id=4,
            company_scope_source="text",
        )
        gl_state["detected_domain"] = "accounting"
        gl_state["detected_tool"] = "get_general_ledger"
        gl_state["verification_status"] = "VALID"
        gl_state["tool_result"] = {
            "company_id": 4,
            "company_name": "Main Store",
            "date_from": "2026-10-01",
            "date_to": "2026-11-30",
            "accounts": [
                {
                    "account_code": "11200MS",
                    "account_name": "KAS-MS",
                    "opening_balance": 0.0,
                    "period_debit": 100000.0,
                    "period_credit": 0.0,
                    "closing_balance": 100000.0,
                    "items": [],
                }
            ],
        }
        md = format_deterministic_answer(gl_state)
        self.assertIn("(Cabang: Main Store)", md)
        self.assertIn("11200MS", md)

        empty_state = create_initial_state(
            user_query="cek akun bank mainstore",
            company_id=4,
            company_scope_source="text",
        )
        empty_state["detected_domain"] = "accounting"
        empty_state["detected_tool"] = "get_account_balance"
        empty_state["verification_status"] = "EMPTY"
        empty_state["extracted_params"] = {"company_id": 4, "account_type": "Bank and Cash"}
        empty_md = format_deterministic_answer(empty_state)
        self.assertIn("Cabang: **Main Store**", empty_md)

    def test_chips_embed_active_branch_only_on_in_chat_override(self):
        state_override = create_initial_state(
            user_query="Tampilkan buku besar akun kas di Main Store",
            company_id=4,
            company_scope_source="text",
        )
        chips = _finalize_followups(
            [
                {"label": "Saldo Kas & Bank", "query": "Berapa total saldo kas dan rekening bank aktif saat ini?"},
                {"label": "Cek Semua Cabang", "query": "Cek saldo kas di semua cabang"},
                {"label": "Panduan AI", "query": "/scope"},
            ],
            user_query=state_override["user_query"],
            state=state_override,
        )
        self.assertEqual(
            chips[0]["query"],
            "Berapa total saldo kas dan rekening bank aktif saat ini di Main Store?",
        )
        self.assertEqual(chips[1]["query"], "Cek saldo kas di semua cabang")
        self.assertEqual(chips[2]["query"], "/scope")

        # Clarification chips also embed active branch when company_scope_source == 'text'
        clar_state = create_initial_state(
            user_query="Tampilkan data akuntansi untuk periode Kuartal Ini",
            company_id=4,
            company_scope_source="text",
        )
        clar_state["query_mode"] = "clarification"
        clar_state["needs_clarification"] = True
        clar_state["clarification_question"] = "Data akuntansi apa yang ingin Anda lihat?"
        clar_state["clarification_options"] = [
            {"label": "Laporan Laba Rugi", "query": "Tampilkan laporan laba rugi untuk kuartal ini"},
            {"label": "Ringkasan Saldo", "query": "Tampilkan ringkasan saldo akun Main Store untuk kuartal ini"},
        ]
        clar_out = _build_clarification_synthesis_output(clar_state)
        self.assertEqual(
            clar_out["suggested_followups"][0]["query"],
            "Tampilkan laporan laba rugi untuk kuartal ini di Main Store",
        )
        # Already mentions Main Store -> not duplicated
        self.assertEqual(
            clar_out["suggested_followups"][1]["query"],
            "Tampilkan ringkasan saldo akun Main Store untuk kuartal ini",
        )

        # Payload default scope (not in-chat override) leaves chip queries untouched
        state_payload = create_initial_state(
            user_query="Laba rugi YTD",
            company_id=3,
            company_scope_source="payload",
        )
        chips_payload = _finalize_followups(
            [{"label": "Saldo Kas & Bank", "query": "Berapa total saldo kas dan rekening bank aktif saat ini?"}],
            user_query=state_payload["user_query"],
            state=state_payload,
        )
        self.assertEqual(
            chips_payload[0]["query"],
            "Berapa total saldo kas dan rekening bank aktif saat ini?",
        )


if __name__ == "__main__":
    unittest.main()
