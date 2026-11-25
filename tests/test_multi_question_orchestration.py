"""
Comprehensive tests for multi-question orchestration, per-clause scoping,
smart same-tool merging/splitting, mixed domain/RAG/guardrail routing,
verbatim synthesis, and per-thread concurrency locking.
"""
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage

from app.graph.state import create_initial_state
from app.graph.nodes.router import (
    decompose_user_query,
    router_node,
)
from app.graph.nodes.extractor import extractor_node
from app.graph.nodes.tool_node import tool_node
from app.graph.nodes.rag_node import rag_node
from app.graph.nodes.verifier import verifier_node
from app.graph.nodes.synthesizer import (
    asynthesizer_node,
    format_deterministic_answer,
)
from app.graph.workflow import (
    arun_workflow,
    decide_after_router,
)
from app.services.django_client import DjangoCoreClient


class TestQueryDecompositionAndRouting(unittest.IsolatedAsyncioTestCase):
    async def test_punctuation_and_newline_multi_question_with_strict_clause_scoping(self):
        query = (
            "Berapa stok kertas A4 di cabang Surabaya? "
            "Dan berapa laba rugi bulan ini di cabang Jakarta?"
        )
        clauses = decompose_user_query(query)
        self.assertGreaterEqual(len(clauses), 2)

        state = create_initial_state(user_query=query, company_id=1)
        routed = router_node(state)

        self.assertTrue(routed["is_multi_tool"])
        active = routed["active_tools"]
        self.assertEqual(len(active), 2)
        self.assertEqual(active[0]["tool"], "get_stock_level")
        self.assertEqual(active[0]["company_id"], 7)  # Surabaya
        self.assertEqual(active[1]["tool"], "get_profit_and_loss")
        self.assertEqual(active[1]["company_id"], 8)  # Jakarta

        merged_state = {**state, **routed}
        extracted = extractor_node(merged_state)
        ext_tools = extracted["active_tools"]
        self.assertEqual(ext_tools[0]["company_id"], 7)
        self.assertEqual(ext_tools[1]["company_id"], 8)
        # Date isolation: stock query has no temporal expression so date_from/date_to must not bleed from Q2
        self.assertNotIn("date_from", ext_tools[0]["params"])
        self.assertIn("date_from", ext_tools[1]["params"])
        self.assertIn("date_to", ext_tools[1]["params"])

    async def test_newline_and_period_split_without_breaking_abbreviations(self):
        # Abbreviations like PT. and No. must NOT be split as separate questions
        clauses_abbrev = decompose_user_query("Cek tagihan vendor PT. Maju Jaya No. 12")
        self.assertEqual(len(clauses_abbrev), 1)

        clauses_nl = decompose_user_query("Cek stok Tinta Epson 664\nBerapa saldo kas kecil?")
        self.assertEqual(len(clauses_nl), 2)

        state = create_initial_state(user_query="Cek stok Tinta Epson 664\nBerapa saldo kas kecil?", company_id=1)
        routed = router_node(state)
        tools = [t["tool"] for t in routed["active_tools"]]
        self.assertEqual(tools, ["get_stock_level", "get_account_balance"])

    async def test_shared_leading_branch_prefix_inherits_to_clauses(self):
        query = "Untuk cabang Bandung: cek stok HVS A4 dan cek saldo bank BCA"
        state = create_initial_state(user_query=query, company_id=1)
        routed = router_node(state)
        active = routed["active_tools"]
        self.assertEqual(len(active), 2)
        self.assertEqual(active[0]["company_id"], 6)  # Bandung
        self.assertEqual(active[1]["company_id"], 6)

    async def test_same_branch_stock_merges_preserving_dedup_contract(self):
        query = "Cek sisa stok kertas A4 dan cek stok tinta hitam"
        state = create_initial_state(user_query=query, company_id=1)
        routed = router_node(state)
        self.assertFalse(routed["is_multi_tool"])
        self.assertEqual(len(routed["active_tools"]), 1)
        self.assertEqual(routed["active_tools"][0]["tool"], "get_stock_level")

        extracted = extractor_node({**state, **routed})
        params = extracted["extracted_params"]
        codes = params.get("codes") or []
        self.assertGreaterEqual(len(codes), 2)

    async def test_cross_branch_stock_and_scalar_product_info_split(self):
        query_cross = "Berapa stok kertas A4 di Jakarta dan berapa stok tinta hitam di Surabaya?"
        st_cross = create_initial_state(user_query=query_cross, company_id=1)
        rt_cross = router_node(st_cross)
        self.assertTrue(rt_cross["is_multi_tool"])
        self.assertEqual(len(rt_cross["active_tools"]), 2)
        self.assertEqual([t["tool"] for t in rt_cross["active_tools"]], ["get_stock_level", "get_stock_level"])
        self.assertEqual([t["company_id"] for t in rt_cross["active_tools"]], [8, 7])

        query_prod = "Berapa harga produk Art Paper 260 dan berapa harga produk Sticker Vinyl?"
        st_prod = create_initial_state(user_query=query_prod, company_id=1)
        rt_prod = router_node(st_prod)
        self.assertTrue(rt_prod["is_multi_tool"])
        self.assertEqual(len(rt_prod["active_tools"]), 2)
        self.assertEqual([t["tool"] for t in rt_prod["active_tools"]], ["get_product_info", "get_product_info"])

        ext_prod = extractor_node({**st_prod, **rt_prod})
        p0 = ext_prod["active_tools"][0]["params"]
        p1 = ext_prod["active_tools"][1]["params"]
        self.assertIn("art paper", (p0.get("code") or p0.get("search") or "").lower())
        self.assertIn("sticker vinyl", (p1.get("code") or p1.get("search") or "").lower())

    async def test_suppression_fixes_sales_inventory_and_hybrid(self):
        # 1. Sales: get_contacts + get_sales_summary must both remain
        st1 = create_initial_state(user_query="Siapa kontak Budi dan berapa omzet bulan ini?", company_id=1)
        rt1 = router_node(st1)
        tools1 = {t["tool"] for t in rt1["active_tools"]}
        self.assertIn("get_contacts", tools1)
        self.assertIn("get_sales_summary", tools1)

        # 2. Inventory: get_low_stock_alerts + get_stock_level must both remain
        st2 = create_initial_state(
            user_query="Barang apa yang stoknya menipis? Dan cek stok HVS A4 di Surabaya.",
            company_id=1,
        )
        rt2 = router_node(st2)
        tools2 = {t["tool"] for t in rt2["active_tools"]}
        self.assertIn("get_low_stock_alerts", tools2)
        self.assertIn("get_stock_level", tools2)

        # 3. Hybrid: RAG specs + get_stock_level must both survive without suppressing product info/stock
        st3 = create_initial_state(
            user_query="Jelaskan spesifikasi teknis Art Paper dan cek stok Art Paper di Jakarta",
            company_id=1,
        )
        rt3 = router_node(st3)
        self.assertEqual(rt3["query_mode"], "hybrid")
        self.assertTrue(rt3["rag_queries"])
        self.assertEqual(rt3["active_tools"][0]["tool"], "get_stock_level")

    async def test_anaphora_inheritance_for_product_and_branch_date(self):
        query_prod = "Berapa harga beli Art Paper 260 dan berapa stoknya?"
        st = create_initial_state(user_query=query_prod, company_id=1)
        rt = router_node(st)
        self.assertTrue(rt["is_multi_tool"])
        self.assertEqual([t["tool"] for t in rt["active_tools"]], ["get_product_info", "get_stock_level"])
        ext = extractor_node({**st, **rt})
        p0 = ext["active_tools"][0]["params"]
        p1 = ext["active_tools"][1]["params"]
        code0 = (p0.get("code") or p0.get("search") or "").lower()
        code1 = (p1.get("code") or p1.get("search") or "").lower()
        self.assertIn("art paper", code0)
        self.assertIn("art paper", code1)

        query_loc = "Berapa omzet Jakarta bulan ini dan bagaimana laba ruginya di sana?"
        st2 = create_initial_state(user_query=query_loc, company_id=1)
        rt2 = router_node(st2)
        self.assertEqual([t["company_id"] for t in rt2["active_tools"]], [8, 8])
        ext2 = extractor_node({**st2, **rt2})
        self.assertIn("date_from", ext2["active_tools"][0]["params"])
        self.assertIn("date_from", ext2["active_tools"][1]["params"])

    async def test_mixed_guardrail_and_single_guardrail_preservation(self):
        # Single-clause regression check preserved
        single_guard = router_node(
            create_initial_state(user_query="Tagihan customer Budi dan stok barang", company_id=1)
        )
        self.assertEqual(single_guard["detected_domain"], "general")
        self.assertIsNone(single_guard["detected_tool"])

        # Mixed multi-question: customer billing Q1 + real ERP Q2
        mixed = create_initial_state(
            user_query="Cek tagihan customer Budi? Dan berapa stok kertas A4 di Jakarta?",
            company_id=1,
        )
        rt_mixed = router_node(mixed)
        self.assertEqual(len(rt_mixed["active_tools"]), 1)
        self.assertEqual(rt_mixed["active_tools"][0]["tool"], "get_stock_level")
        self.assertEqual(len(rt_mixed["non_tool_questions"]), 1)
        self.assertEqual(rt_mixed["non_tool_questions"][0]["category"], "customer_billing")
        self.assertEqual(decide_after_router({**mixed, **rt_mixed}), "extractor")


class TestMultiQuestionExecutionAndSynthesis(unittest.IsolatedAsyncioTestCase):
    async def test_cross_branch_same_tool_e2e_tool_verifier_synthesizer(self):
        query = "Berapa stok kertas A4 di Jakarta? Dan berapa stok tinta hitam di Surabaya?"
        state = create_initial_state(user_query=query, company_id=1)
        routed = router_node(state)
        extracted = extractor_node({**state, **routed})

        mock_client = AsyncMock(spec=DjangoCoreClient)
        mock_client.fetch_locations.return_value = []

        async def _fake_stock(params=None, user_token=None, company_id=None, **kwargs):
            if company_id == 8:
                return [
                    {
                        "code": "HVS-A4",
                        "name": "Kertas HVS A4",
                        "location_name": "Gudang Jakarta",
                        "quantity": 50,
                        "available_quantity": 45,
                    }
                ]
            return [
                {
                    "code": "TINTA-BLK",
                    "name": "Tinta Hitam",
                    "location_name": "Gudang Surabaya",
                    "quantity": 12,
                    "available_quantity": 12,
                }
            ]

        mock_client.fetch_stock_level.side_effect = _fake_stock

        executed = await tool_node({**state, **routed, **extracted}, client=mock_client)
        self.assertIn("get_stock_level_0", executed["multi_tool_results"])
        self.assertIn("get_stock_level_1", executed["multi_tool_results"])

        verified = verifier_node({**state, **routed, **extracted, **executed})
        self.assertEqual(verified["verification_status"], "VALID")

        full_state = {**state, **routed, **extracted, **executed, **verified}
        # Mock LLM that tries to drop Q1; verbatim multi-question guard must preserve both
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(
                content='Tinta Hitam stok 12 unit.\n```json:followups\n[{"label": "Detail Gudang", "query": "Cek stok gudang Surabaya", "icon": "fa-arrow-right"}]\n```'
            )
        )
        synthesized = await asynthesizer_node(full_state, llm=mock_llm)
        ans = synthesized["final_answer"]
        self.assertIn("Kertas HVS A4", ans)
        self.assertIn("Tinta Hitam", ans)
        self.assertIn("Pertanyaan 1", ans)
        self.assertIn("Pertanyaan 2", ans)
        self.assertTrue(synthesized["suggested_followups"])

    async def test_multi_tool_with_ui_company_ids_executes_per_plan_multi_company(self):
        query = "Cek laba rugi bulan ini? Dan cek stok HVS A4?"
        state = create_initial_state(
            user_query=query,
            company_id=1,
            company_ids=[1, 2],
            is_multi_company=True,
        )
        routed = router_node(state)
        extracted = extractor_node({**state, **routed})

        mock_client = AsyncMock(spec=DjangoCoreClient)
        mock_client.fetch_locations.return_value = []

        async def _fake_pnl(params=None, user_token=None, company_id=None, **kwargs):
            return {
                "summary": {
                    "total_revenue": 1000000 * (company_id or 1),
                    "cogs": 400000,
                    "gross_profit": 600000,
                    "net_profit": 500000 * (company_id or 1),
                },
            }

        async def _fake_stock(params=None, user_token=None, company_id=None, **kwargs):
            return {
                "total_count": 1,
                "stocks": [
                    {
                        "product_code": "HVS-A4",
                        "product_name": f"HVS A4 Co{company_id}",
                        "location_name": "Gudang",
                        "quantity": 10 * (company_id or 1),
                        "available_quantity": 10,
                    }
                ],
            }

        mock_client.fetch_profit_and_loss.side_effect = _fake_pnl
        mock_client.fetch_stock_level.side_effect = _fake_stock

        executed = await tool_node({**state, **routed, **extracted}, client=mock_client)
        self.assertEqual(mock_client.fetch_profit_and_loss.await_count, 2)
        self.assertEqual(mock_client.fetch_stock_level.await_count, 2)
        verified = verifier_node({**state, **routed, **extracted, **executed})
        self.assertEqual(verified["verification_status"], "VALID")

        rendered = format_deterministic_answer({**state, **routed, **extracted, **executed, **verified})
        self.assertIn("Laporan Laba Rugi Multi-Perusahaan", rendered)
        self.assertIn("HVS A4 Co1", rendered)
        self.assertIn("HVS A4 Co2", rendered)

    async def test_multi_rag_queries_execute_concurrently_and_render_both(self):
        query = "Bagaimana SOP retur barang? Dan berapa gramasi Art Paper?"
        state = create_initial_state(user_query=query, company_id=1)
        routed = router_node(state)
        self.assertEqual(len(routed["rag_queries"]), 2)

        mock_client = AsyncMock(spec=DjangoCoreClient)
        async def _fake_rag(params, user_token=None, company_id=None):
            q = params["query"].lower()
            if "retur" in q:
                return {"chunks": [{"title": "SOP Retur.pdf", "content": "Maksimal 7 hari kerja.", "similarity_score": 0.9}]}
            return {"chunks": [{"title": "Spek Produk.pdf", "content": "Art Paper tersedia 120 dan 150 GSM.", "similarity_score": 0.88}]}
        mock_client.fetch_knowledge.side_effect = _fake_rag

        with patch("app.graph.nodes.rag_node.settings.VECTOR_DB_URL", ""):
            rag_out = await rag_node({**state, **routed}, client=mock_client)

        self.assertEqual(len(rag_out["rag_results_by_query"]), 2)
        rendered = format_deterministic_answer({**state, **routed, **rag_out})
        self.assertIn("Maksimal 7 hari kerja.", rendered)
        self.assertIn("120 dan 150 GSM", rendered)

    async def test_mixed_general_and_erp_uses_llm_for_general_only(self):
        query = "Halo, kamu bisa membantu apa saja? Dan berapa saldo kas bulan ini?"
        state = create_initial_state(user_query=query, company_id=1)
        routed = router_node(state)
        extracted = extractor_node({**state, **routed})

        mock_client = AsyncMock(spec=DjangoCoreClient)
        mock_client.fetch_account_balance.return_value = {
            "totals": {"opening_balance": 1000000, "period_debit": 250000, "period_credit": 50000, "ending_balance": 1200000},
            "accounts": [{"account_code": "1101", "account_name": "Kas Kecil", "ending_balance": 1200000}],
        }

        executed = await tool_node({**state, **routed, **extracted}, client=mock_client)
        verified = verifier_node({**state, **routed, **extracted, **executed})
        full_state = {**state, **routed, **extracted, **executed, **verified}

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            side_effect=[
                AIMessage(content="Saya dapat membantu mengecek data akuntansi, stok gudang, penjualan, marketplace, dan SOP."),
                AIMessage(content='```json:followups\n[{"label": "Jurnal Kas", "query": "Cek entri jurnal kas", "icon": "fa-arrow-right"}]\n```'),
            ]
        )
        synthesized = await asynthesizer_node(full_state, llm=mock_llm)
        ans = synthesized["final_answer"]
        self.assertIn("Saya dapat membantu mengecek data akuntansi", ans)
        self.assertIn("Kas Kecil", ans)
        self.assertIn("1.200.000", ans)


class TestThreadConcurrencySerialization(unittest.IsolatedAsyncioTestCase):
    async def test_same_thread_id_requests_serialize_while_distinct_threads_overlap(self):
        timeline = []
        active_by_thread = {}

        class _FakeGraph:
            async def aget_state(self, config):
                return None

            async def ainvoke(self, initial_state, config=None):
                tid = config["configurable"]["thread_id"]
                active_by_thread[tid] = active_by_thread.get(tid, 0) + 1
                timeline.append((tid, "enter", active_by_thread[tid]))
                await asyncio.sleep(0.04)
                active_by_thread[tid] -= 1
                timeline.append((tid, "exit", active_by_thread[tid]))
                return {"final_answer": initial_state["user_query"]}

        fake_graph = _FakeGraph()
        r1, r2 = await asyncio.gather(
            arun_workflow("Pertanyaan 1", thread_id="shared-thread-A", graph=fake_graph),
            arun_workflow("Pertanyaan 2", thread_id="shared-thread-A", graph=fake_graph),
        )
        self.assertEqual(r1["final_answer"], "Pertanyaan 1")
        self.assertEqual(r2["final_answer"], "Pertanyaan 2")

        same_thread_events = [ev for ev in timeline if ev[0] == "shared-thread-A"]
        self.assertEqual(
            [ev[1] for ev in same_thread_events],
            ["enter", "exit", "enter", "exit"],
        )


if __name__ == "__main__":
    unittest.main()
