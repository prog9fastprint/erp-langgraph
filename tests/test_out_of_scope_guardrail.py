"""
Unit tests for strict general vs out_of_scope routing, post-LLM coercion,
multi-turn Step 1.5 off-topic isolation, mixed ERP + off-topic synthesis,
and deterministic out-of-scope capability messaging.
"""
import unittest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from app.graph.nodes.router import MultiRouterDecision, router_node
from app.graph.nodes.router_patterns import (
    is_allowed_general_query,
    is_general_help_inquiry,
    is_pure_greeting,
)
from app.graph.nodes.synthesizer import (
    asynthesizer_node,
    format_deterministic_answer,
    synthesizer_node,
)
from app.graph.state import create_initial_state
from app.graph.workflow import decide_after_router


class TestStrictGeneralVsOutOfScopeRouting(unittest.TestCase):
    """Validate strict boundary between general and out_of_scope."""

    def test_pure_greetings_and_help_stay_general(self):
        allowed = [
            "Halo",
            "Hai min",
            "Selamat pagi",
            "Assalamu'alaikum",
            "terima kasih banyak ya min",
            "tes",
            "ping",
            "bisa bantu apa",
            "kamu bisa membantu apa saja?",
            "anda bisa apa aja?",
            "Halo, kamu bisa membantu apa saja?",
            "/scope",
            "help",
            "menu",
            "siapa kamu?",
        ]
        for query in allowed:
            with self.subTest(query=query):
                res = router_node(create_initial_state(query, company_id=1), llm=None)
                self.assertEqual(res["detected_domain"], "general", f"Expected general for '{query}'")
                self.assertEqual(res["query_mode"], "general")
                self.assertIsNone(res["detected_tool"])

    def test_greeting_prefix_or_menu_word_with_off_topic_routes_to_out_of_scope(self):
        off_topic = [
            "Halo, resep nasi goreng apa?",
            "Selamat pagi, berapa harga bitcoin sekarang?",
            "Hai min, cuaca Surabaya hari ini gimana?",
            "Tes bikin script python untuk scraping",
            "menu makan siang enak hari ini apa?",
            "help me write a python sorting function",
        ]
        for query in off_topic:
            with self.subTest(query=query):
                self.assertFalse(is_pure_greeting(query))
                self.assertFalse(is_general_help_inquiry(query))
                self.assertFalse(is_allowed_general_query(query))
                res = router_node(create_initial_state(query, company_id=1), llm=None)
                self.assertEqual(res["detected_domain"], "out_of_scope", f"Expected out_of_scope for '{query}'")
                self.assertEqual(res["query_mode"], "out_of_scope")
                self.assertIsNone(res["detected_tool"])

    def test_llm_returning_general_for_off_topic_is_coerced_to_out_of_scope(self):
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = MultiRouterDecision(
            domain="general",
            query_mode="general",
            tool=None,
            planned_tools=[],
            reasoning="General world knowledge question",
        )
        mock_llm.with_structured_output.return_value = mock_structured

        off_topic_queries = [
            "Siapa juara Piala Dunia 2022?",
            "Jelaskan proses fotosintesis pada tumbuhan",
            "Rekomendasi film horor terbaru",
            "Bagaimana cara belajar gitar cepat?",
            "Cek gaji karyawan bulan ini",
        ]
        for query in off_topic_queries:
            with self.subTest(query=query):
                res = router_node(create_initial_state(query, company_id=1), llm=mock_llm)
                self.assertEqual(res["detected_domain"], "out_of_scope", f"LLM general must coerce for '{query}'")
                self.assertEqual(res["query_mode"], "out_of_scope")
                self.assertIsNone(res["detected_tool"])
                self.assertEqual(res["active_tools"], [])

    def test_customer_billing_guardrail_preserved_as_general(self):
        for query in ("Cek tagihan pelanggan PT Sumber Makmur", "coba cek tagihan Frendy Krisnata di mainstore"):
            with self.subTest(query=query):
                res = router_node(create_initial_state(query, company_id=1), llm=None)
                self.assertEqual(res["detected_domain"], "general")
                self.assertIsNone(res["detected_tool"])

    def test_accounting_salary_expense_does_not_collide_with_hr_out_of_scope(self):
        res_pnl = router_node(create_initial_state("Cek laba rugi beban gaji bulan ini", company_id=1), llm=None)
        self.assertEqual(res_pnl["detected_domain"], "accounting")
        self.assertEqual(res_pnl["detected_tool"], "get_profit_and_loss")

        res_gl = router_node(create_initial_state("Buku besar beban gaji karyawan bulan ini", company_id=1), llm=None)
        self.assertEqual(res_gl["detected_domain"], "accounting")
        self.assertEqual(res_gl["detected_tool"], "get_general_ledger")


class TestMultiTurnShortOffTopicIsolation(unittest.TestCase):
    """Validate Step 1.5 short follow-up tightening after an active ERP turn."""

    def setUp(self):
        self.stock_history = [
            HumanMessage(content="Cek stok HVS A4"),
            AIMessage(content="### Informasi Ketersediaan Stok Barang\nTotal stok HVS A4: 50 unit."),
        ]

    def test_short_off_topic_after_erp_turn_does_not_hijack_prior_tool(self):
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = MultiRouterDecision(
            domain="general",
            query_mode="general",
            tool=None,
            planned_tools=[],
        )
        mock_llm.with_structured_output.return_value = mock_structured

        off_topic_turn2 = [
            "resep nasi goreng apa?",
            "siapa pencipta python?",
            "cuaca surabaya hari ini?",
            "jelaskan fotosintesis",
            "rekomendasi film bagus",
            "cek gaji karyawan",
        ]
        for q in off_topic_turn2:
            with self.subTest(query=q):
                st = create_initial_state(
                    user_query=q,
                    company_id=1,
                    messages=self.stock_history + [HumanMessage(content=q)],
                )
                res = router_node(st, llm=mock_llm)
                self.assertEqual(res["detected_domain"], "out_of_scope", f"Turn 2 '{q}' hijacked prior tool")
                self.assertIsNone(res["detected_tool"])

    def test_legitimate_short_followups_still_inherit_prior_erp_tool(self):
        valid_followups = [
            "di mainstore",
            "cabang bandung",
            "bulan lalu",
            "kalau tinta epson?",
            "Tinta Epson 664",
        ]
        for q in valid_followups:
            with self.subTest(query=q):
                st = create_initial_state(
                    user_query=q,
                    company_id=1,
                    messages=self.stock_history + [HumanMessage(content=q)],
                )
                res = router_node(st, llm=None)
                self.assertEqual(res["detected_domain"], "inventory", f"Valid follow-up '{q}' failed")
                self.assertEqual(res["detected_tool"], "get_stock_level", f"Valid follow-up '{q}' lost tool")


class TestOutOfScopeSynthesizerAndMixedPrompts(unittest.IsolatedAsyncioTestCase):
    """Validate deterministic capability message, follow-up chips, and mixed ERP + off-topic handling."""

    def test_single_out_of_scope_deterministic_response_and_followups(self):
        mock_llm = MagicMock()
        st = create_initial_state("Siapa juara Piala Dunia 2022?", company_id=1)
        st["detected_domain"] = "out_of_scope"
        st["query_mode"] = "out_of_scope"
        st["detected_tool"] = None

        out = synthesizer_node(st, llm=mock_llm)
        ans = out["final_answer"]

        self.assertIn("Permintaan di Luar Cakupan (Out of Scope)", ans)
        self.assertIn("di luar cakupan dan kemampuan Asisten AI FastPrint ERP", ans)
        self.assertIn("Akuntansi & Keuangan", ans)
        self.assertIn("Inventaris & Gudang", ans)
        self.assertIn("Penjualan (Sales)", ans)
        self.assertIn("Marketplace (Read-Only)", ans)
        self.assertEqual(len(out["suggested_followups"]), 3)
        mock_llm.invoke.assert_not_called()

    async def test_mixed_erp_and_off_topic_answers_erp_and_refuses_off_topic_without_llm_leak(self):
        mock_router_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = MultiRouterDecision(
            domain="general",
            query_mode="general",
            tool=None,
            planned_tools=[],
        )
        mock_router_llm.with_structured_output.return_value = mock_structured

        query = "Berapa stok kertas HVS A4? Dan siapa pencipta bahasa Python?"
        state = create_initial_state(query, company_id=1)
        routed = router_node(state, llm=mock_router_llm)

        self.assertEqual(len(routed["active_tools"]), 1)
        self.assertEqual(routed["active_tools"][0]["tool"], "get_stock_level")
        self.assertEqual(len(routed["non_tool_questions"]), 1)
        self.assertEqual(routed["non_tool_questions"][0]["category"], "out_of_scope")
        self.assertEqual(routed["non_tool_questions"][0]["domain"], "out_of_scope")
        self.assertEqual(decide_after_router({**state, **routed}), "extractor")

        # Synthesize mixed result: verify LLM is NEVER called to answer the off-topic clause
        merged = {
            **state,
            **routed,
            "verification_status": "VALID",
            "multi_verification_status": {"get_stock_level": "VALID"},
            "multi_tool_results": {
                "get_stock_level": {
                    "code": "HVS-A4",
                    "name": "Kertas HVS A4",
                    "qty_on_hand": 40.0,
                }
            },
        }
        merged["active_tools"][0]["result"] = merged["multi_tool_results"]["get_stock_level"]

        mock_synth_llm = MagicMock()
        mock_synth_llm.ainvoke = AsyncMock(
            return_value=AIMessage(
                content='```json:followups\n[{"label": "Kartu Stok", "query": "Cek kartu stok HVS A4", "icon": "fa-arrow-right"}]\n```'
            )
        )
        synthesized = await asynthesizer_node(merged, llm=mock_synth_llm)
        ans = synthesized["final_answer"]

        self.assertIn("Kertas HVS A4", ans)
        self.assertIn("40.0 unit", ans)
        self.assertIn("di luar cakupan dan kemampuan Asisten AI FastPrint ERP", ans)
        # Only followup-only prompt should hit LLM (never general Q&A prompt for Python question)
        for call in mock_synth_llm.ainvoke.await_args_list:
            prompt_text = str(call.args[0][-1].content)
            self.assertNotIn("Balas sapaan", prompt_text)
            self.assertNotIn("Jawab pertanyaan umum", prompt_text)

    async def test_synthesizer_defense_in_depth_blocks_off_topic_labeled_as_general(self):
        state = create_initial_state("Cek stok HVS A4 dan siapa pencipta python?", company_id=1)
        state["sub_questions"] = ["Cek stok HVS A4", "siapa pencipta python?"]
        state["non_tool_questions"] = [
            {
                "question_index": 1,
                "question_text": "siapa pencipta python?",
                "category": "general",
                "domain": "general",
            }
        ]
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Guido van Rossum."))

        out = await asynthesizer_node(state, llm=mock_llm)
        self.assertNotIn("Guido van Rossum", out["final_answer"])
        mock_llm.ainvoke.assert_not_called()

    def test_format_deterministic_answer_single_out_of_scope_heading(self):
        st = create_initial_state("buatkan puisi cinta")
        st["detected_domain"] = "out_of_scope"
        ans = format_deterministic_answer(st)
        self.assertIn("### Permintaan di Luar Cakupan (Out of Scope)", ans)
        self.assertIn("di luar cakupan dan kemampuan", ans)


if __name__ == "__main__":
    unittest.main()
