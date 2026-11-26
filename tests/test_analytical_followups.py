"""
Tests for analytical ("thinking") follow-up and causal ERP question handling across
router, extractor, tool_node, formatters, workflow, and synthesizer.
"""
from datetime import date
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.core.context import (
    extract_period_from_messages,
    has_sufficient_prior_erp_context,
    is_analytical_thinking_query,
)
from app.graph.nodes.extractors.accounting import (
    _fallback_extract_balance_sheet,
    _fallback_extract_profit_and_loss,
)
from app.graph.nodes.formatters.accounting import (
    format_get_balance_sheet,
    format_get_profit_and_loss,
)
from app.graph.nodes.router import arouter_node, router_node
from app.graph.nodes.synthesizer import (
    _should_keep_verbatim_deterministic,
    asynthesizer_node,
    synthesizer_node,
)
from app.graph.nodes.tool_node import _execute_single_tool
from app.graph.state import create_initial_state
from app.graph.workflow import arun_workflow, create_workflow_graph
from app.main import app
from app.services.django_client import DjangoCoreClient


def _sample_pnl_payload(num_extra_expenses: int = 0):
    expenses_accounts = [
        {"account_code": "50101HO", "account_name": "BIAYA GAJI KARYAWAN", "balance": 15992442741.87},
        {"account_code": "50112HO", "account_name": "BIAYA PENDIDIKAN/SEMINAR/BUKU", "balance": 708043306.0},
        {"account_code": "50117HO", "account_name": "BIAYA INTERNET SUPPORT SYSTEM", "balance": 622395853.51},
        {"account_code": "50104HO", "account_name": "BIAYA THR KARYAWAN", "balance": 651001044.0},
        {"account_code": "50108HO", "account_name": "BIAYA TUNJANGAN PERSALINAN KARYAWAN", "balance": 0.0},
        {"account_code": "50123HO", "account_name": "BIAYA EKS WL (IM)", "balance": -298044500.0},
    ]
    for i in range(num_extra_expenses):
        expenses_accounts.append({
            "account_code": f"509{i:02d}HO",
            "account_name": f"BIAYA OPERASIONAL TAMBAHAN {i}",
            "balance": 1000000.0,
        })
    return {
        "date_from": "2026-01-01",
        "date_to": "2026-12-31",
        "summary": {
            "total_revenue": 115440844730.82,
            "cogs": 89002644045.88,
            "gross_profit": 26438200684.94,
            "gross_margin_percent": 22.9,
            "operating_expenses": 30638268545.47,
            "operating_profit": -4200067860.53,
            "other_net": 1532377713.42,
            "net_profit": -2667690147.11,
        },
        "breakdown": [
            {
                "group_name": "Income",
                "types": [
                    {
                        "type_name": "Income",
                        "accounts": [
                            {"account_code": "40000HO", "account_name": "PENJUALAN BARANG", "balance": 118044357424.64},
                            {"account_code": "40002HO", "account_name": "DISCOUNT PENJUALAN BARANG", "balance": -560634852.0},
                            {"account_code": "40004HO", "account_name": "PENDAPATAN JASA", "balance": -2022496086.82},
                        ],
                    },
                    {
                        "type_name": "Other Income",
                        "accounts": [
                            {"account_code": "41000HO", "account_name": "PENDAPATAN JASA GIRO", "balance": 0.0},
                            {"account_code": "41002HO", "account_name": "PENDAPATAN LAIN-LAIN", "balance": 1117311739.49},
                        ],
                    },
                ],
            },
            {
                "group_name": "Expense",
                "types": [
                    {
                        "type_name": "Cost of Revenue",
                        "accounts": [
                            {"account_code": "40005HO", "account_name": "HARGA POKOK PENJUALAN BARANG", "balance": 88657254710.88},
                            {"account_code": "40006HO", "account_name": "HARGA POKOK PENJUALAN JASA", "balance": 345389335.0},
                        ],
                    },
                    {
                        "type_name": "Depreciation",
                        "accounts": [
                            {"account_code": "50147HO", "account_name": "BEBAN PENYUSUTAN KENDARAAN HO", "balance": 892543244.0},
                            {"account_code": "50149HO-A", "account_name": "(-)BEBAN PENYUSUTAN ASET TETAP HO", "balance": -2421918563.0},
                        ],
                    },
                    {
                        "type_name": "Expenses",
                        "accounts": expenses_accounts,
                    },
                ],
            },
        ],
    }


class TestAnalyticalIntentHelpers(unittest.TestCase):
    def test_is_analytical_thinking_query_classification(self):
        self.assertTrue(is_analytical_thinking_query("kok minus ya laba bersih nya"))
        self.assertTrue(is_analytical_thinking_query("kenapa laba rugi tahun ini minus?"))
        self.assertTrue(is_analytical_thinking_query("apa penyebab beban operasional besar?"))
        self.assertTrue(is_analytical_thinking_query("akun mana yang paling besar pengeluarannya?"))
        self.assertTrue(is_analytical_thinking_query("jelaskan rincian laporan di atas"))
        self.assertTrue(is_analytical_thinking_query("apakah margin 22.9% ini normal?"))

        self.assertFalse(is_analytical_thinking_query("Laba rugi & gross margin YTD"))
        self.assertFalse(is_analytical_thinking_query("coba berikan detail untuk tahun ini"))
        self.assertFalse(is_analytical_thinking_query("coba berikan detail dari Penjualan Surabaya"))
        self.assertFalse(is_analytical_thinking_query("tampilkan rincian akunnya"))
        self.assertFalse(is_analytical_thinking_query("jabarkan semua datanya dalam table"))
        self.assertFalse(is_analytical_thinking_query("mana produk yang stocknya tersedikit"))

    def test_has_sufficient_prior_erp_context_distinguishes_summary_vs_detail(self):
        summary_msgs = [
            HumanMessage(content="Laba rugi & gross margin YTD"),
            AIMessage(content="Laba bersih periode berjalan adalah Rp -2.667.690.147,11 dengan gross profit margin sebesar 22.9%."),
            HumanMessage(content="kok minus ya laba bersih nya"),
        ]
        self.assertFalse(has_sufficient_prior_erp_context(summary_msgs))

        detail_md = format_get_profit_and_loss("get_profit_and_loss", _sample_pnl_payload())
        detail_msgs = [
            HumanMessage(content="coba berikan detail untuk tahun ini"),
            AIMessage(content=detail_md),
            HumanMessage(content="kok minus ya laba bersih nya"),
        ]
        self.assertTrue(has_sufficient_prior_erp_context(detail_msgs))
        self.assertEqual(extract_period_from_messages(detail_msgs), ("2026-01-01", "2026-12-31", None))


class TestPayloadAndFormatterHygiene(unittest.IsolatedAsyncioTestCase):
    async def test_tool_node_preserves_over_50_pnl_accounts(self):
        mock_client = MagicMock(spec=DjangoCoreClient)
        payload = _sample_pnl_payload(num_extra_expenses=70)
        mock_client.fetch_profit_and_loss = AsyncMock(return_value=payload)
        dispatch_map = {"get_profit_and_loss": mock_client.fetch_profit_and_loss}

        res, err = await _execute_single_tool(
            tool_name="get_profit_and_loss",
            params={"include_breakdown": True},
            erp_client=mock_client,
            user_token="tok",
            default_company_id=3,
            dispatch_map=dispatch_map,
            user_query="coba berikan detail untuk tahun ini",
        )
        self.assertIsNone(err)
        expense_accs = res["breakdown"][1]["types"][2]["accounts"]
        self.assertEqual(len(expense_accs), 76)
        self.assertEqual(expense_accs[-1]["account_code"], "50969HO")

    def test_pnl_and_bs_formatters_omit_zero_accounts_with_footnote(self):
        pnl_md = format_get_profit_and_loss("get_profit_and_loss", _sample_pnl_payload())
        self.assertNotIn("50108HO", pnl_md)
        self.assertNotIn("41000HO", pnl_md)
        self.assertIn("50101HO", pnl_md)
        self.assertIn("2 akun bersaldo Rp 0,00 tidak ditampilkan", pnl_md)

        bs_payload = {
            "as_of_date": "2026-11-26",
            "summary": {
                "total_assets": 1000000.0,
                "total_liabilities": 400000.0,
                "total_equity": 600000.0,
                "total_liabilities_and_equity": 1000000.0,
                "net_balance_check": {"is_balanced": True, "difference": 0.0},
            },
            "breakdown": [
                {
                    "group_name": "Asset",
                    "types": [
                        {
                            "type_name": "Cash",
                            "accounts": [
                                {"account_code": "1101", "account_name": "Kas", "balance": 1000000.0},
                                {"account_code": "1102", "account_name": "Kas Kosong", "balance": 0.0},
                            ],
                        }
                    ],
                }
            ],
        }
        bs_md = format_get_balance_sheet("get_balance_sheet", bs_payload)
        self.assertIn("1101", bs_md)
        self.assertNotIn("1102", bs_md)
        self.assertIn("1 akun bersaldo Rp 0,00 tidak ditampilkan", bs_md)


class TestExtractorAndSynthesizerAnalytical(unittest.IsolatedAsyncioTestCase):
    def test_extractor_auto_enables_breakdown_and_inherits_period_for_analytical_queries(self):
        msgs = [
            HumanMessage(content="coba berikan detail untuk tahun ini"),
            AIMessage(content="### Laporan Laba Rugi (Profit & Loss) — Periode: 2026-01-01 s/d 2026-12-31\n"),
            HumanMessage(content="kok minus ya laba bersih nya"),
        ]
        p_pnl = _fallback_extract_profit_and_loss(
            "kok minus ya laba bersih nya",
            ref_date=date(2026, 11, 26),
            messages=msgs,
        )
        self.assertTrue(p_pnl["include_breakdown"])
        self.assertEqual(p_pnl.get("date_from"), "2026-01-01")
        self.assertEqual(p_pnl.get("date_to"), "2026-12-31")

        bs_msgs = [
            HumanMessage(content="cek neraca bulan lalu"),
            AIMessage(content="### Laporan Neraca Keuangan — Posisi per: 2026-10-31\n"),
            HumanMessage(content="kenapa kewajiban besar?"),
        ]
        p_bs = _fallback_extract_balance_sheet(
            "kenapa kewajiban besar?",
            ref_date=date(2026, 11, 26),
            messages=bs_msgs,
        )
        self.assertTrue(p_bs["include_breakdown"])
        self.assertEqual(p_bs.get("as_of_date"), "2026-10-31")

    async def test_synthesizer_does_not_suppress_llm_on_large_analytical_pnl(self):
        payload = _sample_pnl_payload(num_extra_expenses=20)
        det_md = format_get_profit_and_loss("get_profit_and_loss", payload)
        self.assertGreater(len(det_md), 1500)

        state = create_initial_state("kenapa laba rugi tahun ini minus?", company_id=3)
        state.update({
            "detected_domain": "accounting",
            "detected_tool": "get_profit_and_loss",
            "verification_status": "VALID",
            "extracted_params": {"include_breakdown": True, "date_from": "2026-01-01", "date_to": "2026-12-31"},
            "tool_result": payload,
        })
        self.assertFalse(_should_keep_verbatim_deterministic(det_md, state["user_query"], state, is_multi_q=False))

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(
                content=(
                    "Laba bersih minus Rp -2.667.690.147,11 disebabkan Beban Operasional (Rp 30,64 M) melebihi Laba Kotor (Rp 26,44 M).\n"
                    "- Pemicu utama: `50101HO` BIAYA GAJI KARYAWAN Rp 15.992.442.741,87.\n"
                    '```json:followups\n[{"label": "Cek Gaji", "query": "Cek buku besar akun 50101HO tahun ini", "icon": "fa-arrow-right"}]\n```'
                )
            )
        )
        out = await asynthesizer_node(state, llm=mock_llm)
        mock_llm.ainvoke.assert_awaited_once()
        prompt_sent = mock_llm.ainvoke.call_args[0][0][1].content
        self.assertIn("50101HO", prompt_sent)
        self.assertIn("40004HO", prompt_sent)
        self.assertIn("Laba bersih minus", out["final_answer"])
        self.assertNotIn("50919HO", out["final_answer"])

        # Offline deterministic fallback also explains cause rather than dumping raw 80-row table
        offline_out = synthesizer_node(state, llm=None)
        self.assertIn("Beban Operasional", offline_out["final_answer"])
        self.assertIn("50101HO", offline_out["final_answer"])
        self.assertIn("40004HO", offline_out["final_answer"])
        self.assertNotIn("50919HO", offline_out["final_answer"])


class TestAnalyticalFollowupWorkflow(unittest.IsolatedAsyncioTestCase):
    async def test_exact_three_turn_pnl_followup_reuses_context_and_explains_loss(self):
        mock_client = MagicMock(spec=DjangoCoreClient)
        mock_client.aclose = AsyncMock()
        summary_only = {
            "date_from": "2026-01-01",
            "date_to": "2026-11-26",
            "summary": _sample_pnl_payload()["summary"],
        }
        detailed_payload = _sample_pnl_payload()
        mock_client.fetch_profit_and_loss = AsyncMock(side_effect=[summary_only, detailed_payload])

        graph = create_workflow_graph(checkpointer=MemorySaver())
        cfg = {"configurable": {"thread_id": "test_pnl_thinking_thread"}}

        with patch("app.graph.nodes.tool_node.DjangoCoreClient", return_value=mock_client):
            r1 = await graph.ainvoke(create_initial_state("Laba rugi & gross margin YTD", company_id=3), config=cfg)
            self.assertEqual(r1["detected_tool"], "get_profit_and_loss")

            r2 = await graph.ainvoke(create_initial_state("coba berikan detail untuk tahun ini", company_id=3), config=cfg)
            self.assertEqual(r2["detected_tool"], "get_profit_and_loss")
            self.assertIn("#### Rincian Hierarki Laba Rugi:", r2["final_answer"])
            self.assertIn("50101HO", r2["final_answer"])

            # Turn 3: follow-up thinking question must NOT re-run fetch_profit_and_loss
            r3 = await graph.ainvoke(create_initial_state("kok minus ya laba bersih nya", company_id=3), config=cfg)

        self.assertEqual(mock_client.fetch_profit_and_loss.await_count, 2)
        self.assertEqual(r3["query_mode"], "contextual_analysis")
        self.assertEqual(r3["detected_domain"], "accounting")
        self.assertIsNone(r3["detected_tool"])
        self.assertEqual(r3["verification_status"], "VALID")
        self.assertIn("Beban Operasional", r3["final_answer"])
        self.assertIn("50101HO", r3["final_answer"])
        self.assertNotIn("#### Rincian Hierarki Laba Rugi:", r3["final_answer"])

    async def test_stateless_history_hydration_contextual_analysis(self):
        detail_md = format_get_profit_and_loss("get_profit_and_loss", _sample_pnl_payload())
        history = [
            {"role": "user", "content": "coba berikan detail untuk tahun ini"},
            {"role": "assistant", "content": detail_md},
        ]
        mock_client = MagicMock(spec=DjangoCoreClient)
        mock_client.fetch_profit_and_loss = AsyncMock()
        graph = create_workflow_graph(checkpointer=MemorySaver())

        with patch("app.graph.nodes.tool_node.DjangoCoreClient", return_value=mock_client):
            res = await arun_workflow(
                user_query="kok minus ya laba bersih nya",
                company_id=3,
                thread_id="stateless_pnl_followup",
                history=history,
                graph=graph,
            )
        mock_client.fetch_profit_and_loss.assert_not_awaited()
        self.assertEqual(res["query_mode"], "contextual_analysis")
        self.assertEqual(res["detected_domain"], "accounting")
        self.assertEqual(res["verification_status"], "VALID")
        self.assertIn("50101HO", res["final_answer"])
        self.assertNotIn("Halo! Saya adalah Asisten AI", res["final_answer"])

    async def test_summary_only_turn1_triggers_breakdown_fetch_on_why_followup(self):
        mock_client = MagicMock(spec=DjangoCoreClient)
        mock_client.fetch_profit_and_loss = AsyncMock(return_value=_sample_pnl_payload())
        msgs = [
            HumanMessage(content="Laba rugi & gross margin YTD"),
            AIMessage(content="Laba bersih periode berjalan adalah Rp -2.667.690.147,11 dengan gross profit margin sebesar 22.9%."),
        ]
        graph = create_workflow_graph(checkpointer=MemorySaver())
        with patch("app.graph.nodes.tool_node.DjangoCoreClient", return_value=mock_client):
            res = await arun_workflow(
                user_query="kok minus ya laba bersih nya",
                company_id=3,
                history=[
                    {"role": "user", "content": msgs[0].content},
                    {"role": "assistant", "content": msgs[1].content},
                ],
                graph=graph,
            )
        mock_client.fetch_profit_and_loss.assert_awaited_once()
        self.assertTrue(res["extracted_params"].get("include_breakdown"))
        self.assertIn("50101HO", res["final_answer"])
        self.assertNotIn("#### Rincian Hierarki Laba Rugi:", res["final_answer"])

    async def test_cross_domain_contextual_routing_and_regression_guards(self):
        # 1. Cross-domain Sales & Marketplace contextual followups
        sales_table = (
            "### Ringkasan Eksekutif Penjualan & Omzet (Periode 2026-11-01 s/d 2026-11-30)\n\n"
            "| Cabang | Order | Omzet |\n|:---|:---|:---|\n"
            "| Main Store | 100 | Rp 1.000.000.000,00 |\n"
            "| Surabaya | 20 | Rp 150.000.000,00 |\n"
            "| Bandung | 40 | Rp 300.000.000,00 |\n"
            "| Jakarta | 50 | Rp 450.000.000,00 |\n"
        )
        r_sales = await arouter_node(create_initial_state(
            "kenapa omzet cabang tersebut paling kecil?",
            company_id=3,
            messages=[
                HumanMessage(content="Cek ringkasan penjualan semua cabang bulan ini"),
                AIMessage(content=sales_table),
                HumanMessage(content="kenapa omzet cabang tersebut paling kecil?"),
            ],
        ))
        self.assertEqual(r_sales["query_mode"], "contextual_analysis")
        self.assertEqual(r_sales["detected_domain"], "sales")
        self.assertEqual(r_sales["verification_status"], "VALID")

        mp_table = (
            "### Rekapitulasi Potongan & Settlement Marketplace (Periode 2026-11-01 s/d 2026-11-30)\n\n"
            "| Kategori | Nominal |\n|:---|:---|\n"
            "| Komisi Platform | Rp 50.000.000,00 |\n"
            "| Biaya Layanan | Rp 30.000.000,00 |\n"
            "| Ongkos Kirim | Rp 10.000.000,00 |\n"
            "| Voucher | Rp 5.000.000,00 |\n"
        )
        r_mp = router_node(create_initial_state(
            "kategori potongan mana yang paling besar?",
            company_id=3,
            messages=[
                HumanMessage(content="Ringkasan potongan fee marketplace bulan ini"),
                AIMessage(content=mp_table),
                HumanMessage(content="kategori potongan mana yang paling besar?"),
            ],
        ))
        self.assertEqual(r_mp["query_mode"], "contextual_analysis")
        self.assertEqual(r_mp["detected_domain"], "marketplace")

        # 2. Regression guards: period change, branch change, GL drilldown, diagnostic tool
        pnl_md = format_get_profit_and_loss("get_profit_and_loss", _sample_pnl_payload())
        base_hist = [
            HumanMessage(content="coba berikan detail untuk tahun ini"),
            AIMessage(content=pnl_md),
        ]

        r_period = router_node(create_initial_state(
            "kok minus ya laba bersih nya bulan lalu?",
            company_id=3,
            messages=base_hist + [HumanMessage(content="kok minus ya laba bersih nya bulan lalu?")],
        ))
        self.assertEqual(r_period["detected_tool"], "get_profit_and_loss")
        self.assertEqual(r_period["query_mode"], "live_erp")

        r_branch = router_node(create_initial_state(
            "kok minus ya laba bersih nya di cabang Bandung?",
            company_id=3,
            messages=base_hist + [HumanMessage(content="kok minus ya laba bersih nya di cabang Bandung?")],
        ))
        self.assertEqual(r_branch["detected_tool"], "get_profit_and_loss")
        self.assertEqual(r_branch["company_id"], 6)

        r_drill = router_node(create_initial_state(
            "coba berikan detail dari BIAYA GAJI KARYAWAN",
            company_id=3,
            messages=base_hist + [HumanMessage(content="coba berikan detail dari BIAYA GAJI KARYAWAN")],
        ))
        self.assertEqual(r_drill["detected_tool"], "get_general_ledger")

        r_diag = router_node(create_initial_state(
            "Kenapa update stok atau harga marketplace gagal hari ini?",
            company_id=3,
            messages=base_hist + [HumanMessage(content="Kenapa update stok atau harga marketplace gagal hari ini?")],
        ))
        self.assertEqual(r_diag["detected_tool"], "get_sync_execution_logs")

    async def test_stream_endpoint_contextual_analysis_tokens_match_final_answer(self):
        isolated_graph = create_workflow_graph(checkpointer=MemorySaver())
        pnl_md = format_get_profit_and_loss("get_profit_and_loss", _sample_pnl_payload())
        history = [
            {"role": "user", "content": "coba berikan detail untuk tahun ini"},
            {"role": "assistant", "content": pnl_md},
        ]

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
            with patch("app.graph.workflow.default_workflow_graph", isolated_graph):
                resp = await http_client.post(
                    "/api/v1/chat/stream",
                    json={
                        "query": "kok minus ya laba bersih nya",
                        "company_id": 3,
                        "thread_id": "test_stream_ctx_analysis",
                        "history": history,
                    },
                )
        self.assertEqual(resp.status_code, 200)
        tokens = []
        complete_payload = None
        for block in [b for b in resp.text.strip().split("\n\n") if b.strip()]:
            ev_name, ev_data = None, None
            for line in block.split("\n"):
                if line.startswith("event:"):
                    ev_name = line[6:].strip()
                elif line.startswith("data:"):
                    ev_data = json.loads(line[5:].strip())
            if ev_name == "token":
                tokens.append(ev_data["content"])
            elif ev_name == "complete":
                complete_payload = ev_data

        self.assertIsNotNone(complete_payload)
        self.assertEqual(complete_payload["verification_status"], "VALID")
        self.assertEqual(complete_payload["detected_domain"], "accounting")
        self.assertEqual("".join(tokens), complete_payload["final_answer"])
        self.assertNotIn("json:followups", complete_payload["final_answer"])
        self.assertIn("50101HO", complete_payload["final_answer"])


if __name__ == "__main__":
    unittest.main()
