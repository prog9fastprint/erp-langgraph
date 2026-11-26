"""
Pass 2 (WS3, WS4, WS5) regression tests covering analytics, data precision,
scope handling, meta-questions, and P&L-to-GL drill-down improvements.
"""
from datetime import date
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from app.graph.nodes.extractor import (
    _fallback_extract_account_balance,
    _fallback_extract_account_health,
    _fallback_extract_bank_reconciliation,
    _fallback_extract_contacts,
    _fallback_extract_general_ledger,
    _fallback_extract_inventory_velocity_forecast,
    _fallback_extract_stock_level,
    extractor_node,
)
from app.graph.nodes.router import router_node
from app.graph.nodes.synthesizer import (
    _build_table_query_meta,
    asynthesizer_node,
    format_deterministic_answer,
)
from app.graph.nodes.tool_node import _execute_single_tool
from app.graph.nodes.verifier import verifier_node
from app.graph.state import create_initial_state


class TestPass2WS3AnalyticsAndForecasting(unittest.IsolatedAsyncioTestCase):
    def test_velocity_forecast_routing_extraction_verifier_and_synthesis(self):
        query = (
            "tolong forecasting produk mana yg harus segera di order dan mana produk "
            "yg termasuk fastmoving atau lowmoving, dan berikan kenapa alasannya"
        )
        state = create_initial_state(user_query=query, company_id=3)
        routed = router_node(state, llm=None)

        self.assertEqual(routed["detected_domain"], "inventory")
        self.assertEqual(routed["detected_tool"], "get_inventory_velocity_forecast")
        self.assertFalse(routed["is_multi_tool"])
        self.assertEqual(routed.get("non_tool_questions") or [], [])
        self.assertEqual(len(routed.get("sub_questions") or []), 1)

        with patch("app.graph.nodes.extractor.get_current_erp_date", return_value=date(2026, 11, 26)):
            ext = extractor_node({**state, **routed}, llm=None, ref_date=date(2026, 11, 26))
        params = ext["extracted_params"]
        self.assertEqual(params.get("date_from"), "2026-08-28")
        self.assertEqual(params.get("date_to"), "2026-11-26")
        self.assertEqual(params.get("lead_time_days"), 7)
        self.assertEqual(params.get("target_cover_days"), 30)

        payload = {
            "company_ids": [3, 4],
            "date_window": {"date_from": "2026-08-28", "date_to": "2026-11-26", "lookback_days": 91},
            "lead_time_days": 14,
            "target_cover_days": 30,
            "summary": {
                "total_storable_skus_sold": 25,
                "total_units_sold": 420.0,
                "total_storable_skus_in_stock": 80,
                "zero_sales_in_stock_count": 12,
                "urgent_restock_count": 2,
            },
            "fast_moving": [
                {
                    "product_id": 10,
                    "code": "91010600",
                    "name": "KARET STEMPEL FLASH",
                    "category": "Stempel",
                    "uom": "Pcs",
                    "total_qty_sold": 180.0,
                    "total_revenue": 18000000.0,
                    "qty_on_hand": 10.0,
                    "avg_daily_sales": 1.98,
                    "days_of_cover": 5.1,
                    "stock_status": "LOW_COVER",
                }
            ],
            "slow_moving": [
                {
                    "product_id": 22,
                    "code": "OLD-999",
                    "name": "BARANG LAMA",
                    "category": "Sparepart",
                    "qty_on_hand": 250.0,
                    "total_qty_sold": 0.0,
                    "is_dead_stock": True,
                }
            ],
            "restock_recommendations": [
                {
                    "code": "91010600",
                    "name": "KARET STEMPEL FLASH",
                    "uom": "Pcs",
                    "total_qty_sold": 180.0,
                    "avg_daily_sales": 1.98,
                    "qty_on_hand": 10.0,
                    "days_of_cover": 5.1,
                    "recommended_order_qty": 50,
                    "reason": "Cover stok tersisa 5.1 hari (<= lead time 14 hari).",
                }
            ],
        }
        ver = verifier_node({**state, **routed, **ext, "tool_result": payload})
        self.assertEqual(ver["verification_status"], "VALID")

        ans = format_deterministic_answer({**state, **routed, **ext, "tool_result": payload, **ver})
        self.assertIn("Analisis Kecepatan Perputaran Stok", ans)
        self.assertIn("91010600", ans)
        self.assertIn("Cover stok tersisa 5.1 hari", ans)
        self.assertIn("OLD-999", ans)
        self.assertIn("DEAD STOCK", ans)

    def test_marketplace_pull_failure_and_deduction_impact_routing_and_synthesis(self):
        query = (
            "banyak penarikan marketplace yang gagal di proses dan potongan marketplace "
            "cukup banyak, apakah itu memberikan impact signifikan ke omzet?"
        )
        state = create_initial_state(user_query=query, company_id=3)
        routed = router_node(state, llm=None)
        tools = set(routed.get("detected_tools") or [])

        self.assertIn("get_account_health", tools)
        self.assertIn("get_deduction_and_settlement_summary", tools)
        self.assertIn("get_sales_summary", tools)
        self.assertNotIn("get_profit_and_loss", tools)

        ah_res = {
            "error_window_hours": 24,
            "error_window": {"date_from": "2026-10-01", "date_to": "2026-11-26"},
            "summary": {
                "total_accounts": 4,
                "active_accounts": 4,
                "healthy_accounts": 2,
                "accounts_with_issues": 2,
                "expired_tokens": 0,
                "expiring_soon_tokens": 1,
                "missing_configs_count": 0,
                "recent_fetch_errors_count": 250,
                "pulled_orders_count": 1000,
                "error_to_pulled_order_ratio": 0.25,
            },
            "error_breakdown_by_type": [{"error_type": "FETCH_ORDER_TIMEOUT", "count": 200}],
            "error_breakdown_by_company": [{"company_name": "Main Store", "count": 250}],
            "accounts": [],
        }
        ded_res = {
            "date_field": "received_date",
            "date_window": {"date_from": "2026-10-01", "date_to": "2026-11-26"},
            "summary": {
                "total_orders": 900,
                "total_gross": 1000000000.0,
                "total_deduction": 150000000.0,
                "total_tax": 10000000.0,
                "total_nett": 840000000.0,
                "effective_deduction_pct": 15.0,
                "unposted_journal_orders_count": 5,
                "unposted_journal_nett_amount": 5000000.0,
            },
        }
        sales_res = {
            "period": {"date_from": "2026-10-01", "date_to": "2026-11-26"},
            "confirmed_metrics": {
                "realized_revenue": 3000000000.0,
                "orders_count": 2500,
                "average_order_value": 1200000.0,
            },
        }
        active_tools = [
            {"tool": "get_account_health", "domain": "marketplace", "sub_query": query, "question_index": 0, "result": ah_res, "params": {}, "status": "done"},
            {"tool": "get_deduction_and_settlement_summary", "domain": "marketplace", "sub_query": query, "question_index": 0, "result": ded_res, "params": {}, "status": "done"},
            {"tool": "get_sales_summary", "domain": "sales", "sub_query": query, "question_index": 0, "result": sales_res, "params": {}, "status": "done"},
        ]
        synth_state = {
            **state,
            "is_multi_tool": True,
            "detected_domain": "marketplace",
            "detected_tool": "get_account_health",
            "verification_status": "VALID",
            "active_tools": active_tools,
            "multi_verification_status": {
                "get_account_health": "VALID",
                "get_deduction_and_settlement_summary": "VALID",
                "get_sales_summary": "VALID",
            },
        }
        ans = format_deterministic_answer(synth_state)
        self.assertIn("Ringkasan Evaluasi Dampak ke Omzet", ans)
        self.assertIn("250", ans)
        self.assertIn("25.00%", ans)
        self.assertIn("15.00%", ans)
        self.assertIn("5.00%", ans)  # 150jt / 3M omzet ERP
        self.assertIn("FETCH_ORDER_TIMEOUT", ans)

    def test_top_n_customer_ranking_routing_extraction_and_billing_guardrail(self):
        q1 = "list pelanggan peringkat teratas siapa saja yg paling banyak beli"
        r1 = router_node(create_initial_state(q1, company_id=3), llm=None)
        self.assertEqual(r1["detected_domain"], "sales")
        self.assertEqual(r1["detected_tool"], "get_contacts")
        self.assertNotIn("get_top_selling_products", r1.get("detected_tools") or [])

        p1 = _fallback_extract_contacts(q1, date(2026, 11, 26))
        self.assertEqual(p1.get("sort_by"), "total_spent")
        self.assertEqual(p1.get("status"), "confirmed")
        self.assertFalse(p1.get("is_vendor"))
        self.assertIsNone(p1.get("search"))

        q2 = "5 pelanggan dengan pembelian terbanyak di semua cabang"
        r2 = router_node(create_initial_state(q2, company_id=4), llm=None)
        self.assertEqual(r2["detected_domain"], "sales")
        self.assertEqual(r2["detected_tool"], "get_contacts")
        self.assertTrue(r2["is_multi_company"])
        p2 = _fallback_extract_contacts(q2, date(2026, 11, 26))
        self.assertEqual(p2.get("limit"), 5)
        self.assertEqual(p2.get("sort_by"), "total_spent")

        # Customer billing guardrail stays intact
        r_bill = router_node(create_initial_state("status pembayaran pelanggan Toko Kertas", company_id=1), llm=None)
        self.assertEqual(r_bill["detected_domain"], "general")
        self.assertIsNone(r_bill["detected_tool"])

    def test_q4_target_run_rate_evaluation(self):
        query = "menurutmu omzet Q4 tahun ini bisa tembus target 6M engga?"
        state = create_initial_state(user_query=query, company_id=3)
        routed = router_node(state, llm=None)
        self.assertEqual(routed["detected_domain"], "sales")
        self.assertEqual(routed["detected_tool"], "get_sales_summary")
        self.assertTrue(routed["is_multi_company"])

        with patch("app.graph.nodes.extractor.get_current_erp_date", return_value=date(2026, 11, 26)):
            ext = extractor_node({**state, **routed}, llm=None, ref_date=date(2026, 11, 26))
        self.assertEqual(ext["extracted_params"].get("date_from"), "2026-10-01")
        self.assertEqual(ext["extracted_params"].get("date_to"), "2026-12-31")

        tool_res = {
            "company_ids": [3, 4, 5],
            "period": {"date_from": "2026-10-01", "date_to": "2026-12-31"},
            "confirmed_metrics": {
                "realized_revenue": 4000000000.0,
                "orders_count": 3200,
                "average_order_value": 1250000.0,
            },
            "pipeline_metrics": {"quotation_orders_count": 100, "quotation_potential_value": 300000000.0},
            "cancelled_metrics": {"cancelled_orders_count": 50, "cancelled_value": 80000000.0},
            "by_branch": [
                {"company_id": 4, "company_name": "Main Store", "order_count": 2000, "total_revenue": 2500000000.0, "average_order_value": 1250000.0},
            ],
            "monthly_breakdown": [
                {"month": "2026-10", "order_count": 1700, "total_revenue": 2100000000.0},
                {"month": "2026-11", "order_count": 1500, "total_revenue": 1900000000.0},
            ],
            "channels_breakdown": [],
        }
        with patch("app.graph.nodes.synthesizer.get_current_erp_date", return_value=date(2026, 11, 26)):
            ans = format_deterministic_answer({
                **state,
                **routed,
                **ext,
                "verification_status": "VALID",
                "tool_result": tool_res,
            })
        self.assertIn("Analisis Proyeksi Target & Run-Rate", ans)
        self.assertIn("57 hari berjalan", ans)
        self.assertIn("sisa 35 hari", ans)
        self.assertIn("Berpotensi Tercapai", ans)
        self.assertIn("Kontribusi Omzet per Cabang", ans)
        self.assertIn("Tren Omzet Bulanan", ans)


class TestPass2WS4DataPrecisionAndScoping(unittest.IsolatedAsyncioTestCase):
    def test_cancelled_orders_render_cancel_reason_and_summary(self):
        state = create_initial_state("List pesanan batal bulan ini", company_id=4)
        state["detected_domain"] = "sales"
        state["detected_tool"] = "get_sales_orders"
        state["verification_status"] = "VALID"
        state["tool_result"] = {
            "total_count": 2,
            "total_amount_sum": 750000.0,
            "status_filter": "cancel",
            "orders": [
                {
                    "company_name": "Main Store",
                    "so_number": "MS/SO/901",
                    "quotation_date": "2026-11-20",
                    "customer_name": "Budi",
                    "customer_phone": "08123",
                    "channel_name": "Shopee",
                    "status": "Cancel",
                    "cancel_reason": "Pembeli mengubah pesanan",
                    "total_amount": 500000.0,
                },
            ],
            "cancel_reason_summary": [
                {"reason": "Pembeli mengubah pesanan", "count": 12},
            ],
        }
        ans = format_deterministic_answer(state)
        self.assertIn("Alasan Batal", ans)
        self.assertIn("Pembeli mengubah pesanan", ans)
        self.assertIn("Rekapitulasi Alasan Pembatalan", ans)

    def test_account_balance_bank_and_cash_and_bank_recon_name_extraction(self):
        bal_p = _fallback_extract_account_balance("berapa total saldo kas & rekening bank aktif saat ini?")
        self.assertEqual(bal_p.get("account_type"), "Bank and Cash")
        self.assertTrue(bal_p.get("non_zero_only"))

        recon_p = _fallback_extract_bank_reconciliation("tolong cek rekonsiliasi bank BCA bulan ini", date(2026, 11, 26))
        self.assertEqual(recon_p.get("bank_name"), "bank BCA")
        self.assertEqual(recon_p.get("date_from"), "2026-11-01")
        self.assertEqual(recon_p.get("date_to"), "2026-11-30")

    async def test_warehouse_company_scan_vs_missing_identifier_fail_fast_and_multimatch(self):
        scan_p = _fallback_extract_stock_level("Cek stok produk di gudang")
        self.assertTrue(scan_p.get("allow_company_scan"))

        mock_client = MagicMock()
        mock_client.fetch_stock_level = AsyncMock(return_value={
            "mode": "company_scan",
            "company_id": 4,
            "company_name": "Main Store",
            "total_items": 1,
            "total_count": 120,
            "items": [
                {
                    "product_id": 50,
                    "code": "91010600",
                    "name": "KARET STEMPEL FLASH",
                    "warehouse_name": "WH Main",
                    "location_name": "Rak A1",
                    "qty_on_hand": 400.0,
                }
            ],
        })
        dispatch_map = {"get_stock_level": mock_client.fetch_stock_level}
        res_scan, err_scan = await _execute_single_tool(
            tool_name="get_stock_level",
            params=scan_p,
            erp_client=mock_client,
            user_token="tok",
            default_company_id=4,
            dispatch_map=dispatch_map,
            user_query="Cek stok produk di gudang",
        )
        self.assertIsNone(err_scan)
        self.assertEqual(res_scan["mode"], "company_scan")

        # Fail-fast without 'gudang' or SKU remains untouched
        plain_p = _fallback_extract_stock_level("Cek stok barang")
        self.assertFalse(plain_p.get("allow_company_scan", False))
        res_fail, err_fail = await _execute_single_tool(
            tool_name="get_stock_level",
            params=plain_p,
            erp_client=mock_client,
            user_token="tok",
            default_company_id=4,
            dispatch_map=dispatch_map,
            user_query="Cek stok barang",
        )
        self.assertIsNone(res_fail)
        self.assertIn("MISSING_PRODUCT_IDENTIFIER", err_fail)

        # Pagination meta and deterministic table rendering for company_scan & multi_match
        scan_state = create_initial_state("Cek stok produk di gudang", company_id=4)
        scan_state.update({
            "detected_domain": "inventory",
            "detected_tool": "get_stock_level",
            "verification_status": "VALID",
            "extracted_params": scan_p,
            "tool_result": res_scan,
        })
        meta = _build_table_query_meta(scan_state)
        self.assertIsNotNone(meta)
        self.assertEqual(meta["total_count"], 120)
        ans_scan = format_deterministic_answer(scan_state)
        self.assertIn("91010600", ans_scan)
        self.assertIn("WH Main — Rak A1", ans_scan)

        mm_state = create_initial_state("Cek stok head l210 new", company_id=4)
        mm_state.update({
            "detected_domain": "inventory",
            "detected_tool": "get_stock_level",
            "verification_status": "VALID",
            "tool_result": {
                "mode": "multi_match",
                "query": "head l210 new",
                "company_name": "Main Store",
                "total_matches": 2,
                "products": [
                    {"code": "HD-L210-ORI", "name": "Print Head Epson L210 New Original", "qty_on_hand": 5.0},
                    {"code": "HD-L210-OEM", "name": "Head L210 / L360 New OEM", "qty_on_hand": 12.0},
                ],
            },
        })
        ans_mm = format_deterministic_answer(mm_state)
        self.assertIn("Daftar Varian Produk Cocok (Top 2)", ans_mm)
        self.assertIn("HD-L210-ORI", ans_mm)
        self.assertIn("HD-L210-OEM", ans_mm)


class TestPass2WS5ScopeMetaAndDrilldown(unittest.TestCase):
    def test_out_of_scope_non_erp_chatter_standalone_and_contextual(self):
        r1 = router_node(create_initial_state("kamu tau tempe kripik?", company_id=1), llm=None)
        self.assertEqual(r1["detected_domain"], "out_of_scope")
        self.assertEqual(r1["query_mode"], "out_of_scope")
        self.assertIsNone(r1["detected_tool"])

        # Contextual prior accounting turn must NOT hijack 'di akunting kamu tau tempe kripik?'
        hist = [
            HumanMessage(content="cek saldo akun kas bulan ini"),
            AIMessage(content="Saldo kas tercatat Rp 50.000.000,00."),
            HumanMessage(content="di akunting kamu tau tempe kripik?"),
        ]
        r2 = router_node(create_initial_state("di akunting kamu tau tempe kripik?", company_id=1, messages=hist), llm=None)
        self.assertEqual(r2["detected_domain"], "out_of_scope")
        self.assertIsNone(r2["detected_tool"])

        # Legitimate greetings / capabilities stay general
        for g in ("Halo", "anda bisa apa aja?", "bisa bantu apa?"):
            rg = router_node(create_initial_state(g, company_id=1), llm=None)
            self.assertEqual(rg["detected_domain"], "general")

    def test_meta_questions_period_company_latest_date(self):
        hist = [
            HumanMessage(content="Cek omzet Q4 tahun ini"),
            AIMessage(content="### Ringkasan Eksekutif Penjualan (Periode 2026-10-01 s/d 2026-12-31)\nData transaksi 2026-11-25."),
        ]
        q_per = "periode yg kamu pakai kapan?"
        st_per = create_initial_state(q_per, company_id=3, authorized_company_ids=[3, 4], messages=hist + [HumanMessage(content=q_per)])
        rt_per = router_node(st_per, llm=None)
        self.assertEqual(rt_per["detected_domain"], "general")
        ans_per = format_deterministic_answer({**st_per, **rt_per})
        self.assertIn("2026-10-01 s/d 2026-12-31", ans_per)

        q_comp = "company mana aja?"
        st_comp = create_initial_state(q_comp, company_id=3, company_ids=[3, 4], authorized_company_ids=[3, 4])
        rt_comp = router_node(st_comp, llm=None)
        self.assertEqual(rt_comp["detected_domain"], "general")
        ans_comp = format_deterministic_answer({**st_comp, **rt_comp})
        self.assertIn("Head Office", ans_comp)
        self.assertIn("Main Store", ans_comp)

        q_dt = "data terbaru tanggal berapa?"
        st_dt = create_initial_state(q_dt, company_id=3, messages=hist + [HumanMessage(content=q_dt)])
        rt_dt = router_node(st_dt, llm=None)
        self.assertEqual(rt_dt["detected_domain"], "general")
        with patch("app.graph.nodes.synthesizer.get_current_erp_date", return_value=date(2026, 11, 26)):
            ans_dt = format_deterministic_answer({**st_dt, **rt_dt})
        self.assertIn("2026-11-26", ans_dt)
        self.assertIn("2026-11-25", ans_dt)

    def test_pnl_account_drilldown_followup_routes_to_general_ledger(self):
        hist = [
            HumanMessage(content="Cek laporan laba rugi Surabaya bulan November 2026"),
            AIMessage(
                content=(
                    "### Laporan Laba Rugi (Periode: 2026-11-01 s/d 2026-11-30)\n\n"
                    "| Kode Akun | Nama Akun | Saldo |\n"
                    "|:---|:---|---:|\n"
                    "| `4100.00.01` | Penjualan Surabaya | Rp 120.000.000,00 |\n"
                )
            ),
            HumanMessage(content="coba berikan detail dari Penjualan Surabaya"),
        ]
        state = create_initial_state("coba berikan detail dari Penjualan Surabaya", company_id=1, messages=hist)
        routed = router_node(state, llm=None)
        self.assertEqual(routed["detected_domain"], "accounting")
        self.assertEqual(routed["detected_tool"], "get_general_ledger")
        self.assertEqual(routed.get("company_id"), 7)

        gl_params = _fallback_extract_general_ledger("coba berikan detail dari Penjualan Surabaya", date(2026, 11, 26), hist)
        self.assertEqual(gl_params.get("account_code"), "4100.00.01")
        self.assertEqual(gl_params.get("date_from"), "2026-11-01")
        self.assertEqual(gl_params.get("date_to"), "2026-11-30")


if __name__ == "__main__":
    unittest.main()
