"""
Pass 1 (WS1 & WS2) regression tests covering verified defects from production QA history:
- get_stock_history nested key rendering
- get_contacts channels/sales_phones/contact_id dedup and clean stopword hygiene
- Multi-company get_product_info price table & zero-activity branch omission footnote
- Plan deduplication & verbatim preservation on breakdown/table cues
- Temporal extraction (bare years, quarters, semesters, months) & SKU-UNKNOWN-999 preservation
- Multi-turn status="cancel" + multi-company scope recovery and HEAD OFFICE (id=3) auto-expansion
- _build_table_query_meta support for get_account_balance and cross-branch native sales tools
"""
from datetime import date
import unittest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from app.core.context import clean_product_identifier
from app.graph.nodes.extractor import (
    _fallback_extract_contacts,
    _fallback_extract_sales_orders,
    _fallback_extract_stock_level,
    _fallback_extract_top_selling_products,
    extractor_node,
    resolve_relative_dates,
)
from app.graph.nodes.router import router_node
from app.graph.nodes.synthesizer import (
    _build_table_query_meta,
    asynthesizer_node,
    format_deterministic_answer,
)
from app.graph.state import create_initial_state


class TestPass1SynthesizerRegressions(unittest.IsolatedAsyncioTestCase):
    def test_stock_history_maps_nested_backend_keys(self):
        state = create_initial_state("Cek kartu stok SKU 91010600", company_id=3)
        state["detected_domain"] = "inventory"
        state["detected_tool"] = "get_stock_history"
        state["verification_status"] = "VALID"
        state["tool_result"] = {
            "total_count": 65,
            "history": [
                {
                    "id": 1001,
                    "date": "2026-11-25T14:32:00+07:00",
                    "source_ref": "MS/SO/0585111",
                    "identified_number": "TRF/001",
                    "product": {"id": 50, "code": "91010600", "name": "KARET STEMPEL FLASH"},
                    "warehouse": {"id": 4, "name": "Main Store"},
                    "location": {"id": 9, "name": "Stock"},
                    "operation_type": "Delivery Orders",
                    "qty_change": -5.0,
                    "final_stock": 25.0,
                }
            ],
        }
        md = format_deterministic_answer(state)
        self.assertIn("2026-11-25 14:32", md)
        self.assertIn("`MS/SO/0585111`", md)
        self.assertIn("`91010600` KARET STEMPEL FLASH", md)
        self.assertIn("Main Store / Stock", md)
        self.assertIn("**-5**", md)
        self.assertIn("| 25 |", md)

    def test_contacts_renders_channels_sales_phones_and_contact_id_dedup(self):
        state = create_initial_state("Tampilkan detail kontak lengkap dari semua pelanggan", company_id=4)
        state["detected_domain"] = "sales"
        state["detected_tool"] = "get_contacts"
        state["verification_status"] = "VALID"
        state["is_multi_company"] = True
        state["multi_company_results"] = {
            4: {
                "contacts": [
                    {
                        "contact_id": 777,
                        "name": "Budi Santoso",
                        "phone": "081234567890",
                        "city": "Surabaya",
                        "channels": ["WA", "Shopee"],
                        "sales_phones": ["08111222333"],
                        "orders_count": 3,
                        "total_spent": 4500000.0,
                        "recent_orders": [{"so_number": "MS/SO/001", "company_name": "Main Store", "status": "Paid"}],
                    }
                ]
            },
            7: {
                "contacts": [
                    {
                        "contact_id": 777,
                        "name": "Budi Santoso",
                        "phone": "081234567890",
                        "city": "Surabaya",
                        "channels": ["WA", "Shopee"],
                        "sales_phones": ["08111222333"],
                        "orders_count": 3,
                        "total_spent": 4500000.0,
                        "recent_orders": [],
                    }
                ]
            },
        }
        md = format_deterministic_answer(state)
        self.assertEqual(md.count("Budi Santoso"), 1)
        self.assertIn("WA, Shopee", md)
        self.assertIn("08111222333", md)

    def test_multi_company_omits_zero_rows_and_adds_footnote_plus_mc_product_price(self):
        pl_state = create_initial_state("Laba rugi semua cabang", company_id=3)
        pl_state["detected_domain"] = "accounting"
        pl_state["detected_tool"] = "get_profit_and_loss"
        pl_state["verification_status"] = "VALID"
        pl_state["is_multi_company"] = True
        pl_state["extracted_params"] = {"date_from": "2026-10-01", "date_to": "2026-12-31"}
        pl_state["multi_company_results"] = {
            1: {"summary": {"total_revenue": 0.0, "cogs": 0.0, "gross_profit": 0.0, "net_profit": 0.0}},
            4: {"summary": {"total_revenue": 100000000.0, "cogs": 60000000.0, "gross_profit": 40000000.0, "net_profit": 25000000.0}},
            17: {"summary": {"total_revenue": 0.0, "cogs": 0.0, "gross_profit": 0.0, "net_profit": 0.0}},
        }
        pl_md = format_deterministic_answer(pl_state)
        self.assertIn("Main Store", pl_md)
        self.assertIn("Periode: 2026-10-01 s/d 2026-12-31", pl_md)
        self.assertNotIn("| 1 | Kantor Pusat", pl_md)
        self.assertIn("Kantor Pusat", pl_md)
        self.assertIn("Fastmart", pl_md)
        self.assertIn("Cabang tanpa aktivitas", pl_md)

        prod_state = create_initial_state("harga l210 di semua cabang", company_id=3)
        prod_state["detected_domain"] = "inventory"
        prod_state["detected_tool"] = "get_product_info"
        prod_state["verification_status"] = "VALID"
        prod_state["is_multi_company"] = True
        prod_state["multi_company_results"] = {
            4: {
                "products": [
                    {
                        "code": "2610128",
                        "name": "HEAD L210",
                        "effective_sales_price": 850000.0,
                        "price_source": "pricelist:Offline MS",
                        "uom": {"name": "Pcs"},
                    }
                ]
            },
            17: {"products": []},
        }
        prod_md = format_deterministic_answer(prod_state)
        self.assertIn("Perbandingan Harga Produk Multi-Cabang", prod_md)
        self.assertIn("Rp 850.000,00", prod_md)
        self.assertIn("pricelist:Offline MS", prod_md)
        self.assertIn("Fastmart", prod_md)

    async def test_duplicate_plans_deduped_and_breakdown_not_scalar_collapsed(self):
        shared_res = {
            "total_count": 2,
            "total_amount_sum": 3000000.0,
            "status_filter": "cancel",
            "orders": [
                {"so_number": "MS/SO/001", "quotation_date": "2026-11-10", "customer_name": "Adi", "status": "Cancel", "total_amount": 1500000.0},
                {"so_number": "MS/SO/002", "quotation_date": "2026-11-11", "customer_name": "Budi", "status": "Cancel", "total_amount": 1500000.0},
            ],
        }
        dup_state = create_initial_state("jabarkan semua datanya dalam table", company_id=4)
        dup_state["detected_domain"] = "sales"
        dup_state["detected_tool"] = "get_sales_orders"
        dup_state["verification_status"] = "VALID"
        dup_state["sub_questions"] = ["jabarkan semua datanya dalam table"]
        dup_state["is_multi_tool"] = True
        dup_state["active_tools"] = [
            {"tool": "get_sales_orders", "domain": "sales", "company_id": 4, "company_ids": [4], "params": {"status": "cancel"}, "status": "success", "result": shared_res},
            {"tool": "get_sales_orders", "domain": "sales", "company_id": 4, "company_ids": [4], "params": {"status": "cancel"}, "status": "success", "result": shared_res},
        ]

        det = format_deterministic_answer(dup_state)
        self.assertEqual(det.count("MS/SO/001"), 1)

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(content='Total pesanan Rp 3.000.000,00.\n```json:followups\n[]\n```')
        )
        synth_out = await asynthesizer_node(dup_state, llm=mock_llm)
        self.assertIn("| 1 |", synth_out["final_answer"])
        self.assertIn("MS/SO/001", synth_out["final_answer"])
        self.assertIn("MS/SO/002", synth_out["final_answer"])

    def test_table_query_meta_for_account_balance_and_native_mc_sales(self):
        acc_state = {
            "verification_status": "VALID",
            "detected_domain": "accounting",
            "detected_tool": "get_account_balance",
            "company_id": 4,
            "extracted_params": {},
            "is_multi_company": False,
            "is_multi_tool": False,
            "tool_result": {
                "company_id": 4,
                "total_count": 72,
                "accounts": [{"account_code": "11100", "account_name": "Kas", "ending_balance": 1000.0}] * 50,
            },
        }
        acc_meta = _build_table_query_meta(acc_state)
        self.assertIsNotNone(acc_meta)
        self.assertEqual(acc_meta["tool"], "get_account_balance")
        self.assertEqual(acc_meta["total_count"], 72)

        mc_sales_state = {
            "verification_status": "VALID",
            "detected_domain": "sales",
            "detected_tool": "get_sales_orders",
            "company_id": 4,
            "company_ids": [4, 5, 6, 7],
            "is_multi_company": True,
            "multi_company_results": {},
            "is_multi_tool": False,
            "extracted_params": {"status": "cancel"},
            "tool_result": {
                "company_ids": [4, 5, 6, 7],
                "total_count": 120,
                "orders": [{"so_number": "MS/SO/1"}] * 50,
            },
        }
        sales_meta = _build_table_query_meta(mc_sales_state)
        self.assertIsNotNone(sales_meta)
        self.assertEqual(sales_meta["params"]["company_ids"], [4, 5, 6, 7])


class TestPass1ExtractorAndRouterRegressions(unittest.TestCase):
    def test_temporal_resolution_quarters_bare_year_semesters_months(self):
        ref = date(2026, 11, 25)
        self.assertEqual(resolve_relative_dates("laba rugi di mainstore 2025", ref), ("2025-01-01", "2025-12-31"))
        self.assertEqual(resolve_relative_dates("buat q4 tahun 2026 list omzet nya", ref), ("2026-10-01", "2026-12-31"))
        self.assertEqual(resolve_relative_dates("penjualan kuartal 2", ref), ("2026-04-01", "2026-06-30"))
        self.assertEqual(resolve_relative_dates("omzet semester 1 2025", ref), ("2025-01-01", "2025-06-30"))
        self.assertEqual(resolve_relative_dates("laporan bulan november 2026", ref), ("2026-11-01", "2026-11-30"))
        self.assertEqual(resolve_relative_dates("jurnal JE/2026/001", ref), (None, None))
        self.assertEqual(resolve_relative_dates("cek stok SKU-2026", ref), (None, None))

    def test_clean_product_identifier_preserves_sku_unknown_999(self):
        self.assertEqual(clean_product_identifier("SKU-UNKNOWN-999"), "SKU-UNKNOWN-999")
        self.assertEqual(clean_product_identifier("SKU: 100040203"), "100040203")
        fb = _fallback_extract_stock_level("Cek stok untuk SKU-UNKNOWN-999 atau gudang surabaya")
        self.assertEqual(fb.get("code"), "SKU-UNKNOWN-999")

    def test_contact_and_top_sales_stopwords_clean(self):
        ref = date(2026, 11, 25)
        c_params = _fallback_extract_contacts("Tampilkan detail kontak lengkap dari semua pelanggan", ref)
        self.assertIsNone(c_params.get("search"))

        s_params = _fallback_extract_top_selling_products("buat q4 tahun 2026 list omzet nya", ref)
        self.assertIsNone(s_params.get("search"))
        self.assertEqual(s_params.get("date_from"), "2026-10-01")
        self.assertEqual(s_params.get("date_to"), "2026-12-31")

    def test_multiturn_cancel_status_and_multi_company_persistence(self):
        ref = date(2026, 11, 25)
        history = [
            HumanMessage(content="List semua pesanan penjualan atau quotation yang dibatalkan tahun ini"),
            AIMessage(content="### Daftar Pesanan Penjualan & Quotation (Status: `cancel`)..."),
            HumanMessage(content="semua cabang"),
            AIMessage(content="### Ringkasan Pesanan Penjualan Multi-Cabang (Periode: 2026-01-01 s/d 2026-12-31)..."),
            HumanMessage(content="jabarkan semua datanya dalam table"),
        ]
        state = create_initial_state(
            user_query="jabarkan semua datanya dalam table",
            company_id=4,
            is_multi_company=False,
            messages=history,
        )
        routed = router_node(state, llm=None)
        self.assertEqual(routed["detected_tool"], "get_sales_orders")
        self.assertTrue(routed["is_multi_company"])
        self.assertGreater(len(routed["company_ids"]), 1)

        ext = extractor_node({**state, **routed}, llm=None, ref_date=ref)
        self.assertEqual(ext["extracted_params"].get("status"), "cancel")
        self.assertEqual(ext["extracted_params"].get("date_from"), "2026-01-01")
        self.assertEqual(ext["extracted_params"].get("date_to"), "2026-12-31")

    def test_head_office_auto_expands_sales_mp_and_keeps_accounting_stock_strict(self):
        sales_state = create_initial_state(
            "Total omzet penjualan bulan ini",
            company_id=3,
            authorized_company_ids=[3, 4, 5, 6, 7],
        )
        sales_routed = router_node(sales_state, llm=None)
        self.assertTrue(sales_routed["is_multi_company"])
        self.assertEqual(sales_routed["company_ids"], [3, 4, 5, 6, 7])

        mp_state = create_initial_state(
            "Cek order marketplace yang overdue SLA",
            company_id=3,
            authorized_company_ids=[3, 4, 5],
        )
        mp_routed = router_node(mp_state, llm=None)
        self.assertTrue(mp_routed["is_multi_company"])

        # Accounting & warehouse stock stay strict single-company unless 'semua cabang' is asked
        pl_state = create_initial_state("Laba rugi tahun ini", company_id=3, authorized_company_ids=[3, 4, 5])
        self.assertFalse(router_node(pl_state, llm=None).get("is_multi_company", False))

        stk_state = create_initial_state("Cek stok 91010600", company_id=3, authorized_company_ids=[3, 4, 5])
        self.assertFalse(router_node(stk_state, llm=None).get("is_multi_company", False))

        # Explicit HQ pin stays single-company
        hq_pin_state = create_initial_state("Total omzet penjualan head office bulan ini", company_id=3, authorized_company_ids=[3, 4, 5])
        self.assertFalse(router_node(hq_pin_state, llm=None).get("is_multi_company", False))


if __name__ == "__main__":
    unittest.main()
