"""
Unit and integration tests for ERP entity redirect links (entity_links).
Covers safe URL validation, per-tool URL generation across accounting/inventory/
sales/marketplace, multi-company & multi-tool aggregation, synthesizer preservation
of clean Markdown, and FastAPI /api/v1/chat + /api/v1/chat/stream contracts.
"""
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.graph.nodes.formatters.entity_links import (
    build_entity_links,
    build_entity_links_for_tool,
    is_safe_erp_path,
)
from app.graph.state import create_initial_state
from app.schemas.chat import ChatResponse


class TestEntityLinkSafety(unittest.TestCase):
    """Ensure unsafe schemes, external hosts, path traversal, and invalid IDs are rejected."""

    def test_is_safe_erp_path_accepts_relative_internal_paths(self):
        self.assertTrue(is_safe_erp_path("/accounting/journal_entries_edit/42/"))
        self.assertTrue(is_safe_erp_path("/accounting/vendors/bill/edit/7"))
        self.assertTrue(is_safe_erp_path("/inventory/product-variants/99/edit/"))
        self.assertTrue(is_safe_erp_path("/sales/order/quotation/form/15/?tab=lines"))

    def test_is_safe_erp_path_rejects_unsafe_or_external_urls(self):
        for bad in (
            "",
            "   ",
            None,
            "javascript:alert(1)",
            "https://evil.example/accounting/1",
            "http://localhost/foo",
            "//evil.com/phish",
            "/accounting/../../etc/passwd",
            "/path with spaces/1",
            "relative/no-leading-slash",
        ):
            self.assertFalse(is_safe_erp_path(bad), f"Expected rejection for {bad!r}")

    def test_rejects_placeholder_labels_and_invalid_ids(self):
        links = build_entity_links_for_tool(
            "get_journal_entries",
            {
                "entries": [
                    {"id": 0, "number": "JE/2026/001"},
                    {"id": -5, "number": "JE/2026/002"},
                    {"id": True, "number": "JE/2026/003"},
                    {"id": 10, "number": "-"},
                    {"id": 11, "number": "   "},
                    {"id": 12, "number": "N/A"},
                    {"id": 13, "number": "JE/2026/013"},
                ]
            },
        )
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0]["id"], 13)
        self.assertEqual(links[0]["label"], "JE/2026/013")


class TestAccountingEntityLinks(unittest.TestCase):
    """Validate URL routing across JournalEntry subtypes, COA, GL, and Journals."""

    def test_journal_entries_subtype_disambiguation(self):
        payload = {
            "entries": [
                {"id": 10, "number": "JE/2026/010", "type_vendor_bill": None},
                {"id": 11, "number": "BILL/2026/011", "type_vendor_bill": "Purchase Order"},
                {"id": 12, "number": "BILL/2026/012", "type_vendor_bill": "Landed Cost"},
                {"id": 13, "number": "INV/2026/013", "type_vendor_bill": "Sales Order"},
                {"id": 14, "number": "RV/2026/014", "type_vendor_bill": "Refund Vendor"},
                {"id": 15, "number": "RC/2026/015", "type_vendor_bill": "Refund Customer"},
            ]
        }
        links = {item["label"]: item for item in build_entity_links_for_tool("get_journal_entries", payload)}
        self.assertEqual(links["JE/2026/010"]["url"], "/accounting/journal_entries_edit/10/")
        self.assertEqual(links["JE/2026/010"]["type"], "journal_entry")
        self.assertEqual(links["BILL/2026/011"]["url"], "/accounting/vendors/bill/edit/11")
        self.assertEqual(links["BILL/2026/011"]["type"], "vendor_bill")
        self.assertEqual(links["BILL/2026/012"]["url"], "/accounting/vendors/bill/edit/12")
        self.assertEqual(links["INV/2026/013"]["url"], "/accounting/vendors/invoice/edit/13/")
        self.assertEqual(links["INV/2026/013"]["type"], "customer_invoice")
        self.assertEqual(links["RV/2026/014"]["url"], "/accounting/refund-vendor/edit/14")
        self.assertEqual(links["RC/2026/015"]["url"], "/accounting/refund-customer/edit/15")

    def test_vendor_bills_defaults_to_bill_and_links_vendor_contact(self):
        payload = {
            "bills": [
                {"id": 25, "number": "BILL/2026/025", "vendor_id": 88, "vendor_name": "PT Kertas Prima"}
            ]
        }
        links = {item["label"]: item for item in build_entity_links_for_tool("get_vendor_bills", payload)}
        self.assertEqual(links["BILL/2026/025"]["url"], "/accounting/vendors/bill/edit/25")
        self.assertEqual(links["PT Kertas Prima"]["url"], "/contact/contact/88/update/")
        self.assertEqual(links["PT Kertas Prima"]["type"], "contact")

    def test_general_ledger_account_balance_bs_pnl_and_bank_reconciliation(self):
        gl_links = {
            item["label"]: item
            for item in build_entity_links_for_tool(
                "get_general_ledger",
                {
                    "account_id": 5,
                    "account_code": "1101.001",
                    "items": [
                        {"journal_entry_id": 101, "entry_number": "INV/2026/101", "type_vendor_bill": "Sales Order"}
                    ],
                },
            )
        }
        self.assertEqual(gl_links["1101.001"]["url"], "/accounting/chart_of_accounts/edit/5")
        self.assertEqual(gl_links["INV/2026/101"]["url"], "/accounting/vendors/invoice/edit/101/")

        bs_links = {
            item["label"]: item
            for item in build_entity_links_for_tool(
                "get_balance_sheet",
                {
                    "groups": [
                        {
                            "group_name": "Assets",
                            "types": [
                                {"accounts": [{"account_id": 7, "account_code": "1102.001"}]}
                            ],
                        }
                    ]
                },
            )
        }
        self.assertEqual(bs_links["1102.001"]["url"], "/accounting/chart_of_accounts/edit/7")

        bank_links = {
            item["label"]: item
            for item in build_entity_links_for_tool(
                "get_bank_reconciliation_summary",
                {"journal_id": 14, "journal_name": "Bank BCA 2424", "bank_account_id": 6, "bank_account_code": "1102.002"},
            )
        }
        self.assertEqual(bank_links["Bank BCA 2424"]["url"], "/accounting/journals/update/14")
        self.assertEqual(bank_links["1102.002"]["url"], "/accounting/chart_of_accounts/edit/6")


class TestInventorySalesAndMarketplaceEntityLinks(unittest.TestCase):
    """Validate URL generation for inventory, sales, contact, and marketplace tools."""

    def test_inventory_entities(self):
        stock_links = {
            item["label"]: item
            for item in build_entity_links_for_tool(
                "get_stock_level",
                {
                    "product_id": 310,
                    "code": "91010600",
                    "warehouse_breakdown": [
                        {"warehouse_id": 4, "warehouse_name": "Gudang Pusat", "location_id": 12, "location_name": "Rak A1"}
                    ],
                },
            )
        }
        self.assertEqual(stock_links["91010600"]["url"], "/inventory/product-variants/310/edit/")
        self.assertEqual(stock_links["Gudang Pusat"]["url"], "/inventory/warehouse/4/edit/")
        self.assertEqual(stock_links["Rak A1"]["url"], "/inventory/location/12/edit/")

        hist_links = {
            item["label"]: item
            for item in build_entity_links_for_tool(
                "get_stock_history",
                {
                    "movements": [
                        {
                            "id": 1,
                            "product": {"id": 310, "code": "91010600"},
                            "source_id": 55,
                            "source_ref": "WH/OUT/00055",
                            "warehouse": {"id": 4, "name": "Main Store"},
                            "location": {"id": 12, "full_name": "Main Store/Stock"},
                        }
                    ]
                },
            )
        }
        self.assertEqual(hist_links["91010600"]["url"], "/inventory/product-variants/310/edit/")
        self.assertEqual(hist_links["WH/OUT/00055"]["url"], "/inventory/stock_transfer/55/edit/")
        self.assertEqual(hist_links["Main Store"]["url"], "/inventory/warehouse/4/edit/")
        self.assertEqual(hist_links["Main Store/Stock"]["url"], "/inventory/location/12/edit/")

        scrap_links = {
            item["label"]: item
            for item in build_entity_links_for_tool(
                "get_scrap_records",
                {"scrap_records": [{"id": 77, "ref": "SP/00077"}, {"id": 78, "reference": "SP/00078"}]},
            )
        }
        self.assertEqual(scrap_links["SP/00077"]["url"], "/inventory/scrap/77/edit/")
        self.assertEqual(scrap_links["SP/00078"]["url"], "/inventory/scrap/78/edit/")

        cat_links = {
            item["label"]: item
            for item in build_entity_links_for_tool(
                "get_product_categories",
                {"categories": [{"id": 9, "name": "Stempel Flash"}]},
            )
        }
        self.assertEqual(cat_links["Stempel Flash"]["url"], "/inventory/product_categories/9/edit/")

    def test_sales_and_contact_entities(self):
        contact_links = {
            item["label"]: item
            for item in build_entity_links_for_tool(
                "get_contacts",
                {
                    "contacts": [
                        {
                            "id": 501,
                            "name": "Budi Santoso",
                            "recent_orders": [{"id": 9001, "so_number": "SO/2026/9001"}],
                        }
                    ]
                },
            )
        }
        self.assertEqual(contact_links["Budi Santoso"]["url"], "/contact/contact/501/update/")
        self.assertEqual(contact_links["SO/2026/9001"]["url"], "/sales/order/quotation/form/9001/")

        so_detail_links = {
            item["label"]: item
            for item in build_entity_links_for_tool(
                "get_sales_order_detail",
                {
                    "id": 9002,
                    "so_number": "SO/2026/9002",
                    "customer": {"id": 502, "name": "CV Maju"},
                    "lines": [{"product_id": 311, "code": "SKU-311"}],
                },
            )
        }
        self.assertEqual(so_detail_links["SO/2026/9002"]["url"], "/sales/order/quotation/form/9002/")
        self.assertEqual(so_detail_links["CV Maju"]["url"], "/contact/contact/502/update/")
        self.assertEqual(so_detail_links["SKU-311"]["url"], "/inventory/product-variants/311/edit/")

    def test_marketplace_entities(self):
        mp_links = {
            item["label"]: item
            for item in build_entity_links_for_tool(
                "get_deduction_and_settlement_summary",
                {
                    "by_store": [{"account_id": 3, "account_name": "FastPrint Shopee"}],
                    "orders": [
                        {
                            "quotation_id": 701,
                            "so_number": "SO/2026/0701",
                            "journal_entry_id": 801,
                            "journal_number": "JE/2026/0801",
                        }
                    ],
                },
            )
        }
        self.assertEqual(mp_links["FastPrint Shopee"]["url"], "/marketplace/account/3/edit/")
        self.assertEqual(mp_links["SO/2026/0701"]["url"], "/sales/order/quotation/form/701/")
        self.assertEqual(mp_links["JE/2026/0801"]["url"], "/accounting/journal_entries_edit/801/")

        map_links = {
            item["label"]: item
            for item in build_entity_links_for_tool(
                "get_product_mapping_status",
                {
                    "items": [
                        {
                            "mapping_id": 44,
                            "account_id": 3,
                            "account_name": "FastPrint Shopee",
                            "erp_product": {"id": 312, "sku": "ERP-312"},
                            "mp_product": {"sku_mp": "MP-SKU-44"},
                        }
                    ]
                },
            )
        }
        self.assertEqual(map_links["ERP-312"]["url"], "/inventory/product-variants/312/edit/")
        self.assertEqual(map_links["FastPrint Shopee"]["url"], "/marketplace/account/3/edit/")
        self.assertEqual(map_links["MP-SKU-44"]["url"], "/marketplace/mapping/44/edit/")

        log_links = {
            item["label"]: item
            for item in build_entity_links_for_tool(
                "get_sync_execution_logs",
                {
                    "stock_logs": [{"account_id": 3, "account_name": "Store A", "product_id": 21, "sku": "SKU-21"}],
                    "price_logs": [{"account_id": 4, "account_name": "Store B", "product_id": 22, "erp_sku": "SKU-22"}],
                    "pcls_logs": [{"product_id": 23, "sku": "SKU-23"}],
                },
            )
        }
        self.assertEqual(log_links["SKU-21"]["url"], "/inventory/product-variants/21/edit/")
        self.assertEqual(log_links["SKU-22"]["url"], "/inventory/product-variants/22/edit/")
        self.assertEqual(log_links["SKU-23"]["url"], "/inventory/product-variants/23/edit/")
        self.assertEqual(log_links["Store A"]["url"], "/marketplace/account/3/edit/")


class TestStateOrchestrationAndApiEntityLinks(unittest.IsolatedAsyncioTestCase):
    """Verify multi-tool/multi-company state collection, clean Markdown, and FastAPI responses."""

    def test_build_entity_links_unwraps_envelope_and_skips_non_data_modes(self):
        st = create_initial_state(user_query="cek jurnal", company_id=1)
        st["detected_domain"] = "accounting"
        st["detected_tool"] = "get_journal_entries"
        st["verification_status"] = "VALID"
        st["tool_raw_result"] = {
            "status": "success",
            "data": {"entries": [{"id": 100, "number": "JE/2026/100"}]},
        }
        links = build_entity_links(st)
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0]["url"], "/accounting/journal_entries_edit/100/")

        for mode, dom, vstat in (
            ("general", "general", "VALID"),
            ("clarification", "clarification", "CLARIFICATION"),
            ("live_erp", "accounting", "ERROR"),
        ):
            bad_state = dict(st)
            bad_state["query_mode"] = mode
            bad_state["detected_domain"] = dom
            bad_state["verification_status"] = vstat
            self.assertEqual(build_entity_links(bad_state), [])

    def test_multi_company_and_multi_tool_aggregation_deduplicates(self):
        st = create_initial_state(user_query="cek stok semua cabang", company_id=1)
        st["detected_domain"] = "inventory"
        st["detected_tool"] = "get_product_info"
        st["verification_status"] = "VALID"
        st["multi_company_results"] = [
            {
                "company_id": 1,
                "detected_tool": "get_product_info",
                "raw_result": {"products": [{"id": 10, "code": "SKU-10"}]},
            },
            {
                "company_id": 2,
                "detected_tool": "get_product_info",
                "raw_result": {"products": [{"id": 11, "code": "SKU-11"}, {"id": 10, "code": "SKU-10"}]},
            },
        ]
        links = build_entity_links(st)
        urls = {item["url"] for item in links}
        self.assertEqual(urls, {"/inventory/product-variants/10/edit/", "/inventory/product-variants/11/edit/"})

    async def test_synthesizer_emits_entity_links_with_clean_markdown(self):
        from app.core.config import settings
        from app.graph.nodes.synthesizer import asynthesizer_node

        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "Ringkasan 2 jurnal."
        mock_resp.response_metadata = {}
        mock_llm.ainvoke = AsyncMock(return_value=mock_resp)
        state = create_initial_state(user_query="Tampilkan 2 jurnal terbaru", company_id=1)
        state["detected_domain"] = "accounting"
        state["detected_tool"] = "get_journal_entries"
        state["query_mode"] = "live_erp"
        state["verification_status"] = "VALID"
        raw_payload = {
            "total_count": 2,
            "entries": [
                {
                    "id": 201,
                    "number": "JE/2026/0201",
                    "date": "2026-11-26",
                    "journal_name": "General Journal",
                    "partner_name": "PT Abadi",
                    "desc": "Penyesuaian",
                    "status": "post",
                    "total_debit": 100000.0,
                    "total_credit": 100000.0,
                    "is_balanced": True,
                    "type_vendor_bill": None,
                },
                {
                    "id": 202,
                    "number": "BILL/2026/0202",
                    "date": "2026-11-26",
                    "journal_name": "Vendor Bill",
                    "partner_name": "PT Abadi",
                    "desc": "Pembelian",
                    "status": "post",
                    "total_debit": 250000.0,
                    "total_credit": 250000.0,
                    "is_balanced": True,
                    "type_vendor_bill": "Purchase Order",
                },
            ],
        }
        state["tool_raw_result"] = {"status": "success", "data": raw_payload}
        state["tool_result"] = raw_payload

        with patch.object(settings, "SYNTHESIZER_FAST_DETERMINISTIC", True), \
             patch("app.graph.nodes.synthesizer.adispatch_custom_event", new_callable=AsyncMock):
            out = await asynthesizer_node(state, llm=mock_llm)

        final_md = out["final_answer"]
        self.assertIn("JE/2026/0201", final_md)
        self.assertIn("BILL/2026/0202", final_md)
        # Markdown stays pure text with no injected URL links
        self.assertNotIn("/accounting/journal_entries_edit/", final_md)
        self.assertNotIn("/accounting/vendors/bill/edit/", final_md)

        links_map = {item["label"]: item["url"] for item in out["entity_links"]}
        self.assertEqual(links_map["JE/2026/0201"], "/accounting/journal_entries_edit/201/")
        self.assertEqual(links_map["BILL/2026/0202"], "/accounting/vendors/bill/edit/202")

    def test_chat_response_and_endpoints_include_entity_links(self):
        from fastapi.testclient import TestClient
        from app.main import app

        default_resp = ChatResponse(final_answer="Halo")
        self.assertEqual(default_resp.entity_links, [])

        expected_links = [
            {
                "label": "JE/2026/001",
                "type": "journal_entry",
                "id": 99,
                "url": "/accounting/journal_entries_edit/99/",
            }
        ]
        client = TestClient(app)
        with patch("app.main.arun_workflow", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {
                "final_answer": "| Nomor |\n|---|\n| JE/2026/001 |",
                "detected_domain": "accounting",
                "verification_status": "VALID",
                "entity_links": expected_links,
            }
            resp = client.post("/api/v1/chat", json={"query": "cek 1 jurnal", "company_id": 1})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["entity_links"], expected_links)

    async def test_stream_endpoint_complete_event_includes_entity_links(self):
        import httpx
        from langgraph.checkpoint.memory import MemorySaver
        from app.graph.workflow import create_workflow_graph
        from app.main import app
        from app.services.django_client import DjangoCoreClient

        isolated_graph = create_workflow_graph(checkpointer=MemorySaver())
        mock_client = MagicMock(spec=DjangoCoreClient)
        mock_client.aclose = AsyncMock()
        mock_client.fetch_journal_entries = AsyncMock(
            return_value={
                "total_count": 1,
                "entries": [
                    {
                        "id": 512,
                        "number": "JE/2026/0512",
                        "date": "2026-11-26",
                        "journal_name": "General Journal",
                        "partner_name": "CV Jaya",
                        "desc": "Koreksi",
                        "status": "post",
                        "total_debit": 15000.0,
                        "total_credit": 15000.0,
                        "is_balanced": True,
                        "type_vendor_bill": None,
                    }
                ],
            }
        )

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
            with patch("app.graph.workflow.default_workflow_graph", isolated_graph), \
                 patch("app.graph.nodes.tool_node.DjangoCoreClient", return_value=mock_client):
                resp = await http_client.post(
                    "/api/v1/chat/stream",
                    json={"query": "Tampilkan daftar transaksi jurnal terakhir", "company_id": 1, "thread_id": "test_entity_links_stream"},
                )

        self.assertEqual(resp.status_code, 200)
        complete_payload = None
        for block in resp.text.strip().split("\n\n"):
            lines = block.split("\n")
            if any(ln.strip() == "event: complete" for ln in lines):
                for ln in lines:
                    if ln.strip().startswith("data:"):
                        complete_payload = json.loads(ln.strip()[5:].strip())
        self.assertIsNotNone(complete_payload)
        links = complete_payload.get("entity_links") or []
        self.assertTrue(
            any(item["label"] == "JE/2026/0512" and item["url"] == "/accounting/journal_entries_edit/512/" for item in links)
        )


if __name__ == "__main__":
    unittest.main()
