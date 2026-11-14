"""
Unit and integration tests for UI Chat test cases:
- Tanya stok (inventory.get_stock_level)
- Tanya nilai laba rugi semua cabang + HO (accounting.get_profit_and_loss)
- Tanya harga barang (inventory.get_product_info)
- Guardrails & non-ERP queries
"""
import unittest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.graph.nodes.extractor import (
    resolve_all_branch_company_ids,
    COMPANY_NAME_MAP,
)
from app.graph.nodes.synthesizer import (
    format_deterministic_answer,
    _format_multi_company_presentation,
    _format_rupiah,
)
from app.graph.state import AgentState, create_initial_state
from app.schemas.chat import ChatRequest


class TestUIChatScenarios(unittest.TestCase):
    """Test suite validating UI Chat test cases and multi-company rollups."""

    def setUp(self):
        self.client = TestClient(app)

    def test_tc_stk_01_stock_formatting_single_product(self):
        """TC-STK-01: Verifies formatting of stock response for SKU 91010600."""
        state: AgentState = {
            "detected_domain": "inventory",
            "detected_tool": "get_stock_level",
            "tool_result": {
                "products": [
                    {
                        "code": "91010600",
                        "name": "FP KARET STEMPEL FLASH BULAT 42",
                        "company_name": "Kantor Pusat",
                        "location_name": "Gudang Utama",
                        "quantity": 30.0,
                    }
                ],
                "total_count": 1,
            },
        }
        answer = format_deterministic_answer(state)
        self.assertIn("Informasi Ketersediaan Stok Barang", answer)
        self.assertIn("`91010600`", answer)
        self.assertIn("FP KARET STEMPEL FLASH BULAT 42", answer)
        self.assertIn("30", answer)
        self.assertIn("unit", answer)

    def test_tc_stk_03_stock_breakdown_location(self):
        """TC-STK-03: Verifies sub-table rendering for warehouse and location breakdown."""
        state: AgentState = {
            "detected_domain": "inventory",
            "detected_tool": "get_stock_level",
            "tool_result": {
                "products": [
                    {
                        "code": "91010600",
                        "name": "FP KARET STEMPEL FLASH BULAT 42",
                        "quantity": 30.0,
                    }
                ],
                "breakdown": [
                    {
                        "warehouse_name": "HEAD",
                        "location_name": "Stock",
                        "qty": 30.0,
                    }
                ],
            },
        }
        answer = format_deterministic_answer(state)
        self.assertIn("Rincian Stok per Lokasi / Rak", answer)
        self.assertIn("HEAD", answer)
        self.assertIn("Stock", answer)
        self.assertIn("30", answer)
        self.assertIn("unit", answer)

    def test_tc_pl_01_all_companies_checkbox_rollup(self):
        """TC-PL-01: Verifies multi-company P&L table format and TOTAL GABUNGAN calculation."""
        all_cids = sorted(list(COMPANY_NAME_MAP.keys()))
        mc_results = {}
        for cid in all_cids:
            mc_results[cid] = {
                "summary": {
                    "total_revenue": 10000000.0,
                    "cogs": 6000000.0,
                    "gross_profit": 4000000.0,
                    "net_profit": 2000000.0,
                }
            }

        state: AgentState = {
            "detected_domain": "accounting",
            "detected_tool": "get_profit_and_loss",
            "is_multi_company": True,
            "company_ids": all_cids,
            "multi_company_results": mc_results,
        }
        output = _format_multi_company_presentation(state)
        self.assertIn("Laporan Laba Rugi Multi-Perusahaan (Combined Profit & Loss)", output)
        self.assertIn("TOTAL GABUNGAN", output)
        self.assertIn("Kantor Pusat", output)
        self.assertIn("Surabaya", output)
        self.assertIn("Jakarta", output)
        expected_total_net = len(all_cids) * 2000000.0
        self.assertIn(_format_rupiah(expected_total_net), output)

    def test_tc_pl_02_all_companies_keyword_extraction(self):
        """TC-PL-02: Verifies extractor identifies 'semua cabang dan HO' keywords."""
        query = "Tampilkan laporan laba rugi tahun 2026 untuk semua cabang dan HO"
        cids, is_all = resolve_all_branch_company_ids(query)
        self.assertTrue(is_all)
        self.assertEqual(len(cids), len(COMPANY_NAME_MAP))
        self.assertIn(1, cids)  # Kantor Pusat
        self.assertIn(3, cids)  # Head Office
        self.assertIn(7, cids)  # Surabaya
        self.assertIn(8, cids)  # Jakarta

    def test_tc_prc_01_product_price_formatting(self):
        """TC-PRC-01: Verifies product price rendering with Rupiah format."""
        state: AgentState = {
            "detected_domain": "inventory",
            "detected_tool": "get_product_info",
            "tool_result": {
                "products": [
                    {
                        "code": "1000020201",
                        "name": "FP CANON DYE ANTI UV 1000ML - BLACK",
                        "last_unit_price": 116050.0,
                        "default_cost": 75000.0,
                    }
                ],
                "total_count": 1,
            },
        }
        answer = format_deterministic_answer(state)
        self.assertIn("Informasi Master Produk", answer)
        self.assertIn("1000020201", answer)
        self.assertIn("Harga Jual: Rp 116.050", answer)
        self.assertIn("Estimasi HPP: Rp 75.000", answer)

    def test_tc_chat_request_with_checkbox_company_ids(self):
        """Verifies API ChatRequest accepts company_ids list representing checked branches."""
        req = ChatRequest(
            query="Berapa laba rugi tahun 2026?",
            company_ids=[1, 2, 3, 7, 8],
        )
        self.assertEqual(req.company_ids, [1, 2, 3, 7, 8])
        self.assertEqual(req.company_id, 1)


if __name__ == "__main__":
    unittest.main()
