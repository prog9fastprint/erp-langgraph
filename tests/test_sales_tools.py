"""
Unit tests for Sales Domain AI Tools (get_top_selling_products).
Validates schema validation, client dispatch, router routing, parameter extraction,
tool execution, and single & multi-company synthesizer presentation.
"""
import unittest
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

from app.graph.state import create_initial_state
from app.graph.nodes.router import router_node
from app.graph.nodes.extractor import extractor_node
from app.graph.nodes.tool_node import tool_node
from app.graph.nodes.verifier import _calculate_records_count
from app.graph.nodes.synthesizer import synthesizer_node, format_deterministic_answer
from app.schemas.tools import TopSellingProductsFilterSchema, TOOL_DOMAIN_MAP
from app.services.django_client import DjangoCoreClient
from langchain_core.messages import HumanMessage


class TestSalesTools(unittest.IsolatedAsyncioTestCase):
    """Test suite for get_top_selling_products and sales domain orchestration."""

    def test_schema_validation(self):
        # Valid schema
        schema = TopSellingProductsFilterSchema(
            company_id=7,
            date_from="2026-01-01",
            date_to="2026-12-31",
            limit=10,
            search="HVS",
        )
        self.assertEqual(schema.company_id, 7)
        self.assertEqual(schema.date_from, "2026-01-01")
        self.assertEqual(schema.date_to, "2026-12-31")
        self.assertEqual(schema.limit, 10)
        self.assertEqual(schema.search, "HVS")

        # Registered in TOOL_DOMAIN_MAP
        self.assertEqual(TOOL_DOMAIN_MAP.get("get_top_selling_products"), "sales")

    def test_router_routing_for_top_products_query(self):
        query = "Tampilkan 10 produk dengan pendapatan tertinggi cabang Surabaya dan Jakarta tahun 2026"
        state = create_initial_state(user_query=query)
        result = router_node(state)

        self.assertEqual(result.get("detected_domain"), "sales")
        self.assertEqual(result.get("detected_tool"), "get_top_selling_products")
        self.assertTrue(result.get("is_multi_company"))
        self.assertIn(7, result.get("company_ids", []))
        self.assertIn(8, result.get("company_ids", []))

    def test_router_and_extractor_for_single_branch_top_selling(self):
        query = "Daftar 5 produk terlaris di cabang Surabaya"
        state = create_initial_state(user_query=query)
        r_result = router_node(state)

        self.assertEqual(r_result.get("detected_domain"), "sales")
        self.assertEqual(r_result.get("detected_tool"), "get_top_selling_products")

        state.update(r_result)
        e_result = extractor_node(state)
        self.assertEqual(e_result.get("company_id"), 7)
        self.assertEqual(e_result.get("extracted_params", {}).get("limit"), 5)

    def test_extractor_deterministic_extraction(self):
        query = "Tampilkan 10 produk dengan pendapatan tertinggi cabang Surabaya dan Jakarta tahun 2026"
        state = {
            "messages": [HumanMessage(content=query)],
            "company_id": 7,
            "company_ids": [7, 8],
            "is_multi_company": True,
            "detected_domain": "sales",
            "detected_tool": "get_top_selling_products",
            "query_mode": "live_erp",
            "detected_domains": ["sales"],
            "detected_tools": ["get_top_selling_products"],
            "active_tools": [
                {
                    "tool": "get_top_selling_products",
                    "domain": "sales",
                    "sub_query": query,
                    "params": {},
                    "status": "pending",
                }
            ],
            "user_query": query,
        }
        result = extractor_node(state)
        params = result.get("extracted_params") or {}

        self.assertEqual(params.get("limit"), 10)
        self.assertEqual(params.get("date_from"), "2026-01-01")
        self.assertEqual(params.get("date_to"), "2026-12-31")

    def test_verifier_records_count(self):
        tool_result = {
            "company_id": 7,
            "total_products": 2,
            "products": [
                {"rank": 1, "code": "SKU-01", "name": "Prod 1", "total_revenue": 1000.0},
                {"rank": 2, "code": "SKU-02", "name": "Prod 2", "total_revenue": 500.0},
            ],
        }
        count = _calculate_records_count(tool_result)
        self.assertEqual(count, 2)

    @patch.object(DjangoCoreClient, "execute_tool", new_callable=AsyncMock)
    async def test_django_client_fetch_top_selling_products(self, mock_exec):
        mock_exec.return_value = {
            "success": True,
            "data": {"products": [{"rank": 1, "code": "SKU-01", "total_revenue": 500000.0}]},
        }
        client = DjangoCoreClient()
        res = await client.fetch_top_selling_products(
            params={"limit": 5, "date_from": "2026-01-01", "date_to": "2026-12-31"},
            company_id=7,
        )
        self.assertIn("data", res)
        mock_exec.assert_called_once_with(
            domain="sales",
            tool_name="get_top_selling_products",
            params={"limit": 5, "date_from": "2026-01-01", "date_to": "2026-12-31"},
            company_id=7,
            user_token=None,
        )

    def test_synthesizer_single_company_presentation(self):
        state = {
            "user_query": "Tampilkan 5 produk terlaris Surabaya",
            "detected_domain": "sales",
            "detected_tool": "get_top_selling_products",
            "verification_status": "VALID",
            "is_multi_company": False,
            "company_id": 7,
            "tool_result": {
                "company_id": 7,
                "company_name": "Surabaya",
                "period": {"date_from": "2026-01-01", "date_to": "2026-12-31"},
                "total_products": 2,
                "products": [
                    {
                        "rank": 1,
                        "code": "SKU-HVS-01",
                        "name": "Kertas HVS A4",
                        "category": "Kertas",
                        "total_qty": 500,
                        "total_revenue": 25000000.0,
                    },
                    {
                        "rank": 2,
                        "code": "SKU-INK-02",
                        "name": "Tinta 664",
                        "category": "Tinta",
                        "total_qty": 200,
                        "total_revenue": 16000000.0,
                    },
                ],
                "summary": {"total_revenue": 41000000.0, "total_qty": 700},
            },
        }
        output = format_deterministic_answer(state)
        self.assertIn("Laporan Produk Penjualan Tertinggi", output)
        self.assertIn("`SKU-HVS-01`", output)
        self.assertIn("Kertas HVS A4", output)
        self.assertIn("Rp 25.000.000,00", output)
        self.assertIn("TOTAL", output)
        self.assertIn("Rp 41.000.000,00", output)

    def test_synthesizer_multi_company_presentation(self):
        state = {
            "user_query": "Tampilkan 10 produk dengan pendapatan tertinggi cabang Surabaya dan Jakarta tahun 2026",
            "detected_domain": "sales",
            "detected_tool": "get_top_selling_products",
            "verification_status": "VALID",
            "is_multi_company": True,
            "company_ids": [7, 8],
            "extracted_params": {"limit": 10},
            "multi_company_results": {
                7: {
                    "company_id": 7,
                    "products": [
                        {"rank": 1, "code": "SKU-HVS", "name": "Kertas HVS", "total_qty": 100, "total_revenue": 5000000.0},
                    ],
                },
                8: {
                    "company_id": 8,
                    "products": [
                        {"rank": 1, "code": "SKU-HVS", "name": "Kertas HVS", "total_qty": 200, "total_revenue": 10000000.0},
                        {"rank": 2, "code": "SKU-INK", "name": "Tinta Black", "total_qty": 50, "total_revenue": 4000000.0},
                    ],
                },
            },
        }
        output = format_deterministic_answer(state)
        self.assertIn("Laporan Produk Pendapatan Tertinggi Multi-Cabang", output)
        self.assertIn("Omzet Surabaya", output)
        self.assertIn("Omzet Jakarta", output)
        self.assertIn("Total Omzet Gabungan", output)
        self.assertIn("`SKU-HVS`", output)
        self.assertIn("Rp 5.000.000,00", output)
        self.assertIn("Rp 10.000.000,00", output)
        self.assertIn("**Rp 15.000.000,00**", output)
        self.assertIn("TOTAL GABUNGAN", output)


if __name__ == "__main__":
    unittest.main()
