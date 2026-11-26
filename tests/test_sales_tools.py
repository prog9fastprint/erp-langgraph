"""
Unit tests for Sales Domain AI Tools.
Covers:
- Schema validation for get_top_selling_products, get_sales_orders, get_sales_order_detail, get_sales_summary
- Router routing for sales queries
- Extractor parameter resolution
- Verifier record count detection
- DjangoCoreClient method execution
- Single and multi-company synthesizer presentation
"""
import unittest
from unittest.mock import AsyncMock, patch

from app.graph.state import create_initial_state
from app.graph.nodes.router import router_node
from app.graph.nodes.extractor import extractor_node
from app.graph.nodes.verifier import _calculate_records_count
from app.graph.nodes.synthesizer import format_deterministic_answer
from app.schemas.tools import (
    TopSellingProductsFilterSchema,
    SalesOrdersFilterSchema,
    SalesOrderDetailFilterSchema,
    SalesSummaryFilterSchema,
    TOOL_DOMAIN_MAP,
)
from app.services.django_client import DjangoCoreClient
from langchain_core.messages import HumanMessage


class TestSalesTools(unittest.IsolatedAsyncioTestCase):
    """Comprehensive test suite for Sales domain tools & LangGraph orchestration."""

    def test_schema_validation(self):
        # 1. Top Selling Products
        s1 = TopSellingProductsFilterSchema(
            company_id=7,
            date_from="2026-01-01",
            date_to="2026-12-31",
            limit=10,
            search="HVS",
        )
        self.assertEqual(s1.company_id, 7)
        self.assertEqual(s1.limit, 10)

        # 2. Sales Orders
        s2 = SalesOrdersFilterSchema(
            company_id=7,
            status="confirmed",
            date_from="2026-01-01",
            date_to="2026-03-31",
            limit=25,
            search="PT Jaya",
        )
        self.assertEqual(s2.company_id, 7)
        self.assertEqual(s2.status, "confirmed")
        self.assertEqual(s2.limit, 25)

        # 3. Sales Order Detail
        s3 = SalesOrderDetailFilterSchema(
            company_id=7,
            order_ref="SO/2026/001",
        )
        self.assertEqual(s3.order_ref, "SO/2026/001")

        # 4. Sales Summary
        s4 = SalesSummaryFilterSchema(
            company_id=7,
            date_from="2026-01-01",
            date_to="2026-12-31",
            channel="Shopee",
            status="confirmed",
        )
        self.assertEqual(s4.channel, "Shopee")

        # Canonical Domain Map Registration
        self.assertEqual(TOOL_DOMAIN_MAP.get("get_top_selling_products"), "sales")
        self.assertEqual(TOOL_DOMAIN_MAP.get("get_sales_orders"), "sales")
        self.assertEqual(TOOL_DOMAIN_MAP.get("get_sales_order_detail"), "sales")
        self.assertEqual(TOOL_DOMAIN_MAP.get("get_sales_summary"), "sales")

    def test_router_routing_for_sales_orders(self):
        query = "Tampilkan daftar sales order bulan ini"
        state = create_initial_state(user_query=query)
        result = router_node(state)

        self.assertEqual(result.get("detected_domain"), "sales")
        self.assertEqual(result.get("detected_tool"), "get_sales_orders")

    def test_router_routing_for_sales_order_detail(self):
        query = "Cek detail pesanan SO/2026/001"
        state = create_initial_state(user_query=query)
        result = router_node(state)

        self.assertEqual(result.get("detected_domain"), "sales")
        self.assertEqual(result.get("detected_tool"), "get_sales_order_detail")

    def test_router_routing_for_sales_summary(self):
        query = "Berapa total omzet penjualan cabang Surabaya dan Jakarta tahun 2026"
        state = create_initial_state(user_query=query)
        result = router_node(state)

        self.assertEqual(result.get("detected_domain"), "sales")
        self.assertEqual(result.get("detected_tool"), "get_sales_summary")
        self.assertTrue(result.get("is_multi_company"))
        self.assertIn(7, result.get("company_ids", []))
        self.assertIn(8, result.get("company_ids", []))

    def test_router_routing_for_top_products_query(self):
        query = "Tampilkan 10 produk dengan pendapatan tertinggi cabang Surabaya dan Jakarta tahun 2026"
        state = create_initial_state(user_query=query)
        result = router_node(state)

        self.assertEqual(result.get("detected_domain"), "sales")
        self.assertEqual(result.get("detected_tool"), "get_top_selling_products")
        self.assertTrue(result.get("is_multi_company"))

    def test_extractor_for_sales_order_detail(self):
        query = "Tampilkan rincian order SO/2026/088"
        state = {
            "messages": [HumanMessage(content=query)],
            "company_id": 7,
            "detected_domain": "sales",
            "detected_tool": "get_sales_order_detail",
            "query_mode": "live_erp",
            "user_query": query,
        }
        result = extractor_node(state)
        params = result.get("extracted_params") or {}
        self.assertEqual(params.get("order_ref"), "SO/2026/088")

    def test_extractor_for_sales_summary(self):
        query = "Laporan omzet penjualan tahun 2026"
        state = {
            "messages": [HumanMessage(content=query)],
            "company_id": 7,
            "detected_domain": "sales",
            "detected_tool": "get_sales_summary",
            "query_mode": "live_erp",
            "user_query": query,
        }
        result = extractor_node(state)
        params = result.get("extracted_params") or {}
        self.assertEqual(params.get("date_from"), "2026-01-01")
        self.assertEqual(params.get("date_to"), "2026-12-31")
        self.assertEqual(params.get("status"), "confirmed")

    def test_verifier_records_count(self):
        # 1. Orders list
        res_orders = {"total_count": 5, "orders": [{"id": 1}, {"id": 2}]}
        self.assertEqual(_calculate_records_count(res_orders), 5)

        # 2. Order detail
        res_detail = {"id": 10, "so_number": "SO/2026/001", "lines": [{"id": 101}]}
        self.assertEqual(_calculate_records_count(res_detail), 1)

        # 3. Summary
        res_summary = {"confirmed_metrics": {"realized_revenue": 5000000.0, "orders_count": 10}}
        self.assertEqual(_calculate_records_count(res_summary), 1)

    @patch.object(DjangoCoreClient, "execute_tool", new_callable=AsyncMock)
    async def test_django_client_methods(self, mock_exec):
        client = DjangoCoreClient()

        # 1. fetch_top_selling_products
        mock_exec.return_value = {"success": True, "data": {"products": []}}
        await client.fetch_top_selling_products(params={"limit": 5}, company_id=7)
        mock_exec.assert_called_with(
            domain="sales",
            tool_name="get_top_selling_products",
            params={"limit": 5},
            company_id=7,
            user_token=None,
        )

        # 2. fetch_sales_orders
        await client.fetch_sales_orders(params={"status": "confirmed", "limit": 20}, company_id=7)
        mock_exec.assert_called_with(
            domain="sales",
            tool_name="get_sales_orders",
            params={"status": "confirmed", "limit": 20},
            company_id=7,
            user_token=None,
        )

        # 3. fetch_sales_order_detail
        await client.fetch_sales_order_detail(params={"order_ref": "SO/2026/001"}, company_id=7)
        mock_exec.assert_called_with(
            domain="sales",
            tool_name="get_sales_order_detail",
            params={"order_ref": "SO/2026/001"},
            company_id=7,
            user_token=None,
        )

        # 4. fetch_sales_summary
        await client.fetch_sales_summary(params={"status": "confirmed"}, company_id=7)
        mock_exec.assert_called_with(
            domain="sales",
            tool_name="get_sales_summary",
            params={"status": "confirmed"},
            company_id=7,
            user_token=None,
        )

    def test_synthesizer_sales_orders_presentation(self):
        state = {
            "user_query": "Daftar sales order",
            "detected_domain": "sales",
            "detected_tool": "get_sales_orders",
            "verification_status": "VALID",
            "is_multi_company": False,
            "company_id": 7,
            "tool_result": {
                "company_id": 7,
                "total_count": 1,
                "total_amount_sum": 1500000.0,
                "status_filter": "confirmed",
                "orders": [
                    {
                        "id": 101,
                        "so_number": "SO/2026/001",
                        "quotation_date": "2026-03-01",
                        "customer_name": "PT Cipta Grafika",
                        "channel_name": "Tokopedia",
                        "status": "Sales Order",
                        "total_amount": 1500000.0,
                    }
                ],
            },
        }
        output = format_deterministic_answer(state)
        self.assertIn("Daftar Pesanan Penjualan", output)
        self.assertIn("`SO/2026/001`", output)
        self.assertIn("PT Cipta Grafika", output)
        self.assertIn("Rp 1.500.000,00", output)

    def test_synthesizer_sales_order_detail_presentation(self):
        state = {
            "user_query": "Detail order SO/2026/001",
            "detected_domain": "sales",
            "detected_tool": "get_sales_order_detail",
            "verification_status": "VALID",
            "is_multi_company": False,
            "company_id": 7,
            "tool_result": {
                "id": 101,
                "so_number": "SO/2026/001",
                "company_name": "Surabaya",
                "status": "Sales Order",
                "quotation_date": "2026-03-01",
                "payment_terms": "COD",
                "customer": {"name": "PT Jaya", "sales_phone": "0812345678"},
                "channel": "Direct",
                "financial_summary": {"total_amount": 750000.0, "discount_nominal": 0.0},
                "lines": [
                    {
                        "sku": "SKU-01",
                        "product_name": "Tinta Art Paper",
                        "qty": 3,
                        "uom": "Botol",
                        "unit_price": 250000.0,
                        "disc_nominal": 0.0,
                        "sub_total": 750000.0,
                    }
                ],
            },
        }
        output = format_deterministic_answer(state)
        self.assertIn("Detail Sales Order: `SO/2026/001`", output)
        self.assertIn("PT Jaya", output)
        self.assertIn("Tinta Art Paper", output)
        self.assertIn("Rp 750.000,00", output)

    def test_synthesizer_sales_summary_multi_company_presentation(self):
        state = {
            "user_query": "Berapa total omzet penjualan cabang Surabaya dan Jakarta tahun 2026",
            "detected_domain": "sales",
            "detected_tool": "get_sales_summary",
            "verification_status": "VALID",
            "is_multi_company": True,
            "company_ids": [7, 8],
            "multi_company_results": {
                7: {
                    "confirmed_metrics": {
                        "realized_revenue": 50000000.0,
                        "orders_count": 25,
                        "average_order_value": 2000000.0,
                    },
                    "pipeline_metrics": {"quotation_potential_value": 5000000.0},
                },
                8: {
                    "confirmed_metrics": {
                        "realized_revenue": 30000000.0,
                        "orders_count": 15,
                        "average_order_value": 2000000.0,
                    },
                    "pipeline_metrics": {"quotation_potential_value": 2000000.0},
                },
            },
        }
        output = format_deterministic_answer(state)
        self.assertIn("Perbandingan Performa Penjualan Multi-Cabang", output)
        self.assertIn("Surabaya", output)
        self.assertIn("Jakarta", output)
        self.assertIn("Rp 50.000.000,00", output)
        self.assertIn("Rp 30.000.000,00", output)
        self.assertIn("Rp 80.000.000,00", output)
        self.assertIn("40", output)


if __name__ == "__main__":
    unittest.main()
