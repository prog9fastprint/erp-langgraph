"""
Unit & Integration Tests for Phase 1-4:
- Phase 1: inventory.get_price_history (schema, registry, client dispatch, router, extractor, verifier, formatter)
"""
from datetime import date
import unittest
from unittest.mock import AsyncMock

from pydantic import ValidationError

from app.core.context import resolve_tool_record_count
from app.graph.nodes.extractor import TOOL_SCHEMA_REGISTRY, extractor_node
from app.graph.nodes.formatters import SINGLE_TOOL_FORMATTERS, _format_single_tool_presentation
from app.graph.nodes.router import classify_query_rule_based, router_node
from app.graph.nodes.tool_node import tool_node
from app.graph.nodes.verifier import verifier_node
from app.graph.state import create_initial_state
from app.schemas.tools import PriceHistoryFilterSchema, TOOL_DOMAIN_MAP
from app.services.django_client import DjangoCoreClient
from app.services.tool_registry_service import get_tool_registry_service


class TestPhase1PriceHistorySchemaAndRegistry(unittest.TestCase):
    """Phase 1: Schema validation, TOOL_DOMAIN_MAP, ToolRegistryService, and formatter wiring."""

    def test_price_history_registered_everywhere(self):
        self.assertEqual(TOOL_DOMAIN_MAP.get("get_price_history"), "inventory")
        self.assertIs(TOOL_SCHEMA_REGISTRY.get("get_price_history"), PriceHistoryFilterSchema)
        self.assertIn("get_price_history", SINGLE_TOOL_FORMATTERS)
        self.assertTrue(hasattr(DjangoCoreClient, "fetch_price_history"))

        registry = get_tool_registry_service()
        meta = registry.get_tool_by_name("get_price_history")
        self.assertIsNotNone(meta)
        self.assertEqual(meta["domain"], "inventory")
        self.assertFalse(meta["is_write"])

    def test_price_history_schema_validation(self):
        valid = PriceHistoryFilterSchema(
            company_id=4,
            sku="SKU-A01",
            source="marketplace",
            marketplace="Shopee",
            date_from="2026-09-01",
            date_to="2026-11-26",
            limit=30,
        )
        self.assertEqual(valid.source, "marketplace")
        self.assertEqual(valid.limit, 30)

        with self.assertRaises(ValidationError):
            PriceHistoryFilterSchema(date_from="26-11-2026")

        with self.assertRaises(ValidationError):
            PriceHistoryFilterSchema(source="invalid_source")  # type: ignore[arg-type]


class TestPhase1PriceHistoryRouterAndExtractor(unittest.TestCase):
    """Phase 1: Router classification, collision suppression, and deterministic extraction."""

    def test_routes_price_history_and_suppresses_product_info(self):
        res = classify_query_rule_based("Cek perubahan harga barang Kertas Art Paper 210")
        self.assertIsNotNone(res)
        self.assertEqual(res["detected_domain"], "inventory")
        self.assertEqual(res["detected_tool"], "get_price_history")
        self.assertEqual(res["detected_tools"], ["get_price_history"])
        self.assertFalse(res["is_multi_tool"])

    def test_does_not_collide_with_competitor_price_drops(self):
        res = classify_query_rule_based("Cek penurunan harga kompetitor di Shopee minggu ini")
        self.assertIsNotNone(res)
        self.assertEqual(res["detected_domain"], "marketplace")
        self.assertEqual(res["detected_tool"], "get_competitor_insights")
        self.assertNotIn("get_price_history", res["detected_tools"])

    def test_does_not_collide_with_sync_error_logs(self):
        res = classify_query_rule_based("Cek log gagal update harga marketplace")
        self.assertIsNotNone(res)
        self.assertEqual(res["detected_domain"], "marketplace")
        self.assertEqual(res["detected_tool"], "get_sync_execution_logs")
        self.assertNotIn("get_price_history", res["detected_tools"])

    def test_hq_auto_expands_multi_company(self):
        state = create_initial_state(
            user_query="Cek riwayat perubahan harga SKU-A01",
            company_id=3,
            authorized_company_ids=[3, 4, 7, 8],
        )
        delta = router_node(state)
        self.assertEqual(delta["detected_tool"], "get_price_history")
        self.assertTrue(delta["is_multi_company"])
        self.assertEqual(delta["company_ids"], [3, 4, 7, 8])

    def test_extractor_defaults_to_90_days_and_parses_sku_and_source(self):
        state = create_initial_state(
            user_query="Cek riwayat harga SKU-A01 di Shopee",
            company_id=4,
        )
        state.update(router_node(state))
        ext = extractor_node(state, ref_date=date(2026, 11, 26))
        params = ext["extracted_params"]
        self.assertEqual(params["sku"], "SKU-A01")
        self.assertEqual(params["source"], "marketplace")
        self.assertEqual(params["marketplace"], "Shopee")
        self.assertEqual(params["date_from"], "2026-08-28")
        self.assertEqual(params["date_to"], "2026-11-26")

    def test_extractor_parses_product_name_before_multi_metric_list(self):
        state = create_initial_state(
            user_query="coba cek barang A penjualannya, returnya, perubahan harganya",
            company_id=4,
        )
        state["detected_domain"] = "inventory"
        state["detected_tool"] = "get_price_history"
        state["active_tools"] = [{
            "tool": "get_price_history",
            "domain": "inventory",
            "sub_query": state["user_query"],
            "params": {},
            "status": "pending",
        }]
        ext = extractor_node(state, ref_date=date(2026, 11, 26))
        params = ext["extracted_params"]
        self.assertEqual(params.get("search"), "A")
        self.assertEqual(params.get("source"), "all")


class TestPhase1PriceHistoryExecutionAndFormatting(unittest.IsolatedAsyncioTestCase):
    """Phase 1: DjangoCoreClient dispatch, tool_node native multi-company, verifier, and Markdown formatter."""

    async def test_client_dispatch_endpoint(self):
        client = DjangoCoreClient()
        client.execute_tool = AsyncMock(return_value={"changes": [], "total_count": 0})
        await client.fetch_price_history(
            params={"sku": "SKU-A01", "source": "all"},
            user_token="jwt-token",
            company_id=4,
        )
        client.execute_tool.assert_awaited_once_with(
            domain="inventory",
            tool_name="get_price_history",
            params={"sku": "SKU-A01", "source": "all"},
            company_id=4,
            user_token="jwt-token",
        )

    async def test_tool_node_native_mc_verifier_and_formatter(self):
        sample_payload = {
            "date_window": {"date_from": "2026-08-28", "date_to": "2026-11-26"},
            "current_prices": [
                {"pricelist_name": "Pricelist FP Jawa", "region": "jawa", "min_qty": 1, "price": 45000.0},
            ],
            "summary": {
                "total_changes": 2,
                "latest_change_date": "2026-11-18",
                "min_price": 42000.0,
                "max_price": 45000.0,
                "net_change_idr": 3000.0,
                "net_change_pct": 7.14,
            },
            "total_count": 2,
            "changes": [
                {
                    "changed_at": "2026-11-18 10:15:00",
                    "source": "erp_pricelist",
                    "channel_or_pricelist": "Pricelist FP Jawa",
                    "sku": "SKU-A01",
                    "product_name": "Barang A",
                    "qty_tier": "1",
                    "old_price": 42000.0,
                    "new_price": 45000.0,
                    "delta_idr": 3000.0,
                    "delta_pct": 7.14,
                    "updated_by": "Admin Pricing",
                    "status": "SUCCESS",
                },
                {
                    "changed_at": "2026-11-18 10:17:00",
                    "source": "marketplace",
                    "channel_or_pricelist": "Shopee Official",
                    "sku": "SKU-A01",
                    "product_name": "Barang A",
                    "qty_tier": "1",
                    "old_price": 42000.0,
                    "new_price": 45000.0,
                    "delta_idr": 3000.0,
                    "delta_pct": 7.14,
                    "updated_by": "System",
                    "status": "SUCCESS",
                },
            ],
        }
        client = DjangoCoreClient()
        client.execute_tool = AsyncMock(return_value=sample_payload)

        state = create_initial_state(
            user_query="Cek perubahan harga SKU-A01 semua cabang",
            company_id=4,
            company_ids=[4, 7],
            is_multi_company=True,
        )
        state["detected_domain"] = "inventory"
        state["detected_tool"] = "get_price_history"
        state["extracted_params"] = {"sku": "SKU-A01", "source": "all"}
        state["active_tools"] = [{
            "tool": "get_price_history",
            "domain": "inventory",
            "sub_query": state["user_query"],
            "params": {"sku": "SKU-A01", "source": "all"},
            "company_id": 4,
            "company_ids": [4, 7],
            "is_multi_company": True,
            "status": "pending",
        }]

        out = await tool_node(state, client=client)
        self.assertEqual(out["verification_status"], "VALID")
        self.assertEqual(resolve_tool_record_count(out["tool_result"]), 2)
        client.execute_tool.assert_awaited_once()
        sent_params = client.execute_tool.await_args.kwargs["params"]
        self.assertEqual(sent_params["company_ids"], [4, 7])

        state.update(out)
        ver = verifier_node(state)
        self.assertEqual(ver["verification_status"], "VALID")

        md = _format_single_tool_presentation("get_price_history", out["tool_result"], state["user_query"])
        self.assertIn("Riwayat Perubahan Harga Produk", md)
        self.assertIn("Pricelist FP Jawa", md)
        self.assertIn("Shopee Official", md)
        self.assertIn("Rp 42.000", md)
        self.assertIn("Rp 45.000", md)
        self.assertIn("+7.1%", md)


if __name__ == "__main__":
    unittest.main()
