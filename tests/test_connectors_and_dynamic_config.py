"""
Unit Tests for Multi-System Connectors and Dynamic Configuration Architecture.
Verifies BaseSystemConnector, FastPrintConnector, OdooConnector, TalentaConnector,
GenericRestConnector, BranchRegistry, and dynamic tool execution via tool_node.
"""
import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from app.core.branches import (
    BRANCH_COMPANY_MAP,
    COMPANY_NAME_MAP,
    BranchRegistry,
    branch_registry,
    resolve_all_branch_company_ids,
    resolve_branch_company_id,
)
from app.connectors.base import BaseSystemConnector
from app.connectors.fastprint import FastPrintConnector
from app.connectors.odoo import OdooConnector
from app.connectors.talenta import TalentaConnector
from app.connectors.generic_rest import GenericRestConnector
from app.connectors.registry import ConnectorRegistry, get_connector_registry
from app.schemas.tools import (
    TOOL_DOMAIN_MAP,
    OdooStockFilterSchema,
    TalentaTimeOffFilterSchema,
    register_tool_domain,
)
from app.services.tool_registry_service import get_tool_registry_service


class TestDynamicBranchRegistry(unittest.TestCase):
    """Test dynamic tenant and branch mapping without hardcoding."""

    def test_default_branches_preserved(self):
        """Default FastPrint branch mappings must remain intact."""
        self.assertEqual(resolve_branch_company_id("cabang bandung"), 6)
        self.assertEqual(resolve_branch_company_id("kantor pusat"), 1)

    def test_dynamic_tenant_registration(self):
        """External tenant can register its own branch & company mappings."""
        custom_registry = BranchRegistry()
        custom_registry.register_tenant(
            tenant_id="retail_corp",
            branch_map={
                "toko thamrin": 101,
                "gudang cikarang": 102,
                "outlet bali": 103,
            },
            name_map={
                101: "Toko Thamrin Jakarta",
                102: "Gudang Distribusi Cikarang",
                103: "Outlet Retail Bali",
            },
        )

        # Isolated tenant resolution
        b_map = custom_registry.get_branch_map("retail_corp")
        n_map = custom_registry.get_name_map("retail_corp")
        self.assertEqual(b_map.get("toko thamrin"), 101)
        self.assertEqual(n_map.get(102), "Gudang Distribusi Cikarang")

        # Global helper with tenant_id parameter
        branch_registry.register_tenant(
            tenant_id="client_odoo",
            branch_map={"pos grand indonesia": 201},
            name_map={201: "POS Grand Indonesia"},
        )
        cid = resolve_branch_company_id("di pos grand indonesia", tenant_id="client_odoo")
        self.assertEqual(cid, 201)


class TestSystemConnectors(unittest.IsolatedAsyncioTestCase):
    """Test connector implementations and unified contract."""

    async def test_fastprint_connector_contract(self):
        """FastPrintConnector inherits BaseSystemConnector and supports ERP tools."""
        connector = FastPrintConnector()
        self.assertIsInstance(connector, BaseSystemConnector)
        self.assertEqual(connector.connector_type, "erp_django")
        self.assertIn("get_account_balance", connector.get_supported_tools())
        self.assertIn("reconcile_bank_statement", connector.get_supported_tools())
        await connector.aclose()

    async def test_odoo_connector_contract(self):
        """OdooConnector handles retail stock, products, and POS orders."""
        connector = OdooConnector(url="https://odoo.myretail.com", db="production", username="admin", api_key="secret")
        self.assertIsInstance(connector, BaseSystemConnector)
        self.assertEqual(connector.connector_type, "odoo_rpc")
        self.assertIn("odoo_get_stock", connector.get_supported_tools())
        self.assertIn("odoo_get_pos_orders", connector.get_supported_tools())

        tools = await connector.discover_tools()
        tool_names = [t["tool_name"] for t in tools]
        self.assertIn("odoo_get_stock", tool_names)
        self.assertIn("odoo_get_pos_orders", tool_names)
        await connector.aclose()

    async def test_talenta_connector_contract(self):
        """TalentaConnector handles employee info, leave balance, attendance, and payroll."""
        connector = TalentaConnector(api_key="mekari_test_token")
        self.assertIsInstance(connector, BaseSystemConnector)
        self.assertEqual(connector.connector_type, "talenta_api")
        self.assertIn("talenta_get_employee", connector.get_supported_tools())
        self.assertIn("talenta_get_time_off_balance", connector.get_supported_tools())
        self.assertIn("talenta_get_attendance", connector.get_supported_tools())
        self.assertIn("talenta_get_payroll_summary", connector.get_supported_tools())

        tools = await connector.discover_tools()
        tool_names = [t["tool_name"] for t in tools]
        self.assertIn("talenta_get_time_off_balance", tool_names)
        await connector.aclose()

    async def test_generic_rest_connector(self):
        """GenericRestConnector provides plug-and-play REST endpoint mapping."""
        connector = GenericRestConnector(
            name="warehouse_wms",
            base_url="https://wms.enterprise.com",
            tool_endpoints={"wms_get_bin_rack": "/api/v1/racks"},
        )
        self.assertIsInstance(connector, BaseSystemConnector)
        self.assertEqual(connector.connector_type, "generic_rest")
        self.assertIn("wms_get_bin_rack", connector.get_supported_tools())
        await connector.aclose()


class TestConnectorRegistry(unittest.IsolatedAsyncioTestCase):
    """Test central ConnectorRegistry dispatching."""

    async def test_registry_resolution_and_dispatch(self):
        registry = ConnectorRegistry()
        registry.initialize_default_connectors()

        # 1. Resolve FastPrint tool
        fp_conn = registry.resolve_connector_for_tool("get_general_ledger")
        self.assertEqual(fp_conn.name, "fastprint")

        # 2. Resolve Odoo tool
        odoo_conn = registry.resolve_connector_for_tool("odoo_get_stock")
        self.assertEqual(odoo_conn.name, "odoo")

        # 3. Resolve Talenta HRIS tool
        talenta_conn = registry.resolve_connector_for_tool("talenta_get_time_off_balance")
        self.assertEqual(talenta_conn.name, "talenta")

        # 4. Mock execution dispatch
        mock_talenta = TalentaConnector(api_key="mock_key")
        mock_talenta.execute_tool = AsyncMock(return_value={"remaining_leave": 12, "employee": "Alice"})
        registry.register(mock_talenta)

        res, err = await registry.execute_tool(
            tool_name="talenta_get_time_off_balance",
            params={"employee_id": "EMP001"},
        )
        self.assertIsNone(err)
        self.assertEqual(res["remaining_leave"], 12)
        mock_talenta.execute_tool.assert_awaited_once()

    async def test_dynamic_tool_registration(self):
        """Verify dynamic runtime tool registration in global schemas and registry service."""
        register_tool_domain("custom_wms_scan", "logistics")
        self.assertEqual(TOOL_DOMAIN_MAP.get("custom_wms_scan"), "logistics")

        tool_registry_svc = get_tool_registry_service()
        tool_registry_svc.register_dynamic_tool({
            "tool_name": "custom_wms_scan",
            "domain": "logistics",
            "description": "Scan barcode palet WMS",
            "keywords": ["scan barcode", "palet"],
        })
        tool_card = tool_registry_svc.get_tool_by_name("custom_wms_scan")
        self.assertIsNotNone(tool_card)
        self.assertEqual(tool_card["domain"], "logistics")


class TestPydanticSchemas(unittest.TestCase):
    """Test validation schemas for Odoo and Talenta."""

    def test_odoo_schemas(self):
        schema = OdooStockFilterSchema(product_name="Kertas Foto", limit=10)
        self.assertEqual(schema.product_name, "Kertas Foto")
        self.assertEqual(schema.limit, 10)

    def test_talenta_schemas(self):
        schema = TalentaTimeOffFilterSchema(employee_id="EMP-99", year=2026)
        self.assertEqual(schema.employee_id, "EMP-99")
        self.assertEqual(schema.year, 2026)


if __name__ == "__main__":
    unittest.main()
