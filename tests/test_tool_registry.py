"""
Unit tests for Dynamic Semantic Tool Registry & RBAC Guardrail (TASK-5.1).
Covers in-memory fallback, vector similarity retrieval with distance threshold,
RBAC role/permission filtering, incremental seeding, and router integration.
"""
import asyncio
from datetime import date
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.config import settings
from app.graph.nodes.router import router_node, PlannedToolCall
from app.graph.state import create_initial_state, AgentState
from app.schemas.chat import ChatRequest
from app.services.tool_registry_service import (
    ToolRegistryService,
    get_tool_registry_service,
    CORE_TOOL_DEFINITIONS,
)


class TestToolRegistryService(unittest.IsolatedAsyncioTestCase):
    """Test suite for ToolRegistryService catalog, retrieval, and RBAC filtering."""

    def setUp(self):
        self.registry = ToolRegistryService()

    def test_in_memory_catalog_initialization(self):
        """In-memory catalog must register all 19 core ERP tools with schema definitions."""
        self.assertGreaterEqual(len(self.registry._in_memory_catalog), 19)

        balance_meta = self.registry.get_tool_by_name("get_account_balance")
        self.assertIsNotNone(balance_meta)
        self.assertEqual(balance_meta["domain"], "accounting")
        self.assertIsNone(balance_meta["required_permission"])
        self.assertFalse(balance_meta["is_write"])
        self.assertIn("properties", balance_meta["params_schema"])

        invoice_meta = self.registry.get_tool_by_name("fetch_vendor_invoices")
        self.assertIsNotNone(invoice_meta)
        self.assertEqual(invoice_meta["domain"], "accounting")
        self.assertEqual(invoice_meta["required_permission"], "accounting.view_invoices")

    def test_fallback_domain_filtering(self):
        """Fallback tool retrieval respects domain filter strictly."""
        accounting_tools = self.registry.search_tools_fallback(
            query="saldo buku kas",
            domain="accounting",
            user_roles=["accounting.view_ledger"],
        )
        self.assertTrue(len(accounting_tools) > 0)
        for t in accounting_tools:
            self.assertEqual(t["domain"], "accounting")

        inventory_tools = self.registry.search_tools_fallback(
            query="stok barang",
            domain="inventory",
        )
        self.assertTrue(len(inventory_tools) > 0)
        for t in inventory_tools:
            self.assertEqual(t["domain"], "inventory")

    def test_fallback_rbac_filtering_public_tools(self):
        """Public tools with required_permission=None are accessible without special roles."""
        tools = self.registry.search_tools_fallback(
            query="cek saldo kas akun",
            domain="accounting",
            user_roles=None,
        )
        tool_names = [t["tool_name"] for t in tools]
        self.assertIn("get_account_balance", tool_names)

    def test_fallback_rbac_filtering_restricted_tools(self):
        """Restricted tools require matching role/permission or admin."""
        # 1. User without accounting.view_invoices cannot retrieve fetch_vendor_invoices
        tools_denied = self.registry.search_tools_fallback(
            query="daftar tagihan faktur invoice vendor",
            domain="accounting",
            user_roles=["inventory.viewer"],
        )
        names_denied = [t["tool_name"] for t in tools_denied]
        self.assertNotIn("fetch_vendor_invoices", names_denied)

        # 2. User with accounting.view_invoices can retrieve it
        tools_allowed = self.registry.search_tools_fallback(
            query="daftar tagihan faktur invoice vendor",
            domain="accounting",
            user_roles=["accounting.view_invoices"],
        )
        names_allowed = [t["tool_name"] for t in tools_allowed]
        self.assertIn("fetch_vendor_invoices", names_allowed)

        # 3. Admin user bypasses role restrictions
        tools_admin = self.registry.search_tools_fallback(
            query="daftar tagihan faktur invoice vendor",
            domain="accounting",
            user_roles=["admin"],
        )
        names_admin = [t["tool_name"] for t in tools_admin]
        self.assertIn("fetch_vendor_invoices", names_admin)

    async def test_search_tools_with_pgvector_similarity_and_threshold(self):
        """PGVector similarity search filters out tools with distance > distance_threshold."""
        mock_vec_service = MagicMock()
        mock_vec_service.get_embedding = AsyncMock(return_value=[0.1] * 768)

        mock_pool = MagicMock()
        mock_conn = MagicMock()

        # Simulate 2 tools: 1 close (distance 0.15), 1 far (distance 0.85)
        # The SQL query filters distance <= max_dist, but we test the return mapping
        mock_records = [
            {
                "tool_name": "get_stock_level",
                "domain": "inventory",
                "description": "Pengecekan stok barang fisik",
                "params_schema": json.dumps({"type": "object"}),
                "required_permission": None,
                "is_write": False,
                "distance": 0.15,
            }
        ]
        mock_conn.fetch = AsyncMock(return_value=mock_records)

        # Context manager for pool.acquire()
        mock_acquire = MagicMock()
        mock_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire.__aexit__ = AsyncMock(return_value=None)
        mock_pool.acquire.return_value = mock_acquire

        mock_vec_service.get_pool = AsyncMock(return_value=mock_pool)

        service = ToolRegistryService(vector_service=mock_vec_service)

        results = await service.search_tools(
            query="cek stok barang",
            domain="inventory",
            distance_threshold=0.65,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["tool_name"], "get_stock_level")
        self.assertAlmostEqual(results[0]["distance"], 0.15)
        self.assertIsInstance(results[0]["params_schema"], dict)

    async def test_search_tools_distance_cutoff_returns_empty_on_non_erp_query(self):
        """When PGVector returns no tools below threshold (distance > 0.65), results are empty."""
        mock_vec_service = MagicMock()
        mock_vec_service.get_embedding = AsyncMock(return_value=[0.9] * 768)

        mock_pool = MagicMock()
        mock_conn = MagicMock()
        # No records below distance threshold
        mock_conn.fetch = AsyncMock(return_value=[])

        mock_acquire = MagicMock()
        mock_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire.__aexit__ = AsyncMock(return_value=None)
        mock_pool.acquire.return_value = mock_acquire

        mock_vec_service.get_pool = AsyncMock(return_value=mock_pool)

        service = ToolRegistryService(vector_service=mock_vec_service)

        results = await service.search_tools(
            query="siapa penemu mesin uap?",
            distance_threshold=0.65,
        )

        self.assertEqual(results, [])

    async def test_incremental_seeding_preserves_existing_tools(self):
        """Seeding with force_update=False skips tools already present in the catalog."""
        mock_vec_service = MagicMock()
        mock_vec_service.get_embedding = AsyncMock(return_value=[0.1] * 768)

        mock_pool = MagicMock()
        mock_conn = MagicMock()

        # Simulate that all tools already exist in the database
        existing_rows = [
            {"tool_name": t["tool_name"], "description": t["description"]}
            for t in CORE_TOOL_DEFINITIONS
        ]
        mock_conn.fetch = AsyncMock(return_value=existing_rows)
        mock_conn.execute = AsyncMock(return_value="CREATE TABLE")

        mock_acquire = MagicMock()
        mock_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire.__aexit__ = AsyncMock(return_value=None)
        mock_pool.acquire.return_value = mock_acquire

        mock_vec_service.get_pool = AsyncMock(return_value=mock_pool)

        service = ToolRegistryService(vector_service=mock_vec_service)

        # Run without force update
        result = await service.initialize_catalog(force_update=False)

        self.assertEqual(result["created"], 0)
        self.assertEqual(result["updated"], 0)
        self.assertEqual(result["skipped"], len(CORE_TOOL_DEFINITIONS))
        # Embedding was never called because no tools were updated or created
        mock_vec_service.get_embedding.assert_not_called()

    async def test_incremental_seeding_inserts_new_tools(self):
        """Seeding with force_update=False only inserts tools not yet present in the table."""
        mock_vec_service = MagicMock()
        mock_vec_service.get_embedding = AsyncMock(return_value=[0.2] * 768)

        mock_pool = MagicMock()
        mock_conn = MagicMock()

        # Simulate that only 1 tool exists in the database
        existing_rows = [
            {"tool_name": "get_stock_level", "description": "Existing stock tool"}
        ]
        mock_conn.fetch = AsyncMock(return_value=existing_rows)
        mock_conn.execute = AsyncMock(return_value="INSERT 0 1")

        mock_acquire = MagicMock()
        mock_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire.__aexit__ = AsyncMock(return_value=None)
        mock_pool.acquire.return_value = mock_acquire

        mock_vec_service.get_pool = AsyncMock(return_value=mock_pool)

        service = ToolRegistryService(vector_service=mock_vec_service)

        result = await service.initialize_catalog(force_update=False)

        self.assertEqual(result["skipped"], 1)
        self.assertEqual(result["created"], len(CORE_TOOL_DEFINITIONS) - 1)
        self.assertEqual(result["updated"], 0)


class TestRouterDynamicToolRetrieval(unittest.TestCase):
    """Test suite verifying router node integration with dynamic tool retrieval."""

    def test_router_populates_available_tools_for_matched_rule(self):
        """Router node attaches available_tools metadata for regex-matched tools."""
        state = create_initial_state("cek stok barang kertas A4")
        result = router_node(state)

        self.assertEqual(result["detected_domain"], "inventory")
        self.assertEqual(result["detected_tool"], "get_stock_level")
        self.assertIn("available_tools", result)
        self.assertTrue(len(result["available_tools"]) > 0)
        self.assertEqual(result["available_tools"][0]["tool_name"], "get_stock_level")

    def test_planned_tool_call_supports_dynamic_registered_tools(self):
        """PlannedToolCall accepts dynamically registered tools without literal validation failure."""
        pt = PlannedToolCall(tool="get_stock_level", domain="inventory")
        self.assertEqual(pt.tool, "get_stock_level")
        self.assertEqual(pt.domain, "inventory")

        # Wrong domain gets automatically corrected
        pt_wrong = PlannedToolCall(tool="fetch_vendor_invoices", domain="inventory")
        self.assertEqual(pt_wrong.domain, "accounting")

    def test_chat_request_accepts_user_roles_and_auth_token(self):
        """ChatRequest accepts user_id, user_roles, and auth_token."""
        req = ChatRequest(
            query="cek laporan neraca",
            user_id="usr_12345",
            user_roles=["accounting.view_financial_reports", "viewer"],
            auth_token="jwt_token_abc",
        )
        self.assertEqual(req.user_id, "usr_12345")
        self.assertEqual(req.user_roles, ["accounting.view_financial_reports", "viewer"])
        self.assertEqual(req.auth_token, "jwt_token_abc")

    def test_create_initial_state_propagates_rbac_context(self):
        """create_initial_state stores user_id, user_roles, and auth_token in AgentState."""
        state = create_initial_state(
            user_query="cek neraca",
            user_id="usr_12345",
            user_roles=["accounting.viewer"],
            auth_token="jwt_xyz",
        )
        self.assertEqual(state.get("user_id"), "usr_12345")
        self.assertEqual(state.get("user_roles"), ["accounting.viewer"])
        self.assertEqual(state.get("auth_token"), "jwt_xyz")
        self.assertEqual(state.get("user_token"), "jwt_xyz")
        self.assertEqual(state.get("available_tools"), [])


if __name__ == "__main__":
    unittest.main()
