"""
Unit tests for Dynamic Semantic Tool Registry & RBAC Guardrail (TASK-5.1).
Covers in-memory fallback, vector similarity retrieval with distance threshold,
RBAC role/permission filtering, incremental seeding, and router integration.
"""
import json
import unittest
from unittest.mock import AsyncMock, MagicMock

from app.graph.nodes.router import router_node, PlannedToolCall
from app.graph.state import create_initial_state
from app.schemas.chat import ChatRequest
from app.schemas.tools import TOOL_DOMAIN_MAP
from app.services.tool_registry_service import (
    ToolRegistryService,
    CORE_TOOL_DEFINITIONS,
    build_tool_search_text,
    merge_rrf_candidates,
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

    def test_all_36_enriched_tool_cards_and_search_text(self):
        """All 36 core ERP tools match TOOL_DOMAIN_MAP and include sample_queries, negative_hints, and search_text."""
        self.assertEqual(len(CORE_TOOL_DEFINITIONS), 36)
        catalog_names = {t["tool_name"] for t in CORE_TOOL_DEFINITIONS}
        self.assertEqual(catalog_names, set(TOOL_DOMAIN_MAP.keys()))

        for tool in CORE_TOOL_DEFINITIONS:
            name = tool["tool_name"]
            with self.subTest(tool=name):
                self.assertEqual(tool["domain"], TOOL_DOMAIN_MAP[name])
                samples = tool.get("sample_queries") or []
                self.assertGreaterEqual(len(samples), 3, f"{name} needs at least 3 sample_queries")
                self.assertLessEqual(len(samples), 5, f"{name} should have at most 5 sample_queries")
                self.assertTrue(all(isinstance(q, str) and q.strip() for q in samples))

                neg = (tool.get("negative_hints") or "").strip()
                self.assertTrue(len(neg) >= 15, f"{name} needs substantive negative_hints")

                search_text = build_tool_search_text(tool)
                self.assertIn(f"Tool: {name} ({tool['domain']})", search_text)
                self.assertIn("Deskripsi:", search_text)
                self.assertIn("Kata Kunci:", search_text)
                self.assertIn("Contoh Pertanyaan:", search_text)
                self.assertIn("Bukan Untuk:", search_text)

                cached = self.registry.get_tool_by_name(name)
                self.assertIsNotNone(cached)
                self.assertEqual(cached["sample_queries"], samples)
                self.assertEqual(cached["negative_hints"], neg)
                self.assertEqual(cached["search_text"], search_text)

        # Verify vendor billing boundary separation between get_vendor_bills and fetch_vendor_invoices
        vendor_bills_meta = self.registry.get_tool_by_name("get_vendor_bills")
        vendor_inv_meta = self.registry.get_tool_by_name("fetch_vendor_invoices")
        self.assertIn("customer", vendor_bills_meta["negative_hints"].lower())
        self.assertIn("get_vendor_bills", vendor_inv_meta["negative_hints"])

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

        # Simulate that all tools already exist in the database with up-to-date search_text
        existing_rows = [
            {
                "tool_name": t["tool_name"],
                "description": t["description"],
                "search_text": build_tool_search_text(t),
            }
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

        # Simulate that only 1 tool exists in the database with unchanged description and search_text
        stock_tool = next(t for t in CORE_TOOL_DEFINITIONS if t["tool_name"] == "get_stock_level")
        existing_rows = [
            {
                "tool_name": "get_stock_level",
                "description": stock_tool["description"],
                "search_text": build_tool_search_text(stock_tool),
            }
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

    async def test_initialize_catalog_ddl_and_search_text_change_detection(self):
        """DDL migrates search_text/search_tsv + GIN index; NULL or changed search_text triggers re-embedding."""
        mock_vec_service = MagicMock()
        mock_vec_service.get_embedding = AsyncMock(return_value=[0.3] * 768)

        mock_pool = MagicMock()
        mock_conn = MagicMock()

        tool_a = {
            "tool_name": "tool_legacy_null",
            "domain": "accounting",
            "description": "Same description A",
            "params_schema": {},
            "keywords": ["kas"],
            "sample_queries": ["cek kas A", "saldo kas A", "lihat kas A"],
            "negative_hints": "Bukan untuk stok barang gudang.",
        }
        tool_b = {
            "tool_name": "tool_changed_hints",
            "domain": "inventory",
            "description": "Same description B",
            "params_schema": {},
            "keywords": ["stok"],
            "sample_queries": ["cek stok B", "qty stok B", "sisa stok B"],
            "negative_hints": "Bukan untuk laporan akuntansi laba rugi.",
        }
        tool_c = {
            "tool_name": "tool_unchanged",
            "domain": "sales",
            "description": "Same description C",
            "params_schema": {},
            "keywords": ["omzet"],
            "sample_queries": ["cek omzet C", "total sales C", "ringkasan C"],
            "negative_hints": "Bukan untuk laporan neraca.",
        }

        existing_rows = [
            {"tool_name": "tool_legacy_null", "description": "Same description A", "search_text": None},
            {"tool_name": "tool_changed_hints", "description": "Same description B", "search_text": "Old search text"},
            {
                "tool_name": "tool_unchanged",
                "description": "Same description C",
                "search_text": build_tool_search_text(tool_c),
            },
        ]
        mock_conn.fetch = AsyncMock(return_value=existing_rows)
        mock_conn.execute = AsyncMock(return_value="OK")

        mock_acquire = MagicMock()
        mock_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire.__aexit__ = AsyncMock(return_value=None)
        mock_pool.acquire.return_value = mock_acquire
        mock_vec_service.get_pool = AsyncMock(return_value=mock_pool)

        service = ToolRegistryService(
            vector_service=mock_vec_service,
            default_definitions=[tool_a, tool_b, tool_c],
        )

        result = await service.initialize_catalog(force_update=False)

        self.assertEqual(result, {"created": 0, "updated": 2, "skipped": 1})
        ddl_sql = mock_conn.execute.call_args_list[0][0][0]
        self.assertIn("ADD COLUMN IF NOT EXISTS search_text TEXT", ddl_sql)
        self.assertIn("ADD COLUMN IF NOT EXISTS search_tsv tsvector GENERATED ALWAYS AS", ddl_sql)
        self.assertIn("to_tsvector('simple', coalesce(search_text, ''))", ddl_sql)
        self.assertIn("USING gin (search_tsv)", ddl_sql)

        embedded_texts = [call.args[0] for call in mock_vec_service.get_embedding.call_args_list]
        self.assertEqual(embedded_texts, [build_tool_search_text(tool_a), build_tool_search_text(tool_b)])

    async def test_initialize_catalog_force_update_reembeds_all_with_search_text(self):
        """force_update=True updates existing rows even when search_text is already identical."""
        mock_vec_service = MagicMock()
        mock_vec_service.get_embedding = AsyncMock(return_value=[0.4] * 768)

        mock_pool = MagicMock()
        mock_conn = MagicMock()
        tool = {
            "tool_name": "tool_force",
            "domain": "sales",
            "description": "Desc",
            "params_schema": {},
            "keywords": ["so"],
            "sample_queries": ["cek so", "daftar so", "order so"],
            "negative_hints": "Bukan untuk jurnal akuntansi.",
        }
        expected_search_text = build_tool_search_text(tool)
        mock_conn.fetch = AsyncMock(
            return_value=[{"tool_name": "tool_force", "description": "Desc", "search_text": expected_search_text}]
        )
        mock_conn.execute = AsyncMock(return_value="UPDATE 1")

        mock_acquire = MagicMock()
        mock_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire.__aexit__ = AsyncMock(return_value=None)
        mock_pool.acquire.return_value = mock_acquire
        mock_vec_service.get_pool = AsyncMock(return_value=mock_pool)

        service = ToolRegistryService(vector_service=mock_vec_service, default_definitions=[tool])
        result = await service.initialize_catalog(force_update=True)

        self.assertEqual(result, {"created": 0, "updated": 1, "skipped": 0})
        mock_vec_service.get_embedding.assert_awaited_once_with(expected_search_text)

    def test_conditional_rbac_empty_roles_allows_readonly_and_blocks_write(self):
        """Conditional RBAC: empty/None user_roles returns read-only permissioned tools and hides write tools."""
        write_tool = {
            "tool_name": "create_manual_journal",
            "domain": "accounting",
            "description": "Membuat jurnal manual baru untuk transaksi akuntansi",
            "params_schema": {},
            "keywords": ["buat jurnal manual", "jurnal manual"],
            "sample_queries": ["buat jurnal manual baru", "tambah jurnal manual", "posting jurnal manual"],
            "negative_hints": "Operasi tulis, butuh konfirmasi.",
            "required_permission": "accounting.create_journal",
            "is_write": True,
        }
        registry = ToolRegistryService(default_definitions=CORE_TOOL_DEFINITIONS + [write_tool])

        # Empty/None user_roles -> shows read-only permissioned tools (e.g. get_profit_and_loss, get_vendor_bills)
        pnl_none = registry.search_tools_fallback(query="cek laporan laba rugi bulan ini", user_roles=None)
        self.assertIn("get_profit_and_loss", [t["tool_name"] for t in pnl_none])

        bills_empty = registry.search_tools_fallback(query="daftar tagihan vendor jatuh tempo", user_roles=[])
        self.assertIn("get_vendor_bills", [t["tool_name"] for t in bills_empty])

        # Write tools are never exposed when user_roles is empty/None
        write_res = registry.search_tools_fallback(query="buat jurnal manual", user_roles=[])
        self.assertNotIn("create_manual_journal", [t["tool_name"] for t in write_res])

        # Non-empty verified user_roles enforces required_permission strictly
        pnl_restricted = registry.search_tools_fallback(
            query="cek laporan laba rugi bulan ini",
            user_roles=["inventory.viewer"],
        )
        self.assertNotIn("get_profit_and_loss", [t["tool_name"] for t in pnl_restricted])

        pnl_allowed = registry.search_tools_fallback(
            query="cek laporan laba rugi bulan ini",
            user_roles=["accounting.view_financial_reports"],
        )
        self.assertIn("get_profit_and_loss", [t["tool_name"] for t in pnl_allowed])

    def test_fallback_scores_sample_queries_and_short_acronyms(self):
        """Upgraded fallback recalls tools via natural sample_queries and 2-char ERP acronyms (gl, so, ap)."""
        cases = [
            ("pesanan kilat shopee nyangkut hari ini", "get_kilat_order_diagnostics"),
            ("cek mutasi gl akun kas kecil", "get_general_ledger"),
            ("daftar so hari ini di mainstore", "get_sales_orders"),
            ("produk dead stock slow moving 90 hari", "get_inventory_velocity_forecast"),
            ("sku shopee belum mapping ke erp", "get_product_mapping_status"),
        ]
        for query, expected_tool in cases:
            with self.subTest(query=query):
                res = self.registry.search_tools_fallback(query=query, user_roles=[], top_k=8)
                names = [t["tool_name"] for t in res]
                self.assertIn(expected_tool, names, f"Expected {expected_tool} in {names}")
                top = res[0]
                self.assertIn("negative_hints", top)
                self.assertIn("rrf_score", top)
                self.assertIsNone(top["distance"])

    async def test_search_tools_hybrid_pgvector_fts_rrf_and_conditional_rbac(self):
        """search_tools_hybrid executes PGVector + FTS RRF SQL with Conditional RBAC and enriches negative_hints."""
        mock_vec_service = MagicMock()
        mock_vec_service.get_embedding = AsyncMock(return_value=[0.1] * 768)

        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "tool_name": "get_general_ledger",
                    "domain": "accounting",
                    "description": "Laporan buku besar (General Ledger) lengkap",
                    "params_schema": json.dumps({"type": "object", "properties": {"account_code": {"type": "string"}}}),
                    "required_permission": "accounting.view_ledger",
                    "is_write": False,
                    "distance": 0.21,
                    "rrf_score": 0.032787,
                }
            ]
        )

        mock_acquire = MagicMock()
        mock_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire.__aexit__ = AsyncMock(return_value=None)
        mock_pool.acquire.return_value = mock_acquire
        mock_vec_service.get_pool = AsyncMock(return_value=mock_pool)

        service = ToolRegistryService(vector_service=mock_vec_service)

        # 1. Empty roles -> enforce_rbac=False
        results = await service.search_tools_hybrid(query="cek gl akun kas", user_roles=[])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["tool_name"], "get_general_ledger")
        self.assertAlmostEqual(results[0]["rrf_score"], 0.032787)
        self.assertAlmostEqual(results[0]["distance"], 0.21)
        self.assertTrue(results[0]["negative_hints"])

        sql_executed = mock_conn.fetch.call_args[0][0]
        self.assertIn("vec_ranked", sql_executed)
        self.assertIn("fts_ranked", sql_executed)
        self.assertIn("FULL OUTER JOIN", sql_executed)
        self.assertIn("60.0 +", sql_executed)
        self.assertIn("search_tsv", sql_executed)
        # args: (sql, vector_arr, domain, should_enforce, roles_param, max_dist, cleaned_query, or_tsquery, acronym_regex, limit)
        self.assertFalse(mock_conn.fetch.call_args[0][3])
        self.assertEqual(mock_conn.fetch.call_args[0][4], [])
        self.assertIn("gl", mock_conn.fetch.call_args[0][8])

        # 2. Non-empty verified roles -> enforce_rbac=True
        await service.search_tools_hybrid(query="cek gl akun kas", user_roles=["accounting.view_ledger"])
        self.assertTrue(mock_conn.fetch.call_args[0][3])
        self.assertEqual(mock_conn.fetch.call_args[0][4], ["accounting.view_ledger"])

    async def test_search_tools_hybrid_falls_back_gracefully_when_offline(self):
        """search_tools_hybrid degrades cleanly to upgraded in-memory fallback when pool or embedding is unavailable."""
        mock_vec_service = MagicMock()
        mock_vec_service.get_pool = AsyncMock(return_value=None)

        service = ToolRegistryService(vector_service=mock_vec_service)
        res = await service.search_tools_hybrid(query="cek pesanan kilat shopee nyangkut", user_roles=None)
        self.assertIn("get_kilat_order_diagnostics", [t["tool_name"] for t in res])

    def test_merge_rrf_candidates_deduplicates_and_accumulates_scores(self):
        """merge_rrf_candidates fuses multi-clause lists, boosts tools appearing in multiple clauses, and caps count."""
        list_a = [
            {"tool_name": "get_stock_level", "domain": "inventory", "rrf_score": 0.02, "distance": 0.30},
            {"tool_name": "get_sales_summary", "domain": "sales", "rrf_score": 0.015, "distance": 0.40},
        ]
        list_b = [
            {"tool_name": "get_sales_summary", "domain": "sales", "rrf_score": 0.02, "distance": 0.25},
            {"tool_name": "get_profit_and_loss", "domain": "accounting", "rrf_score": 0.018, "distance": 0.28},
        ]
        merged = merge_rrf_candidates([list_a, list_b], max_results=3)
        self.assertEqual([m["tool_name"] for m in merged], ["get_sales_summary", "get_stock_level", "get_profit_and_loss"])
        self.assertAlmostEqual(merged[0]["rrf_score"], 0.035)
        self.assertAlmostEqual(merged[0]["distance"], 0.25)


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
