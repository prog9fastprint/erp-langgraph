"""
Unit and integration tests for FastPrint ERP Inventory AI Tools (10 Tools Specification).
Validates schema contracts, client methods, routing heuristics, parameter extraction,
verifier record counting, and markdown synthesizer formatting.
"""
import unittest
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import ValidationError

from app.schemas.tools import (
    StockLevelFilterSchema,
    ProductInfoFilterSchema,
    LocationsFilterSchema,
    StockHistoryFilterSchema,
    TransfersFilterSchema,
    LowStockAlertsFilterSchema,
    StockAdjustmentsFilterSchema,
    ScrapRecordsFilterSchema,
    WarehousesFilterSchema,
    ProductCategoriesFilterSchema,
    # Response envelopes & data schemas
    StockLevelResponseSchema,
    ProductInfoResponseSchema,
    LocationsResponseSchema,
    StockHistoryResponseSchema,
    TransfersResponseSchema,
    LowStockAlertsResponseSchema,
    StockAdjustmentsResponseSchema,
    ScrapRecordsResponseSchema,
    WarehousesResponseSchema,
    ProductCategoriesResponseSchema,
    TOOL_DOMAIN_MAP,
)
from app.graph.nodes.router import router_node
from app.graph.nodes.extractor import (
    _fallback_extract_stock_level,
    _fallback_extract_stock_history,
    _fallback_extract_transfers,
    _fallback_extract_low_stock_alerts,
    _fallback_extract_stock_adjustments,
    _fallback_extract_scrap_records,
    _fallback_extract_warehouses,
    _fallback_extract_product_categories,
)
from app.graph.nodes.verifier import _calculate_records_count, verifier_node
from app.graph.nodes.synthesizer import (
    _build_table_query_meta,
    _format_multi_company_presentation,
    _format_single_tool_presentation,
    synthesizer_node,
)
from app.graph.nodes.tool_node import tool_node
from app.services.django_client import DjangoCoreClient


class TestInventorySchemas(unittest.TestCase):
    """Test suite validating all 10 inventory Pydantic schemas and response envelopes."""

    def test_all_10_inventory_tools_in_domain_map(self):
        """All 10 inventory tools must be registered in TOOL_DOMAIN_MAP."""
        expected_tools = [
            "get_product_info",
            "get_stock_level",
            "get_locations",
            "get_stock_history",
            "get_transfers",
            "get_low_stock_alerts",
            "get_stock_adjustments",
            "get_scrap_records",
            "get_warehouses",
            "get_product_categories",
        ]
        for tool_name in expected_tools:
            self.assertIn(tool_name, TOOL_DOMAIN_MAP)
            self.assertEqual(TOOL_DOMAIN_MAP[tool_name], "inventory")

    def test_stock_level_filter_warehouse_scan_mode(self):
        """Warehouse-only scan without SKU/product_id is valid in the new specification."""
        schema = StockLevelFilterSchema(warehouse_id=2, limit=50)
        self.assertEqual(schema.warehouse_id, 2)
        self.assertEqual(schema.limit, 50)
        self.assertIsNone(schema.code)
        self.assertIsNone(schema.product_id)

    def test_stock_level_filter_batch_codes_mode(self):
        """Batch SKU query is valid without single code or product_id."""
        schema = StockLevelFilterSchema(codes=["HVS-A4-70", "HVS-A4-80"], company_id=1, include_breakdown=True)
        self.assertEqual(schema.codes, ["HVS-A4-70", "HVS-A4-80"])
        self.assertTrue(schema.include_breakdown)

    def test_stock_level_filter_empty_raises_validation_error(self):
        """Must raise error if no identifier, warehouse, location, or batch list is provided."""
        with self.assertRaises(ValidationError) as ctx:
            StockLevelFilterSchema(company_id=1)
        self.assertIn("Either 'product_id' or 'code'", str(ctx.exception))

    def test_product_info_filter_extended_fields(self):
        """ProductInfoFilterSchema should support category_id, can_be_sold, and include_cost."""
        schema = ProductInfoFilterSchema(
            search="Kertas HVS",
            category_id=12,
            can_be_sold=True,
            include_cost=True,
            limit=5,
        )
        self.assertEqual(schema.category_id, 12)
        self.assertTrue(schema.can_be_sold)
        self.assertTrue(schema.include_cost)
        self.assertEqual(schema.limit, 5)

    def test_stock_history_filter_validation(self):
        """StockHistoryFilterSchema validates dates and bounds."""
        schema = StockHistoryFilterSchema(
            code="HVS-A4-70",
            date_from="2026-01-01",
            date_to="2026-01-31",
            limit=20,
        )
        self.assertEqual(schema.code, "HVS-A4-70")
        self.assertEqual(schema.date_from, "2026-01-01")

        with self.assertRaises(ValidationError):
            StockHistoryFilterSchema(date_from="invalid-date")

    def test_transfers_filter_validation(self):
        """TransfersFilterSchema validates reference, status, and dates."""
        schema = TransfersFilterSchema(
            ref="WH/IN/0001",
            status="Draft",
            include_lines=True,
        )
        self.assertEqual(schema.ref, "WH/IN/0001")
        self.assertEqual(schema.status, "Draft")
        self.assertTrue(schema.include_lines)

    def test_low_stock_alerts_filter(self):
        """LowStockAlertsFilterSchema validates warehouse_id and threshold."""
        schema = LowStockAlertsFilterSchema(warehouse_id=2, threshold=10.0)
        self.assertEqual(schema.warehouse_id, 2)
        self.assertEqual(schema.threshold, 10.0)

    def test_stock_adjustments_filter(self):
        """StockAdjustmentsFilterSchema validates status and dates."""
        schema = StockAdjustmentsFilterSchema(status="accepted", date_from="2026-02-01")
        self.assertEqual(schema.status, "accepted")

    def test_scrap_records_filter(self):
        """ScrapRecordsFilterSchema validates dates and product_id."""
        schema = ScrapRecordsFilterSchema(product_id=1024, date_from="2026-03-01")
        self.assertEqual(schema.product_id, 1024)

    def test_warehouses_and_categories_filters(self):
        """WarehousesFilterSchema and ProductCategoriesFilterSchema validate search queries."""
        wh = WarehousesFilterSchema(search="Surabaya", include_locations=True)
        self.assertEqual(wh.search, "Surabaya")
        self.assertTrue(wh.include_locations)

        cat = ProductCategoriesFilterSchema(search="Kertas")
        self.assertEqual(cat.search, "Kertas")

    def test_inventory_response_envelopes(self):
        """Response envelopes parse valid data payloads correctly."""
        # Stock history envelope
        sh_payload = {
            "success": True,
            "data": {
                "total_count": 1,
                "history": [
                    {
                        "date": "2026-01-15",
                        "reference": "WH/IN/0001",
                        "code": "HVS-A4-70",
                        "operation_type": "Receipts",
                        "qty_done": 50.0,
                        "balance": 150.0,
                    }
                ],
            },
        }
        sh_env = StockHistoryResponseSchema(**sh_payload)
        self.assertTrue(sh_env.success)
        self.assertEqual(sh_env.data.total_count, 1)

        # Transfers envelope
        tr_payload = {
            "success": True,
            "data": {
                "total_count": 1,
                "transfers": [
                    {
                        "id": 101,
                        "ref": "WH/OUT/0022",
                        "status": "Done",
                        "lines": [{"product_code": "HVS-A4-70", "quantity": 10.0}],
                    }
                ],
            },
        }
        tr_env = TransfersResponseSchema(**tr_payload)
        self.assertTrue(tr_env.success)


class TestInventoryRouting(unittest.TestCase):
    """Test suite validating keyword/regex routing across all inventory tools."""

    def test_route_to_stock_history(self):
        """Queries for kartu stok or riwayat mutasi must route to get_stock_history."""
        state = {"messages": [], "user_query": "Cek kartu stok untuk SKU HVS-A4-70 bulan ini"}
        decision = router_node(state)
        self.assertEqual(decision["detected_domain"], "inventory")
        self.assertEqual(decision["detected_tool"], "get_stock_history")

    def test_route_to_transfers(self):
        """Queries for surat jalan or transfer barang must route to get_transfers."""
        state = {"messages": [], "user_query": "Tampilkan dokumen surat jalan yang statusnya Draft"}
        decision = router_node(state)
        self.assertEqual(decision["detected_domain"], "inventory")
        self.assertEqual(decision["detected_tool"], "get_transfers")

    def test_route_to_low_stock_alerts(self):
        """Queries for stok menipis or stok habis must route to get_low_stock_alerts."""
        state = {"messages": [], "user_query": "Tampilkan daftar produk yang stoknya menipis atau habis"}
        decision = router_node(state)
        self.assertEqual(decision["detected_domain"], "inventory")
        self.assertEqual(decision["detected_tool"], "get_low_stock_alerts")

    def test_route_to_stock_adjustments(self):
        """Queries for stock opname or selisih stok must route to get_stock_adjustments."""
        state = {"messages": [], "user_query": "Ada hasil stock opname dan selisih stok bulan ini?"}
        decision = router_node(state)
        self.assertEqual(decision["detected_domain"], "inventory")
        self.assertEqual(decision["detected_tool"], "get_stock_adjustments")

    def test_route_to_scrap_records(self):
        """Queries for barang rusak or catatan scrap must route to get_scrap_records."""
        state = {"messages": [], "user_query": "Cek riwayat catatan barang rusak atau scrap order"}
        decision = router_node(state)
        self.assertEqual(decision["detected_domain"], "inventory")
        self.assertEqual(decision["detected_tool"], "get_scrap_records")

    def test_route_to_warehouses(self):
        """Queries for master daftar gudang must route to get_warehouses."""
        state = {"messages": [], "user_query": "Tampilkan daftar master gudang perusahaan"}
        decision = router_node(state)
        self.assertEqual(decision["detected_domain"], "inventory")
        self.assertEqual(decision["detected_tool"], "get_warehouses")

    def test_route_to_product_categories(self):
        """Queries for kategori produk must route to get_product_categories."""
        state = {"messages": [], "user_query": "Daftar kategori produk yang ada apa saja?"}
        decision = router_node(state)
        self.assertEqual(decision["detected_domain"], "inventory")
        self.assertEqual(decision["detected_tool"], "get_product_categories")

    def test_route_to_stock_level(self):
        """Standard stock check must still route to get_stock_level."""
        state = {"messages": [], "user_query": "Berapa stok kertas HVS A4 saat ini?"}
        decision = router_node(state)
        self.assertEqual(decision["detected_domain"], "inventory")
        self.assertEqual(decision["detected_tool"], "get_stock_level")


class TestInventoryExtractor(unittest.TestCase):
    """Test suite validating fallback parameter extraction for inventory tools."""

    def test_extract_stock_history_parameters(self):
        ref_d = date(2026, 3, 15)
        params = _fallback_extract_stock_history("kartu stok HVS-A4-70 bulan ini", ref_date=ref_d)
        self.assertEqual(params.get("code"), "HVS-A4-70")
        self.assertEqual(params.get("date_from"), "2026-03-01")
        self.assertEqual(params.get("date_to"), "2026-03-31")

    def test_extract_transfers_parameters(self):
        params = _fallback_extract_transfers("cek transfer WH/IN/0045 status draft")
        self.assertEqual(params.get("ref"), "WH/IN/0045")
        self.assertEqual(params.get("status"), "Draft")

    def test_extract_low_stock_alerts_parameters(self):
        params = _fallback_extract_low_stock_alerts("alert stok di bawah 15 unit")
        self.assertEqual(params.get("threshold"), 15.0)

    def test_extract_stock_adjustments_parameters(self):
        params = _fallback_extract_stock_adjustments("penyesuaian stok status accepted")
        self.assertEqual(params.get("status"), "accepted")

    def test_extract_warehouses_and_categories(self):
        wh_params = _fallback_extract_warehouses("daftar gudang surabaya timur")
        self.assertIn("surabaya", wh_params.get("search", "").lower())

        cat_params = _fallback_extract_product_categories("kategori produk tinta printer")
        self.assertIn("tinta", cat_params.get("search", "").lower())


class TestInventoryVerifier(unittest.TestCase):
    """Test suite validating record count calculation across diverse inventory payloads."""

    def test_verifier_counts_transfers_and_history(self):
        self.assertEqual(_calculate_records_count({"history": [{"id": 1}, {"id": 2}]}), 2)
        self.assertEqual(_calculate_records_count({"transfers": [{"id": 1}]}), 1)
        self.assertEqual(_calculate_records_count({"alerts": [{"code": "SKU1"}]}), 1)
        self.assertEqual(_calculate_records_count({"adjustments": []}), 0)
        self.assertEqual(_calculate_records_count({"warehouses": [{"id": 1}, {"id": 2}]}), 2)
        self.assertEqual(_calculate_records_count({"categories": [{"id": 10}]}), 1)
        self.assertEqual(_calculate_records_count({"scraps": [{"id": 5}]}), 1)


class TestInventorySynthesizer(unittest.TestCase):
    """Test suite validating Markdown presentation formatting for inventory tools."""

    def test_format_stock_level_with_breakdown(self):
        payload = {
            "mode": "single_product",
            "code": "HVS-A4-70",
            "name": "Kertas HVS A4 70gr",
            "qty_on_hand": 150.0,
            "breakdown": [
                {
                    "warehouse_name": "Gudang Utama",
                    "location_name": "Rak A1",
                    "qty": 100.0,
                },
                {
                    "warehouse_name": "Gudang Cabang",
                    "location_name": "Rak B2",
                    "qty": 50.0,
                },
            ],
        }
        md = _format_single_tool_presentation("get_stock_level", payload)
        self.assertIn("Kertas HVS A4 70gr", md)
        self.assertIn("150.0 unit", md)
        self.assertIn("Gudang Utama", md)
        self.assertIn("Rak A1", md)
        self.assertIn("100 unit", md)

    def test_format_stock_history(self):
        payload = {
            "total_count": 1,
            "history": [
                {
                    "date": "2026-03-10",
                    "reference": "WH/IN/0001",
                    "code": "HVS-A4-70",
                    "operation_type": "Receipts",
                    "location_name": "Rak A1",
                    "qty_done": 50.0,
                    "balance": 150.0,
                }
            ],
        }
        md = _format_single_tool_presentation("get_stock_history", payload)
        self.assertIn("Riwayat Kartu Stok", md)
        self.assertIn("WH/IN/0001", md)
        self.assertIn("+50", md)
        self.assertIn("150", md)

    def test_format_transfers_with_lines(self):
        payload = {
            "total_count": 1,
            "transfers": [
                {
                    "ref": "WH/OUT/0010",
                    "date": "2026-03-12",
                    "operation_type": "Delivery Orders",
                    "status": "Done",
                    "lines": [
                        {"product_code": "HVS-A4-70", "product_name": "Kertas HVS", "quantity": 10.0, "uom": "Rim"}
                    ],
                }
            ],
        }
        md = _format_single_tool_presentation("get_transfers", payload)
        self.assertIn("Dokumen Transfer & Mutasi", md)
        self.assertIn("WH/OUT/0010", md)
        self.assertIn("DONE", md)
        self.assertIn("Kertas HVS", md)

    def test_format_low_stock_alerts(self):
        payload = {
            "total_found": 1,
            "alerts": [
                {
                    "code": "TINTA-BK-01",
                    "name": "Tinta Hitam Epson",
                    "warehouse_name": "Gudang Surabaya",
                    "qty_on_hand": 2.0,
                    "threshold": 10.0,
                }
            ],
        }
        md = _format_single_tool_presentation("get_low_stock_alerts", payload)
        self.assertIn("Peringatan Stok Menipis", md)
        self.assertIn("TINTA-BK-01", md)
        self.assertIn("MENIPIS", md)


class TestInventoryDjangoClientMethods(unittest.IsolatedAsyncioTestCase):
    """Test suite validating DjangoCoreClient async helper methods for new inventory tools."""

    async def test_all_inventory_client_methods(self):
        mock_client = AsyncMock(spec=DjangoCoreClient)
        mock_client.execute_tool = AsyncMock(return_value={"success": True, "data": {}})

        client = DjangoCoreClient()
        client.execute_tool = mock_client.execute_tool

        await client.fetch_stock_history({"code": "SKU1"})
        client.execute_tool.assert_called_with(
            domain="inventory",
            tool_name="get_stock_history",
            params={"code": "SKU1"},
            company_id=None,
            user_token=None,
        )

        await client.fetch_transfers({"ref": "WH/IN/1"})
        client.execute_tool.assert_called_with(
            domain="inventory",
            tool_name="get_transfers",
            params={"ref": "WH/IN/1"},
            company_id=None,
            user_token=None,
        )

        await client.fetch_low_stock_alerts({"threshold": 5.0})
        client.execute_tool.assert_called_with(
            domain="inventory",
            tool_name="get_low_stock_alerts",
            params={"threshold": 5.0},
            company_id=None,
            user_token=None,
        )

        await client.fetch_stock_adjustments({"status": "accepted"})
        client.execute_tool.assert_called_with(
            domain="inventory",
            tool_name="get_stock_adjustments",
            params={"status": "accepted"},
            company_id=None,
            user_token=None,
        )

        await client.fetch_scrap_records({"product_id": 10})
        client.execute_tool.assert_called_with(
            domain="inventory",
            tool_name="get_scrap_records",
            params={"product_id": 10},
            company_id=None,
            user_token=None,
        )

        await client.fetch_warehouses(WarehousesFilterSchema(search="Pusat"))
        client.execute_tool.assert_called_with(
            domain="inventory",
            tool_name="get_warehouses",
            params={"search": "Pusat", "include_locations": True},
            company_id=None,
            user_token=None,
        )

        await client.fetch_product_categories({"search": "Tinta"})
        client.execute_tool.assert_called_with(
            domain="inventory",
            tool_name="get_product_categories",
            params={"search": "Tinta"},
            company_id=None,
            user_token=None,
        )


class TestStockIdentifierAndWarehouseResolution(unittest.IsolatedAsyncioTestCase):
    """Test suite for warehouse fallback lookup and missing product identifier guidance."""

    def test_fallback_extract_stock_level_with_gudang_prefix(self):
        """Extracts location_name when user query uses 'gudang <name>' or 'di gudang <name>'."""
        params1 = _fallback_extract_stock_level("cek stok gudang surabaya")
        self.assertEqual(params1.get("location_name"), "surabaya")

        params2 = _fallback_extract_stock_level("cek stok di gudang bandung")
        self.assertEqual(params2.get("location_name"), "bandung")

        params3 = _fallback_extract_stock_level("stok di surabaya")
        self.assertEqual(params3.get("location_name"), "surabaya")

    async def test_tool_node_resolves_warehouse_fallback(self):
        """When location lookup yields 0 results, tool_node resolves warehouse_id via fetch_warehouses."""
        mock_client = MagicMock(spec=DjangoCoreClient)
        # 1. fetch_locations returns empty list
        mock_client.fetch_locations = AsyncMock(return_value={"locations": []})
        # 2. fetch_warehouses returns warehouse match
        mock_client.fetch_warehouses = AsyncMock(return_value={"warehouses": [{"id": 7, "name": "Gudang Surabaya"}]})
        # 3. fetch_stock_level succeeds with resolved warehouse_id
        mock_client.fetch_stock_level = AsyncMock(return_value={"products": [], "total_count": 0, "warehouse_name": "Gudang Surabaya"})

        state = {
            "user_query": "cek stok gudang surabaya",
            "detected_tool": "get_stock_level",
            "extracted_params": {"location_name": "surabaya"},
            "company_id": 1,
        }

        res = await tool_node(state, client=mock_client)

        mock_client.fetch_locations.assert_called_once()
        mock_client.fetch_warehouses.assert_called_once()
        mock_client.fetch_stock_level.assert_called_once()

        call_kwargs = mock_client.fetch_stock_level.call_args[1]
        self.assertEqual(call_kwargs["params"].get("warehouse_id"), 7)
        self.assertEqual(res["verification_status"], "VALID")

    def test_synthesizer_missing_product_identifier(self):
        """Synthesizer provides clear Indonesian guidance when MISSING_PRODUCT_IDENTIFIER error occurs."""
        state = {
            "user_query": "cek stok",
            "detected_domain": "inventory",
            "detected_tool": "get_stock_level",
            "verification_status": "ERROR",
            "error_message": (
                'Django Core request failed with status 400: {"success": false, "data": null, '
                '"error": {"code": "MISSING_PRODUCT_IDENTIFIER", "message": "Either a product identifier '
                "('product_id', 'code', 'codes', 'product_ids') or a location filter ('location_id', 'warehouse_id') "
                'must be provided.", "details": {}}}'
            ),
        }

        ans = synthesizer_node(state)
        final_answer = ans["final_answer"]

        self.assertIn("Informasi Produk atau Gudang Belum Spesifik", final_answer)
        self.assertIn("Kode SKU / Barcode / Nama Produk", final_answer)
        self.assertIn("Nama Gudang / Lokasi Rak", final_answer)
        self.assertIn("HVS-A4-70", final_answer)


class TestLowStockCompanyScopingAndTableMeta(unittest.TestCase):
    """Tests for low stock company/warehouse formatting, pagination hints, and table_query_meta."""

    def test_format_low_stock_alerts_includes_company_and_threshold_note(self):
        payload = {
            "company_id": 1,
            "company_name": "FastPrint Pusat",
            "total_found": 120,
            "alerts": [
                {
                    "code": "TINTA-BK-01",
                    "name": "Tinta Hitam Epson",
                    "company_name": "FastPrint Pusat",
                    "warehouse_name": "Gudang Pusat",
                    "qty_on_hand": 0.0,
                    "threshold": 0.0,
                }
            ],
        }
        md = _format_single_tool_presentation("get_low_stock_alerts", payload)
        self.assertIn("| No | SKU | Nama Produk | Perusahaan | Gudang |", md)
        self.assertIn("FastPrint Pusat", md)
        self.assertIn("Gudang Pusat", md)
        self.assertIn("Batas minimum default <= 0 unit", md)
        self.assertIn("Menampilkan 1 dari 120 data.", md)
        self.assertNotIn("filter periode tanggal", md)

    def test_format_multi_company_low_stock_alerts_table(self):
        state = {
            "detected_tool": "get_low_stock_alerts",
            "multi_company_results": {
                1: {
                    "total_found": 2,
                    "alerts": [
                        {
                            "code": "SKU-01",
                            "name": "Kertas HVS",
                            "company_name": "FastPrint Pusat",
                            "warehouse_name": "Gudang Pusat",
                            "qty_on_hand": 0.0,
                            "threshold": 0.0,
                        }
                    ],
                },
                2: {
                    "total_found": 1,
                    "alerts": [
                        {
                            "code": "SKU-02",
                            "name": "Tinta Cyan",
                            "company_name": "FastPrint Surabaya",
                            "warehouse_name": "Gudang Surabaya",
                            "qty_on_hand": 2.0,
                            "threshold": 5.0,
                        }
                    ],
                },
            },
        }
        md = _format_multi_company_presentation(state)
        self.assertIn("Peringatan Stok Menipis / Kritis Multi-Cabang", md)
        self.assertIn("| No | Perusahaan / Cabang | SKU | Nama Produk | Gudang |", md)
        self.assertIn("FastPrint Pusat", md)
        self.assertIn("FastPrint Surabaya", md)
        self.assertIn("Gudang Surabaya", md)

    def test_build_table_query_meta_for_truncated_and_non_truncated_states(self):
        base_state = {
            "verification_status": "VALID",
            "detected_domain": "inventory",
            "detected_tool": "get_low_stock_alerts",
            "company_id": 1,
            "extracted_params": {"threshold": 0.0, "include_zero_stock": True, "limit": 50},
            "is_multi_company": False,
            "is_multi_tool": False,
            "tool_result": {
                "company_id": 1,
                "company_name": "FastPrint Pusat",
                "total_found": 120,
                "alerts": [
                    {"code": "A", "name": "Prod A", "qty_on_hand": 0.0, "threshold": 0.0},
                    {"code": "B", "name": "Prod B", "qty_on_hand": 0.0, "threshold": 0.0},
                ],
            },
        }
        meta = _build_table_query_meta(base_state)
        self.assertIsNotNone(meta)
        self.assertEqual(meta["domain"], "inventory")
        self.assertEqual(meta["tool"], "get_low_stock_alerts")
        self.assertEqual(meta["company_id"], 1)
        self.assertEqual(meta["total_count"], 120)
        self.assertEqual(meta["displayed_count"], 2)

        non_truncated = dict(base_state, tool_result={"total_found": 2, "alerts": base_state["tool_result"]["alerts"]})
        self.assertIsNone(_build_table_query_meta(non_truncated))

        multi_company = dict(base_state, is_multi_company=True)
        self.assertIsNone(_build_table_query_meta(multi_company))


