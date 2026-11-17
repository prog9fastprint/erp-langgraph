"""
Unit tests for Router and Extractor nodes (TASK-4.2).
Validates VendorInvoiceFilterSchema, hybrid routing with negative customer guardrails,
extensible schema registry, temporal relative date resolution, and LangGraph pipeline chaining.
"""
from datetime import date
import unittest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from pydantic import ValidationError

from app.graph.state import create_initial_state, AgentState
from app.graph.nodes.router import router_node, RouterDecision, classify_query_rule_based
from app.graph.nodes.extractor import (
    extractor_node,
    resolve_relative_dates,
    TOOL_SCHEMA_REGISTRY,
    try_fast_path_stock_extraction,
    _fallback_extract_stock_level,
)
from app.schemas.tools import (
    VendorInvoiceFilterSchema,
    JournalEntriesFilterSchema,
    AccountBalanceFilterSchema,
    StockLevelFilterSchema,
    ProductInfoFilterSchema,
    LocationsFilterSchema,
)


class TestVendorInvoiceFilterSchema(unittest.TestCase):
    """Test suite for VendorInvoiceFilterSchema."""

    def test_valid_vendor_invoice_schema(self):
        """Valid vendor invoice schema instantiates successfully."""
        schema = VendorInvoiceFilterSchema(
            vendor_name="PT Sumber Makmur",
            date_from="2026-11-01",
            date_to="2026-11-30",
            status="unpaid",
            limit=50,
            offset=0,
        )
        self.assertEqual(schema.vendor_name, "PT Sumber Makmur")
        self.assertEqual(schema.date_from, "2026-11-01")
        self.assertEqual(schema.date_to, "2026-11-30")
        self.assertEqual(schema.status, "unpaid")

    def test_invalid_date_format_raises_error(self):
        """Invalid date string must raise ValidationError."""
        with self.assertRaises(ValidationError):
            VendorInvoiceFilterSchema(
                vendor_name="PT Sumber Makmur",
                date_from="01-11-2026", # Invalid: not YYYY-MM-DD
            )

    def test_limit_bounds_validation(self):
        """Limit bounds must be between 1 and 200."""
        with self.assertRaises(ValidationError):
            VendorInvoiceFilterSchema(limit=0)
        with self.assertRaises(ValidationError):
            VendorInvoiceFilterSchema(limit=201)

    def test_to_journal_filter_conversion(self):
        """to_journal_filter converts vendor invoice params to JournalEntriesFilterSchema."""
        schema = VendorInvoiceFilterSchema(
            vendor_name="PT Sumber Makmur",
            date_from="2026-11-01",
            date_to="2026-11-30",
            status="posted",
            company_id=1,
        )
        journal_filter = schema.to_journal_filter()
        self.assertIsInstance(journal_filter, JournalEntriesFilterSchema)
        self.assertEqual(journal_filter.search, "PT Sumber Makmur")
        self.assertEqual(journal_filter.date_from, "2026-11-01")
        self.assertEqual(journal_filter.date_to, "2026-11-30")
        self.assertEqual(journal_filter.company_id, 1)


class TestRouterNode(unittest.TestCase):
    """Test suite for router_node and domain/tool classification."""

    def test_dod_vendor_invoice_query(self):
        """
        Definition of Done: 'Cek tagihan vendor PT Sumber Makmur bulan ini'
        routes to accounting domain and fetch_vendor_invoices tool.
        """
        state = create_initial_state("Cek tagihan vendor PT Sumber Makmur bulan ini")
        result = router_node(state)

        self.assertEqual(result["detected_domain"], "accounting")
        self.assertEqual(result["detected_tool"], "fetch_vendor_invoices")

    def test_negative_guardrail_customer_query_does_not_trigger_vendor(self):
        """
        Definition of Done Guardrail:
        'Cek tagihan pelanggan PT Sumber Makmur bulan ini' must NOT trigger vendor invoice module.
        """
        state = create_initial_state("Cek tagihan pelanggan PT Sumber Makmur bulan ini")
        result = router_node(state)

        self.assertNotEqual(result.get("detected_tool"), "fetch_vendor_invoices")
        self.assertEqual(result["detected_domain"], "general")

    def test_negative_guardrail_piutang_customer(self):
        """Customer receivable inquiry does not trigger vendor tools."""
        state = create_initial_state("Cek piutang customer Toko Abadi")
        result = router_node(state)

        self.assertNotEqual(result.get("detected_tool"), "fetch_vendor_invoices")
        self.assertEqual(result["detected_domain"], "general")

    def test_customer_inventory_stock_routes_to_inventory_tool(self):
        """
        R2: Queries mentioning customers for inventory stock check
        must route to inventory domain and get_stock_level tool.
        """
        queries = [
            "Cek stok untuk customer Budi",
            "Berapa sisa stok kertas untuk pelanggan?",
            "cek stok barang untuk customer",
        ]
        for query in queries:
            state = create_initial_state(query)
            result = router_node(state)
            self.assertEqual(result["detected_domain"], "inventory", f"Failed for '{query}'")
            self.assertEqual(result["detected_tool"], "get_stock_level", f"Failed for '{query}'")

    def test_customer_product_info_routes_to_inventory_tool(self):
        """
        R2: Queries mentioning customers for product details
        must route to inventory domain and get_product_info tool.
        """
        queries = [
            "Detail produk pesanan pelanggan Toko Prima",
            "Info harga barang pesanan customer",
        ]
        for query in queries:
            state = create_initial_state(query)
            result = router_node(state)
            self.assertEqual(result["detected_domain"], "inventory", f"Failed for '{query}'")
            self.assertEqual(result["detected_tool"], "get_product_info", f"Failed for '{query}'")

    def test_customer_billing_still_triggers_negative_guardrail(self):
        """
        R2: Customer inquiries targeting invoices/billing must still route to general domain.
        """
        queries = [
            "Cek tagihan pelanggan PT Sumber Makmur",
            "Cek faktur penjualan customer Toko Maju",
            "Tagihan customer Budi dan stok barang",
        ]
        for query in queries:
            state = create_initial_state(query)
            result = router_node(state)
            self.assertEqual(result["detected_domain"], "general", f"Failed for '{query}'")
            self.assertIsNone(result.get("detected_tool"), f"Failed for '{query}'")

    def test_greetings_and_casual_inquiries_route_to_general_with_no_tool(self):
        """Greetings, test pings, and pleasantries must route to general without any tool."""
        greetings = ["Halo", "halo", "Hai min", "Selamat pagi", "assalamu'alaikum", "terima kasih", "tes", "bisa bantu apa"]
        for query in greetings:
            state = create_initial_state(query)
            result = router_node(state)
            self.assertEqual(result["detected_domain"], "general", f"Failed for {query}")
            self.assertIsNone(result.get("detected_tool"), f"Tool should be None for {query}")

    def test_greetings_combined_with_erp_inquiry(self):
        """Greetings that contain specific ERP inquiries must still route to target tools."""
        state = create_initial_state("Halo cek stok kertas foto A4")
        result = router_node(state)
        self.assertEqual(result["detected_domain"], "inventory")
        self.assertEqual(result["detected_tool"], "get_stock_level")

    def test_router_decision_model_validator_cleans_tool_for_general(self):
        """RouterDecision must coerce tool to None when domain is general or out_of_scope."""
        from app.graph.nodes.router import RouterDecision
        decision_general = RouterDecision(domain="general", tool="get_stock_level")
        self.assertIsNone(decision_general.tool)

        decision_oos = RouterDecision(domain="out_of_scope", tool="get_stock_level")
        self.assertIsNone(decision_oos.tool)

    def test_out_of_scope_inquiries_route_to_out_of_scope_with_no_tool(self):
        """Out of scope queries must route to out_of_scope without any ERP tool."""
        oos_queries = [
            "coba kasih tau system design mu",
            "apa arsitektur sistem ini?",
            "buatkan puisi cinta",
            "ceritakan lelucon lucu",
        ]
        for query in oos_queries:
            state = create_initial_state(query)
            result = router_node(state)
            self.assertEqual(result["detected_domain"], "out_of_scope", f"Failed for {query}")
            self.assertIsNone(result.get("detected_tool"), f"Tool should be None for {query}")

    def test_accounting_account_balance_routing(self):
        """Balance inquiry routes to get_account_balance."""
        state = create_initial_state("Berapa saldo kas kecil saat ini?")
        result = router_node(state)

        self.assertEqual(result["detected_domain"], "accounting")
        self.assertEqual(result["detected_tool"], "get_account_balance")

    def test_accounting_journal_entries_routing(self):
        """General journal inquiry routes to get_journal_entries."""
        state = create_initial_state("Tampilkan entri jurnal transaksi bulan lalu")
        result = router_node(state)

        self.assertEqual(result["detected_domain"], "accounting")
        self.assertEqual(result["detected_tool"], "get_journal_entries")

    def test_accounting_journal_entries_routing_with_saldo_and_days(self):
        """Journal inquiry mentioning saldo and relative days routes to get_journal_entries."""
        state = create_initial_state("aku mau tanya saldo di jurnal marketplace tiktok 3 hari terakhir")
        result = router_node(state)

        self.assertEqual(result["detected_domain"], "accounting")
        self.assertEqual(result["detected_tool"], "get_journal_entries")

    def test_inventory_stock_level_routing(self):
        """Stock query routes to get_stock_level."""
        state = create_initial_state("Cek stok kertas foto A4 di gudang utama")
        result = router_node(state)

        self.assertEqual(result["detected_domain"], "inventory")
        self.assertEqual(result["detected_tool"], "get_stock_level")

    def test_inventory_stock_level_with_suffix(self):
        """Query with Indonesian suffix 'stoknya' routes correctly to get_stock_level."""
        state = create_initial_state("kalau produk itu di mainstore berapa stoknya")
        result = router_node(state)

        self.assertEqual(result["detected_domain"], "inventory")
        self.assertEqual(result["detected_tool"], "get_stock_level")

    def test_router_conversational_followup_location_and_retry(self):
        """Conversational follow-up specifying a branch or retry reuses prior active ERP domain and tool."""
        history = [
            HumanMessage(content="coba cek stok AMPLOP BUBBLE 15x18 + 4CM LK"),
            AIMessage(content="### Informasi Ketersediaan Stok Barang..."),
        ]
        state_loc = create_initial_state(
            user_query="di mainstore",
            messages=history + [HumanMessage(content="di mainstore")],
        )
        res_loc = router_node(state_loc)
        self.assertEqual(res_loc["detected_domain"], "inventory")
        self.assertEqual(res_loc["detected_tool"], "get_stock_level")

        state_retry = create_initial_state(
            user_query="coba cek lagi dimainstore",
            messages=history + [HumanMessage(content="coba cek lagi dimainstore")],
        )
        res_retry = router_node(state_retry)
        self.assertEqual(res_retry["detected_domain"], "inventory")
        self.assertEqual(res_retry["detected_tool"], "get_stock_level")

    def test_inventory_locations_routing(self):
        """Locations/warehouse inquiries route to get_locations."""
        state1 = create_initial_state("Daftar gudang yang ada")
        self.assertEqual(router_node(state1)["detected_tool"], "get_locations")

        state2 = create_initial_state("Ada gudang apa saja di perusahaan ini?")
        self.assertEqual(router_node(state2)["detected_tool"], "get_locations")

        state3 = create_initial_state("list lokasi penyimpanan")
        self.assertEqual(router_node(state3)["detected_tool"], "get_locations")

    def test_inventory_product_info_routing(self):
        """Product query routes to get_product_info."""
        state = create_initial_state("Berapa harga jual dan barcode produk Tinta Sublim?")
        result = router_node(state)

        self.assertEqual(result["detected_domain"], "inventory")
        self.assertEqual(result["detected_tool"], "get_product_info")

    def test_empty_query_routing(self):
        """Empty query defaults to general domain and None tool."""
        state = create_initial_state("")
        result = router_node(state)

        self.assertEqual(result["detected_domain"], "general")
        self.assertIsNone(result["detected_tool"])

    def test_llm_fallback_routing_when_ambiguous(self):
        """When query is ambiguous, fallback to LLM structured output."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = RouterDecision(
            domain="accounting",
            tool="get_account_balance",
            reasoning="User inquires about financial balance",
        )
        mock_llm.with_structured_output.return_value = mock_structured

        state = create_initial_state("Tolong bantu periksa status keuangan saya")
        result = router_node(state, llm=mock_llm)

        self.assertEqual(result["detected_domain"], "accounting")
        self.assertEqual(result["detected_tool"], "get_account_balance")
        mock_llm.with_structured_output.assert_called_once()

    def test_product_purchase_price_routes_to_product_info(self):
        """
        Regression test: Inquiries regarding product purchase price ('harga beli')
        must route to inventory domain and get_product_info tool, NOT crash or trigger vendor invoices.
        """
        queries = [
            "Berapa harga beli kertas foto?",
            "Berapa harga beli Art Paper?",
            "cek harga beli tinta sublim",
            "info harga beli produk roll banner",
        ]
        for query in queries:
            state = create_initial_state(query)
            result = router_node(state)
            self.assertEqual(result["detected_domain"], "inventory", f"Failed domain for '{query}'")
            self.assertEqual(result["detected_tool"], "get_product_info", f"Failed tool for '{query}'")
            self.assertFalse(result.get("is_multi_tool", False), f"Should not be multi-tool for '{query}'")
            self.assertEqual(len(result.get("active_tools", [])), 1, f"Failed active_tools length for '{query}'")
            self.assertEqual(result["active_tools"][0]["tool"], "get_product_info")
            self.assertEqual(result["active_tools"][0]["domain"], "inventory")

    def test_multi_intent_product_purchase_price_and_stock_level(self):
        """
        Regression test: Multi-intent queries asking for purchase price AND stock level
        must detect both get_product_info and get_stock_level in active_tools.
        """
        query = "Berapa harga beli Art Paper dan berapa stoknya?"
        state = create_initial_state(query)
        result = router_node(state)

        self.assertTrue(result.get("is_multi_tool"), "Must flag as multi-tool")
        active_tools = result.get("active_tools", [])
        self.assertEqual(len(active_tools), 2, f"Expected 2 planned tools, got {len(active_tools)}")

        tools = [t["tool"] for t in active_tools]
        self.assertEqual(tools, ["get_product_info", "get_stock_level"])
        self.assertEqual(result["detected_tool"], "get_product_info")
        self.assertEqual(result["detected_domain"], "inventory")
        self.assertEqual(result["detected_domains"], ["inventory"])

        # Check sub-queries split cleanly along Indonesian conjunction 'dan'
        self.assertIn("harga beli", active_tools[0].get("sub_query", "").lower())
        self.assertIn("stok", active_tools[1].get("sub_query", "").lower())

    def test_multi_intent_stock_and_purchase_price_inverted_order(self):
        """
        Regression test: Inverted order query (stock first, then purchase price)
        must preserve chronological tool order: get_stock_level first, then get_product_info.
        """
        query = "Cek stok kertas foto dan harga belinya"
        state = create_initial_state(query)
        result = router_node(state)

        self.assertTrue(result.get("is_multi_tool"), "Must flag as multi-tool")
        active_tools = result.get("active_tools", [])
        self.assertEqual(len(active_tools), 2)

        tools = [t["tool"] for t in active_tools]
        self.assertEqual(tools, ["get_stock_level", "get_product_info"])
        self.assertEqual(result["detected_tool"], "get_stock_level")
        self.assertEqual(result["detected_domain"], "inventory")

    def test_substring_beli_words_do_not_falsely_trigger_vendor_invoices(self):
        """
        Regression test: Indonesian words with substring 'beli' (e.g. 'belimbing')
        must NOT trigger vendor invoice routing or crash min() empty sequence.
        """
        test_cases = [
            ("stok buah belimbing", "get_stock_level", "inventory"),
            ("cek stok belimbing wuluh di gudang", "get_stock_level", "inventory"),
            ("berapa sisa stok sari buah belimbing", "get_stock_level", "inventory"),
        ]
        for query, expected_tool, expected_domain in test_cases:
            state = create_initial_state(query)
            result = router_node(state)
            self.assertEqual(result["detected_domain"], expected_domain, f"Failed domain for '{query}'")
            self.assertEqual(result["detected_tool"], expected_tool, f"Failed tool for '{query}'")
            self.assertNotEqual(result.get("detected_tool"), "fetch_vendor_invoices", f"Should not match vendor invoices for '{query}'")

    def test_purchasing_query_without_erp_tool_routes_safely(self):
        """
        Regression test: General purchasing inquiries without purchase order / vendor invoice context
        must not crash and route safely to general domain without active tools.
        """
        queries = [
            "pembelian bahan baku",
            "bagaimana alur pembelian barang?",
            "rencana pembelian perlengkapan",
        ]
        for query in queries:
            state = create_initial_state(query)
            result = router_node(state)
            self.assertIn(result["detected_domain"], ["general", "knowledge", "out_of_scope"])
            self.assertIsNone(result.get("detected_tool"))
            self.assertEqual(result.get("active_tools", []), [])

    def test_legitimate_vendor_invoice_queries_continue_to_match(self):
        """
        Regression test: Legitimate vendor invoice queries with combinations of vendor, supplier,
        pemasok, tagihan, faktur, invoice, hutang must continue routing to fetch_vendor_invoices.
        """
        cases = [
            ("Cek tagihan vendor PT Abadi", "accounting", "fetch_vendor_invoices"),
            ("Faktur supplier CV Sumber Makmur", "accounting", "fetch_vendor_invoices"),
            ("Tagihan dari pemasok PT Surya bulan ini", "accounting", "fetch_vendor_invoices"),
            ("Hutang ke vendor Toko Kertas", "accounting", "fetch_vendor_invoices"),
            ("Invoice vendor yang belum dibayar", "accounting", "fetch_vendor_invoices"),
            ("daftar vendor supplier bahan baku", "accounting", "fetch_vendor_invoices"),
            ("faktur pembelian vendor PT Surya", "accounting", "fetch_vendor_invoices"),
            ("tagihan pembelian dari CV Makmur", "accounting", "fetch_vendor_invoices"),
        ]
        for query, expected_domain, expected_tool in cases:
            state = create_initial_state(query)
            result = router_node(state)
            self.assertEqual(result["detected_domain"], expected_domain, f"Failed domain for '{query}'")
            self.assertEqual(result["detected_tool"], expected_tool, f"Failed tool for '{query}'")
            self.assertIn(expected_tool, [t["tool"] for t in result.get("active_tools", [])], f"Failed active_tools for '{query}'")

    def test_router_node_exception_safety_guard(self):
        """
        Regression test: router_node must intercept unexpected exceptions from
        classify_query_rule_based and gracefully return a safe fallback state.
        """
        from unittest.mock import patch

        state = create_initial_state("Berapa saldo kas?")
        with patch("app.graph.nodes.router.classify_query_rule_based", side_effect=ValueError("Simulated min() crash")):
            result = router_node(state)
            self.assertIsNotNone(result)
            self.assertIn(result.get("detected_domain"), ["general", "out_of_scope", "accounting"])



class TestExtractorNode(unittest.TestCase):
    """Test suite for extractor_node, registry, and date resolution."""

    def setUp(self):
        self.anchor_date = date(2026, 11, 4)

    def test_tool_schema_registry_coverage(self):
        """TOOL_SCHEMA_REGISTRY must register all 6 core tools."""
        self.assertIn("fetch_vendor_invoices", TOOL_SCHEMA_REGISTRY)
        self.assertIn("get_journal_entries", TOOL_SCHEMA_REGISTRY)
        self.assertIn("get_account_balance", TOOL_SCHEMA_REGISTRY)
        self.assertIn("get_stock_level", TOOL_SCHEMA_REGISTRY)
        self.assertIn("get_product_info", TOOL_SCHEMA_REGISTRY)
        self.assertIn("get_locations", TOOL_SCHEMA_REGISTRY)

        self.assertEqual(TOOL_SCHEMA_REGISTRY["fetch_vendor_invoices"], VendorInvoiceFilterSchema)
        self.assertEqual(TOOL_SCHEMA_REGISTRY["get_journal_entries"], JournalEntriesFilterSchema)
        self.assertEqual(TOOL_SCHEMA_REGISTRY["get_account_balance"], AccountBalanceFilterSchema)
        self.assertEqual(TOOL_SCHEMA_REGISTRY["get_stock_level"], StockLevelFilterSchema)
        self.assertEqual(TOOL_SCHEMA_REGISTRY["get_product_info"], ProductInfoFilterSchema)
        self.assertEqual(TOOL_SCHEMA_REGISTRY["get_locations"], LocationsFilterSchema)

    def test_dod_extractor_vendor_invoice_query(self):
        """
        Definition of Done: 'Cek tagihan vendor PT Sumber Makmur bulan ini'
        extracts vendor_name='PT Sumber Makmur' and resolves 'bulan ini' date range.
        """
        state = create_initial_state("Cek tagihan vendor PT Sumber Makmur bulan ini")
        state["detected_domain"] = "accounting"
        state["detected_tool"] = "fetch_vendor_invoices"

        result = extractor_node(state, ref_date=self.anchor_date)
        params = result.get("extracted_params", {})

        self.assertEqual(params.get("vendor_name"), "PT Sumber Makmur")
        self.assertEqual(params.get("date_from"), "2026-11-01")
        self.assertEqual(params.get("date_to"), "2026-11-30")

    def test_resolve_relative_dates_current_month(self):
        """'bulan ini' resolves to first and last day of anchor month."""
        date_from, date_to = resolve_relative_dates("cek transaksi bulan ini", ref_date=self.anchor_date)
        self.assertEqual(date_from, "2026-11-01")
        self.assertEqual(date_to, "2026-11-30")

    def test_resolve_relative_dates_today(self):
        """'hari ini' resolves to anchor date for both from and to."""
        date_from, date_to = resolve_relative_dates("cek transaksi hari ini", ref_date=self.anchor_date)
        self.assertEqual(date_from, "2026-11-04")
        self.assertEqual(date_to, "2026-11-04")

    def test_resolve_relative_dates_yesterday(self):
        """'kemarin' resolves to previous day."""
        date_from, date_to = resolve_relative_dates("cek invoice kemarin", ref_date=self.anchor_date)
        self.assertEqual(date_from, "2026-11-03")
        self.assertEqual(date_to, "2026-11-03")

    def test_resolve_explicit_dates(self):
        """Explicit YYYY-MM-DD dates in query are preserved."""
        query = "filter dari 2026-03-01 sampai 2026-03-31"
        date_from, date_to = resolve_relative_dates(query, ref_date=self.anchor_date)
        self.assertEqual(date_from, "2026-03-01")
        self.assertEqual(date_to, "2026-03-31")

    def test_extractor_with_mock_llm_structured_output(self):
        """Extractor properly invokes LLM structured output when provided."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = VendorInvoiceFilterSchema(
            vendor_name="PT Sumber Makmur",
            date_from="2026-11-01",
            date_to="2026-11-30",
            status="unpaid",
        )
        mock_llm.with_structured_output.return_value = mock_structured

        state = create_initial_state("Cek tagihan vendor PT Sumber Makmur bulan ini")
        state["detected_tool"] = "fetch_vendor_invoices"

        result = extractor_node(state, llm=mock_llm, ref_date=self.anchor_date)
        params = result.get("extracted_params", {})

        self.assertEqual(params.get("vendor_name"), "PT Sumber Makmur")
        self.assertEqual(params.get("date_from"), "2026-11-01")
        self.assertEqual(params.get("date_to"), "2026-11-30")
        self.assertEqual(params.get("status"), "unpaid")

    def test_extractor_resilience_missing_tool(self):
        """When detected_tool is None, extractor safely returns empty dict."""
        state = create_initial_state("Halo selamat pagi")
        state["detected_tool"] = None

        result = extractor_node(state)
        self.assertEqual(result, {"extracted_params": {}})

    def test_extractor_stock_level_with_location_and_history(self):
        """Extractor extracts location_name and resolves product code from conversation history."""
        # Simulated previous turn: User asks about AAEO-NPKG, assistant answers with SKU 3443 / AAEO-NPKG
        history = [
            HumanMessage(content="saya mau tanya stok barang AAEO-NPKG"),
            AIMessage(content="### Informasi Ketersediaan Stok Barang\nSKU 3443 AAEO-NPKG Qty 1.05 unit"),
        ]
        state = create_initial_state(
            user_query="kalau produk itu di mainstore berapa stoknya",
            messages=history + [HumanMessage(content="kalau produk itu di mainstore berapa stoknya")],
        )
        state["detected_domain"] = "inventory"
        state["detected_tool"] = "get_stock_level"

        result = extractor_node(state)
        params = result.get("extracted_params", {})

        self.assertEqual(params.get("code"), "3443")
        self.assertEqual(params.get("location_name"), "mainstore")

    def test_extractor_product_name_with_special_characters(self):
        """Extractor extracts product variant names containing '+', '/', '&', '()' special characters."""
        state = create_initial_state("coba cek stok AMPLOP BUBBLE 15x18 + 4CM LK")
        state["detected_domain"] = "inventory"
        state["detected_tool"] = "get_stock_level"

        result = extractor_node(state)
        params = result.get("extracted_params", {})
        self.assertEqual(params.get("code"), "AMPLOP BUBBLE 15x18 + 4CM LK")

    def test_extractor_product_name_strips_leading_filler_words(self):
        """Extractor cleanly strips leading filler words ('product', 'produk', 'barang', 'sku') from product codes."""
        test_queries = [
            ("coba cek stok untuk product FP BROTHER DYE ANTI UV 100ML - MAGENTA", "FP BROTHER DYE ANTI UV 100ML - MAGENTA"),
            ("cek stok untuk produk FP BROTHER DYE ANTI UV 100ML - MAGENTA", "FP BROTHER DYE ANTI UV 100ML - MAGENTA"),
            ("cek stok barang FP BROTHER DYE ANTI UV 100ML - MAGENTA", "FP BROTHER DYE ANTI UV 100ML - MAGENTA"),
            ("cek stok SKU: 100040203", "100040203"),
            ("cek stok kode 100040203", "100040203"),
        ]
        for query, expected_code in test_queries:
            state = create_initial_state(query)
            state["detected_domain"] = "inventory"
            state["detected_tool"] = "get_stock_level"
            result = extractor_node(state)
            params = result.get("extracted_params", {})
            self.assertEqual(
                params.get("code"),
                expected_code,
                f"Failed stripping prefix for query: {query}",
            )

    def test_extractor_followup_location_slot_filling_from_history(self):
        """Extractor slot-fills product code from previous turn when user only supplies location in follow-up."""
        history = [
            HumanMessage(content="coba cek stok AMPLOP BUBBLE 15x18 + 4CM LK"),
            AIMessage(content="### Informasi Produk atau Gudang Belum Spesifik\nUntuk memeriksa ketersediaan stok barang..."),
        ]
        state = create_initial_state(
            user_query="di mainstore",
            messages=history + [HumanMessage(content="di mainstore")],
        )
        state["detected_domain"] = "inventory"
        state["detected_tool"] = "get_stock_level"

        result = extractor_node(state)
        params = result.get("extracted_params", {})
        self.assertEqual(params.get("code"), "AMPLOP BUBBLE 15x18 + 4CM LK")
        self.assertEqual(params.get("location_name"), "mainstore")

    def test_extractor_get_locations_query(self):
        """Extractor extracts search keyword for get_locations."""
        state = create_initial_state("daftar gudang utama")
        state["detected_domain"] = "inventory"
        state["detected_tool"] = "get_locations"

        result = extractor_node(state)
        params = result.get("extracted_params", {})
        self.assertEqual(params.get("search"), "utama")

    def test_extractor_stock_preposition_not_extracted_as_sku(self):
        """
        R3: Indonesian prepositions ('di', 'ke', 'dari', 'pada', 'untuk', 'terkait', 'tentang')
        must not be extracted as SKU codes in stock level queries.
        """
        # Query 1: 'cek stok di gudang utama' -> location_name='utama', code is None
        state1 = create_initial_state("cek stok di gudang utama")
        state1["detected_domain"] = "inventory"
        state1["detected_tool"] = "get_stock_level"
        params1 = extractor_node(state1).get("extracted_params", {})
        self.assertEqual(params1.get("location_name"), "utama")
        self.assertIsNone(params1.get("code"))

        # Query 2: 'cek stok ke cabang' -> code is None
        state2 = create_initial_state("cek stok ke cabang")
        state2["detected_domain"] = "inventory"
        state2["detected_tool"] = "get_stock_level"
        params2 = extractor_node(state2).get("extracted_params", {})
        self.assertIsNone(params2.get("code"))

        # Query 3: 'cek stok pada rak 2' -> code is None
        state3 = create_initial_state("cek stok pada rak 2")
        state3["detected_domain"] = "inventory"
        state3["detected_tool"] = "get_stock_level"
        params3 = extractor_node(state3).get("extracted_params", {})
        self.assertIsNone(params3.get("code"))

        # Query 4: 'cek stok dari gudang' -> code is None
        state4 = create_initial_state("cek stok dari gudang")
        state4["detected_domain"] = "inventory"
        state4["detected_tool"] = "get_stock_level"
        params4 = extractor_node(state4).get("extracted_params", {})
        self.assertIsNone(params4.get("code"))

        # Query 5: 'cek stok di gudang utama untuk AAEO-NPKG' -> location_name='utama', code='AAEO-NPKG'
        state5 = create_initial_state("cek stok di gudang utama untuk AAEO-NPKG")
        state5["detected_domain"] = "inventory"
        state5["detected_tool"] = "get_stock_level"
        params5 = extractor_node(state5).get("extracted_params", {})
        self.assertEqual(params5.get("location_name"), "utama")
        self.assertEqual(params5.get("code"), "AAEO-NPKG")

        # Query 6: 'cek stok terkait AAEO-NPKG' -> code='AAEO-NPKG'
        state6 = create_initial_state("cek stok terkait AAEO-NPKG")
        state6["detected_domain"] = "inventory"
        state6["detected_tool"] = "get_stock_level"
        params6 = extractor_node(state6).get("extracted_params", {})
        self.assertEqual(params6.get("code"), "AAEO-NPKG")

        # Query 7: 'cek stok tentang AAEO-NPKG' -> code='AAEO-NPKG'
        state7 = create_initial_state("cek stok tentang AAEO-NPKG")
        state7["detected_domain"] = "inventory"
        state7["detected_tool"] = "get_stock_level"
        params7 = extractor_node(state7).get("extracted_params", {})
        self.assertEqual(params7.get("code"), "AAEO-NPKG")

        # Query 8: 'cek sisa stok untuk ART-PPR-210' -> code='ART-PPR-210' (sisa and stok in stopwords)
        state8 = create_initial_state("cek sisa stok untuk ART-PPR-210")
        state8["detected_domain"] = "inventory"
        state8["detected_tool"] = "get_stock_level"
        params8 = extractor_node(state8).get("extracted_params", {})
        self.assertEqual(params8.get("code"), "ART-PPR-210")

        # Query 9: 'cek stok di gudang untuk customer Andi' -> location_name must not be 'untuk'
        state9 = create_initial_state("cek stok di gudang untuk customer Andi")
        state9["detected_domain"] = "inventory"
        state9["detected_tool"] = "get_stock_level"
        params9 = extractor_node(state9).get("extracted_params", {})
        self.assertNotEqual(params9.get("location_name"), "untuk")

    def test_extractor_vendor_delimiters_trim_trailing_words(self):
        """
        R9: Delimiters like 'yang', 'belum', 'lunas', 'semua', 'sudah', 'dan' must stop vendor name
        extraction and prevent trailing fragment absorption.
        """
        cases = [
            ("tagihan vendor PT Sumber Makmur yang belum lunas", "PT Sumber Makmur", "unpaid"),
            ("faktur vendor CV Abadi Jaya belum lunas", "CV Abadi Jaya", "unpaid"),
            ("semua tagihan supplier Toko Kertas semua", "Toko Kertas", None),
            ("Faktur PT Sumber Makmur yang belum dibayar", "PT Sumber Makmur", "unpaid"),
            ("daftar tagihan vendor PT Cahaya semua", "PT Cahaya", None),
            ("tagihan vendor CV Abadi Jaya lunas", "CV Abadi Jaya", "paid"),
            ("faktur vendor UD Maju Bersama sudah dibayar", "UD Maju Bersama", "paid"),
            ("faktur vendor Toko Kertas dan status lunas", "Toko Kertas", "paid"),
        ]
        for query, expected_vendor, expected_status in cases:
            state = create_initial_state(query)
            state["detected_domain"] = "accounting"
            state["detected_tool"] = "fetch_vendor_invoices"
            params = extractor_node(state, ref_date=self.anchor_date).get("extracted_params", {})
            self.assertEqual(params.get("vendor_name"), expected_vendor, f"Failed vendor extraction for '{query}'")
            if expected_status:
                self.assertEqual(params.get("status"), expected_status, f"Failed status extraction for '{query}'")

    def test_multi_sku_stock_extraction(self):
        """Multi-SKU stock queries must extract all SKUs into 'codes' list."""
        query = "Cek ketersediaan stok untuk 91010600 dan 91030300"
        state = create_initial_state(query)
        state["detected_domain"] = "inventory"
        state["detected_tool"] = "get_stock_level"

        res = extractor_node(state)
        params = res.get("extracted_params", {})
        self.assertIn("codes", params)
        self.assertEqual(params["codes"], ["91010600", "91030300"])
        self.assertNotIn("code", params)

        # Fast path directly
        fp_res = try_fast_path_stock_extraction(query, query, company_id=1)
        self.assertIsNotNone(fp_res)
        self.assertEqual(fp_res.get("codes"), ["91010600", "91030300"])

        # Fallback heuristic directly
        fb_res = _fallback_extract_stock_level(query)
        self.assertEqual(fb_res.get("codes"), ["91010600", "91030300"])


class TestLangGraphPipelineChain(unittest.TestCase):
    """Integration test chaining router_node and extractor_node in LangGraph StateGraph."""

    def test_end_to_end_router_extractor_cycle(self):
        """Chains router_node -> extractor_node in StateGraph and verifies DoD output."""
        anchor_date = date(2026, 11, 4)

        def wrapped_router(state: AgentState) -> dict:
            return router_node(state)

        def wrapped_extractor(state: AgentState) -> dict:
            return extractor_node(state, ref_date=anchor_date)

        builder = StateGraph(AgentState)
        builder.add_node("router", wrapped_router)
        builder.add_node("extractor", wrapped_extractor)

        builder.add_edge(START, "router")
        builder.add_edge("router", "extractor")
        builder.add_edge("extractor", END)

        graph = builder.compile()

        # Run DoD test query
        initial_state = create_initial_state("Cek tagihan vendor PT Sumber Makmur bulan ini")
        final_state = graph.invoke(initial_state)

        # Verify router output
        self.assertEqual(final_state["detected_domain"], "accounting")
        self.assertEqual(final_state["detected_tool"], "fetch_vendor_invoices")

        # Verify extractor output (DoD)
        params = final_state.get("extracted_params", {})
        self.assertEqual(params.get("vendor_name"), "PT Sumber Makmur")
        self.assertEqual(params.get("date_from"), "2026-11-01")
        self.assertEqual(params.get("date_to"), "2026-11-30")

    def test_end_to_end_customer_inventory_chain(self):
        """
        R2 + R3 integration: Customer inventory stock check routes to inventory
        and extracts location without confusing preposition with code.
        """
        anchor_date = date(2026, 11, 4)

        builder = StateGraph(AgentState)
        builder.add_node("router", router_node)
        builder.add_node("extractor", lambda s: extractor_node(s, ref_date=anchor_date))

        builder.add_edge(START, "router")
        builder.add_edge("router", "extractor")
        builder.add_edge("extractor", END)

        graph = builder.compile()

        state = create_initial_state("Cek stok di gudang utama untuk customer Budi")
        final_state = graph.invoke(state)

        self.assertEqual(final_state["detected_domain"], "inventory")
        self.assertEqual(final_state["detected_tool"], "get_stock_level")
        params = final_state.get("extracted_params", {})
        self.assertEqual(params.get("location_name"), "utama")
        self.assertIsNone(params.get("code"))

    def test_is_retry_query_pattern_matching(self):
        """Verify strict pattern matching for conversational retry phrases."""
        from app.graph.nodes.router import is_retry_query
        positive_queries = [
            "coba lagi",
            "tolong coba lagi",
            "silakan coba lagi",
            "coba lagi ya",
            "coba lagi dong",
            "coba cek lagi",
            "cek lagi",
            "coba periksa lagi",
            "periksa lagi",
            "coba cari lagi",
            "cari lagi",
            "tolong cek lagi ya",
            "cek sekali lagi dong",
            "bisa coba cek lagi",
            "ulang",
            "ulangi",
            "coba ulang",
            "ulangi lagi",
            "sekali lagi",
            "coba sekali lagi",
            "retry",
            "tes lagi",
            "mohon ulangi lagi dong",
            "coba cek lagi?",
            "coba cek lagi!",
            "bisa dicek lagi",
            "coba dicek lagi",
            "bisa tolong dicek lagi?",
            "coba cek lagi mas",
            "coba cek lagi min",
            "coba cek lagi gan",
            "cek ulang",
            "coba cari ulang",
            "bisa tolong dicek ulang dong",
        ]
        for q in positive_queries:
            self.assertTrue(is_retry_query(q), f"Should match retry query: '{q}'")

        negative_queries = [
            "daur ulang",
            "berapa kali diulang",
            "proses daur ulang limbah",
            "coba kasih tau system design mu",
            "cek stok kertas foto A4",
            "coba cek saldo kas",
        ]
        for q in negative_queries:
            self.assertFalse(is_retry_query(q), f"Should NOT match retry query: '{q}'")

    def test_find_previous_user_query(self):
        """find_previous_user_query must locate the latest substantive HumanMessage."""
        from langchain_core.messages import AIMessage, HumanMessage
        from app.graph.nodes.router import find_previous_user_query

        # Case 1: Standard previous message
        msgs = [
            HumanMessage(content="Cek stok kertas foto A4"),
            AIMessage(content="Terjadi kendala akses data"),
            HumanMessage(content="coba lagi"),
        ]
        self.assertEqual(find_previous_user_query(msgs), "Cek stok kertas foto A4")

        # Case 2: Consecutive retries
        msgs_consecutive = [
            HumanMessage(content="Berapa saldo kas kecil saat ini?"),
            AIMessage(content="Error"),
            HumanMessage(content="coba lagi"),
            AIMessage(content="Still error"),
            HumanMessage(content="ulangi"),
        ]
        self.assertEqual(find_previous_user_query(msgs_consecutive), "Berapa saldo kas kecil saat ini?")

        # Case 3: Empty or only retry
        self.assertIsNone(find_previous_user_query([]))
        self.assertIsNone(find_previous_user_query([HumanMessage(content="coba lagi")]))

    def test_router_node_conversational_retry_flow(self):
        """Router node must replay routing on prior query when 'coba lagi' is received."""
        from langchain_core.messages import AIMessage, HumanMessage

        state = create_initial_state("coba lagi")
        state["messages"] = [
            HumanMessage(content="Cek stok kertas foto A4"),
            AIMessage(content="Terjadi kendala akses data"),
            HumanMessage(content="coba lagi"),
        ]

        result = router_node(state)
        self.assertEqual(result["detected_domain"], "inventory")
        self.assertEqual(result["detected_tool"], "get_stock_level")
        self.assertEqual(result["user_query"], "Cek stok kertas foto A4")
        self.assertIn("Mengulang permintaan:", result["retry_instruction"])

    def test_router_node_coba_cek_lagi_replays_product_inquiry(self):
        """Router node must replay routing on prior query when 'coba cek lagi' is received."""
        from langchain_core.messages import AIMessage, HumanMessage

        state = create_initial_state("coba cek lagi")
        state["messages"] = [
            HumanMessage(content="coba cek stok untuk product FP BROTHER DYE ANTI UV 100ML - MAGENTA"),
            AIMessage(content="### Produk Tidak Ditemukan\nProduk tidak ditemukan."),
            HumanMessage(content="coba cek lagi"),
        ]

        result = router_node(state)
        self.assertEqual(result["detected_domain"], "inventory")
        self.assertEqual(result["detected_tool"], "get_stock_level")
        self.assertEqual(result["user_query"], "coba cek stok untuk product FP BROTHER DYE ANTI UV 100ML - MAGENTA")
        self.assertIn("Mengulang permintaan:", result.get("retry_instruction", ""))

    def test_router_node_retry_with_no_history(self):
        """Router node with empty history and 'coba lagi' must route to general with NO_PREVIOUS_QUERY_TO_RETRY."""
        state = create_initial_state("coba lagi")
        result = router_node(state)
        self.assertEqual(result["detected_domain"], "general")
        self.assertIsNone(result.get("detected_tool"))
        self.assertEqual(result.get("error_message"), "NO_PREVIOUS_QUERY_TO_RETRY")

    def test_end_to_end_product_purchase_price_chain(self):
        """
        Pipeline integration: 'Berapa harga beli kertas foto?' routes through router -> extractor
        without raising ValueError: min() arg is an empty sequence.
        """
        anchor_date = date(2026, 11, 4)

        builder = StateGraph(AgentState)
        builder.add_node("router", router_node)
        builder.add_node("extractor", lambda s: extractor_node(s, ref_date=anchor_date))
        builder.add_edge(START, "router")
        builder.add_edge("router", "extractor")
        builder.add_edge("extractor", END)
        graph = builder.compile()

        state = create_initial_state("Berapa harga beli kertas foto?")
        final_state = graph.invoke(state)

        self.assertEqual(final_state["detected_domain"], "inventory")
        self.assertEqual(final_state["detected_tool"], "get_product_info")
        self.assertIsNotNone(final_state.get("extracted_params"))

    def test_end_to_end_multi_intent_purchase_price_and_stock_chain(self):
        """
        Pipeline integration: 'Berapa harga beli Art Paper dan berapa stoknya?' executes
        through router -> extractor with dual active_tools populated.
        """
        anchor_date = date(2026, 11, 4)

        builder = StateGraph(AgentState)
        builder.add_node("router", router_node)
        builder.add_node("extractor", lambda s: extractor_node(s, ref_date=anchor_date))
        builder.add_edge(START, "router")
        builder.add_edge("router", "extractor")
        builder.add_edge("extractor", END)
        graph = builder.compile()

        state = create_initial_state("Berapa harga beli Art Paper dan berapa stoknya?")
        final_state = graph.invoke(state)

        self.assertTrue(final_state.get("is_multi_tool"))
        self.assertEqual(len(final_state.get("active_tools", [])), 2)
        tools = [t["tool"] for t in final_state["active_tools"]]
        self.assertEqual(tools, ["get_product_info", "get_stock_level"])
        self.assertEqual(final_state["active_tools"][0]["params"].get("search"), "Art Paper")
        self.assertEqual(final_state["active_tools"][1]["params"].get("code"), "Art Paper")


class TestMultiToolExtractorNode(unittest.TestCase):
    """Test suite validating multi-tool parameter extraction, 3-tier entity resolution, and domain isolation (Milestone 2)."""

    def setUp(self):
        self.anchor_date = date(2026, 11, 4)

    def test_single_tool_legacy_state_populates_active_tools_and_matches_scalar(self):
        """When state has detected_tool and empty active_tools, extractor populates active_tools[0].params matching extracted_params."""
        state = create_initial_state("Cek tagihan vendor PT Sumber Makmur bulan ini")
        state["detected_domain"] = "accounting"
        state["detected_tool"] = "fetch_vendor_invoices"
        state["active_tools"] = []

        result = extractor_node(state, ref_date=self.anchor_date)
        extracted = result.get("extracted_params", {})
        active = result.get("active_tools", [])

        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["tool"], "fetch_vendor_invoices")
        self.assertEqual(active[0]["domain"], "accounting")
        self.assertEqual(active[0]["params"], extracted)
        self.assertEqual(extracted.get("vendor_name"), "PT Sumber Makmur")
        self.assertEqual(extracted.get("date_from"), "2026-11-01")
        self.assertEqual(extracted.get("date_to"), "2026-11-30")

    def test_single_tool_prepopulated_plan_extracted_consistently(self):
        """When active_tools has 1 plan, extractor populates its params and synchronizes extracted_params."""
        state = create_initial_state("Cek stok AAEO-NPKG")
        state["detected_domain"] = "inventory"
        state["detected_tool"] = "get_stock_level"
        state["active_tools"] = [{
            "tool": "get_stock_level",
            "domain": "inventory",
            "sub_query": "Cek stok AAEO-NPKG",
            "params": {},
            "status": "pending",
        }]

        result = extractor_node(state)
        active = result.get("active_tools", [])
        extracted = result.get("extracted_params", {})

        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["params"], extracted)
        self.assertEqual(active[0]["params"].get("code"), "AAEO-NPKG")

    def test_no_tool_detected_preserves_exact_scalar_dict(self):
        """When no tool is detected, extractor safely returns exact {'extracted_params': {}}."""
        state = create_initial_state("Halo selamat pagi")
        state["detected_tool"] = None
        state["active_tools"] = []

        result = extractor_node(state)
        self.assertEqual(result, {"extracted_params": {}})

    def test_multi_tool_cross_domain_extraction_stock_and_vendor_invoices(self):
        """Multi-tool query spanning inventory and accounting extracts distinct parameters per tool plan."""
        state = create_initial_state("Cek stok AAEO-NPKG dan tagihan vendor PT Sumber Makmur bulan ini")
        state["is_multi_tool"] = True
        state["detected_domain"] = "inventory"
        state["detected_tool"] = "get_stock_level"
        state["active_tools"] = [
            {
                "tool": "get_stock_level",
                "domain": "inventory",
                "sub_query": "Cek stok AAEO-NPKG",
                "params": {},
                "status": "pending",
            },
            {
                "tool": "fetch_vendor_invoices",
                "domain": "accounting",
                "sub_query": "tagihan vendor PT Sumber Makmur bulan ini",
                "params": {},
                "status": "pending",
            },
        ]

        result = extractor_node(state, ref_date=self.anchor_date)
        active = result.get("active_tools", [])
        extracted = result.get("extracted_params", {})

        self.assertEqual(len(active), 2)
        # Plan 0: get_stock_level
        self.assertEqual(active[0]["tool"], "get_stock_level")
        self.assertEqual(active[0]["params"].get("code"), "AAEO-NPKG")
        self.assertNotIn("vendor_name", active[0]["params"])
        self.assertNotIn("date_from", active[0]["params"])

        # Plan 1: fetch_vendor_invoices
        self.assertEqual(active[1]["tool"], "fetch_vendor_invoices")
        self.assertEqual(active[1]["params"].get("vendor_name"), "PT Sumber Makmur")
        self.assertEqual(active[1]["params"].get("date_from"), "2026-11-01")
        self.assertEqual(active[1]["params"].get("date_to"), "2026-11-30")
        self.assertNotIn("code", active[1]["params"])

        # Scalar backward compatibility matches primary plan
        self.assertEqual(extracted, active[0]["params"])

    def test_multi_tool_cross_domain_stock_and_account_balance(self):
        """Multi-tool query spanning inventory and account balance extracts distinct parameters."""
        state = create_initial_state("Cek stok AAEO-NPKG dan saldo rekening 111100 bulan ini")
        state["is_multi_tool"] = True
        state["detected_domain"] = "inventory"
        state["detected_tool"] = "get_stock_level"
        state["active_tools"] = [
            {
                "tool": "get_stock_level",
                "domain": "inventory",
                "sub_query": "Cek stok AAEO-NPKG",
                "params": {},
                "status": "pending",
            },
            {
                "tool": "get_account_balance",
                "domain": "accounting",
                "sub_query": "saldo rekening 111100 bulan ini",
                "params": {},
                "status": "pending",
            },
        ]

        result = extractor_node(state, ref_date=self.anchor_date)
        active = result.get("active_tools", [])

        self.assertEqual(len(active), 2)
        self.assertEqual(active[0]["params"].get("code"), "AAEO-NPKG")
        self.assertNotIn("account_code", active[0]["params"])

        self.assertEqual(active[1]["params"].get("account_code"), "111100")
        self.assertEqual(active[1]["params"].get("date_from"), "2026-11-01")
        self.assertEqual(active[1]["params"].get("date_to"), "2026-11-30")
        self.assertNotIn("code", active[1]["params"])

    def test_multi_tool_anaphora_resolution_same_domain_forward(self):
        """Anaphoric sub-query ('berapa stoknya?') resolves entity from preceding product info clause."""
        state = create_initial_state("Berapa harga beli Art Paper dan berapa stoknya?")
        state["is_multi_tool"] = True
        state["detected_domain"] = "inventory"
        state["detected_tool"] = "get_product_info"
        state["active_tools"] = [
            {
                "tool": "get_product_info",
                "domain": "inventory",
                "sub_query": "Berapa harga beli Art Paper",
                "params": {},
                "status": "pending",
            },
            {
                "tool": "get_stock_level",
                "domain": "inventory",
                "sub_query": "berapa stoknya",
                "params": {},
                "status": "pending",
            },
        ]

        result = extractor_node(state)
        active = result.get("active_tools", [])
        self.assertEqual(len(active), 2)
        self.assertEqual(active[0]["params"].get("search"), "Art Paper")
        self.assertEqual(active[1]["params"].get("code"), "Art Paper")

    def test_multi_tool_anaphora_resolution_same_domain_inverted(self):
        """Inverted anaphoric query ('Berapa stoknya dan berapa harga beli Art Paper?') resolves entity correctly."""
        state = create_initial_state("Berapa stoknya dan berapa harga beli Art Paper?")
        state["is_multi_tool"] = True
        state["detected_domain"] = "inventory"
        state["detected_tool"] = "get_stock_level"
        state["active_tools"] = [
            {
                "tool": "get_stock_level",
                "domain": "inventory",
                "sub_query": "Berapa stoknya",
                "params": {},
                "status": "pending",
            },
            {
                "tool": "get_product_info",
                "domain": "inventory",
                "sub_query": "berapa harga beli Art Paper",
                "params": {},
                "status": "pending",
            },
        ]

        result = extractor_node(state)
        active = result.get("active_tools", [])
        self.assertEqual(len(active), 2)
        self.assertEqual(active[0]["params"].get("code"), "Art Paper")
        self.assertEqual(active[1]["params"].get("search"), "Art Paper")

    def test_multi_tool_domain_schema_isolation_cross_domain(self):
        """Cross-domain query ensures vendor names do not leak into inventory code and product names do not leak into vendor."""
        state = create_initial_state("Cek stok produk dan faktur PT Sumber Makmur")
        state["is_multi_tool"] = True
        state["detected_domain"] = "inventory"
        state["detected_tool"] = "get_stock_level"
        state["active_tools"] = [
            {
                "tool": "get_stock_level",
                "domain": "inventory",
                "sub_query": "Cek stok produk",
                "params": {},
                "status": "pending",
            },
            {
                "tool": "fetch_vendor_invoices",
                "domain": "accounting",
                "sub_query": "faktur PT Sumber Makmur",
                "params": {},
                "status": "pending",
            },
        ]

        result = extractor_node(state, ref_date=self.anchor_date)
        active = result.get("active_tools", [])
        self.assertEqual(len(active), 2)
        # Inventory tool must NOT absorb vendor name PT Sumber Makmur as code
        self.assertIsNone(active[0]["params"].get("code"))
        # Accounting tool extracts vendor name
        self.assertEqual(active[1]["params"].get("vendor_name"), "PT Sumber Makmur")

    def test_multi_tool_journal_entries_and_locations(self):
        """Extraction across journal entries and inventory locations."""
        state = create_initial_state("Cek jurnal JE/2026/001 status posted dan daftar gudang utama")
        state["is_multi_tool"] = True
        state["detected_domain"] = "accounting"
        state["detected_tool"] = "get_journal_entries"
        state["active_tools"] = [
            {
                "tool": "get_journal_entries",
                "domain": "accounting",
                "sub_query": "Cek jurnal JE/2026/001 status posted",
                "params": {},
                "status": "pending",
            },
            {
                "tool": "get_locations",
                "domain": "inventory",
                "sub_query": "daftar gudang utama",
                "params": {},
                "status": "pending",
            },
        ]

        result = extractor_node(state, ref_date=self.anchor_date)
        active = result.get("active_tools", [])
        self.assertEqual(len(active), 2)
        self.assertEqual(active[0]["params"].get("entry_number"), "JE/2026/001")
        self.assertEqual(active[0]["params"].get("status"), "posted")
        self.assertEqual(active[1]["params"].get("search"), "utama")

    def test_multi_tool_mock_llm_structured_output_dispatch(self):
        """Extractor properly dispatches schema-specific mock LLM invocations across active tools."""
        mock_llm = MagicMock()
        mock_stock = MagicMock()
        mock_stock.invoke.return_value = StockLevelFilterSchema(code="AAEO-NPKG")
        mock_invoice = MagicMock()
        mock_invoice.invoke.return_value = VendorInvoiceFilterSchema(vendor_name="PT Sumber", date_from="2026-11-01")

        def fake_structured_output(schema):
            if schema == StockLevelFilterSchema:
                return mock_stock
            elif schema == VendorInvoiceFilterSchema:
                return mock_invoice
            return MagicMock()

        mock_llm.with_structured_output.side_effect = fake_structured_output

        state = create_initial_state("Cek stok AAEO-NPKG dan faktur vendor PT Sumber")
        state["is_multi_tool"] = True
        state["detected_tool"] = "get_stock_level"
        state["active_tools"] = [
            {"tool": "get_stock_level", "domain": "inventory", "sub_query": "Cek stok AAEO-NPKG", "params": {}},
            {"tool": "fetch_vendor_invoices", "domain": "accounting", "sub_query": "faktur vendor PT Sumber", "params": {}},
        ]

        result = extractor_node(state, llm=mock_llm, ref_date=self.anchor_date)
        active = result.get("active_tools", [])

        self.assertEqual(len(active), 2)
        self.assertEqual(active[0]["params"].get("code"), "AAEO-NPKG")
        self.assertEqual(active[1]["params"].get("vendor_name"), "PT Sumber")
        self.assertEqual(result["extracted_params"], active[0]["params"])

    def test_multi_tool_mock_llm_partial_failure_resilience(self):
        """When LLM structured output fails on one tool, it gracefully falls back to deterministic extraction."""
        mock_llm = MagicMock()
        mock_stock = MagicMock()
        mock_stock.invoke.return_value = StockLevelFilterSchema(code="AAEO-NPKG")
        mock_invoice = MagicMock()
        mock_invoice.invoke.side_effect = RuntimeError("OpenRouter 503 Provider Unavailable")

        def fake_structured_output(schema):
            if schema == StockLevelFilterSchema:
                return mock_stock
            elif schema == VendorInvoiceFilterSchema:
                return mock_invoice
            return MagicMock()

        mock_llm.with_structured_output.side_effect = fake_structured_output

        state = create_initial_state("Cek stok AAEO-NPKG dan faktur vendor PT Sumber Makmur")
        state["is_multi_tool"] = True
        state["detected_tool"] = "get_stock_level"
        state["active_tools"] = [
            {"tool": "get_stock_level", "domain": "inventory", "sub_query": "Cek stok AAEO-NPKG", "params": {}},
            {"tool": "fetch_vendor_invoices", "domain": "accounting", "sub_query": "faktur vendor PT Sumber Makmur", "params": {}},
        ]

        result = extractor_node(state, llm=mock_llm, ref_date=self.anchor_date)
        active = result.get("active_tools", [])

        self.assertEqual(len(active), 2)
        self.assertEqual(active[0]["params"].get("code"), "AAEO-NPKG")
        # Plan 1 fell back to deterministic heuristic extraction
        self.assertEqual(active[1]["params"].get("vendor_name"), "PT Sumber Makmur")

    def test_multi_tool_state_immutability(self):
        """extractor_node does not mutate original input active_tools list or inner dictionaries in-place."""
        import copy
        state = create_initial_state("Cek stok AAEO-NPKG dan faktur vendor PT Sumber")
        original_plan_0 = {"tool": "get_stock_level", "domain": "inventory", "sub_query": "Cek stok AAEO-NPKG", "params": {}}
        original_plan_1 = {"tool": "fetch_vendor_invoices", "domain": "accounting", "sub_query": "faktur vendor PT Sumber", "params": {}}
        state["active_tools"] = [original_plan_0, original_plan_1]
        state["detected_tool"] = "get_stock_level"

        baseline_plans = copy.deepcopy(state["active_tools"])
        result = extractor_node(state)

        # Original state active_tools dictionaries must remain unmodified
        self.assertEqual(state["active_tools"][0]["params"], baseline_plans[0]["params"])
        self.assertEqual(state["active_tools"][1]["params"], baseline_plans[1]["params"])
        # Returned active_tools must be a new list with populated params
        self.assertNotEqual(id(result["active_tools"]), id(state["active_tools"]))
        self.assertTrue(bool(result["active_tools"][0]["params"]))

    def test_multi_tool_staged_relaxation_retry_cycle(self):
        """Staged relaxation relaxes date filters on attempt 1 and entity filters on attempt 2."""
        state = create_initial_state("Cek tagihan vendor PT Sumber Makmur bulan ini")
        state["active_tools"] = [{
            "tool": "fetch_vendor_invoices",
            "domain": "accounting",
            "sub_query": "Cek tagihan vendor PT Sumber Makmur bulan ini",
            "params": {},
        }]

        # Retry 0: All filters present
        res0 = extractor_node(state, ref_date=self.anchor_date)
        p0 = res0["active_tools"][0]["params"]
        self.assertEqual(p0.get("vendor_name"), "PT Sumber Makmur")
        self.assertEqual(p0.get("date_from"), "2026-11-01")

        # Retry 1: Dates removed
        state["retry_count"] = 1
        res1 = extractor_node(state, ref_date=self.anchor_date)
        p1 = res1["active_tools"][0]["params"]
        self.assertEqual(p1.get("vendor_name"), "PT Sumber Makmur")
        self.assertNotIn("date_from", p1)
        self.assertNotIn("date_to", p1)

        # Retry 2: Dates and vendor_name removed
        state["retry_count"] = 2
        res2 = extractor_node(state, ref_date=self.anchor_date)
        p2 = res2["active_tools"][0]["params"]
        self.assertNotIn("date_from", p2)
        self.assertNotIn("date_to", p2)
        self.assertNotIn("vendor_name", p2)

    def test_vendor_extraction_with_hyphens_ampersands_and_word_boundaries(self):
        """Verify vendor extraction preserves hyphens, ampersands, and does not truncate on delimiter prefixes."""
        from app.graph.nodes.extractor import _fallback_extract_vendor_invoice

        res1 = _fallback_extract_vendor_invoice("cek tagihan vendor MEGA-COM")
        self.assertEqual(res1.get("vendor_name"), "MEGA-COM")

        res2 = _fallback_extract_vendor_invoice("tagihan vendor PT. A&W bulan ini")
        self.assertEqual(res2.get("vendor_name"), "PT. A&W")

        res3 = _fallback_extract_vendor_invoice("tagihan vendor PT. DIAN MAKMUR")
        self.assertEqual(res3.get("vendor_name"), "PT. DIAN MAKMUR")

        res4 = _fallback_extract_vendor_invoice("cek vendor CV. HARIAN JAYA")
        self.assertEqual(res4.get("vendor_name"), "CV. HARIAN JAYA")

    def test_product_name_not_truncated_by_delimiter_substrings(self):
        """Verify product names containing delimiter prefixes (e.g. DIGITAL, HARIAN) are not cut short."""
        from app.graph.nodes.extractor import _fallback_extract_stock_level, _fallback_extract_product_info

        res1 = _fallback_extract_stock_level("cek stok KERTAS DIGITAL PRINTING")
        self.assertEqual(res1.get("code"), "KERTAS DIGITAL PRINTING")

        res2 = _fallback_extract_stock_level("cek stok BUKU HARIAN")
        self.assertEqual(res2.get("code"), "BUKU HARIAN")

        res3 = _fallback_extract_product_info("info produk KERTAS DIGITAL PRINTING")
        self.assertEqual(res3.get("search"), "KERTAS DIGITAL PRINTING")

    def test_multi_segment_document_number_extraction(self):
        """Verify branch-prefixed journal entries and multi-segment documents are captured in full."""
        from app.graph.nodes.extractor import _fallback_extract_journal_entries, _fallback_extract_vendor_invoice
        from app.graph.nodes.router import classify_query_rule_based

        res1 = _fallback_extract_journal_entries("cek entri jurnal BDG/JE/2026/00009")
        self.assertEqual(res1.get("entry_number"), "BDG/JE/2026/00009")

        res2 = _fallback_extract_vendor_invoice("cek tagihan PC/2026/00002")
        self.assertEqual(res2.get("search"), "PC/2026/00002")

        r_res = classify_query_rule_based("cek entri jurnal BDG/JE/2026/00009")
        self.assertEqual(r_res.get("detected_tool"), "get_journal_entries")

    def test_slash_and_hyphen_date_normalization(self):
        """Verify relative date resolver parses DD/MM/YYYY, DD-MM-YYYY, and YYYY/MM/DD to ISO YYYY-MM-DD."""
        from app.graph.nodes.extractor import resolve_relative_dates

        d1, _ = resolve_relative_dates("transaksi tanggal 10/11/2026")
        self.assertEqual(d1, "2026-11-10")

        d2, _ = resolve_relative_dates("transaksi tanggal 2026/11/10")
        self.assertEqual(d2, "2026-11-10")

        d3, _ = resolve_relative_dates("transaksi tanggal 10-11-2026")
        self.assertEqual(d3, "2026-11-10")

    def test_category_and_warehouse_listing_without_search_keyword(self):
        """Verify generic category/warehouse list requests do not extract the query itself as a search filter."""
        from app.graph.nodes.extractor import _fallback_extract_product_categories, _fallback_extract_warehouses

        cat_res = _fallback_extract_product_categories("daftar kategori")
        self.assertNotIn("search", cat_res)

        cat_res2 = _fallback_extract_product_categories("list kategori produk")
        self.assertNotIn("search", cat_res2)

        wh_res = _fallback_extract_warehouses("daftar gudang")
        self.assertNotIn("search", wh_res)

        wh_res2 = _fallback_extract_warehouses("list master warehouses")
        self.assertNotIn("search", wh_res2)



if __name__ == "__main__":
    unittest.main()
