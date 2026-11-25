"""
Adversarial Stress Test Suite - Challenger 2 (Empirical Verification of R2, R3, R6, R9).
Validates edge cases, boundary conditions, and concurrency under adversarial inputs:
- R2: Customer guardrail in router.py (stock/product vs billing/invoice, mixed terms)
- R3: Indonesian preposition exclusion in extractor.py (di, ke, dari, pada, untuk, terkait, tentang, SKUs, numbers)
- R9: Vendor delimiter boundaries in extractor.py (yang belum lunas, semua, belum dibayar, delimiters, stopwords)
- R6: VectorService connection pooling in vector_service.py (concurrent get_pool, concurrent search, connection reuse, error recovery)
"""
import asyncio
from datetime import date
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.graph.state import create_initial_state
from app.graph.nodes.router import router_node
from app.graph.nodes.extractor import (
    extractor_node,
    _fallback_extract_stock_level,
    _fallback_extract_vendor_invoice,
)
from app.services.vector_service import VectorService


class TestAdversarialRouterCustomerGuardrail(unittest.TestCase):
    """
    R2 Empirical Challenge: Customer Guardrail in router.py
    Verify customer inquiries for inventory stock/locations/products route to inventory,
    while inquiries targeting billing/invoices route to general.
    """

    def test_customer_stock_lookups_route_to_inventory(self):
        """Customer queries asking about stock must route to inventory.get_stock_level."""
        queries = [
            "cek stok untuk customer PT Jaya",
            "cek sisa stok barang pesanan customer Andi",
            "stok kertas untuk pelanggan Toko Abadi",
            "cek sisa barang pesanan customer Andi",
            "berapa stok barang untuk buyer luar kota",
            "cek ketersediaan stok barang klien",
            "on hand stock barang untuk pelanggan",
            "cek qty produk pesanan customer Budi",
            "sisa stok untuk pelanggan setia",
            "stok produk klien Toko Subur",
        ]
        for query in queries:
            state = create_initial_state(query)
            result = router_node(state)
            self.assertEqual(
                result["detected_domain"],
                "inventory",
                f"Query '{query}' failed to route to inventory. Got: {result.get('detected_domain')}",
            )
            self.assertEqual(
                result["detected_tool"],
                "get_stock_level",
                f"Query '{query}' failed to set tool to get_stock_level. Got: {result.get('detected_tool')}",
            )

    def test_customer_locations_lookups_route_to_inventory(self):
        """Customer queries asking about warehouse/locations must route to inventory.get_locations."""
        queries = [
            "daftar gudang tempat penyimpanan barang customer",
            "list gudang penyimpanan barang pelanggan",
            "ada gudang apa saja untuk barang customer?",
        ]
        for query in queries:
            state = create_initial_state(query)
            result = router_node(state)
            self.assertEqual(
                result["detected_domain"],
                "inventory",
                f"Query '{query}' failed to route to inventory. Got: {result.get('detected_domain')}",
            )
            self.assertEqual(
                result["detected_tool"],
                "get_locations",
                f"Query '{query}' failed to set tool to get_locations. Got: {result.get('detected_tool')}",
            )

    def test_customer_product_info_lookups_route_to_inventory(self):
        """Customer queries asking about products must route to inventory.get_product_info."""
        queries = [
            "detail barang pesanan customer",
            "info harga jual produk pesanan customer Budi",
            "barcode produk untuk pelanggan Toko Prima",
            "sku barang pesanan customer",
            "varian produk untuk customer",
            "harga barang pesanan customer Andi",
        ]
        for query in queries:
            state = create_initial_state(query)
            result = router_node(state)
            self.assertEqual(
                result["detected_domain"],
                "inventory",
                f"Query '{query}' failed to route to inventory. Got: {result.get('detected_domain')}",
            )
            self.assertEqual(
                result["detected_tool"],
                "get_product_info",
                f"Query '{query}' failed to set tool to get_product_info. Got: {result.get('detected_tool')}",
            )

    def test_customer_billing_triggers_negative_guardrail(self):
        """
        Customer queries targeting billing/invoices/receivables must NOT route to vendor invoices.
        They must route to domain='general' with detected_tool=None.
        """
        billing_queries = [
            "Cek tagihan pelanggan PT Sumber Makmur",
            "Cek faktur penjualan customer Toko Maju",
            "tagihan invoice pelanggan Toko ABC",
            "piutang customer Andi",
            "status pembayaran pelanggan Toko Kertas",
            "faktur penjualan customer",
            "pembayaran bill customer Toko Abadi",
            "cek piutang klien Surabaya",
            "tagihan customer yang belum bayar",
        ]
        for query in billing_queries:
            state = create_initial_state(query)
            result = router_node(state)
            self.assertEqual(
                result["detected_domain"],
                "general",
                f"Billing query '{query}' failed to trigger guardrail. Got domain: {result.get('detected_domain')}",
            )
            self.assertIsNone(
                result.get("detected_tool"),
                f"Billing query '{query}' must have null tool. Got: {result.get('detected_tool')}",
            )

    def test_mixed_customer_and_vendor_billing_routes_to_vendor(self):
        """When both customer and vendor are present in a billing query, vendor takes precedence."""
        query = "cek tagihan vendor PT Sumber Makmur terkait pesanan customer Budi"
        state = create_initial_state(query)
        result = router_node(state)
        self.assertEqual(result["detected_domain"], "accounting")
        self.assertEqual(result["detected_tool"], "fetch_vendor_invoices")

    def test_customer_stock_with_billing_phrase_triggers_guardrail(self):
        """
        Adversarial: When a customer stock query also includes billing/payment terms,
        the customer billing guardrail triggers and routes to general.
        """
        query = "cek stok barang pesanan customer Andi yang belum bayar"
        state = create_initial_state(query)
        result = router_node(state)
        # Contains 'bayar' which is a CUSTOMER_BILLING_TERM
        self.assertEqual(result["detected_domain"], "general")
        self.assertIsNone(result.get("detected_tool"))


class TestAdversarialStockExtractorPrepositions(unittest.TestCase):
    """
    R3 Empirical Challenge: Indonesian Preposition Exclusion in extractor.py
    Verify 'di', 'ke', 'dari', 'pada', 'untuk', 'terkait', 'tentang' are excluded
    from being parsed as SKU codes in stock extraction.
    """

    def test_prepositions_alone_do_not_become_sku(self):
        """Prepositions alone must not be extracted as SKU codes."""
        preposition_queries = [
            ("cek stok di", None, None),
            ("cek stok ke", None, None),
            ("cek stok dari", None, None),
            ("cek stok pada", None, None),
            ("cek stok untuk", None, None),
            ("cek stok terkait", None, None),
            ("cek stok tentang", None, None),
            ("cek stok di gudang utama", "utama", None),
            ("cek stok ke cabang surabaya", None, None),
            ("cek stok dari gudang pusat", None, None),
            ("cek stok untuk customer", None, None),
        ]
        for query, expected_loc, expected_code in preposition_queries:
            params = _fallback_extract_stock_level(query)
            if expected_loc:
                self.assertEqual(
                    params.get("location_name"),
                    expected_loc,
                    f"Query '{query}' failed location_name",
                )
            self.assertEqual(
                params.get("code"),
                expected_code,
                f"Query '{query}' wrongly extracted code '{params.get('code')}'",
            )

    def test_prepositions_with_valid_hyphenated_skus(self):
        """Prepositions followed by valid SKUs must extract the SKU, not the preposition."""
        test_cases = [
            ("cek stok untuk AAEO-NPKG", "AAEO-NPKG"),
            ("cek stok terkait AAEO-NPKG", "AAEO-NPKG"),
            ("cek stok dari AAEO-NPKG", "AAEO-NPKG"),
            ("cek stok tentang AAEO-NPKG", "AAEO-NPKG"),
            ("cek sisa barang untuk ART-PPR-210", "ART-PPR-210"),
            ("cek stok barang terkait GLOSS-A4-100", "GLOSS-A4-100"),
        ]
        for query, expected_code in test_cases:
            params = _fallback_extract_stock_level(query)
            self.assertEqual(
                params.get("code"),
                expected_code,
                f"Query '{query}' failed to extract SKU '{expected_code}'. Got: {params.get('code')}",
            )

    def test_prepositions_with_numeric_product_codes(self):
        """Numeric product codes (3-8 digits) must be extracted cleanly with prepositions."""
        test_cases = [
            ("cek stok untuk 12345", "12345"),
            ("cek stok terkait 987654", "987654"),
            ("cek sisa barang dari 3443", "3443"),
            ("cek stok tentang 54321", "54321"),
        ]
        for query, expected_code in test_cases:
            params = _fallback_extract_stock_level(query)
            self.assertEqual(
                params.get("code"),
                expected_code,
                f"Query '{query}' failed to extract numeric code '{expected_code}'. Got: {params.get('code')}",
            )

    def test_combined_prepositions_location_and_sku(self):
        """Complex query with both location preposition and SKU preposition."""
        query = "cek stok di gudang utama untuk AAEO-NPKG"
        params = _fallback_extract_stock_level(query)
        self.assertEqual(params.get("location_name"), "utama")
        self.assertEqual(params.get("code"), "AAEO-NPKG")

    def test_extractor_node_integration_stock_prepositions(self):
        """Integration via extractor_node with stock preposition queries."""
        state = create_initial_state("cek stok di gudang barat terkait AAEO-NPKG")
        state["detected_domain"] = "inventory"
        state["detected_tool"] = "get_stock_level"

        result = extractor_node(state)
        params = result.get("extracted_params", {})
        self.assertEqual(params.get("location_name"), "barat")
        self.assertEqual(params.get("code"), "AAEO-NPKG")


class TestAdversarialVendorInvoiceDelimiters(unittest.TestCase):
    """
    R9 Empirical Challenge: Vendor Delimiter Boundaries in extractor.py
    Verify delimiters ('yang', 'belum', 'lunas', 'semua', 'dibayar', etc.)
    prevent trailing fragment absorption and extract clean vendor names.
    """

    def setUp(self):
        self.ref_date = date(2026, 11, 4)

    def test_complex_clauses_with_delimiters(self):
        """Multi-word vendor names with complex trailing clauses."""
        cases = [
            ("tagihan supplier Toko Kertas yang belum lunas dan belum dibayar", "Toko Kertas", "unpaid"),
            ("tagihan vendor PT Sumber Makmur yang belum lunas", "PT Sumber Makmur", "unpaid"),
            ("faktur vendor CV Abadi Jaya belum lunas", "CV Abadi Jaya", "unpaid"),
            ("semua tagihan supplier Toko Kertas semua", "Toko Kertas", None),
            ("Faktur PT Sumber Makmur yang belum dibayar", "PT Sumber Makmur", "unpaid"),
            ("tagihan vendor CV Abadi Jaya lunas", "CV Abadi Jaya", "paid"),
            ("tagihan vendor Toko Buku Berkah terbayar", "Toko Buku Berkah", None),
            ("faktur vendor PT Jaya Abadi status unpaid", "PT Jaya Abadi", "unpaid"),
            ("tagihan vendor PT Cahaya sebesar 10 juta", "PT Cahaya", None),
            ("tagihan vendor PT Sentosa di bulan ini", "PT Sentosa", None),
            ("tagihan vendor PT Sinar pada tanggal 2026-11-01", "PT Sinar", None),
        ]
        for query, expected_vendor, expected_status in cases:
            params = _fallback_extract_vendor_invoice(query, ref_date=self.ref_date)
            self.assertEqual(
                params.get("vendor_name"),
                expected_vendor,
                f"Failed vendor extraction for '{query}'. Got '{params.get('vendor_name')}'",
            )
            if expected_status:
                self.assertEqual(
                    params.get("status"),
                    expected_status,
                    f"Failed status extraction for '{query}'. Got '{params.get('status')}'",
                )

    def test_stopwords_not_extracted_as_vendor_name(self):
        """Stopwords must not be erroneously extracted as vendor names."""
        degenerate_queries = [
            "tagihan vendor semua yang belum lunas",
            "tagihan vendor yang belum lunas",
            "daftar tagihan vendor",
        ]
        for query in degenerate_queries:
            params = _fallback_extract_vendor_invoice(query, ref_date=self.ref_date)
            v_name = params.get("vendor_name")
            self.assertNotIn(
                v_name,
                ["semua", "yang", "belum", "lunas"],
                f"Query '{query}' extracted stopword as vendor: '{v_name}'",
            )

    def test_vendor_legal_entity_prefixes(self):
        """Vendor names with PT., CV., UD., Toko with punctuation and dots."""
        cases = [
            ("tagihan PT. Sumber Makmur yang belum lunas", "PT. Sumber Makmur"),
            ("faktur CV. Mandiri Jaya belum dibayar", "CV. Mandiri Jaya"),
            ("invoice UD. Makmur Sejahtera lunas", "UD. Makmur Sejahtera"),
            ("tagihan Toko Kertas Jaya belum lunas", "Toko Kertas Jaya"),
        ]
        for query, expected_vendor in cases:
            params = _fallback_extract_vendor_invoice(query, ref_date=self.ref_date)
            self.assertEqual(
                params.get("vendor_name"),
                expected_vendor,
                f"Failed prefix extraction for '{query}'. Got '{params.get('vendor_name')}'",
            )


class TestAdversarialVectorServiceConnectionPool(unittest.IsolatedAsyncioTestCase):
    """
    R6 Empirical Challenge: VectorService Connection Pool Concurrency & Lifecycle
    Verify pool initialization, double-checked locking under high concurrency,
    connection reuse across concurrent search calls, and graceful closure.
    """

    def setUp(self):
        self.service = VectorService(
            db_url="postgresql://test_user:test_pass@localhost:5432/test_vector_db",
            gemini_api_key="mock_key",
            embedding_model="gemini-embedding-001",
        )

    @patch("asyncpg.create_pool", new_callable=AsyncMock)
    async def test_concurrent_get_pool_calls_create_pool_only_once(self, mock_create_pool):
        """
        Adversarial Concurrency: 50 simultaneous get_pool calls on uninitialized service.
        Must result in exactly ONE create_pool invocation (no race condition / multiple pools).
        """
        mock_pool = MagicMock()
        mock_pool._closed = False
        mock_create_pool.return_value = mock_pool

        # Fire 50 concurrent get_pool coroutines
        pools = await asyncio.gather(*[self.service.get_pool() for _ in range(50)])

        # All 50 callers must receive the exact same pool instance
        for p in pools:
            self.assertIs(p, mock_pool)

        # create_pool must be called exactly once
        mock_create_pool.assert_awaited_once()

    @patch("app.services.vector_service.VectorService.get_embedding")
    @patch("asyncpg.create_pool", new_callable=AsyncMock)
    async def test_concurrent_search_knowledge_calls_acquire_and_release(
        self, mock_create_pool, mock_get_embedding
    ):
        """
        Adversarial Concurrency: 20 simultaneous search_knowledge calls.
        All must share the single pool, acquire connections, and release them without leaks.
        """
        mock_get_embedding.return_value = [0.05] * 768

        acquire_count = 0
        release_count = 0

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [{
            "id": 1,
            "title": "SOP Retur",
            "category": "sop",
            "content": "Syarat retur 3 hari.",
            "similarity_score": 0.95,
            "metadata": "{}",
        }]

        class MockAcquireContextManager:
            async def __aenter__(self):
                nonlocal acquire_count
                acquire_count += 1
                return mock_conn

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                nonlocal release_count
                release_count += 1
                return None

        mock_pool = MagicMock()
        mock_pool._closed = False
        mock_pool.close = AsyncMock()
        mock_pool.acquire.side_effect = lambda: MockAcquireContextManager()
        mock_create_pool.return_value = mock_pool

        # Execute 20 concurrent searches
        queries = [f"Pertanyaan SOP ke-{i}" for i in range(20)]
        results = await asyncio.gather(*[
            self.service.search_knowledge(q, company_id=1, top_k=2) for q in queries
        ])

        # Verify pool was created once
        mock_create_pool.assert_awaited_once()

        # All 20 searches succeeded
        self.assertEqual(len(results), 20)
        for res in results:
            self.assertEqual(len(res), 1)
            self.assertEqual(res[0]["title"], "SOP Retur")

        # Verify every acquire had a corresponding release (zero connection leaks)
        self.assertEqual(acquire_count, 20)
        self.assertEqual(release_count, 20)

    @patch("app.services.vector_service.VectorService.get_embedding")
    @patch("asyncpg.create_pool", new_callable=AsyncMock)
    async def test_search_knowledge_releases_connection_on_fetch_error(
        self, mock_create_pool, mock_get_embedding
    ):
        """
        Adversarial Error Handling: When conn.fetch raises an unhandled database exception,
        the connection must still be cleanly released by the async context manager.
        """
        mock_get_embedding.return_value = [0.05] * 768

        acquire_count = 0
        release_count = 0

        mock_conn = AsyncMock()
        mock_conn.fetch.side_effect = RuntimeError("PostgreSQL query execution failed")

        class MockAcquireContextManager:
            async def __aenter__(self):
                nonlocal acquire_count
                acquire_count += 1
                return mock_conn

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                nonlocal release_count
                release_count += 1
                return None

        mock_pool = MagicMock()
        mock_pool._closed = False
        mock_pool.acquire.side_effect = lambda: MockAcquireContextManager()
        mock_create_pool.return_value = mock_pool

        results = await self.service.search_knowledge("SOP Retur", company_id=1)

        # Must gracefully return empty list upon error, not crash
        self.assertEqual(results, [])

        # Connection was acquired and released properly
        self.assertEqual(acquire_count, 1)
        self.assertEqual(release_count, 1)

    @patch("asyncpg.create_pool", new_callable=AsyncMock)
    async def test_close_pool_idempotent_and_reinitializable(self, mock_create_pool):
        """
        Adversarial Lifecycle: Closing an already closed pool or closing twice must not crash.
        Subsequent get_pool calls must re-initialize a new pool cleanly.
        """
        mock_pool1 = MagicMock()
        mock_pool1._closed = False
        mock_pool1.close = AsyncMock()

        mock_pool2 = MagicMock()
        mock_pool2._closed = False
        mock_pool2.close = AsyncMock()

        mock_create_pool.side_effect = [mock_pool1, mock_pool2]

        # Initial acquire
        p1 = await self.service.get_pool()
        self.assertIs(p1, mock_pool1)

        # Close pool once
        await self.service.close_pool()
        mock_pool1.close.assert_awaited_once()
        self.assertIsNone(self.service._pool)

        # Close pool second time (idempotent test)
        await self.service.close_pool()
        # mock_pool1.close should not be called again
        self.assertEqual(mock_pool1.close.await_count, 1)

        # Subsequent get_pool call re-initializes
        p2 = await self.service.get_pool()
        self.assertIs(p2, mock_pool2)
        self.assertEqual(mock_create_pool.await_count, 2)


class TestAdversarialEmpiricalBugReproductions(unittest.TestCase):
    """
    Empirical Bug Reproductions discovered during adversarial stress testing.
    Remediated and verified to pass cleanly without expectedFailure.
    """

    def test_bug_r3_sisa_stok_extracts_stok_word_as_sku(self):
        """
        BUG R3.1: Queries starting with 'cek sisa stok...' extract the word 'stok'
        as product code instead of the real SKU (e.g. 'ART-PPR-210') because
        'stok', 'stock', 'sisa' are missing from STOCK_STOPWORDS in extractor.py:157.
        """
        query = "cek sisa stok untuk ART-PPR-210"
        params = _fallback_extract_stock_level(query)
        # Expected: ART-PPR-210. Actual: 'stok'
        self.assertEqual(params.get("code"), "ART-PPR-210")

    def test_bug_r3_preposition_following_di_gudang_extracted_as_location(self):
        """
        BUG R3.2: Queries like 'cek stok di gudang untuk customer Andi' extract
        preposition 'untuk' as location_name because prepositions are missing from
        location candidate filter in extractor.py:180.
        """
        query = "cek stok di gudang untuk customer Andi"
        params = _fallback_extract_stock_level(query)
        # Expected: location_name should not be preposition 'untuk'
        self.assertNotEqual(params.get("location_name"), "untuk")

    def test_bug_r9_vendor_delimiter_omits_sudah(self):
        """
        BUG R9.1: Queries like 'faktur vendor UD Maju Bersama sudah dibayar' extract
        'UD Maju Bersama sudah' because 'sudah' is missing from VENDOR_DELIMITERS in extractor.py:107.
        """
        query = "faktur vendor UD Maju Bersama sudah dibayar"
        params = _fallback_extract_vendor_invoice(query, ref_date=date(2026, 11, 4))
        # Expected: 'UD Maju Bersama'. Actual: 'UD Maju Bersama sudah'
        self.assertEqual(params.get("vendor_name"), "UD Maju Bersama")


if __name__ == "__main__":
    unittest.main()
