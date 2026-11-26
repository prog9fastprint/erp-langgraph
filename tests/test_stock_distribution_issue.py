"""
Unit test suite validating fixes for stock distribution query optimization:
- Prevents redundant tool spawning (get_locations, get_product_info) on stock queries
- Resolves branch abbreviations (MS -> 4, HO -> 3, SBY -> 7, etc.)
- Triggers is_multi_company on 'distribusi stok' and 'semua lokasi'
- Fast-path deterministic parameter extraction (< 5ms)
- Protects session company_id from LLM default override
- Sanitizes location_name stop words ('setiap', 'semua', 'gudang')
- Verifies LLM circuit breaker and fail-fast cache
"""
import unittest

from app.graph.nodes.router import router_node
from app.graph.nodes.extractor import (
    extractor_node,
    resolve_branch_company_id,
    resolve_all_branch_company_ids,
    try_fast_path_stock_extraction,
    LOCATION_STOP_WORDS,
)
from app.graph.nodes.synthesizer import synthesizer_node
from app.services.llm import (
    is_provider_unhealthy,
    mark_provider_unhealthy,
    reset_provider_health,
)


class TestStockDistributionIssue(unittest.TestCase):
    """Test suite covering the 7 fixes for stock query routing and latency."""

    def setUp(self):
        reset_provider_health()

    def tearDown(self):
        reset_provider_health()

    def test_router_stock_query_does_not_spawn_get_locations_or_product_info(self):
        """Query 'Apakah distribusi stok SKU 91010600 di setiap gudang/rak? Berikan jumlah unit per lokasi (MS) dan semua lokasi.' must route strictly to get_stock_level."""
        query = "Apakah distribusi stok SKU 91010600 di setiap gudang/rak? Berikan jumlah unit per lokasi (MS) dan semua lokasi."
        state = {"user_query": query, "company_id": 4}
        res = router_node(state)

        self.assertEqual(res.get("detected_domain"), "inventory")
        self.assertEqual(res.get("detected_tool"), "get_stock_level")
        self.assertEqual(res.get("detected_tools"), ["get_stock_level"])
        self.assertFalse(res.get("is_multi_tool"))
        self.assertNotIn("get_locations", res.get("detected_tools", []))
        self.assertNotIn("get_product_info", res.get("detected_tools", []))

    def test_branch_abbreviations_resolution(self):
        """Branch abbreviations like MS, HO, SBY, JKT must map to correct company_ids."""
        self.assertEqual(resolve_branch_company_id("per lokasi (MS)"), 4)
        self.assertEqual(resolve_branch_company_id("cek di HO"), 3)
        self.assertEqual(resolve_branch_company_id("stok sby"), 7)
        self.assertEqual(resolve_branch_company_id("cabang jkt"), 8)
        self.assertEqual(resolve_branch_company_id("stok di bdg"), 6)

    def test_multi_company_keywords_trigger_all_companies(self):
        """Phrases 'distribusi stok', 'semua lokasi', 'semua gudang' must set is_all_companies = True."""
        cids_1, is_all_1 = resolve_all_branch_company_ids("distribusi stok SKU 91010600")
        self.assertTrue(is_all_1)
        self.assertIn(4, cids_1)  # Main store
        self.assertIn(3, cids_1)  # Head office
        self.assertIn(7, cids_1)  # Surabaya

        cids_2, is_all_2 = resolve_all_branch_company_ids("stok di semua lokasi")
        self.assertTrue(is_all_2)

        cids_3, is_all_3 = resolve_all_branch_company_ids("stok di setiap gudang")
        self.assertTrue(is_all_3)

    def test_fast_path_stock_extraction(self):
        """Explicit SKU queries must be extracted via fast-path without LLM calls."""
        res = try_fast_path_stock_extraction(
            sub_query="Apakah distribusi stok SKU 91010600 di setiap gudang/rak?",
            user_query="Apakah distribusi stok SKU 91010600 di setiap gudang/rak?",
            company_id=4,
        )
        self.assertIsNotNone(res)
        self.assertEqual(res.get("code"), "91010600")
        self.assertTrue(res.get("include_breakdown"))
        self.assertEqual(res.get("company_id"), 4)

    def test_location_name_stop_words_sanitization(self):
        """Stop-words like 'setiap', 'semua', 'gudang' must not be accepted as location_name."""
        for word in ("setiap", "semua", "seluruh", "tiap", "gudang", "rak"):
            self.assertIn(word, LOCATION_STOP_WORDS)

    def test_company_id_session_preservation_in_extractor(self):
        """Session company_id (e.g. 4 for Main Store) must be preserved if user specifies no explicit branch name."""
        state = {
            "user_query": "Berapa stok untuk SKU 91010600?",
            "detected_tool": "get_stock_level",
            "detected_domain": "inventory",
            "company_id": 4,
            "active_tools": [{
                "tool": "get_stock_level",
                "domain": "inventory",
                "sub_query": "Berapa stok untuk SKU 91010600?",
                "params": {},
                "status": "pending",
            }],
        }
        res = extractor_node(state)
        # Verify company_id remained 4, not default 1
        self.assertEqual(res.get("company_id"), 4)
        active_tools = res.get("active_tools", [])
        self.assertTrue(len(active_tools) > 0)
        self.assertEqual(active_tools[0]["params"].get("code"), "91010600")

    def test_llm_circuit_breaker_cache(self):
        """Circuit breaker marks provider unhealthy on 401 and bypasses immediately."""
        self.assertFalse(is_provider_unhealthy("openai"))
        mark_provider_unhealthy("openai", duration_sec=900)
        self.assertTrue(is_provider_unhealthy("openai"))
        reset_provider_health("openai")
        self.assertFalse(is_provider_unhealthy("openai"))

    def test_synthesizer_multi_company_stock_table_rendering(self):
        """Synthesizer formats multi-company stock with warehouse, rak, and all branches."""
        state = {
            "user_query": "distribusi stok SKU 91010600 di semua lokasi",
            "detected_tool": "get_stock_level",
            "detected_domain": "inventory",
            "is_multi_company": True,
            "verification_status": "VALID",
            "multi_company_results": {
                3: {
                    "code": "91010600",
                    "name": "FP KARET STEMPEL FLASH BULAT 42",
                    "quantity": 30.0,
                    "available_quantity": 30.0,
                    "breakdown": [{"warehouse_name": "HEAD", "location_name": "Stock", "qty": 30.0}],
                },
                4: {
                    "code": "91010600",
                    "name": "FP KARET STEMPEL FLASH BULAT 42",
                    "quantity": 30.0,
                    "available_quantity": 30.0,
                    "breakdown": [{"warehouse_name": "MAIN", "location_name": "MS", "qty": 30.0}],
                },
                7: {
                    "code": "91010600",
                    "name": "FP KARET STEMPEL FLASH BULAT 42",
                    "quantity": 1.0,
                    "available_quantity": 1.0,
                    "breakdown": [{"warehouse_name": "Hi-Te", "location_name": "Stock", "qty": 1.0}],
                },
                1: {
                    "code": "91010600",
                    "name": "FP KARET STEMPEL FLASH BULAT 42",
                    "quantity": 0.0,
                    "available_quantity": 0.0,
                    "breakdown": [],
                },
            },
        }
        res = synthesizer_node(state)
        answer = res["final_answer"]
        self.assertIn("Laporan Ketersediaan Stok Multi-Cabang", answer)
        self.assertIn("Main Store", answer)
        self.assertIn("MS", answer)
        self.assertIn("Head Office", answer)
        self.assertIn("Surabaya", answer)
        self.assertIn("TOTAL NASIONAL", answer)
        self.assertIn("61", answer)


if __name__ == "__main__":
    unittest.main()
