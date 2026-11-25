"""
Regression test suite for product anaphora resolution, price inquiry formatting,
and verifier strict entity guards (Fix for multi-turn product follow-up issue).
"""
import unittest
from unittest.mock import AsyncMock

from langchain_core.messages import HumanMessage, AIMessage

from app.graph.nodes.extractor import (
    PRODUCT_STOPWORDS,
    _fallback_extract_product_info,
    extractor_node,
)
from app.graph.nodes.verifier import verifier_node
from app.graph.nodes.synthesizer import format_deterministic_answer
from app.graph.nodes.tool_node import _execute_single_tool


class TestProductAnaphoraPrice(unittest.IsolatedAsyncioTestCase):
    def test_product_stopwords_includes_anaphora(self):
        """Ensure anaphoric reference tokens are present in PRODUCT_STOPWORDS."""
        for token in ("tersebut", "tadi", "terkait", "dimaksud", "dia", "tersebutnya"):
            self.assertIn(token, PRODUCT_STOPWORDS)

    def test_fallback_extract_product_info_anaphora_sku_in_aimsg(self):
        """Extract exact SKU when present in previous AIMessage."""
        history = [
            HumanMessage(content="Cek stok Karet Stempel Flash Kotak 18x55"),
            AIMessage(content="Distribusi stok untuk FP KARET STEMPEL FLASH KOTAK 18x55 (SKU: 91030300) adalah 30 unit."),
        ]
        res = _fallback_extract_product_info("berapa harga produk tersebut", history)
        self.assertEqual(res.get("code"), "91030300")

    def test_fallback_extract_product_info_anaphora_human_stock_query(self):
        """Extract product name from previous HumanMessage when SKU is not in AIMessage."""
        history = [
            HumanMessage(content="Cek stok Karet Stempel Flash Kotak 18x55"),
            AIMessage(content="Stok fisik tersedia 30 unit."),
        ]
        res = _fallback_extract_product_info("berapa harga produk tersebut", history)
        self.assertEqual(res.get("search"), "Karet Stempel Flash Kotak 18x55")

    def test_extractor_node_tier3_tool_result_inheritance(self):
        """extractor_node inherits product code from prior turn's tool_result products."""
        state = {
            "user_query": "berapa harga produk tersebut",
            "detected_domain": "inventory",
            "detected_tool": "get_product_info",
            "messages": [
                HumanMessage(content="Cek stok Karet Stempel Flash Kotak 18x55"),
                AIMessage(content="Stok tersedia 30 unit."),
                HumanMessage(content="berapa harga produk tersebut"),
            ],
            "company_id": 1,
            "tool_result": {
                "products": [
                    {"code": "91030300", "name": "FP KARET STEMPEL FLASH KOTAK 18x55"}
                ]
            },
        }
        res = extractor_node(state)
        params = res.get("extracted_params") or {}
        self.assertEqual(params.get("code"), "91030300")

    def test_extractor_node_sanitizes_product_id_sku(self):
        """extractor_node converts 8-digit product_id SKU into code and drops product_id."""
        state = {
            "user_query": "berapa harga produk 91030300",
            "detected_domain": "inventory",
            "detected_tool": "get_product_info",
            "messages": [HumanMessage(content="berapa harga produk 91030300")],
            "company_id": 1,
        }
        res = extractor_node(state)
        params = res.get("extracted_params") or {}
        self.assertEqual(params.get("code"), "91030300")
        self.assertNotIn("product_id", params)

    def test_extractor_node_sanitizes_multiword_code_to_search(self):
        """extractor_node converts multi-word code into search for get_product_info."""
        state = {
            "user_query": "detail produk Karet Stempel Flash Kotak 18x55",
            "detected_domain": "inventory",
            "detected_tool": "get_product_info",
            "messages": [HumanMessage(content="detail produk Karet Stempel Flash Kotak 18x55")],
            "company_id": 1,
        }
        res = extractor_node(state)
        params = res.get("extracted_params") or {}
        self.assertEqual(params.get("search"), "Karet Stempel Flash Kotak 18x55")
        self.assertNotIn("code", params)

    async def test_tool_node_defensive_mapping(self):
        """tool_node normalizes multi-word code into search and 8-digit product_id into code."""
        mock_handler = AsyncMock(return_value={"products": [], "total_count": 0})

        # Case 1: multi-word code -> search
        await _execute_single_tool(
            tool_name="get_product_info",
            params={"code": "Karet Stempel Flash Kotak 18x55", "company_id": 1},
            erp_client=AsyncMock(),
            user_token=None,
            default_company_id=1,
            dispatch_map={"get_product_info": mock_handler},
        )
        called_params = mock_handler.call_args.kwargs["params"]
        self.assertEqual(called_params.get("search"), "Karet Stempel Flash Kotak 18x55")
        self.assertNotIn("code", called_params)

        # Case 2: SKU product_id -> code
        await _execute_single_tool(
            tool_name="get_product_info",
            params={"product_id": 91030300, "company_id": 1},
            erp_client=AsyncMock(),
            user_token=None,
            default_company_id=1,
            dispatch_map={"get_product_info": mock_handler},
        )
        called_params = mock_handler.call_args.kwargs["params"]
        self.assertEqual(called_params.get("code"), "91030300")
        self.assertNotIn("product_id", called_params)

    def test_verifier_strict_product_guard(self):
        """verifier_node terminates with EMPTY on 0 records for specific product queries without retry wipe."""
        state = {
            "detected_tool": "get_product_info",
            "extracted_params": {"search": "NonExistentProduct123", "company_id": 1},
            "tool_result": {"products": [], "total_count": 0},
            "retry_count": 0,
        }
        res = verifier_node(state)
        self.assertEqual(res.get("verification_status"), "EMPTY")
        self.assertIsNone(res.get("retry_instruction"))

    def test_synthesizer_single_product_price_card(self):
        """synthesizer formats structured pricing card for single product price query."""
        state = {
            "user_query": "berapa harga produk tersebut",
            "detected_tool": "get_product_info",
            "verification_status": "VALID",
            "tool_result": {
                "products": [
                    {
                        "name": "FP KARET STEMPEL FLASH KOTAK 18x55",
                        "code": "91030300",
                        "category": {"name": "S JD KARET"},
                        "uom": {"name": "PCS"},
                        "list_price": 323.49,
                        "default_cost": 215.00,
                    }
                ]
            },
        }
        ans = format_deterministic_answer(state)
        self.assertIn("### Informasi Harga Produk", ans)
        self.assertIn("FP KARET STEMPEL FLASH KOTAK 18x55", ans)
        self.assertIn("`91030300`", ans)
        self.assertIn("Rp 323,49", ans)
        self.assertIn("Rp 215,00", ans)
        self.assertIn("PCS", ans)

    def test_synthesizer_multi_variant_price_table(self):
        """synthesizer formats pricing comparison table for multiple variant products."""
        state = {
            "user_query": "berapa harga produk tersebut",
            "detected_tool": "get_product_info",
            "verification_status": "VALID",
            "tool_result": {
                "products": [
                    {
                        "name": "FP KARET STEMPEL FLASH KOTAK 18x55",
                        "code": "91030300",
                        "uom": "PCS",
                        "list_price": 323.49,
                        "default_cost": 215.00,
                    },
                    {
                        "name": "FP KARET STEMPEL FLASH KOTAK 22x55",
                        "code": "91030301",
                        "uom": "PCS",
                        "list_price": 380.00,
                        "default_cost": 250.00,
                    },
                ]
            },
        }
        ans = format_deterministic_answer(state)
        self.assertIn("### Informasi Perbandingan Harga Produk", ans)
        self.assertIn("FP KARET STEMPEL FLASH KOTAK 18x55", ans)
        self.assertIn("FP KARET STEMPEL FLASH KOTAK 22x55", ans)
        self.assertIn("Rp 323,49", ans)
        self.assertIn("Rp 380,00", ans)


if __name__ == "__main__":
    unittest.main()
