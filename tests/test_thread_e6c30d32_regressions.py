"""
Regression test suite covering the 7 defects uncovered in thread e6c30d32-09bd-4770-b5aa-592295984db8:
1. Turn 1: "mana produk yang stocknya tersedikit" -> get_low_stock_alerts(exclude_zero=True)
2. Turn 3 & 4: "l210 penjualan terbanyak dimana" & "dimana >" -> get_top_selling_products multi-company + sort_by='qty' + channel breakdown
3. Turn 5: "harga l210 berapa" -> get_product_info search fallback instead of exact code="l210"
4. Turn 6 & 7: "Berapa harga stiker hologram" / "harga di medan" -> branch pricelist waterfall vs last_unit_price Rp 0,02
5. Turn 8: "Cek harga SKU 1362 di marketplace" -> get_price_comparison routing, filter_status="all", branch probing
6. Turn 9 & 10: "Cek omzet penjualan SKU 1362" / "berapa penjualan botol kalkir bulan ini" -> product-scoped sales routing + multi-tool warning + ERROR guard
7. Follow-up chips deduplication & routability validation
"""
import unittest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from app.graph.nodes.extractor import (
    _fallback_extract_low_stock_alerts,
    _fallback_extract_price_comparison,
    _fallback_extract_product_info,
    _fallback_extract_top_selling_products,
)
from app.graph.nodes.router import classify_query_rule_based, router_node
from app.graph.nodes.synthesizer import (
    _finalize_followups,
    _is_routable_followup_query,
    format_deterministic_answer,
    generate_deterministic_followups,
)
from app.graph.nodes.tool_node import _resolve_and_execute_price_comparison, tool_node
from app.graph.nodes.verifier import verifier_node
from app.graph.state import create_initial_state
from app.services.django_client import DjangoCoreClientError


class TestThreadE6C30D32Regressions(unittest.IsolatedAsyncioTestCase):
    """End-to-end regression coverage for every turn in thread e6c30d32."""

    # ------------------------------------------------------------------
    # Turn 1: "mana produk yang stocknya tersedikit"
    # ------------------------------------------------------------------
    def test_turn1_lowest_stock_routes_to_low_stock_alerts_not_product_info(self):
        res = classify_query_rule_based("mana produk yang stocknya tersedikit")
        self.assertIsNotNone(res)
        self.assertEqual(res["detected_domain"], "inventory")
        self.assertEqual(res["detected_tool"], "get_low_stock_alerts")
        self.assertNotIn("get_product_info", res.get("detected_tools", []))

        extracted = _fallback_extract_low_stock_alerts("mana produk yang stocknya tersedikit")
        self.assertTrue(extracted.get("exclude_zero"))
        self.assertNotIn("search", extracted)

        habis_extracted = _fallback_extract_low_stock_alerts("produk apa yang stok habis")
        self.assertFalse(habis_extracted.get("exclude_zero", False))

    def test_turn1_verifier_and_synthesizer_with_exclude_zero_and_out_of_stock_count(self):
        state = create_initial_state("mana produk yang stocknya tersedikit", company_id=1)
        state["detected_domain"] = "inventory"
        state["detected_tool"] = "get_low_stock_alerts"
        state["extracted_params"] = {"exclude_zero": True, "threshold": 0.0, "limit": 50}
        state["tool_result"] = {
            "alerts": [
                {
                    "product_id": 10,
                    "code": "MIN-01",
                    "name": "Head L210",
                    "qty_on_hand": 1.0,
                    "threshold": 0.0,
                    "uom": {"name": "Pcs"},
                }
            ],
            "total_found": 1,
            "exclude_zero": True,
            "out_of_stock_count": 325,
        }
        v_res = verifier_node(state)
        self.assertEqual(v_res["verification_status"], "VALID")

        state.update(v_res)
        rendered = format_deterministic_answer(state)
        self.assertIn("MIN-01", rendered)
        self.assertIn("325", rendered)

        # Even if zero positive items remain, exclude_zero with out_of_stock_count > 0 does not loop RETRY
        empty_pos_state = dict(state)
        empty_pos_state["tool_result"] = {
            "alerts": [],
            "total_found": 0,
            "exclude_zero": True,
            "out_of_stock_count": 120,
        }
        v_empty_pos = verifier_node(empty_pos_state)
        self.assertEqual(v_empty_pos["verification_status"], "VALID")

    # ------------------------------------------------------------------
    # Turn 3 & 4: "l210 penjualan terbanyak dimana" & "dimana >"
    # ------------------------------------------------------------------
    def test_turn3_and_turn4_branch_ranking_and_bare_dimana_followup(self):
        state_t3 = create_initial_state("l210 penjualan terbanyak dimana", company_id=1)
        r3 = router_node(state_t3, llm=None)
        self.assertEqual(r3["detected_domain"], "sales")
        self.assertEqual(r3["detected_tool"], "get_top_selling_products")
        self.assertTrue(r3["is_multi_company"])
        self.assertGreater(len(r3["company_ids"]), 1)

        p3 = _fallback_extract_top_selling_products("l210 penjualan terbanyak dimana")
        self.assertEqual(p3["search"], "l210")
        self.assertEqual(p3["sort_by"], "qty")

        p_rev = _fallback_extract_top_selling_products("omzet penjualan l210 terbanyak")
        self.assertEqual(p_rev["sort_by"], "revenue")

        # Turn 4: bare "dimana >" following Turn 3 stays routed as multi-company sales ranking
        msgs = [
            HumanMessage(content="l210 penjualan terbanyak dimana"),
            AIMessage(content="Hasil penjualan L210..."),
            HumanMessage(content="dimana >"),
        ]
        state_t4 = create_initial_state("dimana >", company_id=1, messages=msgs)
        r4 = router_node(state_t4, llm=None)
        self.assertEqual(r4["detected_domain"], "sales")
        self.assertEqual(r4["detected_tool"], "get_top_selling_products")
        self.assertTrue(r4["is_multi_company"])

    def test_turn3_synthesizer_multi_company_branch_ranking_and_channel_breakdown(self):
        state = create_initial_state("l210 penjualan terbanyak dimana", company_id=1)
        state["detected_domain"] = "sales"
        state["detected_tool"] = "get_top_selling_products"
        state["is_multi_company"] = True
        state["company_ids"] = [1, 2]
        state["extracted_params"] = {"search": "l210", "sort_by": "qty", "limit": 5}
        state["verification_status"] = "VALID"
        state["multi_company_results"] = {
            1: {
                "products": [{"code": "2040", "name": "HEAD L210", "total_qty": 4.0, "total_revenue": 1200000.0}],
                "channel_breakdown": [{"channel_name": "Tokopedia", "order_count": 2, "total_qty": 4.0, "total_revenue": 1200000.0}],
            },
            2: {
                "products": [{"code": "2040", "name": "HEAD L210", "total_qty": 12.0, "total_revenue": 3600000.0}],
                "channel_breakdown": [{"channel_name": "Shopee", "order_count": 5, "total_qty": 12.0, "total_revenue": 3600000.0}],
            },
        }
        rendered = format_deterministic_answer(state)
        self.assertIn("Toko Pusat", rendered)
        self.assertIn("Shopee", rendered)
        self.assertIn("Ringkasan per Channel Penjualan", rendered)
        self.assertLess(rendered.index("Toko Pusat"), rendered.index("Kantor Pusat"))

        # Single-company mode also renders channel_breakdown table
        single_state = create_initial_state("penjualan l210 di medan", company_id=2)
        single_state["detected_domain"] = "sales"
        single_state["detected_tool"] = "get_top_selling_products"
        single_state["verification_status"] = "VALID"
        single_state["tool_result"] = {
            "sort_by": "qty",
            "products": [{"rank": 1, "code": "2040", "name": "HEAD L210", "category": "Head", "total_qty": 12.0, "total_revenue": 3600000.0}],
            "channel_breakdown": [{"channel_name": "Shopee", "order_count": 5, "total_qty": 12.0, "total_revenue": 3600000.0}],
            "summary": {"total_qty": 12.0, "total_revenue": 3600000.0},
        }
        single_rendered = format_deterministic_answer(single_state)
        self.assertIn("Ringkasan per Channel Penjualan", single_rendered)
        self.assertIn("Shopee", single_rendered)

    # ------------------------------------------------------------------
    # Turn 5: "harga l210 berapa"
    # ------------------------------------------------------------------
    async def test_turn5_harga_l210_uses_search_and_tool_node_relaxes_non_exact_code(self):
        p = _fallback_extract_product_info("harga l210 berapa")
        self.assertEqual(p.get("search"), "l210")
        self.assertNotIn("code", p)

        # Even if an upstream extractor sets non-numeric single-word code="l210", tool_node sends search
        mock_client = MagicMock()
        mock_client.fetch_product_info = AsyncMock(return_value={
            "products": [{"code": "2040", "name": "HEAD L210", "effective_sales_price": 325000.0}]
        })
        state = create_initial_state("harga l210 berapa", company_id=1)
        state["detected_domain"] = "inventory"
        state["detected_tool"] = "get_product_info"
        state["extracted_params"] = {"code": "l210"}

        await tool_node(state, client=mock_client)
        called_kwargs = mock_client.fetch_product_info.call_args.kwargs
        self.assertEqual(called_kwargs["params"].get("search"), "l210")
        self.assertNotIn("code", called_kwargs["params"])

    # ------------------------------------------------------------------
    # Turns 6 & 7: Branch pricelist waterfall vs last_unit_price Rp 0,02
    # ------------------------------------------------------------------
    def test_turns6_7_product_price_uses_effective_pricelist_over_tiny_last_unit_price(self):
        state = create_initial_state("harga di medan", company_id=2)
        state["detected_domain"] = "inventory"
        state["detected_tool"] = "get_product_info"
        state["verification_status"] = "VALID"
        state["tool_result"] = {
            "company_id": 2,
            "company_name": "FP MEDAN",
            "products": [
                {
                    "id": 1362,
                    "code": "1362",
                    "name": "CUSTOM STIKER HOLOGRAM IM",
                    "last_unit_price": 0.02,
                    "default_sales_price": 0.02,
                    "effective_sales_price": 15000.0,
                    "price_source": "pricelist:Offline Medan",
                    "pricelist_prices": [
                        {"pricelist_id": 11, "pricelist_name": "Offline Medan", "price": 15000.0, "min_qty": 1},
                        {"pricelist_id": 12, "pricelist_name": "MP Medan", "price": 18000.0, "min_qty": 1},
                    ],
                    "uom": {"name": "PCS"},
                    "category": {"name": "Sticker"},
                }
            ],
        }
        rendered = format_deterministic_answer(state)
        self.assertIn("Rp 15.000", rendered)
        self.assertIn("Offline Medan", rendered)
        self.assertIn("MP Medan", rendered)
        self.assertNotIn("Rp 0,02", rendered)

    # ------------------------------------------------------------------
    # Turn 8: "Cek harga SKU 1362 di marketplace"
    # ------------------------------------------------------------------
    async def test_turn8_marketplace_price_comparison_routing_and_branch_probe(self):
        route = classify_query_rule_based("Cek harga SKU 1362 di marketplace")
        self.assertIsNotNone(route)
        self.assertEqual(route["detected_domain"], "marketplace")
        self.assertEqual(route["detected_tool"], "get_price_comparison")

        ext = _fallback_extract_price_comparison("Cek harga SKU 1362 di marketplace")
        self.assertEqual(ext["search"], "1362")
        self.assertEqual(ext["filter_status"], "all")

        # When active company_id=1 has 0 marketplace accounts, _resolve_and_execute_price_comparison
        # probes authorized branches and ignores 403 errors on unauthorized ones.
        erp_client = MagicMock()

        async def _fetch_health_side_effect(params, user_token, company_id):
            if company_id == 1:
                return {"accounts": []}
            if company_id == 2:
                return {
                    "accounts": [
                        {
                            "account_id": 77,
                            "account_name": "Fastprint Medan Official",
                            "marketplace": "Shopee",
                            "is_active": True,
                            "has_order_config": True,
                            "company_id": 2,
                        }
                    ]
                }
            raise DjangoCoreClientError("Forbidden (403): Tidak memiliki akses ke cabang ini")

        erp_client.fetch_account_health = AsyncMock(side_effect=_fetch_health_side_effect)
        erp_client.fetch_price_comparison = AsyncMock(return_value={
            "summary": {"total_evaluated": 1, "match_count": 1, "mismatch_count": 0},
            "items": [{"sku": "1362", "erp_price": 15000, "mp_price": 18000, "status": "MATCH"}],
            "total_count": 1,
        })

        res, err = await _resolve_and_execute_price_comparison(
            clean_params={"search": "1362", "filter_status": "all"},
            erp_client=erp_client,
            user_token="tok",
            target_company_id=1,
            user_query="Cek harga SKU 1362 di marketplace",
        )
        self.assertIsNone(err)
        self.assertIsNotNone(res)
        erp_client.fetch_price_comparison.assert_awaited_once()
        self.assertEqual(erp_client.fetch_price_comparison.call_args.kwargs["company_id"], 2)
        self.assertEqual(erp_client.fetch_price_comparison.call_args.kwargs["params"]["account_id"], 77)

    # ------------------------------------------------------------------
    # Turn 9 & 10: Product-scoped sales routing + multi-tool warning + followups
    # ------------------------------------------------------------------
    def test_turn9_10_product_scoped_sales_routing_and_warning_followups(self):
        r9 = classify_query_rule_based("Cek omzet penjualan SKU 1362")
        self.assertIsNotNone(r9)
        self.assertEqual(r9["detected_domain"], "sales")
        self.assertEqual(r9["detected_tool"], "get_top_selling_products")

        r10 = classify_query_rule_based("berapa penjualan botol kalkir bulan ini")
        self.assertIsNotNone(r10)
        self.assertEqual(r10["detected_domain"], "sales")
        self.assertEqual(r10["detected_tool"], "get_top_selling_products")

        # Multi-tool partial error surfaces > [!WARNING] while keeping VALID aggregate
        mt_state = create_initial_state("cek penjualan dan stok SKU 1362", company_id=1)
        mt_state["is_multi_tool"] = True
        mt_state["detected_tools"] = ["get_stock_level", "get_sales_summary"]
        mt_state["active_tools"] = [
            {"tool": "get_stock_level", "domain": "inventory", "status": "success"},
            {"tool": "get_sales_summary", "domain": "sales", "status": "error", "error": "Internal Server Error (500): simulated error"},
        ]
        mt_state["multi_tool_results"] = {
            "get_stock_level": {"total_physical": 5.0, "total_available": 5.0, "stocks": [{"code": "1362", "name": "Stiker"}]},
            "get_sales_summary": None,
        }
        mt_state["multi_verification_status"] = {"get_stock_level": "VALID", "get_sales_summary": "ERROR"}
        mt_state["verification_status"] = "VALID"
        mt_rendered = format_deterministic_answer(mt_state)
        self.assertIn("> [!WARNING]", mt_rendered)
        self.assertIn("simulated error", mt_rendered)

        # Error state blocks entity-extracted followup chips from leaking bogus entity names
        err_state = create_initial_state("mana produk yang stocknya tersedikit", company_id=1)
        err_state["detected_domain"] = "inventory"
        err_state["detected_tool"] = "get_low_stock_alerts"
        err_state["verification_status"] = "ERROR"
        err_state["error_message"] = "PRODUCT_NOT_FOUND"
        err_chips = generate_deterministic_followups(err_state)
        self.assertTrue(all("yang stocknya tersedikit" not in c["query"].lower() for c in err_chips))

        # Finalizer strips self-repeating query chip and validates routability
        self.assertFalse(_is_routable_followup_query("Silakan lanjutkan penjelasan tadi"))
        final_chips = _finalize_followups(
            [
                {"label": "Repeat", "query": "Cek harga SKU 1362 di marketplace"},
                {"label": "Cek Stok", "query": "Cek stok SKU 1362 di semua cabang"},
            ],
            user_query="Cek harga SKU 1362 di marketplace",
            state=err_state,
        )
        self.assertTrue(all(c["query"] != "Cek harga SKU 1362 di marketplace" for c in final_chips))
        self.assertTrue(any("1362" in c["query"] for c in final_chips))


if __name__ == "__main__":
    unittest.main()
