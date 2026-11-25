"""
Regression tests for thread 205d100c-413a-4e42-b1bf-fdec33097940:
- Contact detail & channel buyer follow-ups ('di wa itu contactnya siapa saja', 'detail contact ...')
- Daily top quotation lookup ('Hari ini penjualan ter banyak di quotation mana ??', 'detail quotation MS/SO/...')
- Daily product existence check with HQ auto-probe & strict date guard ('apakah ada penjualan stempel jenis apapun hari ini')
"""
from datetime import date
import unittest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from app.graph.nodes.extractor import extractor_node
from app.graph.nodes.router import router_node
from app.graph.nodes.synthesizer import format_deterministic_answer
from app.graph.nodes.tool_node import tool_node
from app.graph.nodes.verifier import verifier_node
from app.graph.state import create_initial_state
from app.graph.workflow import _resolve_effective_company_ids


class TestContactAndQuotationRouting(unittest.TestCase):
    def test_channel_contact_followup_routes_to_get_contacts_without_fake_branch(self):
        state = create_initial_state(
            user_query="di wa itu contactnya siapa saja",
            company_id=3,
            company_ids=[4, 5, 6, 7],
            is_multi_company=True,
        )
        res = router_node(state, llm=None)
        self.assertEqual(res["detected_domain"], "sales")
        self.assertEqual(res["detected_tool"], "get_contacts")
        self.assertTrue(res["is_multi_company"])

    def test_direct_detail_contact_routes_to_get_contacts(self):
        state = create_initial_state(user_query="detail contact Budi Santoso", company_id=4)
        res = router_node(state, llm=None)
        self.assertEqual(res["detected_domain"], "sales")
        self.assertEqual(res["detected_tool"], "get_contacts")

    def test_daily_top_quotation_routes_to_get_sales_orders(self):
        state = create_initial_state(
            user_query="Hari ini penjualan ter banyak di quotation mana ??",
            company_id=3,
            authorized_company_ids=[3, 4, 5, 6, 7, 8, 10],
        )
        res = router_node(state, llm=None)
        self.assertEqual(res["detected_domain"], "sales")
        self.assertEqual(res["detected_tool"], "get_sales_orders")
        self.assertTrue(res["is_multi_company"])
        self.assertIn(4, res["company_ids"])
        self.assertIn(5, res["company_ids"])

    def test_quotation_detail_ref_routes_to_get_sales_order_detail(self):
        state = create_initial_state(user_query="detail quotation MS/SO/0585308", company_id=3)
        res = router_node(state, llm=None)
        self.assertEqual(res["detected_domain"], "sales")
        self.assertEqual(res["detected_tool"], "get_sales_order_detail")

    def test_daily_stempel_sales_routes_to_top_selling_products(self):
        state = create_initial_state(user_query="apakah ada penjualan stempel jenis apapun hari ini", company_id=3)
        res = router_node(state, llm=None)
        self.assertEqual(res["detected_domain"], "sales")
        self.assertEqual(res["detected_tool"], "get_top_selling_products")


class TestContactAndQuotationExtraction(unittest.TestCase):
    def test_turn4_followup_inherits_product_dates_and_confirmed_status(self):
        today = date(2026, 11, 25)
        state = create_initial_state(
            user_query="di wa itu contactnya siapa saja",
            company_id=4,
            company_ids=[4, 5, 6, 7],
            is_multi_company=True,
            messages=[
                HumanMessage(content="penjualan xp600 di bulan ini berapa di semua cabang"),
                AIMessage(content="Penjualan terbanyak untuk `xp600` tercatat di cabang Jakarta (6 unit)..."),
                HumanMessage(content="di wa itu contactnya siapa saja"),
            ],
        )
        state["detected_domain"] = "sales"
        state["detected_tool"] = "get_contacts"

        res = extractor_node(state, llm=None, ref_date=today)
        params = res["extracted_params"]
        self.assertEqual(params["channel_name"], "WA")
        self.assertEqual(params["product_search"], "xp600")
        self.assertEqual(params["date_from"], "2026-11-01")
        self.assertEqual(params["date_to"], "2026-11-30")
        self.assertEqual(params["status"], "confirmed")
        self.assertFalse(params["is_vendor"])

    def test_turn5_quotation_ranking_uses_active_status_and_amount_desc(self):
        today = date(2026, 11, 25)
        state = create_initial_state(
            user_query="Hari ini penjualan ter banyak di quotation mana ??",
            company_id=4,
        )
        state["detected_domain"] = "sales"
        state["detected_tool"] = "get_sales_orders"

        res = extractor_node(state, llm=None, ref_date=today)
        params = res["extracted_params"]
        self.assertEqual(params["status"], "active")
        self.assertEqual(params["sort_by"], "amount_desc")
        self.assertEqual(params["date_from"], "2026-11-25")
        self.assertEqual(params["date_to"], "2026-11-25")

    def test_turn6_stempel_strips_filler_words(self):
        today = date(2026, 11, 25)
        state = create_initial_state(
            user_query="apakah ada penjualan stempel jenis apapun hari ini",
            company_id=3,
        )
        state["detected_domain"] = "sales"
        state["detected_tool"] = "get_top_selling_products"

        res = extractor_node(state, llm=None, ref_date=today)
        params = res["extracted_params"]
        self.assertEqual(params["search"], "stempel")
        self.assertEqual(params["date_from"], "2026-11-25")
        self.assertEqual(params["date_to"], "2026-11-25")

    def test_branch_prefixed_so_ref_extraction(self):
        today = date(2026, 11, 25)
        state = create_initial_state(
            user_query="detail quotation MS/SO/0585308",
            company_id=3,
        )
        state["detected_domain"] = "sales"
        state["detected_tool"] = "get_sales_order_detail"

        res = extractor_node(state, llm=None, ref_date=today)
        self.assertEqual(res["extracted_params"]["order_ref"], "MS/SO/0585308")

    def test_workflow_preserves_multi_company_on_channel_followup(self):
        primary_cid, eff_cids, is_mc = _resolve_effective_company_ids(
            user_query="di wa itu contactnya siapa saja",
            company_id=3,
            company_ids=None,
            company_id_explicit=False,
            checkpoint_values={
                "company_id": 4,
                "company_ids": [4, 5, 6, 7],
                "is_multi_company": True,
            },
        )
        self.assertEqual(primary_cid, 4)
        self.assertEqual(eff_cids, [4, 5, 6, 7])
        self.assertTrue(is_mc)


class TestHqProbeVerifierAndSynthesizer(unittest.IsolatedAsyncioTestCase):
    async def test_hq_zero_sales_probes_authorized_operating_branches(self):
        state = create_initial_state(
            user_query="apakah ada penjualan stempel jenis apapun hari ini",
            company_id=3,
            authorized_company_ids=[3, 4, 5, 6],
        )
        state["detected_domain"] = "sales"
        state["detected_tool"] = "get_top_selling_products"
        state["extracted_params"] = {
            "search": "stempel",
            "date_from": "2026-11-25",
            "date_to": "2026-11-25",
            "company_id": 3,
        }

        mock_client = MagicMock()
        mock_client.fetch_top_selling_products = AsyncMock(
            side_effect=[
                {"products": [], "summary": {"total_qty": 0, "total_revenue": 0}, "channel_breakdown": []},
                {
                    "products": [
                        {
                            "rank": 1,
                            "code": "1040307",
                            "name": "Stempel Flash",
                            "category": "Stempel",
                            "total_qty": 3,
                            "total_revenue": 450000.0,
                        }
                    ],
                    "summary": {"total_qty": 3, "total_revenue": 450000.0},
                    "channel_breakdown": [{"channel_name": "WA", "order_count": 2, "total_qty": 3, "total_revenue": 450000.0}],
                },
            ]
        )

        res = await tool_node(state, client=mock_client)
        self.assertEqual(mock_client.fetch_top_selling_products.await_count, 2)
        second_call_params = mock_client.fetch_top_selling_products.await_args_list[1].kwargs["params"]
        self.assertEqual(second_call_params["company_ids"], [4, 5, 6])
        self.assertTrue(res["is_multi_company"])
        self.assertEqual(res["tool_result"]["products"][0]["name"], "Stempel Flash")

    def test_explicit_daily_existence_empty_skips_date_dropping_retry(self):
        state = create_initial_state(
            user_query="apakah ada penjualan stempel jenis apapun hari ini",
            company_id=3,
        )
        state["detected_domain"] = "sales"
        state["detected_tool"] = "get_top_selling_products"
        state["extracted_params"] = {"search": "stempel", "date_from": "2026-11-25", "date_to": "2026-11-25"}
        state["tool_result"] = {"products": []}
        state["verification_status"] = "VALID"

        ver = verifier_node(state)
        self.assertEqual(ver["verification_status"], "EMPTY")
        self.assertEqual(state.get("retry_count", 0), 0)

    def test_synthesizer_formats_contacts_and_top_quotation(self):
        c_state = create_initial_state(user_query="di wa itu contactnya siapa saja", company_id=4)
        c_state["detected_domain"] = "sales"
        c_state["detected_tool"] = "get_contacts"
        c_state["verification_status"] = "VALID"
        c_state["tool_result"] = {
            "total_count": 1,
            "filters": {"channel_name": "WA", "product_search": "xp600", "status": "confirmed"},
            "contacts": [
                {
                    "id": 101,
                    "name": "Percetakan Maju",
                    "phone": "08123456789",
                    "city": "Surabaya",
                    "first_time_source": "WA",
                    "sales_phone": "081111222",
                    "orders_count": 2,
                    "total_spent": 9000000.0,
                    "recent_orders": [{"so_number": "MKS/SO/0027824", "company_name": "Makassar", "status": "Paid"}],
                }
            ],
        }
        c_md = format_deterministic_answer(c_state)
        self.assertIn("Percetakan Maju", c_md)
        self.assertIn("08123456789", c_md)
        self.assertIn("MKS/SO/0027824", c_md)

        q_state = create_initial_state(user_query="Hari ini penjualan ter banyak di quotation mana ??", company_id=4)
        q_state["detected_domain"] = "sales"
        q_state["detected_tool"] = "get_sales_orders"
        q_state["verification_status"] = "VALID"
        q_state["tool_result"] = {
            "total_count": 1,
            "total_amount_sum": 4350000.0,
            "status_filter": "active",
            "sort_by": "amount_desc",
            "orders": [
                {
                    "id": 585308,
                    "so_number": "MS/SO/0585308",
                    "company_name": "Main Store",
                    "quotation_date": "2026-11-25",
                    "customer_name": "Santi Indah",
                    "customer_phone": "08199887766",
                    "channel_name": "Shopee",
                    "status": "Quotation",
                    "total_amount": 4350000.0,
                }
            ],
        }
        q_md = format_deterministic_answer(q_state)
        self.assertIn("MS/SO/0585308", q_md)
        self.assertIn("Main Store", q_md)
        self.assertIn("Rp 4.350.000,00", q_md)


if __name__ == "__main__":
    unittest.main()
