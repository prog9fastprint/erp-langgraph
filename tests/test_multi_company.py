"""
Comprehensive unit and integration tests for Multi-Company queries, branch comparisons,
and network-wide rollups in FastPrint ERP AI Orchestrator.
"""
import unittest
from unittest.mock import AsyncMock, patch

from app.graph.state import create_initial_state
from app.graph.nodes.extractor import (
    resolve_all_branch_company_ids,
    resolve_branch_company_id,
    COMPANY_NAME_MAP,
)
from app.graph.workflow import _resolve_effective_company_ids
from app.graph.nodes.tool_node import tool_node
from app.graph.nodes.verifier import verifier_node
from app.graph.nodes.synthesizer import (
    format_deterministic_answer,
    _format_multi_company_presentation,
)
from app.schemas.chat import ChatRequest, ChatResponse
from app.main import app
from fastapi.testclient import TestClient


class TestMultiCompanyExtractor(unittest.TestCase):
    """Test multi-branch extraction, network-wide keywords, and company mapping."""

    def test_resolve_multiple_explicit_branches(self):
        """Extract ordered company IDs for multiple branch mentions in a single prompt."""
        query = "Bandingkan stok tinta di Surabaya dan Bandung"
        cids, is_all = resolve_all_branch_company_ids(query)
        self.assertFalse(is_all)
        self.assertEqual(cids, [7, 6])  # Surabaya=7, Bandung=6

    def test_resolve_all_companies_keywords(self):
        """Keywords like 'semua cabang' or 'seluruh perusahaan' return full company list."""
        all_kw_queries = [
            "Cek saldo kas bank di semua cabang",
            "Laba rugi seluruh perusahaan bulan ini",
            "Total stok kertas di seluruh cabang",
            "Berapa total konsolidasi hutang vendor?",
            "Cek tagihan di semua toko",
        ]
        expected_all = sorted(list(COMPANY_NAME_MAP.keys()))
        for q in all_kw_queries:
            with self.subTest(query=q):
                cids, is_all = resolve_all_branch_company_ids(q)
                self.assertTrue(is_all)
                self.assertEqual(cids, expected_all)

    def test_resolve_single_branch_backward_compatibility(self):
        """Single branch queries return a single-item list and is_all=False."""
        cids, is_all = resolve_all_branch_company_ids("Cek stok di cabang Jakarta")
        self.assertFalse(is_all)
        self.assertEqual(cids, [8])
        self.assertEqual(resolve_branch_company_id("Cek stok di cabang Jakarta"), 8)

    def test_resolve_no_branch(self):
        """Queries without branch return empty list."""
        cids, is_all = resolve_all_branch_company_ids("Berapa saldo kas kecil?")
        self.assertFalse(is_all)
        self.assertEqual(cids, [])


class TestWorkflowCompanyResolution(unittest.TestCase):
    """Test resolution priority for single vs multi-company workflow configurations."""

    def test_explicit_payload_company_ids_take_highest_priority(self):
        """Explicit company_ids in payload overrides query text and default."""
        primary_id, cids, is_multi = _resolve_effective_company_ids(
            user_query="Cek stok di Bandung",
            company_id=1,
            company_ids=[7, 8, 10],
            company_id_explicit=False,
            checkpoint_values=None,
        )
        self.assertTrue(is_multi)
        self.assertEqual(primary_id, 7)
        self.assertEqual(cids, [7, 8, 10])

    def test_all_branches_in_query_triggers_multi_company(self):
        """Query text with 'semua cabang' resolves all companies."""
        primary_id, cids, is_multi = _resolve_effective_company_ids(
            user_query="Total stok tinta di semua cabang",
            company_id=1,
            company_ids=None,
            company_id_explicit=False,
            checkpoint_values=None,
        )
        self.assertTrue(is_multi)
        self.assertEqual(cids, sorted(list(COMPANY_NAME_MAP.keys())))


class TestMultiCompanyToolNodeExecution(unittest.IsolatedAsyncioTestCase):
    """Test bounded concurrency fan-out and partial 403 handling in tool_node."""

    async def test_multi_company_fan_out_success(self):
        """Fan-out across multiple companies collects results mapped by company_id."""
        state = create_initial_state(
            user_query="Bandingkan stok produk INK-01 di Surabaya dan Bandung",
            company_id=7,
            company_ids=[7, 6],
            is_multi_company=True,
        )
        state["detected_tool"] = "get_stock_level"
        state["detected_domain"] = "inventory"
        state["extracted_params"] = {"code": "INK-01"}

        mock_client = AsyncMock()
        mock_client.fetch_stock_level = AsyncMock(side_effect=[
            {"code": "INK-01", "product_name": "Tinta 664", "qty_on_hand": 50.0, "company_id": 7},
            {"code": "INK-01", "product_name": "Tinta 664", "qty_on_hand": 30.0, "company_id": 6},
        ])

        result = await tool_node(state=state, client=mock_client)
        self.assertTrue(result["is_multi_company"])
        self.assertEqual(result["verification_status"], "VALID")
        self.assertIn(7, result["multi_company_results"])
        self.assertIn(6, result["multi_company_results"])
        self.assertEqual(result["unauthorized_companies"], [])

    async def test_multi_company_partial_403_graceful_handling(self):
        """When one company returns 403, successful companies are preserved and 403 is flagged."""
        from app.services.django_client import DjangoCoreClientError

        state = create_initial_state(
            user_query="Bandingkan stok di Surabaya, Bandung, dan Jakarta",
            company_id=7,
            company_ids=[7, 6, 8],
            is_multi_company=True,
        )
        state["detected_tool"] = "get_stock_level"
        state["detected_domain"] = "inventory"
        state["extracted_params"] = {"code": "INK-01"}

        async def _mock_fetch(params, user_token=None, company_id=None):
            if company_id == 8:
                raise DjangoCoreClientError("User tidak memiliki akses ke company 8 (HTTP 403)")
            return {"code": "INK-01", "qty_on_hand": 20.0, "company_id": company_id}

        mock_client = AsyncMock()
        mock_client.fetch_stock_level = AsyncMock(side_effect=_mock_fetch)

        result = await tool_node(state=state, client=mock_client)
        self.assertEqual(result["verification_status"], "VALID")
        self.assertIn(7, result["multi_company_results"])
        self.assertIn(6, result["multi_company_results"])
        self.assertIn(8, result["unauthorized_companies"])
        self.assertIsNone(result["multi_company_results"][8])


class TestMultiCompanyVerifier(unittest.TestCase):
    """Test verifier evaluation for multi-company responses."""

    def test_multi_company_valid_when_any_company_has_records(self):
        """If at least one company returned data, verification is VALID."""
        state = create_initial_state("Cek stok multi cabang")
        state["is_multi_company"] = True
        state["multi_company_results"] = {
            7: {"stocks": [{"code": "INK-01", "quantity": 10}]},
            6: None,
        }
        state["unauthorized_companies"] = [6]

        res = verifier_node(state)
        self.assertEqual(res["verification_status"], "VALID")

    def test_multi_company_error_when_all_unauthorized(self):
        """If all requested companies return 403, verification is ERROR."""
        state = create_initial_state("Cek kas di cabang 8 dan 10")
        state["is_multi_company"] = True
        state["multi_company_results"] = {8: None, 10: None}
        state["unauthorized_companies"] = [8, 10]

        res = verifier_node(state)
        self.assertEqual(res["verification_status"], "ERROR")


class TestMultiCompanySynthesizer(unittest.TestCase):
    """Test synthesis of comparison tables, grand totals, and access warning callouts."""

    def test_account_balance_multi_company_table_and_grand_total(self):
        """Account balance generates a breakdown table per branch and a TOTAL GABUNGAN row."""
        state = create_initial_state("Cek saldo kas di Surabaya dan Bandung")
        state["is_multi_company"] = True
        state["detected_tool"] = "get_account_balance"
        state["verification_status"] = "VALID"
        state["multi_company_results"] = {
            7: {"totals": {"opening_balance": 1000000, "period_debit": 500000, "period_credit": 200000, "ending_balance": 1300000}},
            6: {"totals": {"opening_balance": 2000000, "period_debit": 300000, "period_credit": 100000, "ending_balance": 2200000}},
        }
        state["unauthorized_companies"] = []

        output = format_deterministic_answer(state)
        self.assertIn("Surabaya", output)
        self.assertIn("Bandung", output)
        self.assertIn("TOTAL GABUNGAN", output)
        self.assertIn("Rp 3.500.000,00", output)  # 1.300.000 + 2.200.000

    def test_stock_level_multi_company_table_and_total_nasional(self):
        """Stock level generates a breakdown table and TOTAL NASIONAL row."""
        state = create_initial_state("Stok tinta di Surabaya dan Bandung")
        state["is_multi_company"] = True
        state["detected_tool"] = "get_stock_level"
        state["verification_status"] = "VALID"
        state["multi_company_results"] = {
            7: {"code": "INK-01", "product_name": "Tinta 664", "qty_on_hand": 100, "available_quantity": 90},
            6: {"code": "INK-01", "product_name": "Tinta 664", "qty_on_hand": 50, "available_quantity": 40},
        }
        state["unauthorized_companies"] = []

        output = format_deterministic_answer(state)
        self.assertIn("TOTAL NASIONAL", output)
        self.assertIn("150", output)  # 100 + 50
        self.assertIn("130", output)  # 90 + 40

    def test_unauthorized_alert_rendered(self):
        """When unauthorized companies exist, a warning callout is appended."""
        state = create_initial_state("Stok tinta di Surabaya dan Jakarta")
        state["is_multi_company"] = True
        state["detected_tool"] = "get_stock_level"
        state["verification_status"] = "VALID"
        state["multi_company_results"] = {
            7: {"code": "INK-01", "product_name": "Tinta 664", "qty_on_hand": 100, "available_quantity": 90},
            8: None,
        }
        state["unauthorized_companies"] = [8]

        output = format_deterministic_answer(state)
        self.assertIn("Akses Dibatasi", output)
        self.assertIn("Jakarta", output)


class TestMultiCompanyApiChat(unittest.TestCase):
    """Test API endpoint payload with company_ids."""

    def test_chat_request_accepts_company_ids(self):
        """ChatRequest parses company_ids correctly."""
        req = ChatRequest(query="Cek stok", company_ids=[1, 6, 7])
        self.assertEqual(req.company_ids, [1, 6, 7])

    @patch("app.main.arun_workflow")
    def test_chat_endpoint_forwards_company_ids_and_returns_resolved(self, mock_workflow):
        """POST /api/v1/chat forwards company_ids to workflow and returns resolved_company_ids."""
        mock_workflow.return_value = {
            "final_answer": "Laporan multi perusahaan",
            "detected_domain": "accounting",
            "verification_status": "VALID",
            "company_id": 6,
            "company_ids": [6, 7],
        }

        client = TestClient(app)
        resp = client.post("/api/v1/chat", json={
            "query": "Bandingkan laba rugi Bandung dan Surabaya",
            "company_ids": [6, 7],
        })

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["resolved_company_ids"], [6, 7])
        self.assertEqual(data["resolved_company_id"], 6)
        mock_workflow.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
