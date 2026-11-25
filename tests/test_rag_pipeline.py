"""
Unit and Integration Tests for RAG Dual-Path Pipeline in FastPrint ERP AI Orchestrator.
Tests DjangoCoreClient.fetch_knowledge, router three-way classification, rag_node,
synthesizer dual citations, and StateGraph workflow integration.
"""
import unittest
from unittest.mock import AsyncMock, MagicMock

from app.graph.state import create_initial_state
from app.graph.nodes.router import classify_query_rule_based
from app.graph.nodes.rag_node import rag_node
from app.graph.nodes.synthesizer import synthesizer_node, format_deterministic_answer
from app.graph.workflow import run_workflow_sync
from app.services.django_client import DjangoCoreClient, DjangoCoreClientError


class TestDjangoClientKnowledge(unittest.IsolatedAsyncioTestCase):
    """Test suite for DjangoCoreClient knowledge search dispatch."""

    async def test_fetch_knowledge_request_dispatch(self):
        client = DjangoCoreClient()
        client.execute_tool = AsyncMock(return_value={
            "chunks": [
                {
                    "title": "SOP Retur Barang v1.2",
                    "category": "sop",
                    "content": "Barang cacat cetak dapat diretur maksimal 3x24 jam.",
                    "similarity_score": 0.88,
                }
            ]
        })

        res = await client.fetch_knowledge(
            params={"query": "SOP retur barang"},
            user_token="test_jwt",
            company_id=1,
        )

        client.execute_tool.assert_called_once_with(
            domain="knowledge",
            tool_name="search",
            params={"query": "SOP retur barang"},
            company_id=1,
            user_token="test_jwt",
        )
        self.assertIn("chunks", res)
        self.assertEqual(len(res["chunks"]), 1)


class TestRouterRAGClassification(unittest.TestCase):
    """Test suite for Router classification into live_erp, vector_rag, and hybrid."""

    def test_pure_rag_query_routing(self):
        # 1. SOP query
        res = classify_query_rule_based("Bagaimana prosedur dan SOP retur barang cetak yang cacat?")
        self.assertIsNotNone(res)
        self.assertEqual(res["query_mode"], "vector_rag")
        self.assertEqual(res["detected_domain"], "knowledge")
        self.assertIsNone(res["detected_tool"])

        # 2. Product technical specifications
        res2 = classify_query_rule_based("Apa spesifikasi kertas Art Paper 210 GSM?")
        self.assertIsNotNone(res2)
        self.assertEqual(res2["query_mode"], "vector_rag")
        self.assertEqual(res2["detected_domain"], "knowledge")

    def test_pure_live_query_routing(self):
        # 1. Stock check
        res = classify_query_rule_based("Berapa sisa stok Art Paper 210 di gudang?")
        self.assertIsNotNone(res)
        self.assertEqual(res["query_mode"], "live_erp")
        self.assertEqual(res["detected_domain"], "inventory")
        self.assertEqual(res["detected_tool"], "get_stock_level")

        # 2. Cash balance check
        res2 = classify_query_rule_based("Cek saldo kas kecil hari ini")
        self.assertIsNotNone(res2)
        self.assertEqual(res2["query_mode"], "live_erp")
        self.assertEqual(res2["detected_domain"], "accounting")
        self.assertEqual(res2["detected_tool"], "get_account_balance")

    def test_hybrid_query_routing(self):
        # Combined query: SOP retur + Cek sisa stok
        res = classify_query_rule_based("Apakah kertas Art Paper 210 bisa diretur jika cacat, dan berapa stoknya sekarang?")
        self.assertIsNotNone(res)
        self.assertEqual(res["query_mode"], "hybrid")
        self.assertEqual(res["detected_domain"], "inventory")
        self.assertEqual(res["detected_tool"], "get_stock_level")
        self.assertIsNotNone(res.get("rag_query"))


class TestRAGNode(unittest.IsolatedAsyncioTestCase):
    """Test suite for rag_node execution, similarity filtering, and error handling."""

    async def test_rag_node_similarity_filtering(self):
        mock_client = MagicMock()
        # Return 2 chunks: 1 above threshold (0.85), 1 below threshold (0.40)
        mock_client.fetch_knowledge = AsyncMock(return_value={
            "chunks": [
                {
                    "title": "SOP Retur Barang",
                    "content": "Barang cacat wajib difoto dan dilaporkan.",
                    "similarity_score": 0.85,
                },
                {
                    "title": "Buku Manual Mesin",
                    "content": "Petunjuk pembersihan roller mesin offset.",
                    "similarity_score": 0.40,
                },
            ]
        })

        state = create_initial_state(user_query="Bagaimana SOP retur?")
        state["query_mode"] = "vector_rag"
        state["rag_query"] = "Bagaimana SOP retur?"

        delta = await rag_node(state, client=mock_client)

        self.assertIn("rag_context", delta)
        self.assertEqual(len(delta["rag_context"]), 1)
        self.assertEqual(delta["rag_context"][0]["title"], "SOP Retur Barang")
        self.assertEqual(delta["rag_sources"], ["SOP Retur Barang"])
        self.assertEqual(delta["verification_status"], "VALID")

    async def test_rag_node_empty_results(self):
        mock_client = MagicMock()
        mock_client.fetch_knowledge = AsyncMock(return_value={"chunks": []})

        state = create_initial_state(user_query="SOP misterius yang tidak ada")
        state["query_mode"] = "vector_rag"
        state["rag_query"] = "SOP misterius yang tidak ada"

        delta = await rag_node(state, client=mock_client)
        self.assertEqual(delta["rag_context"], [])
        self.assertEqual(delta["verification_status"], "EMPTY")

    async def test_rag_node_error_handling(self):
        mock_client = MagicMock()
        mock_client.fetch_knowledge = AsyncMock(side_effect=DjangoCoreClientError("PGVector query timeout"))

        state = create_initial_state(user_query="Cari SOP")
        state["query_mode"] = "vector_rag"
        state["rag_query"] = "Cari SOP"

        delta = await rag_node(state, client=mock_client)
        self.assertEqual(delta["rag_context"], [])
        self.assertEqual(delta["verification_status"], "ERROR")
        self.assertIn("PGVector query timeout", delta["error_message"])


class TestSynthesizerDualCitation(unittest.TestCase):
    """Test suite for synthesizer formatting with RAG citations."""

    def test_synthesize_pure_rag_with_citations(self):
        state = create_initial_state(user_query="SOP retur")
        state["query_mode"] = "vector_rag"
        state["detected_domain"] = "knowledge"
        state["verification_status"] = "VALID"
        state["rag_context"] = [
            {
                "title": "SOP Retur Barang v1.2",
                "content": "Pengembalian dapat diajukan dalam kurun waktu 3x24 jam sejak pengiriman.",
                "similarity_score": 0.89,
            }
        ]
        state["rag_sources"] = ["SOP Retur Barang v1.2"]

        answer = format_deterministic_answer(state)
        self.assertIn("Rujukan Pengetahuan & Dokumen SOP", answer)
        self.assertIn("SOP Retur Barang v1.2", answer)
        self.assertIn("3x24 jam", answer)
        self.assertIn("Sumber Rujukan Terverifikasi", answer)

    def test_synthesize_pure_rag_empty_fallback(self):
        state = create_initial_state(user_query="SOP tidak dikenal")
        state["query_mode"] = "vector_rag"
        state["detected_domain"] = "knowledge"
        state["verification_status"] = "EMPTY"
        state["rag_context"] = []

        answer = format_deterministic_answer(state)
        self.assertIn("Informasi Pengetahuan Tidak Ditemukan", answer)
        self.assertIn("belum terdaftar", answer)

    def test_synthesize_hybrid_dual_citation(self):
        state = create_initial_state(user_query="Cek retur dan stok")
        state["query_mode"] = "hybrid"
        state["detected_domain"] = "inventory"
        state["detected_tool"] = "get_stock_level"
        state["verification_status"] = "VALID"
        state["tool_result"] = {
            "product_name": "Kertas Art Paper 210",
            "default_code": "AP-210",
            "quantity": 150.0,
            "available_quantity": 140.0,
            "uom_name": "Rim",
        }
        state["rag_context"] = [
            {
                "title": "Ketentuan Retur Kertas",
                "content": "Kertas rusak akibat kelembaban gudang dapat ditukar utuh.",
                "similarity_score": 0.91,
            }
        ]
        state["rag_sources"] = ["Ketentuan Retur Kertas"]

        res = synthesizer_node(state)
        final_answer = res["final_answer"]

        # Must contain live table data
        self.assertIn("Informasi Ketersediaan Stok Barang", final_answer)
        self.assertIn("150", final_answer)
        # Must contain RAG knowledge section
        self.assertIn("Rujukan Pengetahuan & Dokumen SOP", final_answer)
        self.assertIn("Ketentuan Retur Kertas", final_answer)


class TestWorkflowRAGIntegration(unittest.TestCase):
    """End-to-end integration test of workflow StateGraph with pure RAG and hybrid routing."""

    def test_workflow_greeting_preservation(self):
        # Greetings must continue to route to general synthesizer
        state = run_workflow_sync(user_query="Halo FastPrint!")
        self.assertEqual(state.get("detected_domain"), "general")
        self.assertIn("Halo", state.get("final_answer", ""))


if __name__ == "__main__":
    unittest.main()
