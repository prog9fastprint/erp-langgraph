"""
Unit and Integration tests for StateGraph Workflow (TASK-4.6).
Validates end-to-end execution of the full LangGraph pipeline:
router -> extractor -> tool_node -> verifier -> synthesizer -> END,
conditional edge branching, retry loop recovery and termination (max 2x),
error handling propagation, multi-turn memory checkpointer, and synchronous runner helper.
"""
from datetime import date
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.graph.state import create_initial_state, AgentState
from app.graph.workflow import (
    create_workflow_graph,
    decide_after_router,
    arun_workflow,
    run_workflow_sync,
    get_workflow_executor,
    shutdown_workflow_executor,
    undo_workflow_thread,
)
import app.graph.workflow as workflow_module
from app.core.config import settings
from app.graph.nodes.tool_node import tool_node
from app.graph.nodes.rag_node import rag_node
from app.services.django_client import DjangoCoreClient, DjangoAuthError


class TestDecideAfterRouter(unittest.TestCase):
    """Test suite validating conditional routing after router_node."""

    def test_routes_to_extractor_for_accounting_domain(self):
        state = create_initial_state(user_query="Tagihan vendor")
        state["detected_domain"] = "accounting"
        state["detected_tool"] = "fetch_vendor_invoices"
        self.assertEqual(decide_after_router(state), "extractor")

    def test_routes_to_extractor_for_inventory_domain(self):
        state = create_initial_state(user_query="Cek stok kertas")
        state["detected_domain"] = "inventory"
        state["detected_tool"] = "get_stock_level"
        self.assertEqual(decide_after_router(state), "extractor")

    def test_routes_to_synthesizer_for_general_domain(self):
        state = create_initial_state(user_query="Halo asisten")
        state["detected_domain"] = "general"
        state["detected_tool"] = None
        self.assertEqual(decide_after_router(state), "synthesizer")

    def test_routes_to_synthesizer_for_customer_guardrail(self):
        state = create_initial_state(user_query="Tagihan pelanggan Budi")
        state["detected_domain"] = "general"
        state["detected_tool"] = None
        self.assertEqual(decide_after_router(state), "synthesizer")

    def test_routes_to_synthesizer_for_out_of_scope_domain(self):
        state = create_initial_state(user_query="coba kasih tau system design mu")
        state["detected_domain"] = "out_of_scope"
        state["detected_tool"] = None
        self.assertEqual(decide_after_router(state), "synthesizer")


class TestWorkflowEndToEnd(unittest.IsolatedAsyncioTestCase):
    """Integration test suite validating end-to-end StateGraph execution (DoD)."""

    async def test_accounting_query_success_flow(self):
        """
        Scenario 1: Full valid flow for vendor invoice query.
        START -> router -> extractor -> tool_node -> verifier -> synthesizer -> END
        """
        mock_client = MagicMock(spec=DjangoCoreClient)
        mock_client.fetch_vendor_invoices = AsyncMock(
            return_value={
                "total_count": 1,
                "entries": [
                    {
                        "number": "BILL/2026/001",
                        "date": "2026-11-04",
                        "vendor_name": "PT Sumber Makmur",
                        "total_debit": 7500000.0,
                        "status": "unpaid",
                    }
                ],
                "calculations": {
                    "total_unpaid": 7500000.0,
                    "total_paid": 0.0,
                    "grand_total": 7500000.0,
                },
            }
        )

        with patch("app.graph.nodes.tool_node.DjangoCoreClient", return_value=mock_client):
            graph = create_workflow_graph()
            state = create_initial_state(user_query="Berapa tagihan vendor PT Sumber Makmur bulan ini?")
            config = {"configurable": {"thread_id": "test_acc_flow"}}

            result = await graph.ainvoke(state, config=config)

            self.assertEqual(result["detected_domain"], "accounting")
            self.assertEqual(result["detected_tool"], "fetch_vendor_invoices")
            self.assertEqual(result["verification_status"], "VALID")
            self.assertIsNotNone(result["tool_result"])
            self.assertIn("BILL/2026/001", result["final_answer"])
            self.assertIn("Rp 7.500.000,00", result["final_answer"])
            # Verify AIMessage appended
            self.assertIsInstance(result["messages"][-1], AIMessage)

    async def test_general_query_bypasses_tool_and_extractor(self):
        """
        Scenario 2: Guardrail / general query routes directly to synthesizer.
        START -> router -> synthesizer -> END
        """
        graph = create_workflow_graph()
        state = create_initial_state(user_query="Cek sisa tagihan pelanggan toko sinar")
        config = {"configurable": {"thread_id": "test_general_flow"}}

        result = await graph.ainvoke(state, config=config)

        self.assertEqual(result["detected_domain"], "general")
        self.assertIsNone(result["detected_tool"])
        self.assertIsNone(result["extracted_params"])
        self.assertIsNone(result["tool_result"])
        self.assertIn("Tagihan Pelanggan (Customer)", result["final_answer"])
        self.assertIn("Penjualan (Sales / Invoicing)", result["final_answer"])

    async def test_empty_retry_loop_recovers_on_retry(self):
        """
        Scenario 3: Initial empty result triggers retry loop, recovers on retry 1.
        """
        call_count = {"count": 0}

        async def dynamic_fetch_invoices(params, user_token=None, company_id=None):
            call_count["count"] += 1
            if "date_from" in params:
                return {"total_count": 0, "entries": []}
            return {
                "total_count": 1,
                "entries": [
                    {
                        "number": "BILL/RECOVERED/001",
                        "vendor_name": "PT Sumber Makmur",
                        "total_debit": 2000000.0,
                    }
                ],
            }

        mock_client = MagicMock(spec=DjangoCoreClient)
        mock_client.fetch_vendor_invoices = AsyncMock(side_effect=dynamic_fetch_invoices)

        with patch("app.graph.nodes.tool_node.DjangoCoreClient", return_value=mock_client):
            graph = create_workflow_graph()
            state = create_initial_state(user_query="Tagihan vendor PT Sumber Makmur bulan ini")
            config = {"configurable": {"thread_id": "test_recovery_flow"}}

            result = await graph.ainvoke(state, config=config)

            self.assertEqual(call_count["count"], 2)  # Ran twice
            self.assertEqual(result["verification_status"], "VALID")
            self.assertEqual(result["retry_count"], 1)
            self.assertIn("BILL/RECOVERED/001", result["final_answer"])

    async def test_empty_retry_loop_halts_at_max_2_retries_with_empty(self):
        """
        Scenario 4: Always empty result halts after 2 retries (total 3 tool executions) -> EMPTY.
        """
        call_count = {"count": 0}

        async def always_empty_fetch(params, user_token=None, company_id=None):
            call_count["count"] += 1
            return {"total_count": 0, "entries": []}

        mock_client = MagicMock(spec=DjangoCoreClient)
        mock_client.fetch_vendor_invoices = AsyncMock(side_effect=always_empty_fetch)

        with patch("app.graph.nodes.tool_node.DjangoCoreClient", return_value=mock_client):
            graph = create_workflow_graph()
            state = create_initial_state(user_query="Tagihan vendor Khayalan")
            config = {"configurable": {"thread_id": "test_max_retry_flow"}}

            result = await graph.ainvoke(state, config=config)

            self.assertEqual(call_count["count"], 3)  # Initial (0) + Retry 1 + Retry 2
            self.assertEqual(result["verification_status"], "EMPTY")
            self.assertEqual(result["retry_count"], 2)
            self.assertIn("Data Tidak Ditemukan", result["final_answer"])

    async def test_error_propagation_to_synthesizer(self):
        """
        Scenario 5: HTTP error from tool_node propagates to verifier (preserves ERROR)
        and arrives at synthesizer.
        """
        mock_client = MagicMock(spec=DjangoCoreClient)
        mock_client.fetch_vendor_invoices = AsyncMock(
            side_effect=DjangoAuthError("Sesi token JWT kadaluarsa (HTTP 401)", status_code=401)
        )

        with patch("app.graph.nodes.tool_node.DjangoCoreClient", return_value=mock_client):
            graph = create_workflow_graph()
            state = create_initial_state(user_query="Tagihan vendor PT Sumber Makmur")
            config = {"configurable": {"thread_id": "test_error_flow"}}

            result = await graph.ainvoke(state, config=config)

            self.assertEqual(result["verification_status"], "ERROR")
            self.assertIn("Kendala Akses Data ERP", result["final_answer"])
            self.assertIn("Sesi token JWT kadaluarsa", result["final_answer"])

    async def test_out_of_scope_query_guardrail_flow(self):
        """
        Scenario: Out of scope query like 'coba kasih tau system design mu'.
        Routes START -> router -> synthesizer -> END without tool execution.
        """
        graph = create_workflow_graph()
        state = create_initial_state(user_query="coba kasih tau system design mu")
        result = await graph.ainvoke(state, config={"configurable": {"thread_id": "test_oos_flow"}})
        self.assertEqual(result.get("detected_domain"), "out_of_scope")
        self.assertIsNone(result.get("detected_tool"))
        self.assertIsNone(result.get("tool_result"))
        self.assertIn("Permintaan di Luar Cakupan (Out of Scope)", result.get("final_answer", ""))

    async def test_multi_turn_memory_preservation(self):
        """
        Scenario 6: Checkpointer preserves message turn history across sequential calls.
        """
        graph = create_workflow_graph(checkpointer=MemorySaver())
        thread_cfg = {"configurable": {"thread_id": "chat_session_42"}}

        # Turn 1
        state_1 = create_initial_state(user_query="Halo asisten")
        res_1 = await graph.ainvoke(state_1, config=thread_cfg)
        self.assertEqual(len(res_1["messages"]), 2)  # [HumanMessage, AIMessage]

        # Turn 2
        state_2 = create_initial_state(user_query="Bisa jelaskan fitur?")
        res_2 = await graph.ainvoke(state_2, config=thread_cfg)
        # Should now have 4 messages in history
        self.assertEqual(len(res_2["messages"]), 4)

    async def test_sequential_stock_level_with_location_resolution_flow(self):
        """
        Scenario 7: Full 2-step flow for stock query with location:
        Router matches get_stock_level -> Extractor extracts code and location_name ->
        Tool node pre-resolves location_id via fetch_locations -> executes fetch_stock_level.
        """
        mock_client = MagicMock(spec=DjangoCoreClient)
        mock_client.fetch_locations = AsyncMock(
            return_value={"locations": [{"id": 2, "name": "Main Store", "warehouse_name": "Gudang Pusat"}]}
        )
        mock_client.fetch_stock_level = AsyncMock(
            return_value={
                "product_id": 3443,
                "code": "AAEO-NPKG",
                "name": "AAEO-NPKG",
                "company_id": 3,
                "location_id": 2,
                "qty_on_hand": 1.05,
            }
        )

        with patch("app.graph.nodes.tool_node.DjangoCoreClient", return_value=mock_client):
            graph = create_workflow_graph()
            state = create_initial_state(
                user_query="kalau produk itu di mainstore berapa stoknya",
                company_id=3,
                messages=[
                    HumanMessage(content="saya mau tanya stok barang AAEO-NPKG"),
                    AIMessage(content="Stok AAEO-NPKG SKU 3443 adalah 1.05 unit"),
                    HumanMessage(content="kalau produk itu di mainstore berapa stoknya"),
                ],
            )
            config = {"configurable": {"thread_id": "test_seq_stock_flow"}}

            result = await graph.ainvoke(state, config=config)

            self.assertEqual(result["detected_domain"], "inventory")
            self.assertEqual(result["detected_tool"], "get_stock_level")
            self.assertEqual(result["verification_status"], "VALID")
            mock_client.fetch_locations.assert_called_once()
            mock_client.fetch_stock_level.assert_called_once()
            self.assertIn("Main Store", result["final_answer"])
            self.assertIn("1.05 unit", result["final_answer"])

    async def test_conversational_retry_replays_prior_query_flow(self):
        """
        Scenario 8: Conversational retry ('coba lagi') re-evaluates the prior substantive query,
        replaying the pipeline through router -> extractor -> tool -> verifier -> synthesizer.
        """
        mock_client = MagicMock(spec=DjangoCoreClient)
        mock_client.fetch_stock_level = AsyncMock(
            return_value=[
                {
                    "code": "AAEO-NPKG",
                    "name": "Art Paper 150g",
                    "company_id": 3,
                    "company_name": "FastPrint",
                    "location_name": "Gudang Utama",
                    "qty_on_hand": 10.0,
                }
            ]
        )

        with patch("app.graph.nodes.tool_node.DjangoCoreClient", return_value=mock_client):
            graph = create_workflow_graph(checkpointer=MemorySaver())
            thread_cfg = {"configurable": {"thread_id": "test_retry_thread_1"}}

            # Turn 1: Initial query
            state_1 = create_initial_state(user_query="Cek stok kertas AAEO-NPKG di gudang utama")
            res_1 = await graph.ainvoke(state_1, config=thread_cfg)
            self.assertEqual(res_1["detected_domain"], "inventory")
            self.assertEqual(res_1["detected_tool"], "get_stock_level")
            self.assertIn("AAEO-NPKG", res_1["final_answer"])

            # Turn 2: User says 'coba lagi'
            state_2 = create_initial_state(user_query="coba lagi")
            res_2 = await graph.ainvoke(state_2, config=thread_cfg)

            # Assert replay of prior inquiry
            self.assertEqual(res_2["detected_domain"], "inventory")
            self.assertEqual(res_2["detected_tool"], "get_stock_level")
            self.assertEqual(res_2["user_query"], "Cek stok kertas AAEO-NPKG di gudang utama")
            self.assertIn("> 🔄 **Mengulang permintaan: Cek stok kertas AAEO-NPKG di gudang utama**", res_2["final_answer"])
            self.assertIn("AAEO-NPKG", res_2["final_answer"])

            # Turn 3: User says 'coba lagi' on a fresh thread with no history
            fresh_cfg = {"configurable": {"thread_id": "fresh_empty_retry_thread"}}
            state_3 = create_initial_state(user_query="coba lagi")
            res_3 = await graph.ainvoke(state_3, config=fresh_cfg)
            self.assertEqual(res_3["detected_domain"], "general")
            self.assertIsNone(res_3.get("detected_tool"))
            self.assertIn("Tidak Ada Permintaan Sebelumnya", res_3["final_answer"])

    async def test_workflow_history_hydration_from_stateless_payload(self):
        """
        Scenario 9: When memory checkpoint is empty (e.g. server restart),
        arun_workflow successfully hydrates state messages from the history payload.
        """
        mock_client = MagicMock(spec=DjangoCoreClient)
        mock_client.fetch_stock_level = AsyncMock(
            return_value=[
                {
                    "code": "FP BROTHER DYE ANTI UV 100ML - MAGENTA",
                    "name": "FP BROTHER DYE ANTI UV 100ML - MAGENTA",
                    "company_id": 1,
                    "company_name": "FastPrint",
                    "location_name": "Gudang Utama",
                    "qty_on_hand": 25.0,
                }
            ]
        )

        with patch("app.graph.nodes.tool_node.DjangoCoreClient", return_value=mock_client):
            history_payload = [
                {"role": "user", "content": "coba cek stok untuk product FP BROTHER DYE ANTI UV 100ML - MAGENTA"},
                {"role": "assistant", "content": "### Produk Tidak Ditemukan\nProduk tidak ditemukan."},
            ]

            # Call on a fresh thread with no prior checkpoints, passing history
            res = await arun_workflow(
                user_query="coba cek lagi",
                thread_id="test_hydrated_session_99",
                history=history_payload,
            )

            self.assertEqual(res.get("detected_domain"), "inventory")
            self.assertEqual(res.get("detected_tool"), "get_stock_level")
            self.assertEqual(res.get("verification_status"), "VALID")
            self.assertIn("FP BROTHER DYE ANTI UV 100ML - MAGENTA", res.get("final_answer", ""))
            self.assertIn("25.0 unit", res.get("final_answer", ""))


class TestSynchronousRunnerHelper(unittest.TestCase):
    """Test suite validating run_workflow_sync and arun_workflow wrappers."""

    def test_run_workflow_sync_execution(self):
        mock_client = MagicMock(spec=DjangoCoreClient)
        mock_client.fetch_vendor_invoices = AsyncMock(
            return_value={"total_count": 0, "entries": []}
        )

        with patch("app.graph.nodes.tool_node.DjangoCoreClient", return_value=mock_client):
            result = run_workflow_sync(
                user_query="Tagihan vendor CV Maju",
                thread_id="sync_test_thread",
            )
            self.assertIsNotNone(result)
            self.assertIn("final_answer", result)
            self.assertEqual(result["verification_status"], "EMPTY")

    def test_workflow_executor_singleton_reuse(self):
        """get_workflow_executor must return the same executor singleton instance across calls."""
        e1 = get_workflow_executor()
        e2 = get_workflow_executor()
        self.assertIs(e1, e2)

    def test_workflow_executor_shutdown_and_reinitialization(self):
        """shutdown_workflow_executor resets executor, and next call creates a fresh active one."""
        e1 = get_workflow_executor()
        self.assertFalse(e1._shutdown)
        shutdown_workflow_executor(wait=False)
        self.assertIsNone(workflow_module._WORKFLOW_EXECUTOR)
        e2 = get_workflow_executor()
        self.assertIsNotNone(e2)
        self.assertFalse(e2._shutdown)


class TestHttpClientLifecycle(unittest.IsolatedAsyncioTestCase):
    """Test suite validating DjangoCoreClient lifecycle management in tool_node and rag_node (R4)."""

    async def test_tool_node_own_client_closed_in_finally(self):
        """tool_node must await aclose() when managing its own DjangoCoreClient."""
        mock_instance = MagicMock(spec=DjangoCoreClient)
        mock_instance.fetch_vendor_invoices = AsyncMock(
            return_value={"total_count": 0, "entries": []}
        )
        mock_instance.aclose = AsyncMock()

        with patch("app.graph.nodes.tool_node.DjangoCoreClient", return_value=mock_instance):
            state = create_initial_state(user_query="Tagihan vendor ABC")
            state["detected_domain"] = "accounting"
            state["detected_tool"] = "fetch_vendor_invoices"
            res = await tool_node(state, client=None)
            self.assertEqual(res["verification_status"], "VALID")
            mock_instance.aclose.assert_awaited_once()

    async def test_tool_node_injected_client_not_closed(self):
        """tool_node must NOT await aclose() when client is externally injected."""
        mock_instance = MagicMock(spec=DjangoCoreClient)
        mock_instance.fetch_vendor_invoices = AsyncMock(
            return_value={"total_count": 0, "entries": []}
        )
        mock_instance.aclose = AsyncMock()

        state = create_initial_state(user_query="Tagihan vendor ABC")
        state["detected_domain"] = "accounting"
        state["detected_tool"] = "fetch_vendor_invoices"
        res = await tool_node(state, client=mock_instance)
        self.assertEqual(res["verification_status"], "VALID")
        mock_instance.aclose.assert_not_called()

    async def test_rag_node_own_client_closed_in_finally(self):
        """rag_node must await aclose() when managing its own DjangoCoreClient."""
        mock_instance = MagicMock(spec=DjangoCoreClient)
        mock_instance.fetch_knowledge = AsyncMock(return_value={"chunks": []})
        mock_instance.aclose = AsyncMock()

        with patch("app.graph.nodes.rag_node.DjangoCoreClient", return_value=mock_instance), \
             patch.object(settings, "ENABLE_RAG", True), \
             patch.object(settings, "VECTOR_DB_URL", ""):
            state = create_initial_state(user_query="SOP Retur Barang")
            state["query_mode"] = "vector_rag"
            state["rag_query"] = "SOP Retur Barang"
            res = await rag_node(state, client=None)
            self.assertEqual(res["verification_status"], "EMPTY")
            mock_instance.aclose.assert_awaited_once()

    async def test_rag_node_injected_client_not_closed(self):
        """rag_node must NOT await aclose() when client is externally injected."""
        mock_instance = MagicMock(spec=DjangoCoreClient)
        mock_instance.fetch_knowledge = AsyncMock(return_value={"chunks": []})
        mock_instance.aclose = AsyncMock()

        with patch.object(settings, "ENABLE_RAG", True):
            state = create_initial_state(user_query="SOP Retur Barang")
            state["query_mode"] = "vector_rag"
            state["rag_query"] = "SOP Retur Barang"
            res = await rag_node(state, client=mock_instance)
            self.assertEqual(res["verification_status"], "EMPTY")
            mock_instance.aclose.assert_not_called()


class TestUndoWorkflowThread(unittest.TestCase):
    """Test suite validating undo_workflow_thread rollback on StateGraph checkpoints."""

    def test_undo_nonexistent_or_empty_thread(self):
        success, undone_prompt, remaining = undo_workflow_thread("thread_does_not_exist_xyz")
        self.assertFalse(success)
        self.assertIsNone(undone_prompt)
        self.assertEqual(remaining, 0)

    def test_undo_reverts_last_turn_multi_turn(self):
        graph = create_workflow_graph(checkpointer=MemorySaver())
        tid = "test_undo_multi_turn"
        config = {"configurable": {"thread_id": tid}}

        # Seed 2 full turns
        m1 = HumanMessage(content="Halo turn 1")
        m2 = AIMessage(content="Respons turn 1")
        m3 = HumanMessage(content="Pertanyaan turn 2 yang salah")
        m4 = AIMessage(content="Respons turn 2")
        graph.update_state(config, {"messages": [m1, m2, m3, m4]})

        # Invoke undo
        success, undone_prompt, remaining = undo_workflow_thread(tid, graph=graph)
        self.assertTrue(success)
        self.assertEqual(undone_prompt, "Pertanyaan turn 2 yang salah")
        self.assertEqual(remaining, 2)

        snapshot = graph.get_state(config)
        remaining_msgs = snapshot.values.get("messages", [])
        self.assertEqual(len(remaining_msgs), 2)
        self.assertEqual(remaining_msgs[0].content, "Halo turn 1")
        self.assertEqual(remaining_msgs[1].content, "Respons turn 1")

    def test_undo_single_turn_cleans_thread_entirely(self):
        graph = create_workflow_graph(checkpointer=MemorySaver())
        tid = "test_undo_single_turn"
        config = {"configurable": {"thread_id": tid}}

        m1 = HumanMessage(content="Pesan tunggal")
        m2 = AIMessage(content="Jawaban tunggal")
        graph.update_state(config, {"messages": [m1, m2]})

        success, undone_prompt, remaining = undo_workflow_thread(tid, graph=graph)
        self.assertTrue(success)
        self.assertEqual(undone_prompt, "Pesan tunggal")
        self.assertEqual(remaining, 0)

        snapshot = graph.get_state(config)
        self.assertEqual(len(snapshot.values.get("messages", [])), 0)


if __name__ == "__main__":
    unittest.main()
