"""
Unit tests for Data Verification Node (TASK-4.4).
Validates adaptive records count calculation, verification status transitions (VALID, RETRY, EMPTY, ERROR),
retry counter increment and limit (max 2x), staged filter correction instructions,
conditional edge routing (decide_after_verification), and end-to-end StateGraph retry cycle.
"""
from datetime import date
import unittest
from langgraph.graph import StateGraph, START, END

from app.graph.state import create_initial_state, AgentState
from app.graph.nodes.extractor import extractor_node
from app.graph.nodes.verifier import (
    verifier_node,
    decide_after_verification,
    _calculate_records_count,
)


class TestRecordsCountCalculation(unittest.TestCase):
    """Test suite validating hierarchical / adaptive records counting across ERP payloads."""

    def test_count_from_total_count_field(self):
        """Explicit total_count field takes precedence."""
        self.assertEqual(_calculate_records_count({"total_count": 10, "entries": []}), 10)
        self.assertEqual(_calculate_records_count({"total_count": 0, "entries": []}), 0)

    def test_count_from_count_field(self):
        """Explicit count field is respected."""
        self.assertEqual(_calculate_records_count({"count": 4}), 4)
        self.assertEqual(_calculate_records_count({"count": 0}), 0)

    def test_count_from_entries_list(self):
        """Falls back to len(entries) if total_count is absent."""
        self.assertEqual(_calculate_records_count({"entries": [{"id": 1}, {"id": 2}]}), 2)
        self.assertEqual(_calculate_records_count({"entries": []}), 0)

    def test_count_from_accounts_list(self):
        """Accounting balance payload with accounts list."""
        self.assertEqual(_calculate_records_count({"accounts": [{"account_id": 1}], "total_debit": 100}), 1)
        self.assertEqual(_calculate_records_count({"accounts": [], "total_debit": 0}), 0)

    def test_count_from_products_list(self):
        """Inventory product info payload with products list."""
        self.assertEqual(_calculate_records_count({"products": [{"id": 10}, {"id": 20}]}), 2)
        self.assertEqual(_calculate_records_count({"products": []}), 0)

    def test_count_from_root_list(self):
        """Payload is a direct list of items."""
        self.assertEqual(_calculate_records_count([{"sku": "A"}, {"sku": "B"}, {"sku": "C"}]), 3)
        self.assertEqual(_calculate_records_count([]), 0)

    def test_count_from_single_record_dict(self):
        """Payload is a single entity dictionary (e.g. stock level)."""
        self.assertEqual(_calculate_records_count({"product_id": 1, "qty_on_hand": 50.0}), 1)

    def test_count_none_and_empty_dict(self):
        """None or empty dict evaluates to 0 records."""
        self.assertEqual(_calculate_records_count(None), 0)
        self.assertEqual(_calculate_records_count({}), 0)

    def test_count_from_zero_aggregate_balance_dict(self):
        """Payload with only zero totals evaluates to 0 records."""
        self.assertEqual(_calculate_records_count({"total_debit": 0.0, "total_credit": 0.0, "net_balance": 0.0}), 0)
        self.assertEqual(_calculate_records_count({"accounts": [], "total_debit": 0.0, "total_credit": 0.0, "net_balance": 0.0}), 0)
        self.assertEqual(_calculate_records_count({"total_debit": 50000.0, "total_credit": 0.0, "net_balance": 50000.0}), 1)

    def test_count_from_error_and_detail_payloads(self):
        """Error and detail payloads must evaluate to 0 records instead of defaulting to 1."""
        self.assertEqual(_calculate_records_count({"detail": "Warehouse not found"}), 0)
        self.assertEqual(_calculate_records_count({"error": "Product variant not found"}), 0)
        self.assertEqual(_calculate_records_count({"errors": ["Missing company_id"]}), 0)
        self.assertEqual(_calculate_records_count({"success": False, "message": "Failed"}), 0)
        self.assertEqual(_calculate_records_count({"detail": "Not found", "status_code": 404}), 0)

    def test_count_from_unrecognized_arbitrary_dict(self):
        """Unexpected dictionary without known ERP business attributes evaluates to 0 records."""
        self.assertEqual(_calculate_records_count({"status": "ok"}), 0)
        self.assertEqual(_calculate_records_count({"foo": "bar", "unknown_key": 123}), 0)
        self.assertEqual(_calculate_records_count({"product_id": None, "qty_on_hand": None}), 0)

    def test_count_from_locations_list(self):
        """Payload with locations collection key evaluates to list length."""
        self.assertEqual(_calculate_records_count({"locations": [{"id": 1, "name": "Gudang Utama"}]}), 1)
        self.assertEqual(_calculate_records_count({"locations": []}), 0)


class TestVerifierNodeTransitions(unittest.TestCase):
    """Test suite validating verifier_node status transitions, retries, and instructions."""

    def test_records_found_sets_valid_status(self):
        """When records_count > 0, verification_status must be 'VALID'."""
        state = create_initial_state(user_query="Tagihan vendor PT Sumber Makmur")
        state["tool_result"] = {
            "total_count": 2,
            "entries": [{"id": 1}, {"id": 2}],
        }
        state["retry_count"] = 0

        update = verifier_node(state)
        self.assertEqual(update["verification_status"], "VALID")
        self.assertIsNone(update["retry_instruction"])

    def test_empty_records_first_retry_cycle(self):
        """
        When records_count == 0 and retry_count == 0:
        Sets verification_status='RETRY', increments retry_count to 1,
        and gives date relaxation instruction.
        """
        state = create_initial_state(user_query="Tagihan vendor PT Sumber Makmur bulan ini")
        state["tool_result"] = {"total_count": 0, "entries": []}
        state["retry_count"] = 0

        update = verifier_node(state)
        self.assertEqual(update["verification_status"], "RETRY")
        self.assertEqual(update["retry_count"], 1)
        self.assertIsNotNone(update["retry_instruction"])
        self.assertIn("tanggal", update["retry_instruction"].lower())

    def test_empty_records_second_retry_cycle(self):
        """
        When records_count == 0 and retry_count == 1:
        Sets verification_status='RETRY', increments retry_count to 2,
        and gives search/vendor relaxation instruction.
        """
        state = create_initial_state(user_query="Tagihan vendor PT Sumber Makmur")
        state["tool_result"] = {"total_count": 0, "entries": []}
        state["retry_count"] = 1

        update = verifier_node(state)
        self.assertEqual(update["verification_status"], "RETRY")
        self.assertEqual(update["retry_count"], 2)
        self.assertIsNotNone(update["retry_instruction"])
        self.assertIn("vendor", update["retry_instruction"].lower())

    def test_empty_records_retry_limit_reached_sets_empty(self):
        """
        Definition of Done:
        When records_count == 0 and retry_count >= 2, halts retry cycle and sets 'EMPTY'.
        """
        state = create_initial_state(user_query="Tagihan vendor TidakAda")
        state["tool_result"] = {"total_count": 0, "entries": []}
        state["retry_count"] = 2

        update = verifier_node(state)
        self.assertEqual(update["verification_status"], "EMPTY")
        self.assertIsNone(update.get("retry_instruction"))
        # Must not increment beyond 2
        self.assertNotIn("retry_count", update)

    def test_preexisting_error_status_preserved(self):
        """
        When tool execution had an error, verifier preserves ERROR status
        and does not trigger RETRY.
        """
        state = create_initial_state(user_query="Kueri kas")
        state["tool_result"] = None
        state["verification_status"] = "ERROR"
        state["error_message"] = "Connection refused to ERP (HTTP 500)"
        state["retry_count"] = 0

        update = verifier_node(state)
        self.assertEqual(update["verification_status"], "ERROR")
        self.assertIsNone(update.get("retry_instruction"))
        self.assertNotIn("retry_count", update)


class TestDecideAfterVerification(unittest.TestCase):
    """Test suite validating conditional edge helper decide_after_verification."""

    def test_routes_to_extractor_on_retry(self):
        """Status 'RETRY' routes back to 'extractor'."""
        state = create_initial_state(user_query="test")
        state["verification_status"] = "RETRY"
        self.assertEqual(decide_after_verification(state), "extractor")

    def test_routes_to_synthesizer_on_valid(self):
        """Status 'VALID' routes forward to 'synthesizer'."""
        state = create_initial_state(user_query="test")
        state["verification_status"] = "VALID"
        self.assertEqual(decide_after_verification(state), "synthesizer")

    def test_routes_to_synthesizer_on_empty(self):
        """Status 'EMPTY' routes forward to 'synthesizer'."""
        state = create_initial_state(user_query="test")
        state["verification_status"] = "EMPTY"
        self.assertEqual(decide_after_verification(state), "synthesizer")

    def test_routes_to_synthesizer_on_error(self):
        """Status 'ERROR' routes forward to 'synthesizer' for user explanation."""
        state = create_initial_state(user_query="test")
        state["verification_status"] = "ERROR"
        self.assertEqual(decide_after_verification(state), "synthesizer")


class TestExtractorStagedRelaxation(unittest.TestCase):
    """Test suite validating extractor_node responding to retry_count and retry_instruction."""

    def test_extractor_relaxes_date_on_first_retry(self):
        """
        On retry_count=1, deterministic fallback relaxes date filters.
        """
        anchor = date(2026, 11, 4)
        query = "Tagihan vendor PT Sumber Makmur bulan ini"

        # Initial attempt: date_from and date_to extracted
        state_initial = create_initial_state(user_query=query)
        state_initial["detected_tool"] = "fetch_vendor_invoices"
        state_initial["retry_count"] = 0
        ext_initial = extractor_node(state_initial, ref_date=anchor)
        self.assertIn("date_from", ext_initial["extracted_params"])
        self.assertIn("date_to", ext_initial["extracted_params"])
        self.assertEqual(ext_initial["extracted_params"]["vendor_name"], "PT Sumber Makmur")

        # Retry 1: date filters relaxed
        state_retry1 = create_initial_state(user_query=query)
        state_retry1["detected_tool"] = "fetch_vendor_invoices"
        state_retry1["retry_count"] = 1
        state_retry1["retry_instruction"] = "Longgarkan tanggal"
        ext_retry1 = extractor_node(state_retry1, ref_date=anchor)
        self.assertNotIn("date_from", ext_retry1["extracted_params"])
        self.assertNotIn("date_to", ext_retry1["extracted_params"])
        self.assertEqual(ext_retry1["extracted_params"]["vendor_name"], "PT Sumber Makmur")

    def test_extractor_relaxes_vendor_and_date_on_second_retry(self):
        """
        On retry_count=2, deterministic fallback relaxes both date and vendor filters.
        """
        anchor = date(2026, 11, 4)
        query = "Tagihan vendor PT Sumber Makmur bulan ini"

        state_retry2 = create_initial_state(user_query=query)
        state_retry2["detected_tool"] = "fetch_vendor_invoices"
        state_retry2["retry_count"] = 2
        state_retry2["retry_instruction"] = "Hapus nama vendor"
        ext_retry2 = extractor_node(state_retry2, ref_date=anchor)
        self.assertNotIn("date_from", ext_retry2["extracted_params"])
        self.assertNotIn("date_to", ext_retry2["extracted_params"])
        self.assertNotIn("vendor_name", ext_retry2["extracted_params"])


class TestStateGraphRetryLoop(unittest.TestCase):
    """
    Test suite validating full LangGraph StateGraph cycle with verification retry loop.
    Demonstrates max 2x loop termination and recovery when data is found on retry.
    """

    def test_retry_loop_recovers_when_data_found_on_retry(self):
        """
        Scenario A:
        1. Extractor extracts vendor + dates
        2. Tool call with date filters returns 0 records
        3. Verifier triggers RETRY (count=1), routes back to Extractor
        4. Extractor relaxes dates
        5. Tool call without date filters returns 2 records
        6. Verifier marks VALID, routes to Synthesizer!
        """
        call_counter = {"count": 0}

        def mock_tool_node(state: AgentState):
            call_counter["count"] += 1
            params = state.get("extracted_params") or {}
            # If date_from is present (strict query), return empty
            if "date_from" in params:
                return {"tool_result": {"total_count": 0, "entries": []}}
            # If date_from was relaxed, return records
            return {
                "tool_result": {
                    "total_count": 2,
                    "entries": [{"id": 101}, {"id": 102}],
                }
            }

        def mock_synthesizer_node(state: AgentState):
            return {"final_answer": "Ditemukan 2 tagihan setelah relaksasi filter tanggal."}

        # Build StateGraph
        builder = StateGraph(AgentState)
        builder.add_node("extractor", extractor_node)
        builder.add_node("tool_node", mock_tool_node)
        builder.add_node("verifier", verifier_node)
        builder.add_node("synthesizer", mock_synthesizer_node)

        builder.add_edge(START, "extractor")
        builder.add_edge("extractor", "tool_node")
        builder.add_edge("tool_node", "verifier")
        builder.add_conditional_edges(
            "verifier",
            decide_after_verification,
            {"extractor": "extractor", "synthesizer": "synthesizer"},
        )
        builder.add_edge("synthesizer", END)

        graph = builder.compile()

        state = create_initial_state(user_query="Tagihan vendor PT Sumber Makmur bulan ini")
        state["detected_tool"] = "fetch_vendor_invoices"

        final_state = graph.invoke(state)
        self.assertEqual(final_state["verification_status"], "VALID")
        self.assertEqual(final_state["retry_count"], 1)
        self.assertEqual(call_counter["count"], 2)  # Ran twice
        self.assertIn("Ditemukan 2 tagihan", final_state["final_answer"])

    def test_retry_loop_terminates_at_max_2_retries_with_empty(self):
        """
        Definition of Done Scenario B:
        Tool continuously returns empty. The system must retry at most 2 times,
        then terminate with verification_status='EMPTY' and route to synthesizer.
        """
        call_counter = {"count": 0}

        def mock_always_empty_tool(state: AgentState):
            call_counter["count"] += 1
            return {"tool_result": {"total_count": 0, "entries": []}}

        def mock_synthesizer_node(state: AgentState):
            return {"final_answer": "Tidak ada data tagihan ditemukan di sistem."}

        # Build StateGraph
        builder = StateGraph(AgentState)
        builder.add_node("extractor", extractor_node)
        builder.add_node("tool_node", mock_always_empty_tool)
        builder.add_node("verifier", verifier_node)
        builder.add_node("synthesizer", mock_synthesizer_node)

        builder.add_edge(START, "extractor")
        builder.add_edge("extractor", "tool_node")
        builder.add_edge("tool_node", "verifier")
        builder.add_conditional_edges(
            "verifier",
            decide_after_verification,
            {"extractor": "extractor", "synthesizer": "synthesizer"},
        )
        builder.add_edge("synthesizer", END)

        graph = builder.compile()

        state = create_initial_state(user_query="Tagihan vendor Khayalan")
        state["detected_tool"] = "fetch_vendor_invoices"

        final_state = graph.invoke(state)
        # Attempt 0 -> empty (retry 1) -> Attempt 1 -> empty (retry 2) -> Attempt 2 -> empty (EMPTY)
        # Total tool calls = 3 (initial + 2 retries)
        self.assertEqual(call_counter["count"], 3)
        self.assertEqual(final_state["verification_status"], "EMPTY")
        self.assertEqual(final_state["retry_count"], 2)
        self.assertIn("Tidak ada data tagihan ditemukan", final_state["final_answer"])


if __name__ == "__main__":
    unittest.main()
