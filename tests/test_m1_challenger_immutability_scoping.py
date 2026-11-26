"""
Milestone 1 Adversarial & Empirical Test Suite - Challenger 2.
Empirically verifies:
1. State Immutability: router_node returns deltas without in-place mutation of input state.
2. Thread Safety: Concurrent router_node invocations with shared state dict.
3. Multi-Company Scoping & company_id propagation: custom company_id, 0 company_id, None fallback.
4. ToolCallPlan & MultiRouterDecision type safety and validation constraints.
5. Edge cases: empty/whitespace queries, missing messages, extreme payload size,
   multi-intent combinations (2 tools, 3 tools, same-domain multi-intent),
   customer-vendor guardrails, conversational retry chains.
"""
import copy
import concurrent.futures
from typing import List
import unittest

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from pydantic import ValidationError

from app.core.config import settings
from app.graph.state import (
    AgentState,
    create_initial_state,
)
from app.graph.nodes.router import (
    router_node,
    find_previous_user_query,
    PlannedToolCall,
    MultiRouterDecision,
)


class TestStateImmutabilityAndThreadSafety(unittest.TestCase):
    """Empirical verification that router_node guarantees state immutability."""

    def test_router_node_does_not_mutate_input_state_single_intent(self):
        """router_node must not alter the original state dictionary or any of its nested objects."""
        original_messages = [HumanMessage(content="Cek stok kertas A4")]
        input_state: AgentState = {
            "messages": original_messages,
            "user_query": "Cek stok kertas A4",
            "company_id": 7,
            "user_token": "token-xyz",
            "retry_count": 0,
            "detected_domain": None,
            "detected_tool": None,
            "active_tools": [],
            "multi_tool_results": {},
        }

        # Take a deep copy for strict baseline comparison
        baseline_state = copy.deepcopy(input_state)
        messages_obj_id = id(input_state["messages"])
        active_tools_id = id(input_state["active_tools"])

        delta = router_node(input_state)

        # 1. Delta must be returned
        self.assertIsInstance(delta, dict)
        self.assertEqual(delta.get("detected_domain"), "inventory")
        self.assertEqual(delta.get("detected_tool"), "get_stock_level")

        # 2. Input state must be bit-for-bit identical to baseline
        self.assertEqual(input_state["user_query"], baseline_state["user_query"])
        self.assertEqual(input_state["company_id"], baseline_state["company_id"])
        self.assertEqual(input_state["detected_domain"], baseline_state["detected_domain"])
        self.assertEqual(input_state["detected_tool"], baseline_state["detected_tool"])
        self.assertEqual(input_state["active_tools"], baseline_state["active_tools"])
        self.assertEqual(len(input_state["messages"]), 1)
        self.assertEqual(id(input_state["messages"]), messages_obj_id)
        self.assertEqual(id(input_state["active_tools"]), active_tools_id)

    def test_router_node_does_not_mutate_input_state_multi_intent(self):
        """router_node must not mutate input state on multi-intent queries."""
        input_state = create_initial_state(
            user_query="Cek sisa stok kertas dan saldo kas kecil",
            company_id=3,
        )
        baseline_state = copy.deepcopy(input_state)

        delta = router_node(input_state)

        self.assertIsInstance(delta, dict)
        self.assertTrue(delta.get("is_multi_tool"))
        self.assertEqual(len(delta.get("active_tools", [])), 2)

        # Assert input_state is completely unchanged
        self.assertEqual(input_state["user_query"], baseline_state["user_query"])
        self.assertEqual(input_state["is_multi_tool"], baseline_state["is_multi_tool"])
        self.assertEqual(input_state["active_tools"], baseline_state["active_tools"])
        self.assertEqual(input_state["detected_domain"], baseline_state["detected_domain"])
        self.assertEqual(input_state["detected_tool"], baseline_state["detected_tool"])

    def test_router_node_concurrent_thread_safety(self):
        """Simultaneous calls to router_node using the same input state dictionary must be safe."""
        shared_state = create_initial_state(
            user_query="Berapa saldo kas dan cek stok tinta hitam?",
            company_id=4,
        )

        def worker():
            return router_node(shared_state)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker) for _ in range(50)]
            results = [f.result() for f in futures]

        # Verify all 50 executions produced identical valid results without race conditions
        for res in results:
            self.assertTrue(res.get("is_multi_tool"))
            self.assertEqual(len(res.get("active_tools", [])), 2)
            self.assertEqual(res.get("detected_tools"), ["get_account_balance", "get_stock_level"])


class TestCompanyIdScopingAndPropagation(unittest.TestCase):
    """Empirical verification of company_id scoping and StateGraph lifecycle propagation."""

    def test_default_company_id_fallback(self):
        """Omitting company_id must fall back to settings.DEFAULT_COMPANY_ID."""
        state = create_initial_state(user_query="Cek kas")
        self.assertEqual(state["company_id"], settings.DEFAULT_COMPANY_ID)

    def test_explicit_custom_company_id(self):
        """Custom positive company_id must be strictly preserved."""
        state = create_initial_state(user_query="Cek kas", company_id=42)
        self.assertEqual(state["company_id"], 42)

    def test_company_id_zero_is_preserved_not_defaulted(self):
        """Boundary condition: company_id=0 must NOT be treated as falsy and overridden."""
        state = create_initial_state(user_query="Cek kas", company_id=0)
        self.assertEqual(state["company_id"], 0)

    def test_company_id_propagation_across_stategraph_with_router_node(self):
        """In a compiled StateGraph, company_id must persist untouched through router_node."""
        def dummy_extractor(state: AgentState) -> dict:
            return {"extracted_params": {"account_code": "111100"}}

        workflow = StateGraph(AgentState)
        workflow.add_node("router", router_node)
        workflow.add_node("extractor", dummy_extractor)
        workflow.add_edge(START, "router")
        workflow.add_edge("router", "extractor")
        workflow.add_edge("extractor", END)

        app = workflow.compile()

        # Test with custom company_id=99
        initial_state = create_initial_state("Cek saldo kas 111100", company_id=99)
        final_state = app.invoke(initial_state)

        self.assertEqual(final_state["company_id"], 99)
        self.assertEqual(final_state["detected_domain"], "accounting")
        self.assertEqual(final_state["detected_tool"], "get_account_balance")
        self.assertEqual(final_state["extracted_params"], {"account_code": "111100"})


class TestToolCallPlanAndModelValidation(unittest.TestCase):
    """Type safety and edge cases for ToolCallPlan and Pydantic router schemas."""

    def test_planned_tool_call_valid_and_domain_mapping(self):
        """PlannedToolCall corrects wrong domain for known tools."""
        pt = PlannedToolCall(tool="fetch_vendor_invoices", domain="inventory")
        self.assertEqual(pt.domain, "accounting")

    def test_planned_tool_call_invalid_tool_raises_validation_error(self):
        """PlannedToolCall must reject unregistered tools."""
        with self.assertRaises(ValidationError):
            PlannedToolCall(tool="non_existent_tool", domain="accounting")

    def test_multi_router_decision_domain_clearing_rules(self):
        """Non-live domains must strip tool and planned_tools completely."""
        for dom in ["general", "out_of_scope", "knowledge"]:
            d = MultiRouterDecision(
                domain=dom,
                tool="get_stock_level",
                planned_tools=[PlannedToolCall(tool="get_stock_level", domain="inventory")],
            )
            self.assertIsNone(d.tool)
            self.assertEqual(d.planned_tools, [])

    def test_multi_router_decision_subquery_preserved(self):
        """sub_query in PlannedToolCall is preserved in MultiRouterDecision."""
        decision = MultiRouterDecision(
            domain="inventory",
            planned_tools=[
                PlannedToolCall(
                    tool="get_stock_level",
                    domain="inventory",
                    sub_query="stok kertas foto glossy",
                )
            ],
        )
        self.assertEqual(decision.planned_tools[0].sub_query, "stok kertas foto glossy")


class TestRouterAdversarialEdgeCases(unittest.TestCase):
    """Stress-testing router_node on edge cases, boundary inputs, and complex multi-intents."""

    def test_empty_and_whitespace_user_query(self):
        """Empty or whitespace queries must cleanly route to general without throwing exceptions."""
        for empty_val in ["", "   ", "\t\n  \r", None]:
            state: AgentState = {"user_query": empty_val, "messages": []}
            result = router_node(state)
            self.assertEqual(result["detected_domain"], "general")
            self.assertIsNone(result["detected_tool"])
            self.assertFalse(result["is_multi_tool"])
            self.assertEqual(result["active_tools"], [])

    def test_fallback_to_messages_when_user_query_empty(self):
        """When user_query is empty, router_node extracts query from last message."""
        state: AgentState = {
            "user_query": "",
            "messages": [HumanMessage(content="Cek saldo kas")],
        }
        result = router_node(state)
        self.assertEqual(result["detected_domain"], "accounting")
        self.assertEqual(result["detected_tool"], "get_account_balance")

    def test_non_human_last_message_fallback(self):
        """When user_query is empty and last message is AIMessage, handle gracefully."""
        state: AgentState = {
            "user_query": "",
            "messages": [AIMessage(content="Ada yang bisa saya bantu?")],
        }
        result = router_node(state)
        self.assertEqual(result["detected_domain"], "general")

    def test_extreme_query_length_handling(self):
        """Massive query payload (10,000 chars) must not crash regex or router."""
        large_query = "Cek sisa stok barang " + ("dan buku besar " * 500)
        state = create_initial_state(large_query)
        result = router_node(state)
        self.assertIn(result["detected_domain"], ["inventory", "accounting"])
        self.assertTrue(result["is_multi_tool"])

    def test_sql_injection_adversarial_query(self):
        """SQL injection / adversarial query should be handled safely as text."""
        query = "Cek saldo kas'; DROP TABLE accounts; SELECT * FROM pg_user; --"
        state = create_initial_state(query)
        result = router_node(state)
        self.assertEqual(result["detected_domain"], "accounting")
        self.assertEqual(result["detected_tool"], "get_account_balance")

    def test_multi_intent_three_tools_across_two_domains(self):
        """Complex triple intent: stock (inventory) + balance (accounting) + vendor invoice (accounting)."""
        query = "Cek sisa stok kertas A4, saldo kas kecil, dan tagihan vendor PT Sumber Makmur"
        state = create_initial_state(query)
        result = router_node(state)

        self.assertTrue(result["is_multi_tool"])
        self.assertEqual(len(result["active_tools"]), 3)

        expected_tools = ["get_stock_level", "get_account_balance", "fetch_vendor_invoices"]
        actual_tools = [t["tool"] for t in result["active_tools"]]
        self.assertEqual(actual_tools, expected_tools)

        # Primary tool backward compatibility
        self.assertEqual(result["detected_tool"], "get_stock_level")
        self.assertEqual(result["detected_domain"], "inventory")
        self.assertCountEqual(result["detected_domains"], ["inventory", "accounting"])
        self.assertEqual(result["detected_tools"], expected_tools)

        # Verify each active_tools item structure complies with ToolCallPlan
        for plan in result["active_tools"]:
            self.assertIn("tool", plan)
            self.assertIn("domain", plan)
            self.assertIn("sub_query", plan)
            self.assertIn("params", plan)
            self.assertEqual(plan["status"], "pending")

    def test_multi_intent_same_domain_stock_and_locations(self):
        """Multi-intent within the same domain (inventory): stock + locations."""
        query = "Cek sisa stok kertas foto dan daftar gudang aktif"
        state = create_initial_state(query)
        result = router_node(state)

        self.assertTrue(result["is_multi_tool"])
        tools = [t["tool"] for t in result["active_tools"]]
        self.assertIn("get_stock_level", tools)
        self.assertIn("get_locations", tools)
        self.assertEqual(result["detected_domains"], ["inventory"])

    def test_deduplication_of_identical_tools(self):
        """Query with duplicate intent for the same tool must deduplicate in active_tools."""
        query = "Cek sisa stok kertas A4 dan cek stok tinta hitam"
        state = create_initial_state(query)
        result = router_node(state)

        # Should not have duplicate get_stock_level in active_tools
        self.assertFalse(result["is_multi_tool"])
        self.assertEqual(len(result["active_tools"]), 1)
        self.assertEqual(result["active_tools"][0]["tool"], "get_stock_level")

    def test_conversational_retry_skips_interleaved_retries(self):
        """find_previous_user_query correctly skips consecutive retry messages."""
        messages: List[BaseMessage] = [
            HumanMessage(content="Cek saldo bank BCA"),
            AIMessage(content="Saldo kosong"),
            HumanMessage(content="coba lagi"),
            AIMessage(content="Masih kosong"),
            HumanMessage(content="ulang ya"),
        ]
        prev_query = find_previous_user_query(messages)
        self.assertEqual(prev_query, "Cek saldo bank BCA")

    def test_conversational_retry_with_no_history_fails_cleanly(self):
        """'coba lagi' without prior human query returns clean error."""
        state = create_initial_state("coba lagi")
        state["messages"] = [HumanMessage(content="coba lagi")]
        result = router_node(state)
        self.assertEqual(result["detected_domain"], "general")
        self.assertEqual(result.get("error_message"), "NO_PREVIOUS_QUERY_TO_RETRY")


if __name__ == "__main__":
    unittest.main()
