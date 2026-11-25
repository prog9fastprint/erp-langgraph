"""
Unit tests for LangGraph AgentState and StateGraph cycle compatibility (TASK-4.1).
Validates schema definition, helper initialization, multi-company scoping,
message reducer integration (add_messages), and StateGraph execution cycle (DoD).
"""
import unittest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from app.graph.state import (
    AgentState,
    ToolCallPlan,
    create_initial_state,
)
from app.graph.nodes.router import (
    router_node,
    PlannedToolCall,
    MultiRouterDecision,
)
from app.graph.workflow import decide_after_router
from app.core.config import settings


class TestAgentStateCreation(unittest.TestCase):
    """Test suite for AgentState schema definition and initial state constructor."""

    def test_create_initial_state_defaults(self):
        """create_initial_state should populate standard defaults and wrap query in HumanMessage."""
        state = create_initial_state("Berapa saldo kas kecil?")

        self.assertEqual(state["user_query"], "Berapa saldo kas kecil?")
        self.assertIsNone(state["user_token"])
        self.assertEqual(state["company_id"], settings.DEFAULT_COMPANY_ID)
        self.assertEqual(state["retry_count"], 0)
        self.assertIsNone(state["detected_domain"])
        self.assertIsNone(state["detected_tool"])
        self.assertIsNone(state["extracted_params"])
        self.assertIsNone(state["tool_result"])
        self.assertIsNone(state["verification_status"])
        self.assertIsNone(state["final_answer"])
        self.assertIsNone(state["error_message"])

        # Check messages list
        self.assertEqual(len(state["messages"]), 1)
        self.assertIsInstance(state["messages"][0], HumanMessage)
        self.assertEqual(state["messages"][0].content, "Berapa saldo kas kecil?")

    def test_create_initial_state_with_explicit_parameters(self):
        """create_initial_state respects custom user_token, company_id, and pre-existing messages."""
        existing_msgs = [
            SystemMessage(content="You are an ERP assistant"),
            HumanMessage(content="Cek stok tinta"),
        ]
        state = create_initial_state(
            user_query="Cek stok tinta",
            user_token="test-jwt-token-xyz",
            company_id=3,
            messages=existing_msgs,
        )

        self.assertEqual(state["user_token"], "test-jwt-token-xyz")
        self.assertEqual(state["company_id"], 3)
        self.assertEqual(len(state["messages"]), 2)
        self.assertIsInstance(state["messages"][0], SystemMessage)
        self.assertIsInstance(state["messages"][1], HumanMessage)

    def test_partial_state_dict_validity(self):
        """AgentState supports partial dictionary representations (total=False)."""
        partial_state: AgentState = {
            "detected_domain": "accounting",
            "detected_tool": "get_journal_entries",
        }
        self.assertEqual(partial_state["detected_domain"], "accounting")
        self.assertEqual(partial_state["detected_tool"], "get_journal_entries")


class TestLangGraphCycleCompatibility(unittest.TestCase):
    """Test suite verifying AgentState compatibility with LangGraph StateGraph cycles (DoD)."""

    def test_stategraph_compilation_and_execution_cycle(self):
        """Definition of Done: StateGraph(AgentState) compiles and executes through multiple nodes."""

        # Define 3 sequential pipeline nodes returning partial state updates
        def router_node(state: AgentState) -> dict:
            return {
                "detected_domain": "accounting",
                "detected_tool": "get_account_balance",
                "extracted_params": {"account_code": "111100"},
            }

        def tool_mock_node(state: AgentState) -> dict:
            return {
                "tool_result": {
                    "account_code": state.get("extracted_params", {}).get("account_code"),
                    "net_balance": 1500000.0,
                },
                "verification_status": "VALID",
            }

        def answer_node(state: AgentState) -> dict:
            ans = f"Saldo akun {state['tool_result']['account_code']} adalah Rp {state['tool_result']['net_balance']:,.0f}"
            return {
                "final_answer": ans,
                "messages": [AIMessage(content=ans)],
            }

        # Build LangGraph workflow using AgentState
        workflow = StateGraph(AgentState)
        workflow.add_node("router", router_node)
        workflow.add_node("tool_exec", tool_mock_node)
        workflow.add_node("present_answer", answer_node)

        workflow.add_edge(START, "router")
        workflow.add_edge("router", "tool_exec")
        workflow.add_edge("tool_exec", "present_answer")
        workflow.add_edge("present_answer", END)

        app = workflow.compile()

        # Execute graph with initial state
        initial_state = create_initial_state("Berapa saldo kas 111100?", company_id=2)
        result = app.invoke(initial_state)

        # Assert full state propagation across nodes
        self.assertEqual(result["user_query"], "Berapa saldo kas 111100?")
        self.assertEqual(result["company_id"], 2)
        self.assertEqual(result["detected_domain"], "accounting")
        self.assertEqual(result["detected_tool"], "get_account_balance")
        self.assertEqual(result["extracted_params"], {"account_code": "111100"})
        self.assertEqual(result["tool_result"]["net_balance"], 1500000.0)
        self.assertEqual(result["verification_status"], "VALID")
        self.assertIn("Saldo akun 111100", result["final_answer"])

        # Assert messages reducer appended the AIMessage
        self.assertEqual(len(result["messages"]), 2)
        self.assertIsInstance(result["messages"][0], HumanMessage)
        self.assertIsInstance(result["messages"][1], AIMessage)

    def test_conditional_routing_on_verification_status(self):
        """Verifies conditional routing in StateGraph using verification_status."""

        def evaluate_node(state: AgentState) -> dict:
            return {"verification_status": "RETRY", "retry_count": state.get("retry_count", 0) + 1}

        def retry_node(state: AgentState) -> dict:
            return {"error_message": "Needs clarification"}

        def success_node(state: AgentState) -> dict:
            return {"final_answer": "Success"}

        def route_condition(state: AgentState) -> str:
            if state.get("verification_status") == "VALID":
                return "success"
            return "retry"

        workflow = StateGraph(AgentState)
        workflow.add_node("eval", evaluate_node)
        workflow.add_node("retry", retry_node)
        workflow.add_node("success", success_node)

        workflow.add_edge(START, "eval")
        workflow.add_conditional_edges("eval", route_condition, {"success": "success", "retry": "retry"})
        workflow.add_edge("retry", END)
        workflow.add_edge("success", END)

        app = workflow.compile()
        result = app.invoke(create_initial_state("Query retry test"))

        self.assertEqual(result["verification_status"], "RETRY")
        self.assertEqual(result["retry_count"], 1)
        self.assertEqual(result["error_message"], "Needs clarification")
        self.assertIsNone(result["final_answer"])


class TestMultiToolStateExtension(unittest.TestCase):
    """Test suite verifying multi-tool state schema extensions in AgentState."""

    def test_tool_call_plan_partial_and_full(self):
        """ToolCallPlan supports progressive population across pipeline nodes."""
        plan_partial: ToolCallPlan = {
            "tool": "get_stock_level",
            "domain": "inventory",
            "sub_query": "stok kertas photo a4",
            "status": "pending",
        }
        self.assertEqual(plan_partial["tool"], "get_stock_level")
        self.assertEqual(plan_partial["domain"], "inventory")
        self.assertEqual(plan_partial["status"], "pending")
        self.assertNotIn("params", plan_partial)

        # Extractor stage: params populated
        plan_partial["params"] = {"product_name": "kertas photo a4"}
        self.assertEqual(plan_partial["params"]["product_name"], "kertas photo a4")

        # Tool node stage: status updated
        plan_partial["status"] = "success"
        self.assertEqual(plan_partial["status"], "success")

    def test_create_initial_state_multi_tool_defaults(self):
        """create_initial_state populates multi-tool defaults while preserving scalar defaults."""
        state = create_initial_state("Cek stok kertas dan saldo kas")

        # Verify multi-tool defaults
        self.assertIs(state["is_multi_tool"], False)
        self.assertEqual(state["detected_domains"], [])
        self.assertEqual(state["detected_tools"], [])
        self.assertEqual(state["active_tools"], [])
        self.assertEqual(state["multi_tool_results"], {})
        self.assertEqual(state["multi_verification_status"], {})

        # Verify scalar defaults are untouched
        self.assertIsNone(state["detected_domain"])
        self.assertIsNone(state["detected_tool"])
        self.assertIsNone(state["extracted_params"])
        self.assertIsNone(state["tool_result"])
        self.assertIsNone(state["verification_status"])
        self.assertEqual(state["retry_count"], 0)

    def test_create_initial_state_custom_is_multi_tool(self):
        """create_initial_state accepts custom is_multi_tool parameter."""
        state = create_initial_state("Cek multi", is_multi_tool=True)
        self.assertIs(state["is_multi_tool"], True)

    def test_stategraph_multi_tool_data_flow(self):
        """StateGraph compiles with AgentState and updates multi-tool state."""
        def mock_router(state: AgentState) -> dict:
            return {
                "is_multi_tool": True,
                "detected_domains": ["inventory", "accounting"],
                "detected_tools": ["get_stock_level", "get_account_balance"],
                "active_tools": [
                    {"tool": "get_stock_level", "domain": "inventory", "sub_query": "stok", "status": "pending"},
                    {"tool": "get_account_balance", "domain": "accounting", "sub_query": "saldo", "status": "pending"},
                ],
                # Dual-write primary tool for backward compatibility
                "detected_domain": "inventory",
                "detected_tool": "get_stock_level",
            }

        def mock_extractor(state: AgentState) -> dict:
            active = list(state.get("active_tools") or [])
            for plan in active:
                if plan["tool"] == "get_stock_level":
                    plan["params"] = {"product_name": "kertas"}
                elif plan["tool"] == "get_account_balance":
                    plan["params"] = {"account_code": "111100"}
            return {
                "active_tools": active,
                "extracted_params": active[0].get("params"),
            }

        def mock_tool_node(state: AgentState) -> dict:
            return {
                "multi_tool_results": {
                    "get_stock_level": {"total_quantity": 150},
                    "get_account_balance": {"net_balance": 5000000.0},
                },
                "multi_verification_status": {
                    "get_stock_level": "VALID",
                    "get_account_balance": "VALID",
                },
                "tool_result": {"total_quantity": 150},
                "verification_status": "VALID",
            }

        workflow = StateGraph(AgentState)
        workflow.add_node("router", mock_router)
        workflow.add_node("extractor", mock_extractor)
        workflow.add_node("tool_node", mock_tool_node)

        workflow.add_edge(START, "router")
        workflow.add_edge("router", "extractor")
        workflow.add_edge("extractor", "tool_node")
        workflow.add_edge("tool_node", END)

        app = workflow.compile()
        initial_state = create_initial_state("Cek stok kertas dan saldo kas")
        result = app.invoke(initial_state)

        self.assertTrue(result["is_multi_tool"])
        self.assertEqual(len(result["active_tools"]), 2)
        self.assertEqual(result["active_tools"][0]["params"], {"product_name": "kertas"})
        self.assertEqual(result["multi_tool_results"]["get_stock_level"]["total_quantity"], 150)
        self.assertEqual(result["multi_tool_results"]["get_account_balance"]["net_balance"], 5000000.0)
        self.assertEqual(result["tool_result"]["total_quantity"], 150)
        self.assertEqual(result["verification_status"], "VALID")


class TestDecideAfterRouterEdge(unittest.TestCase):
    """Test suite validating decide_after_router with multi-tool active_tools."""

    def test_routes_to_extractor_for_multi_tool_active_tools(self):
        """Active tools list with multiple planned calls routes directly to extractor."""
        state = create_initial_state(user_query="Cek stok kertas dan saldo kas")
        state["is_multi_tool"] = True
        state["active_tools"] = [
            {"tool": "get_stock_level", "domain": "inventory"},
            {"tool": "get_account_balance", "domain": "accounting"},
        ]
        state["detected_domain"] = "inventory"
        state["detected_tool"] = "get_stock_level"
        self.assertEqual(decide_after_router(state), "extractor")

    def test_routes_to_extractor_when_active_tools_present_without_scalar_tool(self):
        """active_tools with ERP plans routes to extractor even if scalar detected_tool is None."""
        state = create_initial_state(user_query="Cek stok kertas dan saldo kas")
        state["is_multi_tool"] = True
        state["active_tools"] = [
            {"tool": "get_stock_level", "domain": "inventory"},
        ]
        state["detected_tool"] = None
        state["detected_domain"] = None
        self.assertEqual(decide_after_router(state), "extractor")

    def test_guardrail_precedence_over_active_tools(self):
        """Guardrail domain general routes to synthesizer even if active_tools is non-empty."""
        state = create_initial_state(user_query="Halo")
        state["detected_domain"] = "general"
        state["active_tools"] = [{"tool": "get_stock_level", "domain": "inventory"}]
        self.assertEqual(decide_after_router(state), "synthesizer")


class TestMultiDomainRouter(unittest.TestCase):
    """Test suite for multi-domain routing, planned tools models, and state integration."""

    def test_planned_tool_call_validator(self):
        """PlannedToolCall corrects domain to align with registered tool domain."""
        pt = PlannedToolCall(tool="get_stock_level", domain="accounting")
        self.assertEqual(pt.domain, "inventory")

    def test_multi_router_decision_reconcile_planned_tools(self):
        """MultiRouterDecision reconciles primary tool from planned_tools."""
        decision = MultiRouterDecision(
            domain="inventory",
            planned_tools=[PlannedToolCall(tool="get_stock_level", domain="inventory")],
        )
        self.assertEqual(decision.tool, "get_stock_level")

    def test_multi_router_decision_synthesizes_planned_tools(self):
        """MultiRouterDecision synthesizes planned_tools if tool is provided alone."""
        decision = MultiRouterDecision(
            domain="accounting",
            tool="get_account_balance",
        )
        self.assertEqual(len(decision.planned_tools), 1)
        self.assertEqual(decision.planned_tools[0].tool, "get_account_balance")

    def test_multi_router_decision_cleans_general(self):
        """MultiRouterDecision cleans tools when domain is general."""
        decision = MultiRouterDecision(
            domain="general",
            tool="get_stock_level",
            planned_tools=[PlannedToolCall(tool="get_stock_level", domain="inventory")],
        )
        self.assertIsNone(decision.tool)
        self.assertEqual(decision.planned_tools, [])

    def test_single_tool_retains_scalar_and_populates_active_tools(self):
        """Single-intent query retains exact scalar domain/tool and sets active_tools (len 1)."""
        state = create_initial_state("Cek sisa stok kertas foto A4 di gudang utama")
        result = router_node(state)

        self.assertEqual(result["detected_domain"], "inventory")
        self.assertEqual(result["detected_tool"], "get_stock_level")
        self.assertFalse(result.get("is_multi_tool", False))
        self.assertEqual(len(result.get("active_tools", [])), 1)
        self.assertEqual(result["active_tools"][0]["tool"], "get_stock_level")
        self.assertEqual(result["active_tools"][0]["domain"], "inventory")
        self.assertEqual(result.get("detected_domains"), ["inventory"])
        self.assertEqual(result.get("detected_tools"), ["get_stock_level"])

    def test_cross_module_stock_and_balance_multi_intent(self):
        """Cross-module query detects both inventory and accounting intents."""
        state = create_initial_state("Berapa stok kertas Art Paper dan berapa saldo kas kecil?")
        result = router_node(state)

        self.assertTrue(result.get("is_multi_tool"))
        self.assertEqual(len(result.get("active_tools", [])), 2)

        tools = [t["tool"] for t in result["active_tools"]]
        domains = [t["domain"] for t in result["active_tools"]]
        self.assertIn("get_stock_level", tools)
        self.assertIn("get_account_balance", tools)
        self.assertIn("inventory", domains)
        self.assertIn("accounting", domains)

        # Dual representation verification
        self.assertEqual(result["detected_tool"], result["active_tools"][0]["tool"])
        self.assertEqual(result["detected_domain"], result["active_tools"][0]["domain"])
        self.assertCountEqual(result["detected_domains"], ["inventory", "accounting"])
        self.assertCountEqual(result["detected_tools"], ["get_stock_level", "get_account_balance"])

    def test_cross_module_stock_and_vendor_invoice(self):
        """Cross-module query detects stock level and vendor invoice intents."""
        state = create_initial_state("Cek stok tinta sublim dan tagihan vendor PT Sumber Makmur bulan ini")
        result = router_node(state)

        self.assertTrue(result.get("is_multi_tool"))
        tools = [t["tool"] for t in result["active_tools"]]
        self.assertIn("get_stock_level", tools)
        self.assertIn("fetch_vendor_invoices", tools)

    def test_customer_billing_guardrail_remains_general_not_multi_tool(self):
        """Customer billing guardrail remains general and produces empty active_tools."""
        state = create_initial_state("Cek tagihan pelanggan PT Sumber Makmur bulan ini")
        result = router_node(state)

        self.assertEqual(result["detected_domain"], "general")
        self.assertIsNone(result.get("detected_tool"))
        self.assertFalse(result.get("is_multi_tool", False))
        self.assertEqual(result.get("active_tools", []), [])

    def test_conversational_retry_replays_multi_tool_query(self):
        """Conversational retry ('coba lagi') correctly replays prior multi-tool query."""
        state = create_initial_state("coba lagi")
        state["messages"] = [
            HumanMessage(content="Cek stok kertas foto dan saldo kas"),
            AIMessage(content="Data diproses"),
            HumanMessage(content="coba lagi"),
        ]
        result = router_node(state)

        self.assertTrue(result.get("is_multi_tool"))
        self.assertEqual(len(result.get("active_tools", [])), 2)
        self.assertIn("Mengulang permintaan:", result.get("retry_instruction", ""))


if __name__ == "__main__":
    unittest.main()
