"""
Unit tests for Phase 3 Golden Eval Dataset & Repeatable NLU Eval Runner (Task 3.1).
Validates dataset schema, full 36-tool coverage, offline eval runner execution,
zero guardrail regressions, and Phase 3 exit gate evaluation math.
"""
import unittest

from app.schemas.tools import TOOL_DOMAIN_MAP
from app.services.tool_registry_service import CORE_TOOL_DEFINITIONS
from scripts.eval_nlu_router import (
    DEFAULT_DATASET_PATH,
    evaluate_exit_gates,
    evaluate_golden_cases,
    format_markdown_report,
    load_golden_cases,
)


class TestGoldenEvalDatasetAndRunner(unittest.IsolatedAsyncioTestCase):
    """Validate golden dataset completeness and repeatable offline/gate evaluation."""

    def test_golden_dataset_schema_and_all_36_tools_coverage(self):
        cases = load_golden_cases(DEFAULT_DATASET_PATH)
        self.assertGreaterEqual(len(cases), 80)
        self.assertLessEqual(len(cases), 120)

        ids = [c["id"] for c in cases]
        self.assertEqual(len(ids), len(set(ids)), "Golden case IDs must be unique")

        covered_tools = set()
        guardrail_count = 0
        clarify_count = 0
        clear_count = 0

        for case in cases:
            self.assertTrue(case.get("query", "").strip())
            self.assertIn(
                case.get("expected_domain"),
                {"accounting", "inventory", "sales", "marketplace", "knowledge", "general", "out_of_scope", "clarification"},
            )
            self.assertIn(
                case.get("expected_query_mode"),
                {"live_erp", "vector_rag", "hybrid", "general", "out_of_scope", "contextual_analysis", "clarification"},
            )
            for t in case.get("expected_tools") or []:
                self.assertIn(t, TOOL_DOMAIN_MAP, f"Unknown tool '{t}' in case {case['id']}")
                covered_tools.add(t)
            if case.get("is_guardrail"):
                guardrail_count += 1
            if case.get("must_clarify"):
                clarify_count += 1
            else:
                clear_count += 1

        all_core_tools = {t["tool_name"] for t in CORE_TOOL_DEFINITIONS}
        self.assertEqual(covered_tools, all_core_tools, "Golden dataset must cover all 36 core ERP tools")
        self.assertGreaterEqual(guardrail_count, 10)
        self.assertGreaterEqual(clarify_count, 5)
        self.assertGreaterEqual(clear_count, 70)

    async def test_offline_evaluation_zero_guardrail_regressions_and_markdown_output(self):
        cases = load_golden_cases(DEFAULT_DATASET_PATH)
        report = await evaluate_golden_cases(cases, mode="offline")

        summary = report["summary"]
        self.assertEqual(summary["mode"], "offline")
        self.assertEqual(summary["total_cases"], len(cases))
        self.assertEqual(summary["guardrail_regressions"], 0)
        self.assertTrue(report["exit_gates"]["all_passed"])
        self.assertGreaterEqual(summary["retrieval_topk_recall_pct"], 80.0)

        md = format_markdown_report(report)
        self.assertIn("# NLU Router Evaluation Report", md)
        self.assertIn("zero_guardrail_regressions", md)
        self.assertIn("PASS", md)

    def test_live_exit_gate_thresholds(self):
        passing_summary = {
            "retrieval_topk_recall_pct": 99.0,
            "semantic_accuracy_pct": 96.5,
            "legacy_accuracy_pct": 91.0,
            "false_clarification_rate_pct": 1.5,
            "p95_latency_delta_ms": 620.0,
            "guardrail_regressions": 0,
        }
        pass_res = evaluate_exit_gates(passing_summary, mode="live")
        self.assertTrue(pass_res["all_passed"])
        self.assertEqual(len(pass_res["gates"]), 5)

        failing_summary = {
            "retrieval_topk_recall_pct": 97.5,
            "semantic_accuracy_pct": 94.0,
            "legacy_accuracy_pct": 95.0,
            "false_clarification_rate_pct": 4.2,
            "p95_latency_delta_ms": 910.0,
            "guardrail_regressions": 1,
        }
        fail_res = evaluate_exit_gates(failing_summary, mode="live")
        self.assertFalse(fail_res["all_passed"])
        for gate_info in fail_res["gates"].values():
            self.assertFalse(gate_info["passed"])


if __name__ == "__main__":
    unittest.main()
