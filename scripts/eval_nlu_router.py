"""
Repeatable Golden Evaluation Runner for FastPrint ERP NLU Modernization (Phase 3).

Supports two execution modes:
  - --mode offline : Zero-API-cost evaluation of golden dataset schema, hard-guardrail safety,
                     in-memory hybrid retrieval Top-K recall, and legacy_regex baseline accuracy.
  - --mode live    : Full staging/production benchmark comparing legacy_regex vs semantic_llm
                     (live PGVector + configured LLM) against all 5 Phase 3 exit gates:
                       1. Retrieval Top-K Recall >= 98.0%
                       2. Semantic Routing Accuracy >= 95.0% and >= Legacy Accuracy
                       3. False-Clarification Rate <= 3.0%
                       4. P95 Additional Router Latency <= 800ms
                       5. Zero Guardrail Regressions (== 0)

Usage:
    python scripts/eval_nlu_router.py --mode offline
    python scripts/eval_nlu_router.py --mode live --output eval_report.json --markdown eval_report.md --fail-on-gate
"""
import argparse
import asyncio
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set
from unittest.mock import patch

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

# Ensure repository root is importable when invoked as `python scripts/eval_nlu_router.py`
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.config import settings  # noqa: E402
from app.graph.nodes.router import (  # noqa: E402
    _enforce_general_vs_out_of_scope,
    _try_route_contextual_analysis,
    arouter_node,
)
from app.graph.nodes.router_decomposer import decompose_user_query  # noqa: E402
from app.graph.nodes.router_semantic import (  # noqa: E402
    aretrieve_candidate_tools,
    aroute_semantic_nlu,
    check_hard_guardrails,
)
from app.graph.state import create_initial_state  # noqa: E402
from app.services.llm import get_chat_model  # noqa: E402

DEFAULT_DATASET_PATH = REPO_ROOT / "tests" / "fixtures" / "nlu_golden_cases.json"

EXIT_GATES = {
    "min_retrieval_recall_pct": 98.0,
    "min_semantic_accuracy_pct": 95.0,
    "max_false_clarification_pct": 3.0,
    "max_p95_latency_delta_ms": 800.0,
    "max_guardrail_regressions": 0,
}


def load_golden_cases(path: Path = DEFAULT_DATASET_PATH) -> List[Dict[str, Any]]:
    """Load golden evaluation cases JSON list from disk."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected JSON array in {path}")
    return raw


def _build_state_for_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """Construct AgentState for a single golden case including optional multi-turn history."""
    msgs: List[BaseMessage] = []
    for item in case.get("history") or []:
        role = (item.get("role") or "user").lower()
        content = str(item.get("content") or "")
        if role in ("assistant", "ai"):
            msgs.append(AIMessage(content=content))
        else:
            msgs.append(HumanMessage(content=content))
    msgs.append(HumanMessage(content=case["query"]))

    state = create_initial_state(
        user_query=case["query"],
        company_id=int(case.get("company_id") or 1),
        messages=msgs,
        user_roles=case.get("user_roles") or [],
    )
    if case.get("prior_detected_tool"):
        state["prior_detected_tool"] = case["prior_detected_tool"]
    if case.get("prior_detected_domain"):
        state["prior_detected_domain"] = case["prior_detected_domain"]
    return state


def _acceptable_tool_sets(case: Dict[str, Any]) -> List[Set[str]]:
    """Return list of acceptable tool-name sets for matching a golden case."""
    acceptable = case.get("acceptable_tools")
    if acceptable and isinstance(acceptable, list):
        return [set(group) for group in acceptable]
    return [set(case.get("expected_tools") or [])]


def _is_route_match(case: Dict[str, Any], result: Optional[Dict[str, Any]]) -> bool:
    """Evaluate whether a router output matches the golden case expectation."""
    if not isinstance(result, dict):
        return False
    must_clarify = bool(case.get("must_clarify", False))
    got_clarify = bool(result.get("needs_clarification") or result.get("query_mode") == "clarification")
    if must_clarify:
        return got_clarify

    if got_clarify:
        return False

    expected_mode = case.get("expected_query_mode")
    expected_domain = case.get("expected_domain")
    got_mode = result.get("query_mode")
    got_domain = result.get("detected_domain")
    got_tools = set(result.get("detected_tools") or ([] if not result.get("detected_tool") else [result["detected_tool"]]))

    if expected_mode in ("general", "out_of_scope"):
        return got_domain == expected_domain and not got_tools

    if expected_mode == "vector_rag":
        return got_mode == "vector_rag" or got_domain == "knowledge"

    if expected_mode == "hybrid":
        if got_mode != "hybrid":
            return False
        return any(req.issubset(got_tools) for req in _acceptable_tool_sets(case))

    # live_erp
    return any(got_tools == req for req in _acceptable_tool_sets(case))


def _is_retrieval_hit(case: Dict[str, Any], candidates: List[Dict[str, Any]]) -> bool:
    """Return True when every expected tool in at least one acceptable set appears in Top-K candidates."""
    cand_names = {c.get("tool_name") for c in candidates if isinstance(c, dict)}
    return any(req.issubset(cand_names) for req in _acceptable_tool_sets(case))


def _percentile(values: List[float], p: float) -> float:
    """Compute nearest-rank percentile p in [0, 100]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil((p / 100.0) * len(ordered)) - 1))
    return round(ordered[idx], 2)


def evaluate_exit_gates(summary: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """Check computed benchmark summary against Phase 3 cutover exit gates."""
    gates: Dict[str, Dict[str, Any]] = {}

    grd_regressions = int(summary.get("guardrail_regressions", 0))
    gates["zero_guardrail_regressions"] = {
        "passed": grd_regressions <= EXIT_GATES["max_guardrail_regressions"],
        "actual": grd_regressions,
        "threshold": EXIT_GATES["max_guardrail_regressions"],
    }

    recall = float(summary.get("retrieval_topk_recall_pct", 0.0))
    if mode == "live":
        gates["retrieval_topk_recall"] = {
            "passed": recall >= EXIT_GATES["min_retrieval_recall_pct"],
            "actual": recall,
            "threshold": EXIT_GATES["min_retrieval_recall_pct"],
        }
        sem_acc = float(summary.get("semantic_accuracy_pct", 0.0))
        leg_acc = float(summary.get("legacy_accuracy_pct", 0.0))
        gates["semantic_routing_accuracy"] = {
            "passed": sem_acc >= EXIT_GATES["min_semantic_accuracy_pct"] and sem_acc >= leg_acc,
            "actual": sem_acc,
            "legacy_baseline": leg_acc,
            "threshold": EXIT_GATES["min_semantic_accuracy_pct"],
        }
        fc_rate = float(summary.get("false_clarification_rate_pct", 0.0))
        gates["false_clarification_rate"] = {
            "passed": fc_rate <= EXIT_GATES["max_false_clarification_pct"],
            "actual": fc_rate,
            "threshold": EXIT_GATES["max_false_clarification_pct"],
        }
        p95_delta = float(summary.get("p95_latency_delta_ms", 0.0))
        gates["p95_router_latency_delta"] = {
            "passed": p95_delta <= EXIT_GATES["max_p95_latency_delta_ms"],
            "actual": p95_delta,
            "threshold": EXIT_GATES["max_p95_latency_delta_ms"],
        }

    all_passed = all(g["passed"] for g in gates.values())
    return {"all_passed": all_passed, "gates": gates}


async def evaluate_golden_cases(
    cases: List[Dict[str, Any]],
    mode: str = "offline",
    llm: Optional[Any] = None,
    concurrency: int = 5,
    show_progress: bool = False,
) -> Dict[str, Any]:
    """Run repeatable evaluation over golden cases in 'offline' or 'live' mode."""
    live_model = llm
    if mode == "live" and live_model is None:
        live_model = get_chat_model()

    effective_concurrency = max(1, int(concurrency)) if mode == "live" else 1
    sem = asyncio.Semaphore(effective_concurrency)
    total_count = len(cases)
    completed_count = 0

    async def _eval_one(case: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal completed_count
        async with sem:
            state = _build_state_for_case(case)
            query = case["query"]
            clauses = decompose_user_query(query) or [query]
            expected_tools = case.get("expected_tools") or []
            is_guardrail = bool(case.get("is_guardrail", False))
            must_clarify = bool(case.get("must_clarify", False))

            # 1. Legacy Regex Baseline
            t0 = time.perf_counter()
            with patch.object(settings, "NLU_ROUTER_MODE", "legacy_regex"):
                legacy_res = await arouter_node(state, llm=None)
            legacy_ms = (time.perf_counter() - t0) * 1000.0
            legacy_match = None if must_clarify else _is_route_match(case, legacy_res)

            # 2. Semantic Router + Single-Pass Retrieval Recall
            semantic_res: Optional[Dict[str, Any]] = None
            semantic_ms = 0.0
            semantic_match: Optional[bool] = None
            retrieval_hit: Optional[bool] = None
            retrieved_names: List[str] = []

            if mode == "offline":
                if expected_tools and not must_clarify:
                    with patch.object(settings, "DYNAMIC_TOOL_RETRIEVAL_ENABLED", False):
                        cands = await aretrieve_candidate_tools(query, clauses, state)
                    retrieved_names = [c.get("tool_name") for c in cands if isinstance(c, dict)]
                    hard_res = check_hard_guardrails(
                        query, state, clauses, contextual_analysis_fn=_try_route_contextual_analysis
                    )
                    if hard_res and hard_res.get("detected_tool") in set(expected_tools):
                        retrieval_hit = True
                    else:
                        retrieval_hit = _is_retrieval_hit(case, cands)

                if is_guardrail:
                    hard_res = check_hard_guardrails(
                        query, state, clauses, contextual_analysis_fn=_try_route_contextual_analysis
                    )
                    semantic_res = hard_res
                    semantic_match = _is_route_match(case, hard_res)
            else:
                t1 = time.perf_counter()
                semantic_res = await aroute_semantic_nlu(
                    state=state,
                    user_query=query,
                    get_model_fn=lambda: live_model,
                    contextual_analysis_fn=_try_route_contextual_analysis,
                    enforce_general_fn=_enforce_general_vs_out_of_scope,
                )
                semantic_ms = (time.perf_counter() - t1) * 1000.0
                semantic_match = _is_route_match(case, semantic_res)

                if expected_tools and not must_clarify:
                    avail = (semantic_res or {}).get("available_tools") or []
                    retrieved_names = [c.get("tool_name") for c in avail if isinstance(c, dict)]
                    if (semantic_res or {}).get("detected_tool") in set(expected_tools) and avail:
                        retrieval_hit = _is_retrieval_hit(case, avail) or (
                            set(expected_tools).issubset(set((semantic_res or {}).get("detected_tools") or []))
                        )
                    else:
                        retrieval_hit = _is_retrieval_hit(case, avail)

            completed_count += 1
            if show_progress and mode == "live":
                status_tag = "OK" if semantic_match else "MISS"
                print(
                    f"[{completed_count}/{total_count}] {case['id']} {status_tag} ({semantic_ms:.0f}ms)",
                    file=sys.stderr,
                    flush=True,
                )

            return {
                "id": case["id"],
                "category": case.get("category"),
                "query": query,
                "expected_domain": case.get("expected_domain"),
                "expected_query_mode": case.get("expected_query_mode"),
                "expected_tools": expected_tools,
                "must_clarify": must_clarify,
                "is_guardrail": is_guardrail,
                "retrieval_hit": retrieval_hit,
                "retrieved_tools": retrieved_names,
                "legacy_match": legacy_match,
                "legacy_tools": legacy_res.get("detected_tools") if isinstance(legacy_res, dict) else [],
                "legacy_domain": legacy_res.get("detected_domain") if isinstance(legacy_res, dict) else None,
                "legacy_ms": round(legacy_ms, 2),
                "semantic_match": semantic_match,
                "semantic_tools": semantic_res.get("detected_tools") if isinstance(semantic_res, dict) else [],
                "semantic_domain": semantic_res.get("detected_domain") if isinstance(semantic_res, dict) else None,
                "semantic_needs_clarification": bool(
                    isinstance(semantic_res, dict)
                    and (semantic_res.get("needs_clarification") or semantic_res.get("query_mode") == "clarification")
                ),
                "semantic_ms": round(semantic_ms, 2),
            }

    if effective_concurrency > 1:
        case_reports = list(await asyncio.gather(*(_eval_one(c) for c in cases)))
    else:
        case_reports = [await _eval_one(c) for c in cases]

    retrieval_total = sum(1 for r in case_reports if r["retrieval_hit"] is not None)
    retrieval_hits = sum(1 for r in case_reports if r["retrieval_hit"] is True)
    legacy_total = sum(1 for r in case_reports if r["legacy_match"] is not None)
    legacy_hits = sum(1 for r in case_reports if r["legacy_match"] is True)

    legacy_latencies_ms = [float(r["legacy_ms"]) for r in case_reports]
    semantic_latencies_ms = [float(r["semantic_ms"]) for r in case_reports] if mode == "live" else []

    if mode == "offline":
        semantic_total = 0
        semantic_hits = 0
        clarify_total = 0
        clarify_hits = 0
        clear_total = 0
        false_clarifications = 0
        guardrail_total = sum(1 for r in case_reports if r["is_guardrail"])
        guardrail_regressions = sum(1 for r in case_reports if r["is_guardrail"] and not r["semantic_match"])
    else:
        semantic_total = len(case_reports)
        semantic_hits = sum(1 for r in case_reports if r["semantic_match"] is True)
        clarify_total = sum(1 for r in case_reports if r["must_clarify"])
        clarify_hits = sum(1 for r in case_reports if r["must_clarify"] and r["semantic_needs_clarification"])
        clear_total = sum(1 for r in case_reports if not r["must_clarify"])
        false_clarifications = sum(
            1 for r in case_reports if not r["must_clarify"] and r["semantic_needs_clarification"]
        )
        guardrail_total = sum(1 for r in case_reports if r["is_guardrail"])
        guardrail_regressions = sum(1 for r in case_reports if r["is_guardrail"] and not r["semantic_match"])

    def _pct(num: int, den: int) -> float:
        return round((num / den) * 100.0, 2) if den > 0 else 0.0

    legacy_p95 = _percentile(legacy_latencies_ms, 95)
    semantic_p50 = _percentile(semantic_latencies_ms, 50)
    semantic_p95 = _percentile(semantic_latencies_ms, 95)

    summary: Dict[str, Any] = {
        "mode": mode,
        "total_cases": len(cases),
        "retrieval_cases": retrieval_total,
        "retrieval_topk_recall_pct": _pct(retrieval_hits, retrieval_total),
        "legacy_cases": legacy_total,
        "legacy_accuracy_pct": _pct(legacy_hits, legacy_total),
        "semantic_cases": semantic_total,
        "semantic_accuracy_pct": _pct(semantic_hits, semantic_total),
        "clarification_cases": clarify_total,
        "clarification_recall_pct": _pct(clarify_hits, clarify_total),
        "clear_cases": clear_total,
        "false_clarification_count": false_clarifications,
        "false_clarification_rate_pct": _pct(false_clarifications, clear_total),
        "guardrail_cases": guardrail_total,
        "guardrail_regressions": guardrail_regressions,
        "legacy_p95_ms": legacy_p95,
        "semantic_p50_ms": semantic_p50,
        "semantic_p95_ms": semantic_p95,
        "p95_latency_delta_ms": round(max(0.0, semantic_p95 - legacy_p95), 2) if mode == "live" else 0.0,
    }
    gate_eval = evaluate_exit_gates(summary, mode=mode)
    return {
        "summary": summary,
        "exit_gates": gate_eval,
        "cases": case_reports,
    }


def format_markdown_report(report: Dict[str, Any]) -> str:
    """Format benchmark report as concise GitHub-flavored Markdown."""
    s = report["summary"]
    g = report["exit_gates"]
    lines = [
        f"# NLU Router Evaluation Report (`mode={s['mode']}`)",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| Total Golden Cases | {s['total_cases']} |",
        f"| Retrieval Top-K Recall | {s['retrieval_topk_recall_pct']}% ({s['retrieval_cases']} cases) |",
        f"| Legacy Regex Accuracy | {s['legacy_accuracy_pct']}% ({s['legacy_cases']} cases) |",
        f"| Semantic LLM Accuracy | {s['semantic_accuracy_pct']}% ({s['semantic_cases']} cases) |",
        f"| Clarification Recall | {s['clarification_recall_pct']}% ({s['clarification_cases']} cases) |",
        f"| False-Clarification Rate | {s['false_clarification_rate_pct']}% ({s['false_clarification_count']}/{s['clear_cases']}) |",
        f"| Guardrail Regressions | {s['guardrail_regressions']} / {s['guardrail_cases']} |",
        f"| Legacy P95 Latency | {s['legacy_p95_ms']} ms |",
        f"| Semantic Latency | P50 {s.get('semantic_p50_ms', 0.0)} ms / P95 {s['semantic_p95_ms']} ms (delta: {s['p95_latency_delta_ms']} ms) |",
        "",
        "## Exit Gates",
        "",
        "| Gate | Status | Actual | Threshold |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for name, info in g["gates"].items():
        status = "PASS" if info["passed"] else "FAIL"
        lines.append(f"| `{name}` | {status} | {info['actual']} | {info['threshold']} |")
    return "\n".join(lines) + "\n"


async def _cli_main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate FastPrint ERP NLU Router against golden cases.")
    parser.add_argument("--mode", choices=["offline", "live"], default="offline")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of cases to run.")
    parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent cases in live mode (default 5).")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write full JSON report.")
    parser.add_argument("--markdown", type=Path, default=None, help="Optional path to write Markdown summary.")
    parser.add_argument("--fail-on-gate", action="store_true", help="Exit with code 1 if any exit gate fails.")
    args = parser.parse_args(argv)

    cases = load_golden_cases(args.dataset)
    if args.limit is not None and args.limit > 0:
        cases = cases[: args.limit]

    report = await evaluate_golden_cases(
        cases,
        mode=args.mode,
        concurrency=args.concurrency,
        show_progress=(args.mode == "live"),
    )
    md = format_markdown_report(report)
    print(md)

    if args.output:
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.markdown:
        args.markdown.write_text(md, encoding="utf-8")

    if args.fail_on_gate and not report["exit_gates"]["all_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_cli_main()))
