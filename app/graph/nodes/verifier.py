"""
Data Verification Node for FastPrint ERP LangGraph Agent Workflow (TASK-4.4).
Evaluates tool execution results, implements an automated retry/correction cycle
for empty query results (with maximum 2 retries), and provides conditional edge routing.
"""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.core.context import (
    compute_multi_tool_counts,
    get_multi_tool_key,
    has_strict_document_identifier,
    resolve_tool_record_count,
)
from app.graph.state import AgentState, VerificationStatus

logger = logging.getLogger("verifier_node")


def _calculate_records_count(tool_result: Any) -> int:
    """Delegate record count inspection to unified context helper."""
    return resolve_tool_record_count(tool_result)


def _is_explicit_daily_existence(
    state: AgentState,
    tool_name: Optional[str],
    params: Optional[Dict[str, Any]],
) -> bool:
    """Return True when the user asked an explicit yes/no daily sales existence check."""
    user_q_low = (state.get("user_query") or "").lower()
    ext_params = params or {}
    return bool(
        (
            state.get("detected_domain") == "sales"
            or tool_name in ("get_top_selling_products", "get_sales_orders", "get_contacts", "get_sales_summary")
        )
        and (ext_params.get("date_from") or ext_params.get("date_to"))
        and re.search(r"\b(?:hari\s+ini|kemarin)\b", user_q_low)
        and re.search(
            r"\b(?:apakah\s+ada|ada\s+(?:tidak|penjualan|transaksi|order|quotation)|cek\s+(?:ada|apakah))\b",
            user_q_low,
        )
    )


def verifier_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph pipeline node that evaluates state['tool_result'] to verify data availability.

    Rules (TASK-4.4):
    1. If tool execution previously encountered an ERROR, preserve ERROR status.
    2. If records_count > 0: Set verification_status = "VALID".
    3. If records_count == 0 and retry_count < 2:
       Set verification_status = "RETRY", increment retry_count += 1,
       and provide staged filter correction instruction in retry_instruction.
    4. If records_count == 0 and retry_count >= 2:
       Set verification_status = "EMPTY".

    Args:
        state: Current AgentState.

    Returns:
        Partial AgentState update dictionary.
    """
    current_instruction = state.get("retry_instruction")
    preserved_replay = current_instruction if current_instruction and current_instruction.startswith("Mengulang permintaan:") else None

    multi_results = state.get("multi_tool_results") or {}
    multi_status = state.get("multi_verification_status") or {}
    active_tools = state.get("active_tools") or []
    is_single_tool_mc = (
        not state.get("is_multi_tool")
        and len(active_tools) <= 1
        and bool(state.get("is_multi_company"))
        and bool(state.get("multi_company_results"))
    )
    is_multi = state.get("is_multi_tool") or (len(active_tools) > 1) or (len(multi_results) > 1 and not is_single_tool_mc)

    # ------------------------------------------------------------------
    # Path MC: Verification for Multi-Company Execution (Single-Tool Only)
    # ------------------------------------------------------------------
    if is_single_tool_mc:
        mc_results = state.get("multi_company_results") or {}
        total_records = sum(_calculate_records_count(r) for r in mc_results.values() if r)
        unauth = state.get("unauthorized_companies") or []

        if total_records > 0:
            return {
                "verification_status": "VALID",
                "retry_instruction": None,
            }
        elif len(unauth) == len(mc_results) and len(unauth) > 0:
            return {
                "verification_status": "ERROR",
                "error_message": f"Akses ditolak ke seluruh cabang yang diminta (HTTP 403): {unauth}",
            }
        else:
            is_explicit_daily_existence = _is_explicit_daily_existence(
                state,
                state.get("detected_tool"),
                state.get("extracted_params"),
            )
            current_retry = int(state.get("retry_count") or 0)
            if current_retry < 2 and not is_explicit_daily_existence:
                return {
                    "verification_status": "RETRY",
                    "retry_count": current_retry + 1,
                    "retry_instruction": "Hasil query multi-cabang kosong (0 data). Longgarkan batasan tanggal untuk memperluas pencarian.",
                }
            return {
                "verification_status": "EMPTY",
                "retry_instruction": None,
            }

    # ------------------------------------------------------------------
    # Path A: Aggregate Verification for Multi-Tool Execution (Milestone 4)
    # ------------------------------------------------------------------
    if is_multi and (multi_results or active_tools):
        tool_counts = compute_multi_tool_counts(active_tools)

        items: List[Tuple[str, Any, Optional[str]]] = []

        if active_tools:
            for idx, plan in enumerate(active_tools):
                tn = plan.get("tool", "unnamed_tool")
                key = get_multi_tool_key(tn, idx, tool_counts)
                res = plan.get("result")
                if res is None and key in multi_results:
                    res = multi_results[key]
                elif res is None and tn in multi_results:
                    res = multi_results[tn]
                st = plan.get("status") or multi_status.get(key) or multi_status.get(tn)
                items.append((key, res, st))
        else:
            for key, res in multi_results.items():
                items.append((key, res, multi_status.get(key)))

        per_tool_status: Dict[str, str] = {}
        total_records = 0
        has_empty_tool = False
        has_error_tool = False
        has_valid_tool = False

        for idx, (key, res, st) in enumerate(items):
            plan = active_tools[idx] if idx < len(active_tools) else {}
            plan_mc_results = plan.get("multi_company_results") if isinstance(plan, dict) else None

            if isinstance(plan_mc_results, dict) and plan_mc_results:
                plan_unauth = plan.get("unauthorized_companies") or []
                mc_total = sum(_calculate_records_count(r) for r in plan_mc_results.values() if r)
                if len(plan_unauth) == len(plan_mc_results) and len(plan_unauth) > 0:
                    per_tool_status[key] = "ERROR"
                    has_error_tool = True
                elif mc_total > 0:
                    total_records += mc_total
                    per_tool_status[key] = "VALID"
                    has_valid_tool = True
                else:
                    per_tool_status[key] = "EMPTY"
                    plan_params = (plan.get("params") or {}) if isinstance(plan, dict) else {}
                    if not has_strict_document_identifier(plan_params):
                        has_empty_tool = True
                continue

            if st in ("error", "ERROR") or (res is None and (plan.get("error") if isinstance(plan, dict) else None)) or (res is None and state.get("error_message")):
                per_tool_status[key] = "ERROR"
                has_error_tool = True
            else:
                cnt = _calculate_records_count(res)
                total_records += cnt
                if cnt > 0:
                    per_tool_status[key] = "VALID"
                    has_valid_tool = True
                else:
                    per_tool_status[key] = "EMPTY"
                    plan_params = (active_tools[idx].get("params") or {}) if idx < len(active_tools) else {}
                    if not has_strict_document_identifier(plan_params):
                        has_empty_tool = True

        current_retry_count = int(state.get("retry_count") or 0)

        # 1. If all tools failed with error and none succeeded or returned empty data
        if has_error_tool and not has_valid_tool and not has_empty_tool:
            logger.warning("Multi-tool verification: all tools encountered errors -> ERROR.")
            return {
                "verification_status": "ERROR",
                "multi_verification_status": per_tool_status,
                "retry_instruction": preserved_replay,
            }

        # 2. If any tool returned empty and we have retry budget
        if has_empty_tool and current_retry_count < 2:
            next_retry = current_retry_count + 1
            if next_retry == 1:
                instruction = (
                    "Sebagian atau seluruh query data ERP kosong (0 data). Longgarkan batasan filter tanggal "
                    "(date_from / date_to) untuk memperluas pencarian transaksi."
                )
            else:
                instruction = (
                    "Hasil query masih kosong setelah pelonggaran tanggal. Hapus batasan pencarian "
                    "kata kunci spesifik untuk memeriksa transaksi umum."
                )
            logger.warning(f"Multi-tool verification empty for some tools. Triggering RETRY {next_retry}/2.")
            return {
                "verification_status": "RETRY",
                "multi_verification_status": per_tool_status,
                "retry_count": next_retry,
                "retry_instruction": instruction,
            }

        # 3. If at least one tool returned valid records (whether retries exhausted or other tools errored/empty)
        if total_records > 0 or has_valid_tool:
            if has_error_tool:
                logger.info("Multi-tool verification: partial data available with some tool errors -> VALID.")
            else:
                logger.info(f"Multi-tool verification successful: all {len(items)} tools found records.")
            return {
                "verification_status": "VALID",
                "multi_verification_status": per_tool_status,
                "retry_instruction": preserved_replay,
            }

        # 4. If all tools empty and retry limit reached
        logger.info("Multi-tool verification finished: all tools empty and retry limit reached -> EMPTY.")
        return {
            "verification_status": "EMPTY",
            "multi_verification_status": per_tool_status,
            "retry_instruction": preserved_replay,
        }

    # ------------------------------------------------------------------
    # Path B: Single-Tool Verification (Backward Compatibility)
    # ------------------------------------------------------------------
    # 1. Preserve pre-existing ERROR status from tool_node
    if state.get("verification_status") == "ERROR" or (
        state.get("error_message") and state.get("tool_result") is None
    ):
        logger.info("Preserving ERROR verification status from previous node failure.")
        return {
            "verification_status": "ERROR",
            "retry_instruction": preserved_replay,
        }

    tool_result = state.get("tool_result")
    records_count = _calculate_records_count(tool_result)
    current_retry_count = int(state.get("retry_count") or 0)

    # 2. Valid data found
    if records_count > 0:
        logger.info(f"Verification successful: {records_count} record(s) found.")
        return {
            "verification_status": "VALID",
            "retry_instruction": preserved_replay,
        }

    # Strict Identifier Guard: If querying a specific document number or specific product catalog item
    # and 0 records found, terminate immediately with EMPTY to avoid wasting retries on hallucinated relaxations
    # or dumping random generic catalog products.
    extracted_params = state.get("extracted_params") or {}
    tool = state.get("detected_tool")
    is_strict_product_search = (
        tool == "get_product_info"
        and isinstance(extracted_params, dict)
        and bool(extracted_params.get("code") or extracted_params.get("search") or extracted_params.get("product_id"))
    )
    is_strict_marketplace_lookup = (
        isinstance(extracted_params, dict)
        and (
            (tool == "get_kilat_order_diagnostics" and bool(extracted_params.get("reference")))
            or (
                tool in (
                    "get_account_health",
                    "get_marketplace_orders",
                    "get_product_mapping_status",
                    "get_stock_sync_simulation",
                    "get_sync_execution_logs",
                    "get_price_comparison",
                    "get_fee_and_escrow_anomalies",
                    "get_deduction_and_settlement_summary",
                    "get_cashflow_and_balance_summary",
                )
                and extracted_params.get("account_id") is not None
            )
        )
    )

    is_explicit_daily_existence = _is_explicit_daily_existence(
        state,
        tool,
        extracted_params if isinstance(extracted_params, dict) else None,
    )

    if (
        has_strict_document_identifier(extracted_params)
        or is_strict_product_search
        or is_strict_marketplace_lookup
        or is_explicit_daily_existence
    ):
        guard_type = (
            "Strict Daily Sales Existence Guard"
            if is_explicit_daily_existence
            else (
                "Strict Product Guard"
                if is_strict_product_search
                else ("Strict Marketplace Guard" if is_strict_marketplace_lookup else "Strict Identifier Guard")
            )
        )
        logger.info(f"Specific query returned 0 records -> EMPTY ({guard_type}).")
        return {
            "verification_status": "EMPTY",
            "retry_instruction": preserved_replay,
        }

    # 3. Empty data with remaining retry budget (retry_count < 2)
    if current_retry_count < 2:
        next_retry = current_retry_count + 1
        if next_retry == 1:
            instruction = (
                "Hasil query kosong (0 data). Longgarkan atau hilangkan batasan filter tanggal "
                "(date_from / date_to) untuk memperluas pencarian transaksi."
            )
        else:
            instruction = (
                "Hasil query masih kosong setelah pelonggaran tanggal. Hapus batasan pencarian "
                "kata kunci / nama vendor yang terlalu spesifik untuk memeriksa transaksi umum."
            )

        logger.warning(
            f"Verification empty (records_count=0). Triggering RETRY {next_retry}/2 with instruction: {instruction}"
        )
        return {
            "verification_status": "RETRY",
            "retry_count": next_retry,
            "retry_instruction": instruction,
        }

    # 4. Empty data and retry limit reached (retry_count >= 2)
    logger.info("Verification finished: 0 records found and retry limit (2x) reached -> EMPTY.")
    return {
        "verification_status": "EMPTY",
        "retry_instruction": preserved_replay,
    }


def decide_after_verification(state: AgentState) -> str:
    """
    Conditional edge routing helper function.

    Routing rules:
    - Returns 'extractor' if verification_status == 'RETRY'.
    - Returns 'synthesizer' if verification_status in ('VALID', 'EMPTY', 'ERROR').

    Args:
        state: Current AgentState.

    Returns:
        Target node identifier name ('extractor' or 'synthesizer').
    """
    status: Optional[VerificationStatus] = state.get("verification_status")

    if status == "RETRY":
        return "extractor"

    # Default to synthesizer for VALID, EMPTY, or ERROR
    return "synthesizer"
