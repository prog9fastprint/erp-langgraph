"""
Data Verification Node for FastPrint ERP LangGraph Agent Workflow (TASK-4.4).
Evaluates tool execution results, implements an automated retry/correction cycle
for empty query results (with maximum 2 retries), and provides conditional edge routing.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from app.core.context import (
    compute_multi_tool_counts,
    get_multi_tool_key,
    has_strict_document_identifier,
)
from app.graph.state import AgentState, VerificationStatus

logger = logging.getLogger("verifier_node")


def _calculate_records_count(tool_result: Any) -> int:
    """
    Hierarchically inspect and extract the record count from diverse ERP response structures.

    Handles:
    - total_count or count fields (e.g. Journal entries, Vendor invoices)
    - list entries/records (e.g. accounts, entries, products, items, locations)
    - root lists (e.g. multiple items array)
    - single record dictionary with recognized business attributes (e.g. stock quant, product, balance)
    - error or detail payload dicts, empty structures or None -> 0
    """
    if tool_result is None:
        return 0

    if isinstance(tool_result, list):
        return len(tool_result)

    if isinstance(tool_result, dict):
        if not tool_result:
            return 0

        # Check explicit error payloads from upstream ERP / DRF endpoints
        if (
            tool_result.get("error") is not None
            or tool_result.get("detail") is not None
            or tool_result.get("errors") is not None
            or tool_result.get("success") is False
        ):
            return 0

        # Check explicit count fields
        if "total_count" in tool_result and isinstance(tool_result["total_count"], (int, float)):
            return int(tool_result["total_count"])

        if "count" in tool_result and isinstance(tool_result["count"], (int, float)):
            return int(tool_result["count"])

        # Check common collection list keys
        for list_key in (
            "entries", "accounts", "products", "items", "records", "lines", "locations",
            "transfers", "history", "adjustments", "scraps", "warehouses", "categories", "alerts", "bills",
        ):
            val = tool_result.get(list_key)
            if isinstance(val, list):
                return len(val)

        # Check financial report summary or totals blocks (e.g. P&L, Balance Sheet, Bank Recon, Account Totals)
        if isinstance(tool_result.get("summary"), dict) and tool_result["summary"]:
            return 1
        if isinstance(tool_result.get("totals"), dict) and tool_result["totals"]:
            return 1

        # Nested data field inspection (if envelope wasn't stripped)
        if "data" in tool_result and isinstance(tool_result["data"], (dict, list)):
            return _calculate_records_count(tool_result["data"])

        # Specific check for aggregate balance payloads without account records
        balance_metric_keys = {
            "net_balance", "total_debit", "total_credit",
            "grand_net_balance", "grand_total_debit", "grand_total_credit",
            "total_revenue", "total_assets", "bank_statement_balance",
        }
        if any(k in balance_metric_keys for k in tool_result.keys()):
            has_value = any(float(v or 0) != 0.0 for v in tool_result.values() if isinstance(v, (int, float)))
            return 1 if has_value else 0

        # Single record entity check: verify presence of known ERP business record attributes
        KNOWN_RECORD_ATTRS = {
            "id",
            "product_id",
            "product_name",
            "sku",
            "code",
            "default_code",
            "barcode",
            "account_id",
            "account_code",
            "account_name",
            "location_id",
            "location_name",
            "partner_id",
            "partner_name",
            "vendor_id",
            "vendor_name",
            "invoice_id",
            "journal_id",
            "move_id",
            "reference",
            "name",
            "uom",
            "qty_on_hand",
            "quantity",
            "available_quantity",
            "virtual_available",
            "incoming_qty",
            "outgoing_qty",
            "net_balance",
            "total_debit",
            "total_credit",
            "grand_net_balance",
            "amount_total",
            "price_unit",
            "list_price",
            "standard_price",
            "debit",
            "credit",
            "balance",
            "total_revenue",
            "gross_profit",
            "net_profit",
            "cogs",
            "operating_profit",
            "gross_margin_percent",
            "total_assets",
            "total_liabilities",
            "total_equity",
            "total_liabilities_and_equity",
            "bank_statement_balance",
            "erp_book_balance",
            "residual_balance",
            "bill_date",
            "due_date",
            "due_status",
            "closing_balance",
            "running_balance",
        }
        has_business_attr = any(k in KNOWN_RECORD_ATTRS for k in tool_result.keys())
        has_non_none_value = any(v is not None for v in tool_result.values())
        if has_business_attr and has_non_none_value:
            return 1

        # Fallback for unrecognized payload dictionaries without ERP business attributes
        return 0

    return 0


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

    # ------------------------------------------------------------------
    # Path MC: Verification for Multi-Company Execution
    # ------------------------------------------------------------------
    if state.get("is_multi_company") and state.get("multi_company_results"):
        mc_results = state.get("multi_company_results") or {}
        total_records = sum(_calculate_records_count(r) for r in mc_results.values() if r)
        unauth = state.get("unauthorized_companies") or []

        if total_records > 0 or (len(mc_results) > len(unauth) and any(r is not None for r in mc_results.values())):
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
            current_retry = int(state.get("retry_count") or 0)
            if current_retry < 2:
                return {
                    "verification_status": "RETRY",
                    "retry_count": current_retry + 1,
                    "retry_instruction": "Hasil query multi-cabang kosong (0 data). Longgarkan batasan tanggal untuk memperluas pencarian.",
                }
            return {
                "verification_status": "EMPTY",
                "retry_instruction": None,
            }

    multi_results = state.get("multi_tool_results") or {}
    multi_status = state.get("multi_verification_status") or {}
    active_tools = state.get("active_tools") or []
    is_multi = state.get("is_multi_tool") or (len(multi_results) > 1) or (len(active_tools) > 1)

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
            if st in ("error", "ERROR") or (res is None and state.get("error_message")):
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

    # Strict Identifier Guard: If querying a specific document number and 0 records found,
    # terminate immediately with EMPTY to avoid wasting retries on hallucinated relaxations.
    extracted_params = state.get("extracted_params") or {}
    if has_strict_document_identifier(extracted_params):
        logger.info("Specific document identifier search returned 0 records -> EMPTY (Strict Identifier Guard).")
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
