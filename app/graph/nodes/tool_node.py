"""
Tool Execution Node for FastPrint ERP LangGraph Agent Workflow (TASK-4.3).
Executes ERP tools via DjangoCoreClient using dynamic dispatching, multi-company prioritization,
graceful error capture, and preserves transaction data and calculations block intact.
"""
import asyncio
import concurrent.futures
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.core.context import (
    clean_product_identifier,
    compute_multi_tool_counts,
    get_multi_tool_key,
    sanitize_erp_payload,
    should_preserve_plan_on_retry,
)
from app.graph.state import AgentState
from app.services.django_client import (
    DjangoCoreClient,
    DjangoCoreClientError,
)

logger = logging.getLogger("tool_node")


async def _execute_single_tool(
    tool_name: str,
    params: Dict[str, Any],
    erp_client: DjangoCoreClient,
    user_token: Optional[str],
    default_company_id: Optional[int],
    dispatch_map: Dict[str, Any],
    user_query: str = "",
    allow_empty_stock_params: bool = False,
) -> Tuple[Any, Optional[str]]:
    """Execute a single ERP tool, handling parameter adaptation, location resolution, and error capturing."""
    clean_params = dict(params or {})
    target_company_id = clean_params.get("company_id") or default_company_id

    # Adapt parameters for balance sheet cut-off date
    if tool_name == "get_balance_sheet":
        if "date_to" in clean_params and not clean_params.get("as_of_date"):
            clean_params["as_of_date"] = clean_params["date_to"]

    # Adapt parameters for vendor invoices/journal entries
    if tool_name in ("fetch_vendor_invoices", "get_journal_entries"):
        if "vendor_name" in clean_params:
            current_search = clean_params.get("search")
            if not current_search or clean_params["vendor_name"].lower() in current_search.lower():
                clean_params["search"] = clean_params["vendor_name"]
        elif "search" in clean_params and not clean_params.get("vendor_name"):
            cand = str(clean_params["search"]).strip()
            # Do not promote search to vendor_name if it is a document number pattern
            if not re.search(r"\b(?:[A-Za-z0-9_\-]+/)+\d{4}/\d+\b", cand) and not any(cand.upper().startswith(p) for p in ("JE/", "PC/", "INV/", "BILL/")):
                clean_params["vendor_name"] = clean_params["search"]

    # Adapt parameters for stock level inquiries
    if tool_name == "get_stock_level":
        if "search" in clean_params and not clean_params.get("code"):
            clean_params["code"] = clean_params["search"]
        if "sku" in clean_params and not clean_params.get("code"):
            clean_params["code"] = clean_params["sku"]
        if "product_code" in clean_params and not clean_params.get("code"):
            clean_params["code"] = clean_params["product_code"]

    # Sanitize product identifier fields across inventory and product tools
    for k in ("code", "search", "sku", "product_code"):
        if k in clean_params and isinstance(clean_params[k], str):
            clean_params[k] = clean_product_identifier(clean_params[k])
    if "codes" in clean_params and isinstance(clean_params["codes"], list):
        clean_params["codes"] = [
            clean_product_identifier(c) for c in clean_params["codes"] if c and isinstance(c, str)
        ]

    # Pre-lookup for stock location_name -> location_id or warehouse_id
    if tool_name == "get_stock_level" and clean_params.get("location_name") and not clean_params.get("location_id") and not clean_params.get("warehouse_id"):
        try:
            loc_res = await erp_client.fetch_locations(
                params={"search": clean_params["location_name"]},
                user_token=user_token,
                company_id=target_company_id,
            )
            locations = loc_res.get("locations", []) if isinstance(loc_res, dict) else (loc_res if isinstance(loc_res, list) else [])
            if locations:
                matched_loc = locations[0]
                clean_params["location_id"] = matched_loc.get("id")
                clean_params["matched_location_name"] = matched_loc.get("name")
            else:
                # Fallback to warehouse lookup if location was not found
                wh_res = await erp_client.fetch_warehouses(
                    params={"search": clean_params["location_name"]},
                    user_token=user_token,
                    company_id=target_company_id,
                )
                warehouses = wh_res.get("warehouses", []) if isinstance(wh_res, dict) else (wh_res if isinstance(wh_res, list) else [])
                if warehouses:
                    matched_wh = warehouses[0]
                    clean_params["warehouse_id"] = matched_wh.get("id")
                    clean_params["matched_location_name"] = matched_wh.get("name")
        except Exception as loc_err:
            logger.warning(f"Could not resolve location_name '{clean_params.get('location_name')}': {loc_err}")

    # Auto-resolve branch warehouse or fail-fast locally when stock query has no identifier
    if tool_name == "get_stock_level":
        has_product_identifier = bool(
            clean_params.get("product_id") is not None
            or clean_params.get("code")
            or clean_params.get("codes")
            or clean_params.get("product_ids")
        )
        has_resolved_location = bool(
            clean_params.get("warehouse_id") is not None
            or clean_params.get("location_id") is not None
        )
        has_location_name = bool(clean_params.get("location_name"))

        if not has_product_identifier and not has_resolved_location:
            from app.graph.nodes.extractor import resolve_branch_company_id

            has_explicit_branch = bool(
                user_query and resolve_branch_company_id(user_query) is not None
            )
            if has_explicit_branch:
                try:
                    wh_res = await erp_client.fetch_warehouses(
                        params={},
                        user_token=user_token,
                        company_id=target_company_id,
                    )
                    warehouses = (
                        wh_res.get("warehouses", [])
                        if isinstance(wh_res, dict)
                        else (wh_res if isinstance(wh_res, list) else [])
                    )
                    if warehouses and isinstance(warehouses[0], dict) and warehouses[0].get("id") is not None:
                        clean_params["warehouse_id"] = warehouses[0]["id"]
                        clean_params["matched_location_name"] = warehouses[0].get("name")
                        has_resolved_location = True
                except Exception as wh_err:
                    logger.warning(
                        "Could not auto-resolve branch warehouse for company_id=%s: %s",
                        target_company_id,
                        wh_err,
                    )

            if not allow_empty_stock_params and not has_resolved_location and not has_location_name:
                return (
                    None,
                    "MISSING_PRODUCT_IDENTIFIER: Mohon sertakan kode SKU, nama produk, atau nama gudang/cabang.",
                )

    handler = dispatch_map.get(tool_name)
    if not handler:
        return None, f"Tool '{tool_name}' not registered in dispatch map"

    try:
        raw_result = await handler(
            params=clean_params,
            user_token=user_token,
            company_id=target_company_id,
        )
    except DjangoCoreClientError as err:
        logger.warning(f"DjangoCoreClientError in _execute_single_tool for tool '{tool_name}': {err}")
        return None, str(err)
    except Exception as exc:
        logger.error(f"Unexpected error in _execute_single_tool for tool '{tool_name}': {exc}", exc_info=True)
        return None, f"Unexpected execution error: {str(exc)}"

    if isinstance(raw_result, dict):
        loc_label = clean_params.get("matched_location_name") or clean_params.get("location_name")
        if loc_label and not raw_result.get("location_name"):
            raw_result["location_name"] = loc_label

    final_result = (
        sanitize_erp_payload(raw_result)
        if settings.ENABLE_PAYLOAD_SANITIZATION
        else raw_result
    )
    return final_result, None


async def tool_node(
    state: AgentState,
    client: Optional[DjangoCoreClient] = None,
) -> Dict[str, Any]:
    """
    Asynchronous LangGraph node that executes ERP tools based on state parameters and routing.
    Supports both single-tool legacy workflows and concurrent multi-tool subagent execution (Milestone 3).

    Definition of Done:
    - Passes user_token and company_id to Django Core.
    - Preserves transaction data and the calculations block intact in state['tool_result'] and state['multi_tool_results'].
    - Concurrently executes multi-tool plans via asyncio.gather while isolating individual failures.
    - Captures errors gracefully into state['error_message'] and sets verification_status='ERROR'.
    """
    user_token = state.get("user_token")
    own_client = client is None
    erp_client = client or DjangoCoreClient(require_user_token=settings.REQUIRE_USER_TOKEN)

    try:
        # Dynamic Tool Dispatch Registry
        dispatch_map = {
            "fetch_vendor_invoices": erp_client.fetch_vendor_invoices,
            "get_journal_entries": erp_client.fetch_journal_entries,
            "get_account_balance": erp_client.fetch_account_balance,
            "get_general_ledger": erp_client.fetch_general_ledger,
            "get_profit_and_loss": erp_client.fetch_profit_and_loss,
            "get_balance_sheet": erp_client.fetch_balance_sheet,
            "get_vendor_bills": erp_client.fetch_vendor_bills,
            "get_bank_reconciliation_summary": erp_client.fetch_bank_reconciliation_summary,
            "get_stock_level": erp_client.fetch_stock_level,
            "get_product_info": erp_client.fetch_product_info,
            "get_locations": erp_client.fetch_locations,
            "get_stock_history": erp_client.fetch_stock_history,
            "get_transfers": erp_client.fetch_transfers,
            "get_low_stock_alerts": erp_client.fetch_low_stock_alerts,
            "get_stock_adjustments": erp_client.fetch_stock_adjustments,
            "get_scrap_records": erp_client.fetch_scrap_records,
            "get_warehouses": erp_client.fetch_warehouses,
            "get_product_categories": erp_client.fetch_product_categories,
        }

        active_tools = state.get("active_tools")
        is_multi_tool = state.get("is_multi_tool") or (bool(active_tools) and len(active_tools) > 1)

        # ------------------------------------------------------------------
        # Path A: Multi-Tool Execution via asyncio.gather (Milestone 3)
        # ------------------------------------------------------------------
        if is_multi_tool and active_tools:
            tool_counts = compute_multi_tool_counts(active_tools)
            retry_count = int(state.get("retry_count") or 0)
            prev_multi_status = state.get("multi_verification_status") or {}

            async def _reuse_cached_result(cached_res: Any) -> Tuple[Any, Optional[str]]:
                return cached_res, None

            tasks = []
            for idx, plan in enumerate(active_tools):
                t_name = plan.get("tool", "unnamed_tool")
                key = get_multi_tool_key(t_name, idx, tool_counts)
                if (
                    retry_count > 0
                    and plan.get("status") == "success"
                    and plan.get("result") is not None
                    and should_preserve_plan_on_retry(plan, key, prev_multi_status)
                ):
                    tasks.append(_reuse_cached_result(plan.get("result")))
                else:
                    tasks.append(
                        _execute_single_tool(
                            tool_name=plan.get("tool", "fetch_vendor_invoices"),
                            params=plan.get("params") or {},
                            erp_client=erp_client,
                            user_token=user_token,
                            default_company_id=state.get("company_id"),
                            dispatch_map=dispatch_map,
                            user_query=plan.get("sub_query") or state.get("user_query") or "",
                            allow_empty_stock_params=not bool(plan.get("sub_query")),
                        )
                    )
            results = await asyncio.gather(*tasks, return_exceptions=True)

            multi_tool_results: Dict[str, Any] = {}
            multi_verification_status: Dict[str, str] = {}
            updated_active_tools: List[Dict[str, Any]] = []
            errors: List[str] = []

            for idx, (plan, res) in enumerate(zip(active_tools, results)):
                p_copy = dict(plan)
                t_name = p_copy.get("tool", "unnamed_tool")
                key = get_multi_tool_key(t_name, idx, tool_counts)

                if isinstance(res, Exception):
                    err_msg = str(res)
                    multi_tool_results[key] = None
                    multi_verification_status[key] = "ERROR"
                    p_copy["status"] = "error"
                    p_copy["error"] = err_msg
                    errors.append(f"{t_name}: {err_msg}")
                else:
                    final_res, err = res
                    if err:
                        multi_tool_results[key] = None
                        multi_verification_status[key] = "ERROR"
                        p_copy["status"] = "error"
                        p_copy["error"] = err
                        errors.append(f"{t_name}: {err}")
                    else:
                        multi_tool_results[key] = final_res
                        multi_verification_status[key] = "VALID"
                        p_copy["status"] = "success"
                        p_copy["result"] = final_res

                updated_active_tools.append(p_copy)

            primary_tool = state.get("detected_tool") or active_tools[0].get("tool", "fetch_vendor_invoices")
            primary_result = multi_tool_results.get(primary_tool)
            if primary_result is None and f"{primary_tool}_0" in multi_tool_results:
                primary_result = multi_tool_results.get(f"{primary_tool}_0")
            if primary_result is None:
                primary_result = next((r for r in multi_tool_results.values() if r is not None), None)

            has_success = any(st == "VALID" for st in multi_verification_status.values())
            all_failed = not has_success and len(multi_verification_status) > 0

            return {
                "active_tools": updated_active_tools,
                "multi_tool_results": multi_tool_results,
                "multi_verification_status": multi_verification_status,
                "tool_result": primary_result,
                "verification_status": "ERROR" if all_failed else "VALID",
                "error_message": "; ".join(errors) if all_failed else (errors[0] if errors else None),
            }

        # ------------------------------------------------------------------
        # Path B: Single-Tool Legacy Execution (Full Backward Compatibility)
        # ------------------------------------------------------------------
        extracted_params = dict(state.get("extracted_params") or {})
        detected_tool = state.get("detected_tool") or "fetch_vendor_invoices"
        target_company_id = extracted_params.get("company_id") or state.get("company_id")

        final_tool_result, err = await _execute_single_tool(
            tool_name=detected_tool,
            params=extracted_params,
            erp_client=erp_client,
            user_token=user_token,
            default_company_id=target_company_id,
            dispatch_map=dispatch_map,
            user_query=state.get("user_query") or "",
        )

        if err:
            return {
                "tool_result": None,
                "multi_tool_results": {detected_tool: None},
                "verification_status": "ERROR",
                "error_message": err,
            }

        return {
            "company_id": target_company_id,
            "tool_result": final_tool_result,
            "multi_tool_results": {detected_tool: final_tool_result},
            "multi_verification_status": {detected_tool: "VALID"},
            "verification_status": "VALID",
            "error_message": None,
        }

    except DjangoCoreClientError as err:
        det_tool = state.get("detected_tool") or "fetch_vendor_invoices"
        logger.warning(f"DjangoCoreClientError in tool_node for tool '{det_tool}': {err}")
        return {
            "tool_result": None,
            "multi_tool_results": {det_tool: None},
            "verification_status": "ERROR",
            "error_message": str(err),
        }
    except Exception as exc:
        det_tool = state.get("detected_tool") or "fetch_vendor_invoices"
        logger.error(f"Unexpected error in tool_node for tool '{det_tool}': {exc}", exc_info=True)
        return {
            "tool_result": None,
            "multi_tool_results": {det_tool: None},
            "verification_status": "ERROR",
            "error_message": f"Unexpected execution error: {str(exc)}",
        }
    finally:
        if own_client:
            await erp_client.aclose()


def tool_node_sync(
    state: AgentState,
    client: Optional[DjangoCoreClient] = None,
) -> Dict[str, Any]:
    """
    Synchronous wrapper for tool_node.
    Enables safe execution in synchronous CLI runners, testing suites, or standard scripts.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Running inside an active event loop (e.g. async runner or Jupyter)
        async def _exec() -> Dict[str, Any]:
            if isinstance(client, DjangoCoreClient) and not hasattr(client, "_mock_return_value"):
                async with DjangoCoreClient(
                    base_url=client.base_url,
                    token=client.token,
                    username=client.username,
                    password=client.password,
                    default_company_id=client.default_company_id,
                    timeout=client.timeout,
                    service_key=client.service_key,
                ) as c:
                    return await tool_node(state, client=c)
            return await tool_node(state, client=client)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            return executor.submit(asyncio.run, _exec()).result()
    else:
        return asyncio.run(tool_node(state, client=client))
