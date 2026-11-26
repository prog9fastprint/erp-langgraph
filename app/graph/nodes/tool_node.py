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

from app.core.branches import COMPANY_NAME_MAP, LOCATION_STOP_WORDS, resolve_branch_company_id
from app.core.config import settings
from app.core.context import (
    clean_product_identifier,
    compute_multi_tool_counts,
    get_multi_tool_key,
    resolve_tool_record_count,
    sanitize_erp_payload,
    should_preserve_plan_on_retry,
)
from app.graph.state import AgentState
from app.schemas.tools import TOOL_DOMAIN_MAP
from app.services.django_client import (
    DjangoCoreClient,
    DjangoCoreClientError,
)

logger = logging.getLogger("tool_node")

MAX_PRICE_COMPARISON_ACCOUNTS = 5

# Single source of truth mapping canonical tool names to DjangoCoreClient fetch_* method names
_TOOL_FETCH_METHOD_MAP: Dict[str, str] = {
    tool_name: (f"fetch_{tool_name[4:]}" if tool_name.startswith("get_") else tool_name)
    for tool_name in TOOL_DOMAIN_MAP
}


def _build_dispatch_map(erp_client: DjangoCoreClient) -> Dict[str, Any]:
    """Bind canonical ERP tools to their DjangoCoreClient coroutine handlers."""
    return {
        tool_name: getattr(erp_client, method_name)
        for tool_name, method_name in _TOOL_FETCH_METHOD_MAP.items()
        if hasattr(erp_client, method_name)
    }


def _is_permission_error(err: Optional[str]) -> bool:
    """Return True if an error message represents HTTP 403 / company access denial."""
    if not err:
        return False
    err_lower = str(err).lower()
    return any(
        k in err_lower
        for k in ("403", "tidak memiliki akses", "akses ditolak", "permission_denied", "permission", "unauthorized", "forbidden")
    )


def _match_marketplace_accounts(
    accounts: List[Dict[str, Any]],
    user_query: str = "",
    marketplace_filter: Optional[str] = None,
    require_order_config: bool = False,
) -> List[Dict[str, Any]]:
    """Filter active marketplace accounts by optional store name mention or platform."""
    candidates = [
        a for a in accounts
        if isinstance(a, dict)
        and a.get("account_id") is not None
        and a.get("is_active", True)
        and (not require_order_config or a.get("has_order_config", False))
    ]
    if marketplace_filter:
        mp_low = marketplace_filter.strip().lower()
        plat_matched = [a for a in candidates if mp_low in str(a.get("marketplace") or "").lower()]
        if plat_matched:
            candidates = plat_matched

    q_low = (user_query or "").lower()
    if q_low and candidates:
        name_matched = [
            a for a in candidates
            if a.get("account_name") and str(a["account_name"]).strip().lower() in q_low
        ]
        if name_matched:
            return name_matched
    return candidates


async def _resolve_and_execute_price_comparison(
    clean_params: Dict[str, Any],
    erp_client: DjangoCoreClient,
    user_token: Optional[str],
    target_company_id: Optional[int],
    user_query: str = "",
) -> Tuple[Any, Optional[str]]:
    """
    Pre-resolve account_id via get_account_health when both account_id and pricelist_id
    are omitted on get_price_comparison, supporting single-match injection or bounded
    multi-account fan-out across matching active accounts with AccountOrder.
    """
    health_params: Dict[str, Any] = {}
    if clean_params.get("company_ids"):
        health_params["company_ids"] = clean_params["company_ids"]
    mp_hint = clean_params.pop("marketplace", None)
    if not mp_hint:
        q_low = (user_query or "").lower()
        if "tiktok" in q_low:
            mp_hint = "TikTok Shop"
        elif "shopee" in q_low:
            mp_hint = "Shopee"
        elif "tokopedia" in q_low:
            mp_hint = "Tokopedia"
        elif "lazada" in q_low:
            mp_hint = "Lazada"
    if mp_hint:
        health_params["marketplace"] = mp_hint

    try:
        health_res = await erp_client.fetch_account_health(
            params=health_params,
            user_token=user_token,
            company_id=target_company_id,
        )
    except DjangoCoreClientError as err:
        return None, str(err)

    accounts = health_res.get("accounts", []) if isinstance(health_res, dict) else []
    matched = _match_marketplace_accounts(
        accounts,
        user_query=user_query,
        marketplace_filter=mp_hint,
        require_order_config=True,
    )

    # Fallback: if active single branch has no MarketplaceAccount/AccountOrder configured
    # and user did not explicitly pin a branch in query, scan authorized branches safely.
    if not matched and not clean_params.get("company_ids"):
        if not (user_query and resolve_branch_company_id(user_query) is not None):
            other_cids = [cid for cid in COMPANY_NAME_MAP.keys() if cid != target_company_id]
            sem = asyncio.Semaphore(getattr(settings, "ERP_MULTI_COMPANY_CONCURRENCY", 5))

            async def _probe_branch_accounts(cid: int) -> List[Dict[str, Any]]:
                async with sem:
                    try:
                        b_res = await erp_client.fetch_account_health(
                            params=dict(health_params),
                            user_token=user_token,
                            company_id=cid,
                        )
                        return b_res.get("accounts", []) if isinstance(b_res, dict) else []
                    except DjangoCoreClientError as b_err:
                        if _is_permission_error(str(b_err)):
                            return []
                        logger.debug("Optional branch marketplace probe failed for cid=%s: %s", cid, b_err)
                        return []
                    except Exception:
                        return []

            branch_lists = await asyncio.gather(*[_probe_branch_accounts(cid) for cid in other_cids])
            seen_acc_ids = {a.get("account_id") for a in accounts if isinstance(a, dict) and a.get("account_id") is not None}
            merged_accounts = list(accounts)
            for bl in branch_lists:
                for acc_item in bl:
                    aid = acc_item.get("account_id") if isinstance(acc_item, dict) else None
                    if aid is not None and aid not in seen_acc_ids:
                        seen_acc_ids.add(aid)
                        merged_accounts.append(acc_item)
            matched = _match_marketplace_accounts(
                merged_accounts,
                user_query=user_query,
                marketplace_filter=mp_hint,
                require_order_config=True,
            )

    if not matched:
        return (
            None,
            "MISSING_PRICELIST_OR_ACCOUNT: Tidak ditemukan akun marketplace aktif yang memiliki "
            "konfigurasi AccountOrder/Pricelist. Mohon sertakan 'account_id' atau 'pricelist_id' secara spesifik.",
        )

    if len(matched) == 1:
        single_params = dict(clean_params)
        single_params["account_id"] = matched[0]["account_id"]
        try:
            res = await erp_client.fetch_price_comparison(
                params=single_params,
                user_token=user_token,
                company_id=matched[0].get("company_id") or target_company_id,
            )
            return res, None
        except DjangoCoreClientError as err:
            return None, str(err)

    # Bounded fan-out across matched active accounts (max 5)
    selected = matched[:MAX_PRICE_COMPARISON_ACCOUNTS]

    async def _call_acc(acc: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[str]]:
        p = dict(clean_params)
        p["account_id"] = acc["account_id"]
        try:
            res = await erp_client.fetch_price_comparison(
                params=p,
                user_token=user_token,
                company_id=acc.get("company_id") or target_company_id,
            )
            return acc, res, None
        except DjangoCoreClientError as err:
            return acc, None, str(err)

    outcomes = await asyncio.gather(*[_call_acc(a) for a in selected])
    successes = [(acc, res) for acc, res, err in outcomes if res is not None]
    errors = [err for _, res, err in outcomes if err]

    if not successes:
        return None, errors[0] if errors else "MISSING_PRICELIST_ID: Tidak ada akun dengan Pricelist valid."

    if len(successes) == 1:
        return successes[0][1], None

    merged_summary = {
        "total_evaluated": 0,
        "match_count": 0,
        "mismatch_count": 0,
        "missing_mp_price": 0,
        "missing_erp_price": 0,
    }
    merged_items: List[Dict[str, Any]] = []
    merged_live_errors: Dict[str, Any] = {}
    merged_pricelists: List[Dict[str, Any]] = []
    accounts_compared: List[Dict[str, Any]] = []
    total_count_sum = 0

    for acc, res in successes:
        s = res.get("summary") or {}
        for k in merged_summary:
            merged_summary[k] += int(s.get(k) or 0)
        total_count_sum += int(res.get("total_count") or 0)
        merged_items.extend(res.get("items") or [])
        if res.get("pricelist"):
            merged_pricelists.append(res["pricelist"])
        accounts_compared.append({
            "account_id": acc["account_id"],
            "account_name": acc.get("account_name"),
            "marketplace": acc.get("marketplace"),
        })
        if isinstance(res.get("live_fallback_errors"), dict):
            for ek, ev in res["live_fallback_errors"].items():
                merged_live_errors[str(ek)] = ev

    limit_val = int(clean_params.get("limit") or 20)
    first_res = successes[0][1]
    return {
        "company_ids": first_res.get("company_ids") or ([target_company_id] if target_company_id else []),
        "pricelist": first_res.get("pricelist"),
        "pricelists": merged_pricelists,
        "accounts_compared": accounts_compared,
        "filter_status": first_res.get("filter_status", "mismatch"),
        "include_live": bool(clean_params.get("include_live", False)),
        "live_fallback_errors": merged_live_errors or None,
        "summary": merged_summary,
        "total_count": total_count_sum,
        "limit": limit_val,
        "offset": int(clean_params.get("offset") or 0),
        "items": merged_items[:limit_val],
    }, None


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

    # Sanitize product identifier fields across inventory and product/SKU tools
    for k in ("code", "sku", "product_code"):
        if k in clean_params and isinstance(clean_params[k], str):
            clean_params[k] = clean_product_identifier(clean_params[k])
    if "search" in clean_params and isinstance(clean_params["search"], str):
        if tool_name in ("get_product_info", "get_stock_level", "get_stock_history", "get_product_mapping_status", "get_price_comparison"):
            clean_params["search"] = clean_product_identifier(clean_params["search"])
    if "codes" in clean_params and isinstance(clean_params["codes"], list):
        clean_params["codes"] = [
            clean_product_identifier(c) for c in clean_params["codes"] if c and isinstance(c, str)
        ]

    # Pre-lookup for stock location_name -> location_id or warehouse_id
    loc_cand = str(clean_params.get("location_name") or "").strip().lower()
    if loc_cand in LOCATION_STOP_WORDS:
        clean_params.pop("location_name", None)
        loc_cand = ""
    if tool_name == "get_stock_level" and loc_cand and not clean_params.get("location_id") and not clean_params.get("warehouse_id"):
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

            if not allow_empty_stock_params and not has_resolved_location and not has_location_name and not clean_params.get("allow_company_scan"):
                return (
                    None,
                    "MISSING_PRODUCT_IDENTIFIER: Mohon sertakan kode SKU, nama produk, atau nama gudang/cabang.",
                )

    # Defensive parameter mapping for get_product_info
    if tool_name == "get_product_info":
        raw_pid = clean_params.get("product_id")
        if raw_pid and isinstance(raw_pid, (int, str)) and len(str(raw_pid)) >= 5:
            if not clean_params.get("code") and not clean_params.get("search"):
                clean_params["code"] = str(raw_pid).strip()
            clean_params.pop("product_id", None)

        raw_code = clean_params.get("code")
        if raw_code and isinstance(raw_code, str):
            code_str = raw_code.strip()
            is_explicit_sku_ref = bool(
                user_query
                and re.search(rf"\b(?:sku|kode)\s*[:#]?\s*{re.escape(code_str)}\b", user_query, re.IGNORECASE)
            )
            is_exact_sku_shape = bool(re.match(r"^(?:\d{3,14}|[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+)$", code_str))
            if " " in code_str or (not is_exact_sku_shape and not is_explicit_sku_ref):
                if not clean_params.get("search"):
                    clean_params["search"] = code_str
                clean_params.pop("code", None)

    # Pre-resolve account_id for get_price_comparison when both account_id and pricelist_id are omitted
    if tool_name == "get_price_comparison" and not clean_params.get("account_id") and not clean_params.get("pricelist_id"):
        raw_result, pre_err = await _resolve_and_execute_price_comparison(
            clean_params=clean_params,
            erp_client=erp_client,
            user_token=user_token,
            target_company_id=target_company_id,
            user_query=user_query,
        )
        if pre_err:
            return None, pre_err
        final_result = (
            sanitize_erp_payload(raw_result)
            if settings.ENABLE_PAYLOAD_SANITIZATION
            else raw_result
        )
        return final_result, None

    # Pre-resolve explicit store/account name hint -> account_id on other marketplace tools if needed
    if (
        TOOL_DOMAIN_MAP.get(tool_name) == "marketplace"
        and tool_name != "get_account_health"
        and not clean_params.get("account_id")
        and user_query
        and re.search(r"\b(?:akun|toko)\s+(?!(?:marketplace|shopee|tiktok|tokopedia|lazada|yang|mana|ini|itu)\b)[A-Za-z0-9]", user_query, re.IGNORECASE)
    ):
        try:
            hp: Dict[str, Any] = {}
            if clean_params.get("company_ids"):
                hp["company_ids"] = clean_params["company_ids"]
            if clean_params.get("marketplace"):
                hp["marketplace"] = clean_params["marketplace"]
            h_res = await erp_client.fetch_account_health(
                params=hp,
                user_token=user_token,
                company_id=target_company_id,
            )
            h_accounts = h_res.get("accounts", []) if isinstance(h_res, dict) else []
            q_low = user_query.lower()
            explicit_matches = [
                a for a in h_accounts
                if isinstance(a, dict)
                and a.get("account_id") is not None
                and a.get("account_name")
                and str(a["account_name"]).strip().lower() in q_low
            ]
            if len(explicit_matches) == 1:
                clean_params["account_id"] = explicit_matches[0]["account_id"]
        except Exception as acc_err:
            logger.debug("Optional store-name pre-lookup skipped: %s", acc_err)

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

    if isinstance(raw_result, dict) and raw_result:
        raw_result = dict(raw_result)
        loc_label = clean_params.get("matched_location_name") or clean_params.get("location_name")
        if loc_label and not raw_result.get("location_name"):
            raw_result["location_name"] = loc_label
        if target_company_id is not None:
            try:
                cid_int = int(target_company_id)
                if raw_result.get("company_id") is None:
                    raw_result["company_id"] = cid_int
                if not raw_result.get("company_name"):
                    mapped_name = COMPANY_NAME_MAP.get(cid_int)
                    if mapped_name:
                        raw_result["company_name"] = mapped_name
            except (TypeError, ValueError):
                pass

    preserve_keys = {"is_active"} if tool_name == "get_account_health" else None
    max_rec = 200 if tool_name in ("get_profit_and_loss", "get_balance_sheet") else 50
    final_result = (
        sanitize_erp_payload(raw_result, max_records=max_rec, preserve_fields=preserve_keys)
        if settings.ENABLE_PAYLOAD_SANITIZATION
        else raw_result
    )
    return final_result, None


async def _execute_tool_multi_company(
    tool_name: str,
    base_params: Dict[str, Any],
    target_company_ids: List[int],
    erp_client: DjangoCoreClient,
    user_token: Optional[str],
    dispatch_map: Dict[str, Any],
    user_query: str = "",
    default_company_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute one ERP tool across multiple companies (native company_ids or per-branch fan-out)."""
    params_copy = dict(base_params or {})
    supports_native_mc = (
        TOOL_DOMAIN_MAP.get(tool_name) == "marketplace"
        or tool_name in ("get_contacts", "get_sales_orders", "get_sales_summary", "get_inventory_velocity_forecast")
    )
    if supports_native_mc:
        primary_cid = target_company_ids[0]
        native_params = dict(params_copy)
        native_params["company_ids"] = target_company_ids
        native_params["company_id"] = primary_cid
        native_res, native_err = await _execute_single_tool(
            tool_name=tool_name,
            params=native_params,
            erp_client=erp_client,
            user_token=user_token,
            default_company_id=primary_cid,
            dispatch_map=dispatch_map,
            user_query=user_query,
        )
        if not native_err:
            return {
                "company_id": primary_cid,
                "company_ids": target_company_ids,
                "is_multi_company": True,
                "multi_company_results": {},
                "unauthorized_companies": [],
                "tool_result": native_res,
                "verification_status": "VALID",
                "error_message": None,
            }
        if not _is_permission_error(native_err):
            return {
                "company_id": primary_cid,
                "company_ids": target_company_ids,
                "is_multi_company": True,
                "multi_company_results": {},
                "unauthorized_companies": [],
                "tool_result": None,
                "verification_status": "ERROR",
                "error_message": native_err,
            }
        # HTTP 403 fallback: strip company_ids and fan out per branch to isolate unauthorized IDs
        params_copy.pop("company_ids", None)

    concurrency_limit = getattr(settings, "ERP_MULTI_COMPANY_CONCURRENCY", 5)
    semaphore = asyncio.Semaphore(concurrency_limit)
    multi_company_results: Dict[int, Any] = {}
    unauthorized_companies: List[int] = []
    errors: List[str] = []

    async def _fetch_company_tool(cid: int):
        params_for_cid = dict(params_copy)
        params_for_cid["company_id"] = cid
        async with semaphore:
            try:
                res, err = await _execute_single_tool(
                    tool_name=tool_name,
                    params=params_for_cid,
                    erp_client=erp_client,
                    user_token=user_token,
                    default_company_id=cid,
                    dispatch_map=dispatch_map,
                    user_query=user_query,
                    allow_empty_stock_params=True,
                )
                if err:
                    if _is_permission_error(str(err)):
                        unauthorized_companies.append(cid)
                        multi_company_results[cid] = None
                    else:
                        errors.append(f"Company {cid}: {err}")
                        multi_company_results[cid] = None
                else:
                    multi_company_results[cid] = res
            except Exception as exc:
                exc_str = str(exc)
                if _is_permission_error(exc_str):
                    unauthorized_companies.append(cid)
                else:
                    errors.append(f"Company {cid}: {exc_str}")
                multi_company_results[cid] = None

    await asyncio.gather(*[_fetch_company_tool(cid) for cid in target_company_ids], return_exceptions=True)

    primary_res = next((r for r in multi_company_results.values() if r is not None), None)
    valid_companies = [cid for cid, r in multi_company_results.items() if r is not None]
    has_success = len(valid_companies) > 0
    all_failed = not has_success and len(target_company_ids) > 0
    v_status = "VALID" if has_success else ("ERROR" if errors else "EMPTY")
    err_msg = "; ".join(errors) if (all_failed and errors) else None

    return {
        "company_id": default_company_id or target_company_ids[0],
        "company_ids": target_company_ids,
        "is_multi_company": True,
        "tool_result": primary_res,
        "multi_company_results": multi_company_results,
        "unauthorized_companies": sorted(unauthorized_companies),
        "verification_status": v_status,
        "error_message": err_msg,
    }


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
        # Dynamic Tool Dispatch Registry (derived from TOOL_DOMAIN_MAP)
        dispatch_map = _build_dispatch_map(erp_client)

        active_tools = state.get("active_tools")
        is_multi_tool = state.get("is_multi_tool") or (bool(active_tools) and len(active_tools) > 1)
        target_company_ids = [int(c) for c in (state.get("company_ids") or [])]
        is_multi_company = bool(state.get("is_multi_company") or len(target_company_ids) > 1)

        # ------------------------------------------------------------------
        # Path A: Multi-Tool Execution via asyncio.gather (supports per-plan multi-company)
        # ------------------------------------------------------------------
        if is_multi_tool and active_tools:
            tool_counts = compute_multi_tool_counts(active_tools)
            retry_count = int(state.get("retry_count") or 0)
            prev_multi_status = state.get("multi_verification_status") or {}

            async def _run_plan(idx: int, plan: Dict[str, Any]) -> Tuple[str, Any, Optional[str], Optional[Dict[str, Any]]]:
                t_name = plan.get("tool", "unnamed_tool")
                key = get_multi_tool_key(t_name, idx, tool_counts)
                if (
                    retry_count > 0
                    and plan.get("status") == "success"
                    and (plan.get("result") is not None or plan.get("multi_company_results"))
                    and should_preserve_plan_on_retry(plan, key, prev_multi_status)
                ):
                    mc_meta = (
                        {
                            "is_multi_company": True,
                            "company_ids": plan.get("company_ids"),
                            "multi_company_results": plan.get("multi_company_results"),
                            "unauthorized_companies": plan.get("unauthorized_companies", []),
                        }
                        if plan.get("multi_company_results")
                        else None
                    )
                    return "single", plan.get("result"), None, mc_meta

                plan_params = dict(plan.get("params") or {})
                plan_cids = [int(c) for c in (plan.get("company_ids") or []) if c]
                if not plan_cids:
                    if plan.get("company_id") is not None:
                        plan_cids = [int(plan["company_id"])]
                    elif plan_params.get("company_id") is not None:
                        plan_cids = [int(plan_params["company_id"])]
                    elif target_company_ids:
                        plan_cids = list(target_company_ids)

                plan_is_mc = bool(plan.get("is_multi_company")) or len(plan_cids) > 1
                plan_user_query = plan.get("sub_query") or plan.get("question_text") or state.get("user_query") or ""

                if plan_is_mc and len(plan_cids) > 1:
                    mc_out = await _execute_tool_multi_company(
                        tool_name=t_name,
                        base_params=plan_params,
                        target_company_ids=plan_cids,
                        erp_client=erp_client,
                        user_token=user_token,
                        dispatch_map=dispatch_map,
                        user_query=plan_user_query,
                        default_company_id=plan.get("company_id") or plan_cids[0],
                    )
                    if mc_out.get("verification_status") == "ERROR":
                        return "mc", None, mc_out.get("error_message") or "Multi-company tool error", mc_out
                    return "mc", mc_out.get("tool_result"), None, mc_out

                single_cid = plan_cids[0] if plan_cids else state.get("company_id")
                res, err = await _execute_single_tool(
                    tool_name=plan.get("tool", "fetch_vendor_invoices"),
                    params=plan_params,
                    erp_client=erp_client,
                    user_token=user_token,
                    default_company_id=single_cid,
                    dispatch_map=dispatch_map,
                    user_query=plan_user_query,
                    allow_empty_stock_params=not bool(plan.get("sub_query")),
                )
                return "single", res, err, None

            outcomes = await asyncio.gather(
                *[_run_plan(idx, plan) for idx, plan in enumerate(active_tools)],
                return_exceptions=True,
            )

            multi_tool_results: Dict[str, Any] = {}
            multi_verification_status: Dict[str, str] = {}
            updated_active_tools: List[Dict[str, Any]] = []
            errors: List[str] = []

            for idx, (plan, outcome) in enumerate(zip(active_tools, outcomes)):
                p_copy = dict(plan)
                t_name = p_copy.get("tool", "unnamed_tool")
                key = get_multi_tool_key(t_name, idx, tool_counts)

                if isinstance(outcome, Exception):
                    err_msg = str(outcome)
                    multi_tool_results[key] = None
                    multi_verification_status[key] = "ERROR"
                    p_copy["status"] = "error"
                    p_copy["error"] = err_msg
                    errors.append(f"{t_name}: {err_msg}")
                else:
                    mode_tag, final_res, err, mc_meta = outcome
                    if mc_meta:
                        p_copy["is_multi_company"] = True
                        p_copy["company_ids"] = mc_meta.get("company_ids")
                        p_copy["multi_company_results"] = mc_meta.get("multi_company_results") or {}
                        p_copy["unauthorized_companies"] = mc_meta.get("unauthorized_companies") or []
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
            any_mc = any(bool(p.get("is_multi_company")) for p in updated_active_tools)

            return {
                "active_tools": updated_active_tools,
                "multi_tool_results": multi_tool_results,
                "multi_verification_status": multi_verification_status,
                "tool_result": primary_result,
                "verification_status": "ERROR" if all_failed else "VALID",
                "error_message": "; ".join(errors) if all_failed else (errors[0] if errors else None),
                "is_multi_company": any_mc,
                "multi_company_results": {},
            }

        # ------------------------------------------------------------------
        # Path MC: Single-Tool Multi-Company Execution
        # ------------------------------------------------------------------
        if is_multi_company and len(target_company_ids) > 1:
            tool_name = (
                state.get("detected_tool")
                or (active_tools[0].get("tool") if active_tools else None)
                or "fetch_vendor_invoices"
            )
            base_params = dict(
                (active_tools[0].get("params") if active_tools else None)
                or state.get("extracted_params")
                or {}
            )
            mc_out = await _execute_tool_multi_company(
                tool_name=tool_name,
                base_params=base_params,
                target_company_ids=target_company_ids,
                erp_client=erp_client,
                user_token=user_token,
                dispatch_map=dispatch_map,
                user_query=state.get("user_query") or "",
                default_company_id=state.get("company_id") or target_company_ids[0],
            )
            mc_results_dict = mc_out.get("multi_company_results") or {}
            primary_res = mc_out.get("tool_result")
            multi_tool_results_compat: Dict[str, Any] = {}
            if mc_results_dict:
                for cid, r in mc_results_dict.items():
                    multi_tool_results_compat[f"{tool_name}_{cid}"] = r
                if primary_res is not None:
                    multi_tool_results_compat[tool_name] = primary_res
            else:
                multi_tool_results_compat[tool_name] = primary_res

            res_payload: Dict[str, Any] = {
                "company_id": mc_out["company_id"],
                "company_ids": mc_out["company_ids"],
                "is_multi_company": True,
                "tool_result": primary_res,
                "multi_company_results": mc_results_dict,
                "unauthorized_companies": mc_out.get("unauthorized_companies", []),
                "multi_tool_results": multi_tool_results_compat,
                "verification_status": mc_out["verification_status"],
                "error_message": mc_out.get("error_message"),
            }
            if not mc_results_dict:
                res_payload["multi_verification_status"] = {
                    tool_name: "ERROR" if mc_out["verification_status"] == "ERROR" else "VALID"
                }
            return res_payload

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

        # Hybrid HQ Auto-Probe: when active session company is HQ (1 or 3), no explicit branch
        # was pinned in user_query, and HQ returned 0 sales records, auto-probe authorized branches.
        user_q_str = str(state.get("user_query") or "")
        user_q_low = user_q_str.lower()
        hq_pinned = bool(re.search(r"\b(?:kantor\s+pusat|head\s*office|hq|pusat)\b", user_q_low))

        if (
            TOOL_DOMAIN_MAP.get(detected_tool) == "sales"
            and detected_tool in ("get_contacts", "get_sales_orders", "get_top_selling_products")
            and int(target_company_id or 0) in (1, 3)
            and not hq_pinned
            and resolve_branch_company_id(user_q_str) is None
            and resolve_tool_record_count(final_tool_result) == 0
        ):
            raw_auth = state.get("authorized_company_ids") or []
            probe_cids = [int(c) for c in raw_auth if c and int(c) not in (1, 3)]
            if not probe_cids:
                probe_cids = [int(c) for c in COMPANY_NAME_MAP.keys() if int(c) not in (1, 3)]
            if probe_cids:
                probe_params = dict(extracted_params)
                probe_params["company_ids"] = probe_cids
                probe_params["company_id"] = probe_cids[0]
                probe_res, probe_err = await _execute_single_tool(
                    tool_name=detected_tool,
                    params=probe_params,
                    erp_client=erp_client,
                    user_token=user_token,
                    default_company_id=probe_cids[0],
                    dispatch_map=dispatch_map,
                    user_query=user_q_str,
                )
                if not probe_err and probe_res is not None:
                    return {
                        "company_id": probe_cids[0],
                        "company_ids": probe_cids,
                        "is_multi_company": True,
                        "multi_company_results": {},
                        "unauthorized_companies": [],
                        "tool_result": probe_res,
                        "multi_tool_results": {detected_tool: probe_res},
                        "multi_verification_status": {detected_tool: "VALID"},
                        "verification_status": "VALID",
                        "error_message": None,
                    }

        return {
            "company_id": target_company_id,
            "company_ids": [target_company_id],
            "is_multi_company": False,
            "multi_company_results": {},
            "unauthorized_companies": [],
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
            if isinstance(client, DjangoCoreClient) and type(client) is DjangoCoreClient:
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
