"""
Extractor Node for FastPrint ERP LangGraph Agent Workflow (TASK-4.2 / Milestone 2).
Uses an extensible schema registry (TOOL_SCHEMA_REGISTRY) to extract typed parameters
across multi-tool active_tools subagent plans with LLM structured output, temporal
date resolution, 3-tier entity resolution, and deterministic heuristic fallbacks.
"""
import calendar
import copy
from datetime import date
import logging
import os
import re
import sys
import threading
from typing import Any, Callable, Dict, List, Optional, Type
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel

from app.core.config import settings
from app.core.context import (
    trim_conversation_messages,
    build_contextual_prompt,
    clean_product_identifier,
    compute_multi_tool_counts,
    get_current_erp_date,
    get_multi_tool_key,
    should_preserve_plan_on_retry,
)
from app.graph.state import AgentState
from app.schemas.tools import (
    AccountBalanceFilterSchema,
    JournalEntriesFilterSchema,
    GeneralLedgerFilterSchema,
    ProfitAndLossFilterSchema,
    BalanceSheetFilterSchema,
    VendorBillsFilterSchema,
    BankReconciliationFilterSchema,
    StockLevelFilterSchema,
    ProductInfoFilterSchema,
    VendorInvoiceFilterSchema,
    LocationsFilterSchema,
    StockHistoryFilterSchema,
    TransfersFilterSchema,
    LowStockAlertsFilterSchema,
    StockAdjustmentsFilterSchema,
    ScrapRecordsFilterSchema,
    WarehousesFilterSchema,
    ProductCategoriesFilterSchema,
    TopSellingProductsFilterSchema,
    SalesOrdersFilterSchema,
    SalesOrderDetailFilterSchema,
    SalesSummaryFilterSchema,
    ContactsFilterSchema,
    AccountHealthFilterSchema,
    MarketplaceOrdersFilterSchema,
    KilatOrderDiagnosticsFilterSchema,
    ProductMappingStatusFilterSchema,
    StockSyncSimulationFilterSchema,
    SyncExecutionLogsFilterSchema,
    PriceComparisonFilterSchema,
    FeeAndEscrowAnomaliesFilterSchema,
    DeductionAndSettlementSummaryFilterSchema,
    CashflowAndBalanceSummaryFilterSchema,
    CompetitorInsightsFilterSchema,
    TOOL_DOMAIN_MAP,
)
from app.services.llm import get_chat_model

from app.core.branches import (
    ALL_COMPANIES_KEYWORDS,
    BRANCH_COMPANY_MAP,
    COMPANY_NAME_MAP,
    LOCATION_STOP_WORDS,
    resolve_all_branch_company_ids,
    resolve_branch_company_id,
    strip_company_scope_keywords,
)
from app.core.context import (
    ANAPHORA_TOKENS,
    ANAPHORIC_CLITICS,
    has_anaphoric_reference,
)
from app.graph.nodes.extractors import (
    ALLOWED_JOURNAL_STATUSES,
    LOCATION_STOPWORDS,
    PRODUCT_STOPWORDS,
    STOCK_DELIMITERS,
    STOCK_STOPWORDS,
    TEMPORAL_KEYWORDS,
    VENDOR_DELIMITERS,
    VENDOR_NAME_STOPWORDS,
    VENDOR_PREFIXES,
    _CONTACT_STOPWORDS,
    _NON_PRODUCT_PHRASE_RE,
    _SALES_CHANNEL_PATTERNS,
    _TOP_SALES_STOPWORDS,
    _apply_sales_date_filters,
    _extract_mp_account_id,
    _extract_mp_platform,
    _extract_mp_sku_token,
    _extract_prior_sales_context,
    _extract_sales_channel,
    _fallback_extract_account_balance,
    _fallback_extract_account_health,
    _fallback_extract_balance_sheet,
    _fallback_extract_bank_reconciliation,
    _fallback_extract_cashflow_balance,
    _fallback_extract_competitor_insights,
    _fallback_extract_contacts,
    _fallback_extract_deduction_summary,
    _fallback_extract_fee_anomalies,
    _fallback_extract_general_ledger,
    _fallback_extract_journal_entries,
    _fallback_extract_kilat_diagnostics,
    _fallback_extract_locations,
    _fallback_extract_low_stock_alerts,
    _fallback_extract_marketplace_orders,
    _fallback_extract_price_comparison,
    _fallback_extract_product_categories,
    _fallback_extract_product_info,
    _fallback_extract_product_mapping_status,
    _fallback_extract_profit_and_loss,
    _fallback_extract_sales_order_detail,
    _fallback_extract_sales_orders,
    _fallback_extract_sales_summary,
    _fallback_extract_scrap_records,
    _fallback_extract_stock_adjustments,
    _fallback_extract_stock_history,
    _fallback_extract_stock_level,
    _fallback_extract_stock_sync_simulation,
    _fallback_extract_sync_execution_logs,
    _fallback_extract_top_selling_products,
    _fallback_extract_transfers,
    _fallback_extract_vendor_bills,
    _fallback_extract_vendor_invoice,
    _fallback_extract_warehouses,
    _has_live_marketplace_intent,
    _is_pure_branch_token,
    _is_valid_product_identifier,
    has_temporal_expression,
    is_account_code_or_name,
    is_accounting_term,
    is_vendor_name,
    resolve_relative_dates,
    try_fast_path_stock_extraction,
)

logger = logging.getLogger("extractor_node")

# Extensible registry mapping tool names to input filter schemas
TOOL_SCHEMA_REGISTRY: Dict[str, Type[BaseModel]] = {
    "fetch_vendor_invoices": VendorInvoiceFilterSchema,
    "get_journal_entries": JournalEntriesFilterSchema,
    "get_account_balance": AccountBalanceFilterSchema,
    "get_general_ledger": GeneralLedgerFilterSchema,
    "get_profit_and_loss": ProfitAndLossFilterSchema,
    "get_balance_sheet": BalanceSheetFilterSchema,
    "get_vendor_bills": VendorBillsFilterSchema,
    "get_bank_reconciliation_summary": BankReconciliationFilterSchema,
    "get_stock_level": StockLevelFilterSchema,
    "get_product_info": ProductInfoFilterSchema,
    "get_locations": LocationsFilterSchema,
    "get_stock_history": StockHistoryFilterSchema,
    "get_transfers": TransfersFilterSchema,
    "get_low_stock_alerts": LowStockAlertsFilterSchema,
    "get_stock_adjustments": StockAdjustmentsFilterSchema,
    "get_scrap_records": ScrapRecordsFilterSchema,
    "get_warehouses": WarehousesFilterSchema,
    "get_product_categories": ProductCategoriesFilterSchema,
    "get_top_selling_products": TopSellingProductsFilterSchema,
    "get_sales_orders": SalesOrdersFilterSchema,
    "get_sales_order_detail": SalesOrderDetailFilterSchema,
    "get_sales_summary": SalesSummaryFilterSchema,
    "get_contacts": ContactsFilterSchema,
    "get_account_health": AccountHealthFilterSchema,
    "get_marketplace_orders": MarketplaceOrdersFilterSchema,
    "get_kilat_order_diagnostics": KilatOrderDiagnosticsFilterSchema,
    "get_product_mapping_status": ProductMappingStatusFilterSchema,
    "get_stock_sync_simulation": StockSyncSimulationFilterSchema,
    "get_sync_execution_logs": SyncExecutionLogsFilterSchema,
    "get_price_comparison": PriceComparisonFilterSchema,
    "get_fee_and_escrow_anomalies": FeeAndEscrowAnomaliesFilterSchema,
    "get_deduction_and_settlement_summary": DeductionAndSettlementSummaryFilterSchema,
    "get_cashflow_and_balance_summary": CashflowAndBalanceSummaryFilterSchema,
    "get_competitor_insights": CompetitorInsightsFilterSchema,
}



def _extract_single_tool_params(
    plan: Dict[str, Any],
    all_plans: List[Dict[str, Any]],
    user_query: str,
    state: AgentState,
    model: Optional[BaseChatModel],
    today: date,
    retry_count: int,
    retry_instruction: Optional[str],
    model_factory: Optional[Callable[[], Optional[BaseChatModel]]] = None,
) -> Dict[str, Any]:
    """Extract and validate parameters for a single planned ERP tool."""
    tool_name = plan.get("tool")
    domain = plan.get("domain") or TOOL_DOMAIN_MAP.get(tool_name, "general")
    sub_query = (plan.get("sub_query") or user_query).strip()
    is_decomposed_clause = len(state.get("sub_questions") or []) > 1 and sub_query != user_query
    allow_global_backfill = (sub_query != user_query) and (
        not is_decomposed_clause or has_anaphoric_reference(sub_query)
    )

    target_schema = TOOL_SCHEMA_REGISTRY.get(tool_name)
    if target_schema is None:
        if domain == "accounting":
            target_schema = JournalEntriesFilterSchema
        else:
            return {}

    supports_dates = "date_from" in target_schema.model_fields
    rel_date_from, rel_date_to = None, None
    if supports_dates:
        rel_date_from, rel_date_to = resolve_relative_dates(sub_query, today)
        if not rel_date_from and not rel_date_to and allow_global_backfill:
            rel_date_from, rel_date_to = resolve_relative_dates(user_query, today)

    extracted_dict: Optional[Dict[str, Any]] = None

    # 0. Fast-Path Deterministic Extraction (Tier 1)
    if tool_name == "get_stock_level" and retry_count == 0:
        fast_params = try_fast_path_stock_extraction(sub_query, user_query, state.get("company_id"))
        if fast_params:
            extracted_dict = dict(fast_params)
    elif tool_name == "get_profit_and_loss" and retry_count == 0:
        fast_params = _fallback_extract_profit_and_loss(sub_query, today)
        if fast_params.get("date_from") or fast_params.get("date_to") or fast_params.get("include_breakdown"):
            extracted_dict = dict(fast_params)
    elif tool_name == "get_balance_sheet" and retry_count == 0:
        fast_params = _fallback_extract_balance_sheet(sub_query, today)
        if fast_params.get("as_of_date") or fast_params.get("include_breakdown"):
            extracted_dict = dict(fast_params)
    elif tool_name == "get_top_selling_products" and retry_count == 0:
        fast_params = _fallback_extract_top_selling_products(sub_query, today)
        if fast_params:
            extracted_dict = dict(fast_params)
    elif tool_name == "get_contacts" and retry_count == 0:
        fast_params = _fallback_extract_contacts(sub_query, today, state.get("messages"), state=state)
        if fast_params:
            extracted_dict = dict(fast_params)
    elif tool_name == "get_sales_orders" and retry_count == 0:
        fast_params = _fallback_extract_sales_orders(sub_query, today, state.get("messages"), state=state)
        if fast_params.get("status") == "active" or fast_params.get("sort_by") == "amount_desc" or fast_params.get("channel_name") or fast_params.get("product_search"):
            extracted_dict = dict(fast_params)

    # 1. LLM Structured Output Attempt (resolve model lazily only after Tier-1 fast path misses)
    active_model = model
    if extracted_dict is None and active_model is None and model_factory is not None:
        active_model = model_factory()
    if extracted_dict is None and active_model is not None:
        try:
            _, last_day_num = calendar.monthrange(today.year, today.month)
            month_start = today.replace(day=1).strftime("%Y-%m-%d")
            month_end = today.replace(day=last_day_num).strftime("%Y-%m-%d")

            system_prompt = (
                f"You are an expert ERP Parameter Extractor for FastPrint ERP.\n"
                f"Target Tool: {tool_name}\n"
                f"Target Domain: {domain}\n"
                f"Reference Today's Date: {today.strftime('%Y-%m-%d')}\n"
                f"Current Month Range: {month_start} to {month_end}\n\n"
                f"Extract filter entities strictly adhering to the schema.\n"
                f"Guidelines for extracted fields:\n"
                f"- For 'search': Extract ONLY key entity names or document numbers (e.g. 'TikTok', 'PT Sumber Makmur', 'BCA'). Do NOT include full conversational phrases like 'aku mau tanya', 'saldo di jurnal', or relative dates.\n"
                f"- For 'vendor_name': Extract only clean vendor / supplier company names (e.g. 'MEGACOM BEC', 'PT Sumber Makmur'). Do NOT append branch or city names into vendor_name.\n"
                f"- For 'account_code' (in get_account_balance): Only set if user provided an actual account code (e.g. '11285', '111100'). If the user provided a name like 'bank tiktok' or 'marketplace tiktok', do NOT set 'account_code' to the name.\n"
                f"- For 'code' (in get_stock_level): Extract the product SKU code, barcode, OR product name into 'code'. Strip filler words like 'product', 'produk', 'barang', 'item', 'sku', 'kode'.\n"
                f"- For 'code' vs 'search' (in get_product_info): Put ONLY explicit SKU/barcode identifiers (e.g. '1362', '91030300', 'HVS-A4-70') into 'code'. Put product names or partial model keywords (e.g. 'L210', 'stiker hologram', 'botol kalkir') into 'search'.\n"
                f"- For 'codes' (in get_stock_level): If the user specifies multiple product SKU codes or barcodes (e.g. '91010600 dan 91030300', 'SKU A, B, dan C'), extract all of them as a list of strings into 'codes' rather than a single 'code'.\n"
                f"- For 'location_name' (in get_stock_level): Extract warehouse or location names (e.g. 'mainstore', 'gudang surabaya', 'pusat') into 'location_name'.\n"
                f"- STRICT DATE RULE: Do NOT populate 'date_from' or 'date_to' unless the user explicitly specifies a date, relative date (e.g. 'bulan ini', 'bulan lalu', 'kemarin'), or time window. If no time expression is mentioned, leave both None.\n"
                f"- STRICT STATUS RULE: For accounting journal/invoice/bill tools, valid status values are only 'draft', 'posted', 'paid', 'unpaid', or 'cancel'. For get_sync_execution_logs, status must be 'FAILED', 'SKIPPED', 'SUCCESS', or 'all'. For get_marketplace_orders, use mp_status and sla_status.\n"
                f"- MARKETPLACE LIVE RULE: Set include_live=True ONLY when user explicitly asks for live/real-time marketplace API data; otherwise keep include_live=False.\n"
                f"- If the user specifies relative dates like 'bulan ini' or '3 hari terakhir', map them to date_from and date_to.\n"
                f"- If the user refers to entities from previous turns (e.g. 'produk itu', 'barang tadi', 'vendor tersebut', 'itu'), "
                f"resolve the specific SKU, code, or entity name from the conversation history."
            )

            if retry_instruction:
                system_prompt += (
                    f"\n\nIMPORTANT RETRY INSTRUCTION (Attempt {retry_count}):\n"
                    f"{retry_instruction}\n"
                    f"Adjust and relax your extracted filter parameters accordingly."
                )

            system_prompt = build_contextual_prompt(
                system_prompt,
                state.get("conversation_summary"),
            )

            structured_model = active_model.with_structured_output(target_schema)
            raw_history = (state.get("messages") or [])[:-1]
            history_msgs = trim_conversation_messages(
                raw_history,
                max_messages=settings.MAX_WINDOW_MESSAGES,
            )

            effective_query = user_query if (has_anaphoric_reference(sub_query) and not is_decomposed_clause) else sub_query

            messages = [
                SystemMessage(content=system_prompt),
                *history_msgs,
                HumanMessage(content=effective_query),
            ]
            extracted_obj = structured_model.invoke(messages)

            if isinstance(extracted_obj, BaseModel):
                extracted_dict = extracted_obj.model_dump(exclude_none=True)
            elif isinstance(extracted_obj, dict):
                extracted_dict = extracted_obj
            else:
                extracted_dict = {}

            if retry_count == 0 and supports_dates:
                if rel_date_from and not extracted_dict.get("date_from"):
                    extracted_dict["date_from"] = rel_date_from
                if rel_date_to and not extracted_dict.get("date_to"):
                    extracted_dict["date_to"] = rel_date_to
        except Exception as exc:
            logger.debug("LLM parameter extraction failed for tool '%s': %s", tool_name, exc)
            extracted_dict = None

    # 2. Deterministic Fallback if LLM unavailable or failed
    if extracted_dict is None:
        if tool_name == "fetch_vendor_invoices":
            extracted_dict = _fallback_extract_vendor_invoice(sub_query, today)
        elif tool_name == "get_stock_level":
            extracted_dict = _fallback_extract_stock_level(sub_query, state.get("messages"))
        elif tool_name == "get_locations":
            extracted_dict = _fallback_extract_locations(sub_query)
        elif tool_name == "get_product_info":
            extracted_dict = _fallback_extract_product_info(sub_query, state.get("messages"))
        elif tool_name == "get_account_balance":
            extracted_dict = _fallback_extract_account_balance(sub_query, today)
        elif tool_name == "get_journal_entries":
            extracted_dict = _fallback_extract_journal_entries(sub_query, today)
        elif tool_name == "get_general_ledger":
            extracted_dict = _fallback_extract_general_ledger(sub_query, today)
        elif tool_name == "get_profit_and_loss":
            extracted_dict = _fallback_extract_profit_and_loss(sub_query, today)
        elif tool_name == "get_balance_sheet":
            extracted_dict = _fallback_extract_balance_sheet(sub_query, today)
        elif tool_name == "get_vendor_bills":
            extracted_dict = _fallback_extract_vendor_bills(sub_query, today)
        elif tool_name == "get_bank_reconciliation_summary":
            extracted_dict = _fallback_extract_bank_reconciliation(sub_query, today)
        elif tool_name == "get_stock_history":
            extracted_dict = _fallback_extract_stock_history(sub_query, today)
        elif tool_name == "get_transfers":
            extracted_dict = _fallback_extract_transfers(sub_query, today)
        elif tool_name == "get_low_stock_alerts":
            extracted_dict = _fallback_extract_low_stock_alerts(sub_query)
        elif tool_name == "get_stock_adjustments":
            extracted_dict = _fallback_extract_stock_adjustments(sub_query, today)
        elif tool_name == "get_scrap_records":
            extracted_dict = _fallback_extract_scrap_records(sub_query, today)
        elif tool_name == "get_warehouses":
            extracted_dict = _fallback_extract_warehouses(sub_query)
        elif tool_name == "get_product_categories":
            extracted_dict = _fallback_extract_product_categories(sub_query)
        elif tool_name == "get_top_selling_products":
            extracted_dict = _fallback_extract_top_selling_products(sub_query, today)
        elif tool_name == "get_sales_orders":
            extracted_dict = _fallback_extract_sales_orders(sub_query, today, state.get("messages"), state=state)
        elif tool_name == "get_sales_order_detail":
            extracted_dict = _fallback_extract_sales_order_detail(sub_query)
        elif tool_name == "get_sales_summary":
            extracted_dict = _fallback_extract_sales_summary(sub_query, today)
        elif tool_name == "get_contacts":
            extracted_dict = _fallback_extract_contacts(sub_query, today, state.get("messages"), state=state)
        elif tool_name == "get_account_health":
            extracted_dict = _fallback_extract_account_health(sub_query)
        elif tool_name == "get_marketplace_orders":
            extracted_dict = _fallback_extract_marketplace_orders(sub_query, today)
        elif tool_name == "get_kilat_order_diagnostics":
            extracted_dict = _fallback_extract_kilat_diagnostics(sub_query, today)
        elif tool_name == "get_product_mapping_status":
            extracted_dict = _fallback_extract_product_mapping_status(sub_query)
        elif tool_name == "get_stock_sync_simulation":
            extracted_dict = _fallback_extract_stock_sync_simulation(sub_query)
        elif tool_name == "get_sync_execution_logs":
            extracted_dict = _fallback_extract_sync_execution_logs(sub_query, today)
        elif tool_name == "get_price_comparison":
            extracted_dict = _fallback_extract_price_comparison(sub_query)
        elif tool_name == "get_fee_and_escrow_anomalies":
            extracted_dict = _fallback_extract_fee_anomalies(sub_query, today)
        elif tool_name == "get_deduction_and_settlement_summary":
            extracted_dict = _fallback_extract_deduction_summary(sub_query, today)
        elif tool_name == "get_cashflow_and_balance_summary":
            extracted_dict = _fallback_extract_cashflow_balance(sub_query, today)
        elif tool_name == "get_competitor_insights":
            extracted_dict = _fallback_extract_competitor_insights(sub_query, today)
        else:
            extracted_dict = {}
            if supports_dates and retry_count == 0:
                if rel_date_from:
                    extracted_dict["date_from"] = rel_date_from
                if rel_date_to:
                    extracted_dict["date_to"] = rel_date_to

    # Pre-clean invalid superlative/interrogative phrases before peer/history harvesting
    if tool_name in ("get_stock_level", "get_product_info"):
        for k in ("code", "search"):
            val = extracted_dict.get(k)
            if isinstance(val, str) and not _is_valid_product_identifier(val):
                extracted_dict.pop(k, None)

    # 3. 3-Tier Entity Resolution & Anaphora Handling
    if tool_name == "get_stock_level":
        if not extracted_dict.get("code") and not extracted_dict.get("codes") and not extracted_dict.get("product_id"):
            # Tier 1: Peer harvesting (Inventory domain)
            for peer in all_plans:
                if peer is plan:
                    continue
                if peer.get("domain") == "inventory":
                    peer_p = peer.get("params") or {}
                    peer_cand = peer_p.get("code") or peer_p.get("search")
                    if not peer_cand:
                        peer_sub = peer.get("sub_query") or ""
                        peer_ext = _fallback_extract_product_info(peer_sub, state.get("messages"))
                        peer_cand = peer_ext.get("code") or peer_ext.get("search")
                    if peer_cand and not is_vendor_name(peer_cand) and not is_account_code_or_name(peer_cand) and not is_accounting_term(peer_cand):
                        extracted_dict["code"] = peer_cand
                        break
            # Tier 2: Global user_query inspection (guarded on decomposed non-anaphoric clauses)
            if not extracted_dict.get("code") and allow_global_backfill:
                u_ext = _fallback_extract_product_info(user_query, state.get("messages"))
                cand = u_ext.get("code") or u_ext.get("search")
                if not cand:
                    u_stock = _fallback_extract_stock_level(user_query, state.get("messages"))
                    cand = u_stock.get("code")
                if cand and not is_vendor_name(cand) and not is_account_code_or_name(cand) and not is_accounting_term(cand):
                    extracted_dict["code"] = cand

            # Tier 3: Conversation History Harvesting (Multi-turn Slot Filling)
            sub_has_prod = bool(_fallback_extract_stock_level(sub_query).get("code"))
            if not extracted_dict.get("code") or not sub_has_prod:
                prior_extracted = state.get("extracted_params") or {}
                if prior_extracted.get("code"):
                    extracted_dict["code"] = prior_extracted["code"]
                elif prior_extracted.get("product_id"):
                    extracted_dict["product_id"] = prior_extracted["product_id"]

            if not extracted_dict.get("code"):
                history_msgs = state.get("messages") or []
                prior_msgs = (
                    history_msgs[:-1]
                    if (history_msgs and user_query and user_query.lower() in str(getattr(history_msgs[-1], "content", "") or "").lower())
                    else history_msgs
                )
                for prev_msg in reversed(prior_msgs):
                    if not isinstance(prev_msg, HumanMessage) and getattr(prev_msg, "type", "") != "human":
                        continue
                    content = getattr(prev_msg, "content", "") or ""
                    prev_stock = _fallback_extract_stock_level(content)
                    if prev_stock.get("code"):
                        extracted_dict["code"] = prev_stock["code"]
                        break
                    prev_info = _fallback_extract_product_info(content)
                    cand = prev_info.get("code") or prev_info.get("search")
                    if cand and not is_vendor_name(cand) and not is_account_code_or_name(cand) and not is_accounting_term(cand):
                        extracted_dict["code"] = cand
                        break

        if not extracted_dict.get("location_name"):
            if is_decomposed_clause:
                s_stock = _fallback_extract_stock_level(sub_query, state.get("messages"))
                if s_stock.get("location_name"):
                    extracted_dict["location_name"] = s_stock["location_name"]
            else:
                u_stock = _fallback_extract_stock_level(user_query, state.get("messages"))
                if u_stock.get("location_name"):
                    extracted_dict["location_name"] = u_stock["location_name"]
                elif sub_query != user_query:
                    s_stock = _fallback_extract_stock_level(sub_query, state.get("messages"))
                    if s_stock.get("location_name"):
                        extracted_dict["location_name"] = s_stock["location_name"]

    elif tool_name == "get_product_info":
        if not extracted_dict.get("code") and not extracted_dict.get("search") and not extracted_dict.get("product_id"):
            # Tier 1: Peer harvesting (Inventory domain)
            for peer in all_plans:
                if peer is plan:
                    continue
                if peer.get("domain") == "inventory":
                    peer_p = peer.get("params") or {}
                    peer_cand = peer_p.get("code") or peer_p.get("search")
                    if not peer_cand:
                        peer_sub = peer.get("sub_query") or ""
                        peer_ext = _fallback_extract_stock_level(peer_sub, state.get("messages"))
                        peer_cand = peer_ext.get("code")
                    if peer_cand and not is_vendor_name(peer_cand) and not is_account_code_or_name(peer_cand) and not is_accounting_term(peer_cand):
                        extracted_dict["search"] = peer_cand
                        break
            # Tier 2: Global user_query inspection (guarded on decomposed non-anaphoric clauses)
            if not extracted_dict.get("code") and not extracted_dict.get("search") and allow_global_backfill:
                u_ext = _fallback_extract_product_info(user_query, state.get("messages"))
                if u_ext.get("code") and not is_vendor_name(u_ext["code"]) and not is_account_code_or_name(u_ext["code"]) and not is_accounting_term(u_ext["code"]):
                    extracted_dict["code"] = u_ext["code"]
                elif u_ext.get("search") and not is_vendor_name(u_ext["search"]) and not is_account_code_or_name(u_ext["search"]) and not is_accounting_term(u_ext["search"]):
                    extracted_dict["search"] = u_ext["search"]

        # Tier 3: Conversation history entity resolution & previous turn inheritance
        is_ref_inquiry = has_anaphoric_reference(sub_query) or has_anaphoric_reference(user_query)
        needs_resolution = (not extracted_dict.get("code") and not extracted_dict.get("search") and not extracted_dict.get("product_id"))

        # A. If referential query and exact SKU code is missing, prioritize exact SKU from prior tool_result
        if (needs_resolution or is_ref_inquiry) and not extracted_dict.get("code"):
            prior_tool_result = state.get("tool_result") or {}
            if isinstance(prior_tool_result, dict):
                res_prods = prior_tool_result.get("products")
                if isinstance(res_prods, list) and res_prods and isinstance(res_prods[0], dict):
                    first_prod = res_prods[0]
                    prod_code = first_prod.get("code") or first_prod.get("default_code") or first_prod.get("sku")
                    prod_name = first_prod.get("name")
                    if prod_code and not is_accounting_term(str(prod_code)):
                        extracted_dict["code"] = str(prod_code).strip()
                        extracted_dict.pop("search", None)
                    elif prod_name and not extracted_dict.get("search"):
                        extracted_dict["search"] = str(prod_name).strip()

        # B. Inspect previous turn's extracted_params if still unresolved
        if not extracted_dict.get("code") and not extracted_dict.get("search") and not extracted_dict.get("product_id"):
            prior_extracted = state.get("extracted_params") or {}
            if prior_extracted.get("code"):
                p_code = prior_extracted["code"]
                if " " in str(p_code):
                    extracted_dict["search"] = str(p_code).strip()
                else:
                    extracted_dict["code"] = str(p_code).strip()
            elif prior_extracted.get("search"):
                extracted_dict["search"] = prior_extracted["search"]

    elif tool_name == "fetch_vendor_invoices":
        if not extracted_dict.get("vendor_name") and not extracted_dict.get("vendor_id"):
            # Tier 1: Peer harvesting (Accounting domain)
            for peer in all_plans:
                if peer is plan:
                    continue
                if peer.get("domain") == "accounting":
                    peer_p = peer.get("params") or {}
                    peer_vendor = peer_p.get("vendor_name")
                    if peer_vendor and is_vendor_name(peer_vendor):
                        extracted_dict["vendor_name"] = peer_vendor
                        break
            # Tier 2: Global user_query inspection
            if not extracted_dict.get("vendor_name") and allow_global_backfill:
                v_ext = _fallback_extract_vendor_invoice(user_query, today)
                if v_ext.get("vendor_name") and is_vendor_name(v_ext["vendor_name"]):
                    extracted_dict["vendor_name"] = v_ext["vendor_name"]
                if not extracted_dict.get("status") and v_ext.get("status"):
                    extracted_dict["status"] = v_ext["status"]

    elif tool_name == "get_account_balance":
        if not extracted_dict.get("account_code") and not extracted_dict.get("account_id") and allow_global_backfill:
            b_ext = _fallback_extract_account_balance(user_query, today)
            if b_ext.get("account_code"):
                extracted_dict["account_code"] = b_ext["account_code"]

    elif tool_name == "get_journal_entries":
        if not extracted_dict.get("entry_number") and not extracted_dict.get("status") and not extracted_dict.get("search") and allow_global_backfill:
            j_ext = _fallback_extract_journal_entries(user_query, today)
            for k in ("entry_number", "status", "search"):
                if j_ext.get(k):
                    extracted_dict[k] = j_ext[k]

    elif tool_name == "get_locations":
        if not extracted_dict.get("search") and allow_global_backfill:
            l_ext = _fallback_extract_locations(user_query)
            if l_ext.get("search"):
                extracted_dict["search"] = l_ext["search"]

    # 4. Global Temporal Date Backfill (Accounting Domain Only)
    if supports_dates and retry_count == 0:
        if not extracted_dict.get("date_from") and rel_date_from:
            extracted_dict["date_from"] = rel_date_from
        if not extracted_dict.get("date_to") and rel_date_to:
            extracted_dict["date_to"] = rel_date_to

    has_inherited_dates = bool(extracted_dict.pop("_inherited_dates", False))

    # Strict date enforcement: if query mentions no explicit dates or temporal keywords, drop hallucinated dates
    # Exception: referential sales follow-ups inheriting dates from prior turn (e.g. 'di wa itu contactnya siapa saja').
    has_explicit_anaphora = bool(
        any(clitic in sub_query.lower() for clitic in ANAPHORIC_CLITICS)
        or any(re.search(rf"\b{re.escape(token)}\b", sub_query.lower()) for token in ANAPHORA_TOKENS)
    )
    has_scope_temporal = has_temporal_expression(sub_query) or (
        (not is_decomposed_clause or has_explicit_anaphora) and has_temporal_expression(user_query)
    )
    if supports_dates and not has_inherited_dates and not has_scope_temporal:
        extracted_dict.pop("date_from", None)
        extracted_dict.pop("date_to", None)

    # Marketplace include_live guardrail: enforce False unless explicit live intent in query
    if "include_live" in target_schema.model_fields:
        extracted_dict["include_live"] = _has_live_marketplace_intent(sub_query) or (
            not is_decomposed_clause and _has_live_marketplace_intent(user_query)
        )

    # Marketplace platform backfill if LLM omitted it
    if "marketplace" in target_schema.model_fields and not extracted_dict.get("marketplace"):
        mp_plat = _extract_mp_platform(sub_query) or (
            None if is_decomposed_clause else _extract_mp_platform(user_query)
        )
        if mp_plat:
            extracted_dict["marketplace"] = mp_plat

    # Status sanitization scoped strictly to accounting journal/bill/invoice tools
    if tool_name in ("fetch_vendor_invoices", "get_journal_entries", "get_general_ledger", "get_vendor_bills"):
        if "status" in extracted_dict and extracted_dict["status"]:
            st_val = str(extracted_dict["status"]).lower().strip()
            if st_val in ALLOWED_JOURNAL_STATUSES:
                extracted_dict["status"] = st_val
            else:
                extracted_dict.pop("status", None)

    # Branch entity resolution into company_id (per-clause isolated when decomposed)
    detected_branch = (
        plan.get("company_id")
        or resolve_branch_company_id(sub_query)
        or (None if is_decomposed_clause else resolve_branch_company_id(user_query))
    )
    if detected_branch:
        extracted_dict["company_id"] = detected_branch
    elif state.get("company_id"):
        extracted_dict["company_id"] = state.get("company_id")

    # Clean location_name stop-words (e.g. 'setiap', 'semua', 'gudang', 'rak')
    if "location_name" in extracted_dict and extracted_dict["location_name"]:
        loc_val = str(extracted_dict["location_name"]).lower().strip()
        if loc_val in LOCATION_STOP_WORDS:
            extracted_dict.pop("location_name", None)

    # Reject superlative/interrogative phrases masquerading as product identifiers
    if tool_name in ("get_stock_level", "get_product_info"):
        for k in ("code", "search"):
            val = extracted_dict.get(k)
            if isinstance(val, str) and not _is_valid_product_identifier(val):
                extracted_dict.pop(k, None)

    # Low stock alerts: superlative queries ("stocknya tersedikit") exclude <= 0 items
    if tool_name == "get_low_stock_alerts" and re.search(
        r"\b(?:tersedikit|paling\s+sedikit|terkecil|terendah)\b",
        f"{sub_query} {user_query}",
        re.IGNORECASE,
    ):
        extracted_dict["exclude_zero"] = True
        if not extracted_dict.get("limit") or extracted_dict.get("limit") == 50:
            extracted_dict["limit"] = 20

    # Price comparison: for SKU lookup without mismatch intent, show all comparison rows
    if tool_name == "get_price_comparison":
        combined_q = f"{sub_query} {user_query}".lower()
        if extracted_dict.get("search") and not re.search(
            r"\b(selisih|mismatch|beda|lebih\s+mahal|lebih\s+murah)\b", combined_q
        ):
            if extracted_dict.get("filter_status") in (None, "mismatch"):
                extracted_dict["filter_status"] = "all"

    # For get_stock_level & get_product_info: prevent LLM from setting product_id to SKU or internal code
    if tool_name in ("get_stock_level", "get_product_info") and extracted_dict.get("product_id"):
        raw_pid = extracted_dict["product_id"]
        # If product_id looks like a SKU (>= 100000 or string >= 5 digits) and user didn't ask for "id: ..."
        if not re.search(r"\b(?:id|product_id)\s*[:=]?\s*\d+\b", user_query, re.IGNORECASE):
            if isinstance(raw_pid, (int, str)) and len(str(raw_pid)) >= 5:
                if not extracted_dict.get("code") and not extracted_dict.get("search"):
                    extracted_dict["code"] = str(raw_pid).strip()
                extracted_dict.pop("product_id", None)
            elif extracted_dict.get("code") or extracted_dict.get("search"):
                extracted_dict.pop("product_id", None)

    # For get_product_info: move non-exact product names/model keywords (e.g. 'L210', 'stiker hologram')
    # from 'code' into 'search' so exact SKU lookup does not miss partial name matches.
    if tool_name == "get_product_info" and extracted_dict.get("code"):
        code_str = str(extracted_dict["code"]).strip()
        is_explicit_sku_ref = bool(
            re.search(rf"\b(?:sku|kode)\s*[:#]?\s*{re.escape(code_str)}\b", f"{sub_query} {user_query}", re.IGNORECASE)
        )
        is_exact_sku_shape = bool(re.match(r"^(?:\d{3,14}|[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+)$", code_str))
        if " " in code_str or (not is_exact_sku_shape and not is_explicit_sku_ref):
            if not extracted_dict.get("search"):
                extracted_dict["search"] = code_str
            extracted_dict.pop("code", None)

    # Clean branch name noise from vendor_name if accidentally included
    if extracted_dict.get("vendor_name"):
        v_clean = str(extracted_dict["vendor_name"]).strip()
        for b_name in sorted(BRANCH_COMPANY_MAP.keys(), key=lambda k: -len(k)):
            b_pattern = rf"\b(?:di\s*cabang|dicabang|cabang|di)?\s*{re.escape(b_name)}\b"
            v_clean = re.sub(b_pattern, "", v_clean, flags=re.IGNORECASE).strip().strip("-,:; ")
        if v_clean:
            extracted_dict["vendor_name"] = v_clean

    # Document number detection & include_lines
    doc_m = re.search(r"\b((?:[A-Za-z0-9_\-]+/)+\d{4}/\d+)\b", sub_query) or re.search(r"\b((?:[A-Za-z0-9_\-]+/)+\d{4}/\d+)\b", user_query)
    if doc_m:
        doc_code = doc_m.group(1).upper()
        if tool_name in ("fetch_vendor_invoices", "get_journal_entries"):
            if not extracted_dict.get("search") and not extracted_dict.get("entry_number"):
                if tool_name == "get_journal_entries":
                    extracted_dict["entry_number"] = doc_code
                else:
                    extracted_dict["search"] = doc_code
            extracted_dict["include_lines"] = True

    if any(kw in user_query.lower() for kw in ("detail", "rincian", "item", "isi")) and tool_name in ("fetch_vendor_invoices", "get_journal_entries"):
        extracted_dict["include_lines"] = True

    # 5. Staged Retry Relaxation (Strict Identifier Guard: do not pop document number / code / reference)
    has_doc_identifier = bool(
        extracted_dict.get("entry_number")
        or extracted_dict.get("reference")
        or extracted_dict.get("order_ref")
        or (
            extracted_dict.get("search")
            and re.search(
                r"\b(?:(?:[A-Za-z0-9_\-]+/)+\d{4}/\d+|[A-Za-z]{2,5}/SO/\d+)\b",
                str(extracted_dict.get("search")),
                re.IGNORECASE,
            )
        )
    )
    if retry_count == 1:
        extracted_dict.pop("date_from", None)
        extracted_dict.pop("date_to", None)
    elif retry_count >= 2:
        extracted_dict.pop("date_from", None)
        extracted_dict.pop("date_to", None)
        if not has_doc_identifier:
            if tool_name == "fetch_vendor_invoices":
                extracted_dict.pop("vendor_name", None)
                extracted_dict.pop("search", None)
            elif tool_name == "get_journal_entries":
                extracted_dict.pop("status", None)
                extracted_dict.pop("search", None)
            elif tool_name == "get_sales_orders":
                extracted_dict["status"] = "all"
                extracted_dict.pop("search", None)
                extracted_dict.pop("customer_name", None)
                extracted_dict.pop("product_search", None)
            elif tool_name == "get_contacts":
                extracted_dict["status"] = "all"
                extracted_dict.pop("product_search", None)
                extracted_dict.pop("search", None)
            elif domain == "sales":
                extracted_dict.pop("search", None)
                extracted_dict.pop("channel_name", None)
            elif domain == "marketplace":
                extracted_dict.pop("search", None)
                if not extracted_dict.get("account_id"):
                    extracted_dict.pop("marketplace", None)

    # Clean product codes / search terms across inventory and SKU-based marketplace tools
    if "code" in extracted_dict and isinstance(extracted_dict["code"], str):
        extracted_dict["code"] = clean_product_identifier(extracted_dict["code"])
    if "codes" in extracted_dict and isinstance(extracted_dict["codes"], list):
        extracted_dict["codes"] = [
            clean_product_identifier(c) for c in extracted_dict["codes"] if c and isinstance(c, str)
        ]
    if "sku" in extracted_dict and isinstance(extracted_dict["sku"], str):
        extracted_dict["sku"] = clean_product_identifier(extracted_dict["sku"])
    if "search" in extracted_dict and isinstance(extracted_dict["search"], str):
        if tool_name in ("get_product_info", "get_stock_level", "get_stock_history", "get_product_mapping_status", "get_price_comparison"):
            extracted_dict["search"] = clean_product_identifier(extracted_dict["search"])

    # 6. Schema Filtering / Validation
    try:
        validated_obj = target_schema(**extracted_dict)
        clean_params = validated_obj.model_dump(exclude_none=True)
    except Exception as exc:
        logger.debug("Parameter Pydantic validation fallback for tool '%s': %s", tool_name, exc)
        valid_keys = set(target_schema.model_fields.keys())
        clean_params = {k: v for k, v in extracted_dict.items() if k in valid_keys and v is not None}

    return clean_params


def _backfill_peer_entities(plans: List[Dict[str, Any]]) -> None:
    """Propagate resolved product/vendor entities across peer plans after parallel extraction."""
    if len(plans) <= 1:
        return
    for plan in plans:
        if not isinstance(plan, dict):
            continue
        tool_name = plan.get("tool")
        params = plan.setdefault("params", {})
        if (
            tool_name == "get_stock_level"
            and not params.get("code")
            and not params.get("codes")
            and not params.get("product_id")
        ):
            for peer in plans:
                if peer is plan or peer.get("domain") != "inventory":
                    continue
                peer_p = peer.get("params") or {}
                cand = peer_p.get("code") or peer_p.get("search")
                if (
                    cand
                    and _is_valid_product_identifier(str(cand))
                    and not is_vendor_name(str(cand))
                    and not is_account_code_or_name(str(cand))
                    and not is_accounting_term(str(cand))
                ):
                    params["code"] = clean_product_identifier(str(cand))
                    break
        elif (
            tool_name == "get_product_info"
            and not params.get("code")
            and not params.get("search")
            and not params.get("product_id")
        ):
            for peer in plans:
                if peer is plan or peer.get("domain") != "inventory":
                    continue
                peer_p = peer.get("params") or {}
                cand = peer_p.get("code") or peer_p.get("search")
                if (
                    cand
                    and _is_valid_product_identifier(str(cand))
                    and not is_vendor_name(str(cand))
                    and not is_account_code_or_name(str(cand))
                    and not is_accounting_term(str(cand))
                ):
                    params["search"] = clean_product_identifier(str(cand))
                    break
        elif (
            tool_name == "fetch_vendor_invoices"
            and not params.get("vendor_name")
            and not params.get("vendor_id")
        ):
            for peer in plans:
                if peer is plan or peer.get("domain") != "accounting":
                    continue
                peer_vendor = (peer.get("params") or {}).get("vendor_name")
                if peer_vendor and is_vendor_name(peer_vendor):
                    params["vendor_name"] = peer_vendor
                    break


def extractor_node(
    state: AgentState,
    llm: Optional[BaseChatModel] = None,
    ref_date: Optional[date] = None,
) -> Dict[str, Any]:
    """
    LangGraph pipeline node that extracts structured tool filter parameters from user query.
    Iterates across active_tools subagent plans with 3-tier entity resolution and fallback heuristics,
    while maintaining backward compatibility with legacy scalar detected_tool and extracted_params.

    Args:
        state: Current AgentState.
        llm: Optional BaseChatModel instance with structured output capabilities.
        ref_date: Reference anchor date for relative date resolution (defaults to get_current_erp_date()).

    Returns:
        Partial AgentState update dictionary with:
            - active_tools: Updated subagent plans with validated params dictionaries.
            - extracted_params: Dictionary of parsed and validated parameter filters for primary tool.
    """
    user_query = state.get("user_query")
    if not user_query and state.get("messages"):
        last_msg = state["messages"][-1]
        user_query = getattr(last_msg, "content", "") or ""

    user_query = (user_query or "").strip()
    from app.graph.nodes.router_decomposer import find_previous_user_query, is_retry_query
    if is_retry_query(user_query):
        prior_q = find_previous_user_query(state.get("messages"))
        if prior_q:
            user_query = prior_q

    raw_active_tools = state.get("active_tools")
    detected_tool = state.get("detected_tool")
    detected_domain = state.get("detected_domain")
    retry_count = int(state.get("retry_count") or 0)
    retry_instruction = state.get("retry_instruction")
    today = ref_date or get_current_erp_date()

    # Guard: If neither active_tools nor detected_tool was provided, return empty extracted params
    if not raw_active_tools and not detected_tool:
        return {"extracted_params": {}}

    # Normalize active_tools list with deep copies to maintain strict state immutability
    working_active_tools: List[Dict[str, Any]] = []
    if raw_active_tools:
        working_active_tools = [copy.deepcopy(plan) for plan in raw_active_tools]
    elif detected_tool:
        domain = detected_domain or TOOL_DOMAIN_MAP.get(detected_tool, "general")
        working_active_tools = [{
            "tool": detected_tool,
            "domain": domain,
            "sub_query": user_query,
            "params": {},
            "status": "pending",
        }]

    if not working_active_tools:
        return {"extracted_params": {}}

    # Resolve optional LLM model lazily only when Tier-1 fast path misses
    cached_model = llm
    model_resolved = llm is not None
    model_lock = threading.Lock()
    is_test_env = "unittest" in sys.modules or os.environ.get("FASTPRINT_TEST_ENV") == "true"

    def _get_model() -> Optional[BaseChatModel]:
        nonlocal cached_model, model_resolved
        with model_lock:
            if model_resolved:
                return cached_model
            model_resolved = True
            if not is_test_env:
                try:
                    cached_model = get_chat_model()
                except Exception as exc:
                    logger.debug("Lazy extractor LLM initialization unavailable: %s", exc)
                    cached_model = None
            return cached_model

    # Multi-tool extraction iteration (preserve already-VALID or strict-doc plans on retry)
    multi_status = state.get("multi_verification_status") or {}
    tool_counts = compute_multi_tool_counts(working_active_tools)

    plans_to_extract = []
    for idx, plan in enumerate(working_active_tools):
        t_name = plan.get("tool", "unnamed_tool")
        key = get_multi_tool_key(t_name, idx, tool_counts)
        if retry_count > 0 and should_preserve_plan_on_retry(plan, key, multi_status):
            continue
        plans_to_extract.append((idx, plan))

    if len(plans_to_extract) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(len(plans_to_extract), 4)) as executor:
            futures = [
                executor.submit(
                    _extract_single_tool_params,
                    plan,
                    working_active_tools,
                    user_query,
                    state,
                    llm,
                    today,
                    retry_count,
                    retry_instruction,
                    _get_model,
                )
                for _, plan in plans_to_extract
            ]
            extracted_results = [f.result() for f in futures]
        for (_, plan), extracted in zip(plans_to_extract, extracted_results):
            plan["params"] = extracted
    elif len(plans_to_extract) == 1:
        _, plan = plans_to_extract[0]
        extracted = _extract_single_tool_params(
            plan=plan,
            all_plans=working_active_tools,
            user_query=user_query,
            state=state,
            model=llm,
            today=today,
            retry_count=retry_count,
            retry_instruction=retry_instruction,
            model_factory=_get_model,
        )
        plan["params"] = extracted

    _backfill_peer_entities(working_active_tools)

    primary_params = working_active_tools[0].get("params", {}) if working_active_tools else {}

    result_update: Dict[str, Any] = {
        "active_tools": working_active_tools,
        "extracted_params": primary_params,
    }

    has_per_clause_scoping = len(state.get("sub_questions") or []) > 1 or any(
        p.get("company_id") is not None or p.get("is_multi_company") is not None
        for p in working_active_tools
    )
    if has_per_clause_scoping:
        for plan_item in working_active_tools:
            plan_params = plan_item.setdefault("params", {})
            eff_cid = plan_item.get("company_id") or plan_params.get("company_id") or state.get("company_id") or settings.DEFAULT_COMPANY_ID
            plan_params["company_id"] = eff_cid
            if plan_item.get("company_id") is None:
                plan_item["company_id"] = eff_cid
            if not plan_item.get("company_ids"):
                plan_item["company_ids"] = state.get("company_ids") if state.get("is_multi_company") else [eff_cid]
            if plan_item.get("is_multi_company") is None:
                plan_item["is_multi_company"] = len(plan_item["company_ids"]) > 1

        first_plan = working_active_tools[0]
        result_update["company_id"] = first_plan.get("company_id") or state.get("company_id") or settings.DEFAULT_COMPANY_ID
        result_update["company_ids"] = first_plan.get("company_ids") or [result_update["company_id"]]
        result_update["is_multi_company"] = bool(first_plan.get("is_multi_company"))
        if not result_update["is_multi_company"]:
            result_update["multi_company_results"] = {}
        return result_update

    multi_cids, is_all = resolve_all_branch_company_ids(user_query)
    explicit_branch = resolve_branch_company_id(user_query)

    if is_all:
        effective_cids = list(COMPANY_NAME_MAP.keys())
        result_update["is_multi_company"] = True
        result_update["company_ids"] = effective_cids
        primary_cid = effective_cids[0] if effective_cids else (state.get("company_id") or 1)
        result_update["company_id"] = primary_cid
        primary_params["company_id"] = primary_cid
    elif len(multi_cids) > 1:
        result_update["is_multi_company"] = True
        result_update["company_ids"] = multi_cids
        primary_cid = multi_cids[0]
        result_update["company_id"] = primary_cid
        primary_params["company_id"] = primary_cid
    elif explicit_branch is not None or len(multi_cids) == 1:
        # Explicit single company overrides prior multi-company state
        target_cid = explicit_branch if explicit_branch is not None else multi_cids[0]
        result_update["is_multi_company"] = False
        result_update["company_ids"] = [target_cid]
        result_update["company_id"] = target_cid
        result_update["multi_company_results"] = {}
        for plan_item in working_active_tools:
            plan_params = plan_item.setdefault("params", {})
            if plan_params:
                plan_params["company_id"] = target_cid
        if primary_params:
            primary_params["company_id"] = target_cid
    elif state.get("is_multi_company") and state.get("company_ids"):
        # Follow-up without branch mention inherits prior multi-company scope
        effective_cids = [int(c) for c in state.get("company_ids") if c]
        result_update["is_multi_company"] = True
        result_update["company_ids"] = effective_cids
        primary_cid = effective_cids[0] if effective_cids else (state.get("company_id") or 1)
        result_update["company_id"] = primary_cid
        primary_params["company_id"] = primary_cid
    else:
        # Standard single-company fallback
        result_update["is_multi_company"] = False
        result_update["multi_company_results"] = {}
        if state.get("company_id") is not None:
            session_cid = state.get("company_id")
            result_update["company_id"] = session_cid
            result_update["company_ids"] = [session_cid]
            for plan_item in working_active_tools:
                plan_params = plan_item.get("params")
                if isinstance(plan_params, dict) and "company_id" in plan_params:
                    plan_params["company_id"] = session_cid
            if "company_id" in primary_params:
                primary_params["company_id"] = session_cid
        elif primary_params.get("company_id") is not None:
            result_update["company_id"] = primary_params["company_id"]
            result_update["company_ids"] = [primary_params["company_id"]]

    return result_update

__all__ = [
    "TOOL_SCHEMA_REGISTRY",
    "ALLOWED_JOURNAL_STATUSES",
    "BRANCH_COMPANY_MAP",
    "COMPANY_NAME_MAP",
    "ALL_COMPANIES_KEYWORDS",
    "LOCATION_STOP_WORDS",
    "LOCATION_STOPWORDS",
    "PRODUCT_STOPWORDS",
    "STOCK_STOPWORDS",
    "STOCK_DELIMITERS",
    "TEMPORAL_KEYWORDS",
    "VENDOR_DELIMITERS",
    "VENDOR_NAME_STOPWORDS",
    "VENDOR_PREFIXES",
    "ANAPHORA_TOKENS",
    "ANAPHORIC_CLITICS",
    "_NON_PRODUCT_PHRASE_RE",
    "_TOP_SALES_STOPWORDS",
    "_SALES_CHANNEL_PATTERNS",
    "_CONTACT_STOPWORDS",
    "resolve_all_branch_company_ids",
    "resolve_branch_company_id",
    "strip_company_scope_keywords",
    "has_temporal_expression",
    "resolve_relative_dates",
    "is_vendor_name",
    "is_account_code_or_name",
    "is_accounting_term",
    "has_anaphoric_reference",
    "_is_valid_product_identifier",
    "_is_pure_branch_token",
    "_apply_sales_date_filters",
    "_extract_sales_channel",
    "_extract_prior_sales_context",
    "_extract_mp_platform",
    "_extract_mp_account_id",
    "_has_live_marketplace_intent",
    "_extract_mp_sku_token",
    "try_fast_path_stock_extraction",
    "_fallback_extract_vendor_invoice",
    "_fallback_extract_stock_level",
    "_fallback_extract_locations",
    "_fallback_extract_product_info",
    "_fallback_extract_account_balance",
    "_fallback_extract_journal_entries",
    "_fallback_extract_stock_history",
    "_fallback_extract_transfers",
    "_fallback_extract_low_stock_alerts",
    "_fallback_extract_stock_adjustments",
    "_fallback_extract_scrap_records",
    "_fallback_extract_warehouses",
    "_fallback_extract_product_categories",
    "_fallback_extract_general_ledger",
    "_fallback_extract_profit_and_loss",
    "_fallback_extract_balance_sheet",
    "_fallback_extract_vendor_bills",
    "_fallback_extract_bank_reconciliation",
    "_fallback_extract_top_selling_products",
    "_fallback_extract_contacts",
    "_fallback_extract_sales_orders",
    "_fallback_extract_sales_order_detail",
    "_fallback_extract_sales_summary",
    "_fallback_extract_account_health",
    "_fallback_extract_marketplace_orders",
    "_fallback_extract_kilat_diagnostics",
    "_fallback_extract_product_mapping_status",
    "_fallback_extract_stock_sync_simulation",
    "_fallback_extract_sync_execution_logs",
    "_fallback_extract_price_comparison",
    "_fallback_extract_fee_anomalies",
    "_fallback_extract_deduction_summary",
    "_fallback_extract_cashflow_balance",
    "_fallback_extract_competitor_insights",
    "_extract_single_tool_params",
    "_backfill_peer_entities",
    "extractor_node",
]
