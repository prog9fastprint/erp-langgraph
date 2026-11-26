"""
Extractor Node for FastPrint ERP LangGraph Agent Workflow (TASK-4.2 / Milestone 2).
Uses an extensible schema registry (TOOL_SCHEMA_REGISTRY) to extract typed parameters
across multi-tool active_tools subagent plans with LLM structured output, temporal
date resolution, 3-tier entity resolution, and deterministic heuristic fallbacks.
"""
import calendar
import copy
from datetime import date, timedelta
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type
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
from app.graph.state import AgentState, ToolCallPlan
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
    InventoryVelocityForecastFilterSchema,
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
    "get_inventory_velocity_forecast": InventoryVelocityForecastFilterSchema,
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

# Whitelist of valid accounting/journal/vendor bill statuses
ALLOWED_JOURNAL_STATUSES = {"draft", "posted", "post", "cancel", "unpaid", "paid"}

# Canonical mapping from Indonesian branch/city names to FastPrint ERP company_id
BRANCH_COMPANY_MAP: Dict[str, int] = {
    "kantor pusat": 1,
    "toko pusat": 2,
    "head office": 3,
    "head-office": 3,
    "ho": 3,
    "main store": 4,
    "mainstore": 4,
    "ms": 4,
    "makasar": 5,
    "makassar": 5,
    "fp makasar": 5,
    "fp makassar": 5,
    "mks": 5,
    "bandung": 6,
    "fp bandung": 6,
    "bdg": 6,
    "surabaya": 7,
    "fp surabaya": 7,
    "surabaya mt": 7,
    "surabaya mth": 7,
    "sby": 7,
    "jakarta": 8,
    "fp jakarta": 8,
    "jkt": 8,
    "toko fast print surabaya": 9,
    "toko surabaya": 9,
    "medan": 10,
    "fp medan": 10,
    "mdn": 10,
    "hr": 11,
    "sandy": 12,
    "bekasi": 13,
    "fp bekasi": 13,
    "bks": 13,
    "tangerang": 14,
    "fp tangerang": 14,
    "tgr": 14,
    "malang": 15,
    "fp malang": 15,
    "mlg": 15,
    "fastmart": 17,
    "pusat": 1,
}

# Canonical human-readable display names for company IDs
COMPANY_NAME_MAP: Dict[int, str] = {
    1: "Kantor Pusat",
    2: "Toko Pusat",
    3: "Head Office",
    4: "Main Store",
    5: "Makassar",
    6: "Bandung",
    7: "Surabaya",
    8: "Jakarta",
    9: "Toko Fast Print Surabaya",
    10: "Medan",
    11: "HR",
    12: "Sandy",
    13: "Bekasi",
    14: "Tangerang",
    15: "Malang",
    17: "Fastmart",
}

ALL_COMPANIES_KEYWORDS = (
    "semua cabang",
    "seluruh cabang",
    "semua perusahaan",
    "seluruh perusahaan",
    "semua toko",
    "seluruh toko",
    "konsolidasi",
    "semua branch",
    "seluruh branch",
    "all branches",
    "all branch",
    "all companies",
    "total nasional",
    "semua lokasi",
    "setiap lokasi",
    "semua gudang",
    "seluruh gudang",
    "setiap gudang",
    "distribusi stok",
    "semua rak",
    "setiap rak",
)

LOCATION_STOP_WORDS = {
    "setiap", "semua", "seluruh", "tiap", "masing-masing", "all", "every", "each", "gudang", "rak",
}

_MONTH_NAME_MAP: Dict[str, int] = {
    "januari": 1, "jan": 1,
    "februari": 2, "feb": 2,
    "maret": 3, "mar": 3,
    "april": 4, "apr": 4,
    "mei": 5,
    "juni": 6, "jun": 6,
    "juli": 7, "jul": 7,
    "agustus": 8, "agu": 8, "aug": 8,
    "september": 9, "sept": 9, "sep": 9,
    "oktober": 10, "okt": 10, "oct": 10,
    "november": 11, "nov": 11,
    "desember": 12, "des": 12, "dec": 12,
}

TEMPORAL_KEYWORDS = (
    "bulan", "hari", "minggu", "tahun", "kemarin", "today", "yesterday",
    "kuartal", "triwulan", "semester",
    "januari", "februari", "maret", "april", "mei", "juni",
    "juli", "agustus", "september", "oktober", "november", "desember",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "agu", "sept", "sep", "okt", "oct", "nov", "des", "dec",
    "lalu", "terakhir", "kebelakang", "sejak", "sampai", "hingga",
)


def resolve_all_branch_company_ids(text: Optional[str]) -> Tuple[List[int], bool]:
    """
    Resolve multiple branch/city mentions or 'semua cabang' into a list of company_ids.

    Returns:
        Tuple of (list_of_company_ids, is_all_companies_flag)
    """
    if not text:
        return [], False
    cleaned = text.lower().strip()

    # Check for all-companies / network-wide rollup keywords
    if any(re.search(rf"\b{re.escape(kw)}\b", cleaned) for kw in ALL_COMPANIES_KEYWORDS):
        return sorted(list(COMPANY_NAME_MAP.keys())), True

    matched_spans: List[Tuple[int, int, int]] = []  # (start_idx, end_idx, company_id)
    for branch_key in sorted(BRANCH_COMPANY_MAP.keys(), key=lambda k: -len(k)):
        pattern = rf"(?:\b(?:di\s*cabang|dicabang|cabang|di\s*toko|di\s*kantor|kantor|di)\s+)?\b{re.escape(branch_key)}\b"
        for m in re.finditer(pattern, cleaned):
            start, end = m.span()
            if not any(max(start, s) < min(end, e) for s, e, _ in matched_spans):
                matched_spans.append((start, end, BRANCH_COMPANY_MAP[branch_key]))

    matched_spans.sort(key=lambda x: x[0])
    seen = set()
    ordered_ids: List[int] = []
    for _, _, cid in matched_spans:
        if cid not in seen:
            seen.add(cid)
            ordered_ids.append(cid)

    return ordered_ids, False


def resolve_branch_company_id(text: Optional[str]) -> Optional[int]:
    """
    Resolve single Indonesian city or branch mention into official FastPrint ERP company_id.
    Matches phrases like 'cabang bandung', 'dicabang bandung', 'di bandung', 'kantor pusat', etc.
    """
    if not text:
        return None
    ids, is_all = resolve_all_branch_company_ids(text)
    if is_all:
        return None
    return ids[0] if ids else None


def resolve_prior_multi_company_scope(
    messages: Optional[Sequence[Any]],
    query: str,
) -> Optional[List[int]]:
    """
    Recover prior multi-company scope on continuation/breakdown follow-ups when
    upstream session state did not persist is_multi_company across turns.
    Stops immediately if the latest explicit HumanMessage narrowed to a single branch.
    """
    if not messages:
        return None
    q_low = (query or "").lower().strip()
    is_continuation = has_anaphoric_reference(q_low) or bool(
        re.search(r"\b(?:breakdown|jabarkan|rincian(?:nya)?|tabel|table|datanya|detail(?:nya)?|list|daftar)\b", q_low)
    )
    if not is_continuation:
        return None

    prior_msgs = list(messages[:-1]) if (q_low and q_low in str(getattr(messages[-1], "content", "") or "").lower()) else list(messages)
    for msg in reversed(prior_msgs[-6:]):
        content = str(getattr(msg, "content", "") or "").strip()
        if not content:
            continue
        msg_type = getattr(msg, "type", "")
        if isinstance(msg, HumanMessage) or msg_type in ("human", "user"):
            cids, is_all = resolve_all_branch_company_ids(content)
            if is_all:
                return cids
            if len(cids) > 1:
                return cids
            if len(cids) == 1 or resolve_branch_company_id(content) is not None:
                return None
        else:
            if any(
                marker in content
                for marker in (
                    "Multi-Perusahaan",
                    "Multi-Cabang",
                    "Lintas Cabang",
                    "TOTAL GABUNGAN",
                    "TOTAL NASIONAL",
                )
            ):
                return sorted(list(COMPANY_NAME_MAP.keys()))
    return None


def strip_company_scope_keywords(text: Optional[str]) -> str:
    """
    Strip multi-company rollup expressions ('semua cabang', 'seluruh cabang', '+ headoffice')
    and explicit branch mentions from a prior query so a fresh follow-up branch scope
    can be cleanly applied without keyword conflict.
    """
    if not text:
        return ""
    cleaned = text
    # 1. Strip ALL_COMPANIES_KEYWORDS
    for kw in ALL_COMPANIES_KEYWORDS:
        cleaned = re.sub(rf"(?:\buntuk|\bdi)?\s*\b{re.escape(kw)}\b", "", cleaned, flags=re.IGNORECASE)
    # 2. Strip "+ headoffice" / "+ head office" / "+ ho" / "+ pusat"
    cleaned = re.sub(r"\+\s*(?:head\s*office|headoffice|ho|kantor\s*pusat|pusat)\b", "", cleaned, flags=re.IGNORECASE)
    # 3. Strip individual branch mentions (e.g. "di bandung", "cabang surabaya")
    for branch_key in sorted(BRANCH_COMPANY_MAP.keys(), key=lambda k: -len(k)):
        pattern = rf"(?:\b(?:di\s*cabang|dicabang|cabang|di\s*toko|di\s*kantor|kantor|di)\s+)?\b{re.escape(branch_key)}\b"
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    # Clean up leftover punctuation and excessive whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _extract_standalone_year(text_low: str) -> Optional[int]:
    """Extract explicit 4-digit calendar year while ignoring document paths and SKU codes."""
    m_kw = re.search(r"\b(?:tahun|thn|periode)\s+(20\d\d)\b", text_low)
    if m_kw:
        return int(m_kw.group(1))
    for m_yr in re.finditer(r"(?<![/\-\.#])\b(20[2-3]\d)\b(?![/\-])", text_low):
        prefix = text_low[:m_yr.start()].rstrip()
        if re.search(r"\b(?:sku|kode|barcode)\s*[:#]?$", prefix):
            continue
        return int(m_yr.group(1))
    return None


def has_temporal_expression(text: Optional[str]) -> bool:
    """Check if query text contains explicit dates, quarters, months, years, or relative temporal keywords."""
    if not text:
        return False
    cleaned = text.lower()
    if any(re.search(rf"\b{re.escape(k)}\b", cleaned) for k in TEMPORAL_KEYWORDS):
        return True
    if re.search(r"\b(?:q|kuartal|triwulan)\s*[- ]?[1-4]\b", cleaned):
        return True
    if re.search(r"\b(?:semester|smt)\s*[- ]?[12]\b", cleaned):
        return True
    if re.search(r"\b\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?\b", cleaned):
        return True
    if re.search(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", cleaned):
        return True
    if _extract_standalone_year(cleaned) is not None:
        return True
    return False


def resolve_relative_dates(
    text: str,
    ref_date: Optional[date] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve relative Indonesian temporal expressions, quarters, semesters, months,
    and standalone calendar years into ISO YYYY-MM-DD date range.
    """
    today = ref_date or get_current_erp_date()
    cleaned = (text or "").lower()

    # 1. Check for explicit dates (YYYY-MM-DD, YYYY/MM/DD, DD/MM/YYYY, DD-MM-YYYY)
    explicit_dates = []
    for m in re.finditer(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", cleaned):
        try:
            explicit_dates.append(date(int(m.group(1)), int(m.group(2)), int(m.group(3))).strftime("%Y-%m-%d"))
        except ValueError:
            pass
    if not explicit_dates:
        for m in re.finditer(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b", cleaned):
            try:
                explicit_dates.append(date(int(m.group(3)), int(m.group(2)), int(m.group(1))).strftime("%Y-%m-%d"))
            except ValueError:
                pass

    if len(explicit_dates) >= 2:
        return explicit_dates[0], explicit_dates[1]
    elif len(explicit_dates) == 1:
        return explicit_dates[0], None

    # 2. Bulan ini (current month)
    if "bulan ini" in cleaned:
        first_day = today.replace(day=1).strftime("%Y-%m-%d")
        _, last_day_num = calendar.monthrange(today.year, today.month)
        last_day = today.replace(day=last_day_num).strftime("%Y-%m-%d")
        return first_day, last_day

    # 3. Bulan lalu (previous month)
    if "bulan lalu" in cleaned or "bulan kemarin" in cleaned:
        first_current = today.replace(day=1)
        prev_month_end = first_current - timedelta(days=1)
        first_day = prev_month_end.replace(day=1).strftime("%Y-%m-%d")
        last_day = prev_month_end.strftime("%Y-%m-%d")
        return first_day, last_day

    # 4. Hari ini (today)
    if "hari ini" in cleaned:
        today_str = today.strftime("%Y-%m-%d")
        return today_str, today_str

    # 5. Kemarin (yesterday)
    if "kemarin" in cleaned:
        yesterday_str = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        return yesterday_str, yesterday_str

    # 6. Minggu ini (current week)
    if "minggu ini" in cleaned:
        start_week = today - timedelta(days=today.weekday())
        end_week = start_week + timedelta(days=6)
        return start_week.strftime("%Y-%m-%d"), end_week.strftime("%Y-%m-%d")

    # 7. N hari terakhir (e.g. '3 hari terakhir', '7 hari terakhir')
    n_days_match = re.search(r"\b(\d+)\s+hari\s+(?:terakhir|kebelakang|lalu)\b", cleaned)
    if n_days_match:
        n_days = int(n_days_match.group(1))
        start_date = today - timedelta(days=max(0, n_days - 1))
        return start_date.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")

    # 8. Quarters (e.g. 'Q4 2026', 'kuartal 1', 'triwulan 3')
    q_match = re.search(r"\b(?:q|kuartal|triwulan)\s*[- ]?([1-4])\b", cleaned)
    if q_match:
        q_num = int(q_match.group(1))
        yr = _extract_standalone_year(cleaned) or today.year
        quarter_ranges = {
            1: (f"{yr}-01-01", f"{yr}-03-31"),
            2: (f"{yr}-04-01", f"{yr}-06-30"),
            3: (f"{yr}-07-01", f"{yr}-09-30"),
            4: (f"{yr}-10-01", f"{yr}-12-31"),
        }
        return quarter_ranges[q_num]

    # 9. Semesters (e.g. 'semester 1 2026', 'smt 2')
    s_match = re.search(r"\b(?:semester|smt)\s*[- ]?([12])\b", cleaned)
    if s_match:
        s_num = int(s_match.group(1))
        yr = _extract_standalone_year(cleaned) or today.year
        return (f"{yr}-01-01", f"{yr}-06-30") if s_num == 1 else (f"{yr}-07-01", f"{yr}-12-31")

    # 10. Named months (e.g. 'november 2026', 'bulan oktober', 'des 2025')
    month_pat = "|".join(sorted(_MONTH_NAME_MAP.keys(), key=len, reverse=True))
    m_month = re.search(rf"\b({month_pat})\b", cleaned)
    if m_month:
        mo = _MONTH_NAME_MAP[m_month.group(1)]
        yr = _extract_standalone_year(cleaned) or today.year
        _, last_day_num = calendar.monthrange(yr, mo)
        return f"{yr}-{mo:02d}-01", f"{yr}-{mo:02d}-{last_day_num:02d}"

    # 11. Explicit or standalone calendar year (e.g. 'tahun 2026', 'mainstore 2025')
    standalone_yr = _extract_standalone_year(cleaned)
    if standalone_yr is not None:
        return f"{standalone_yr}-01-01", f"{standalone_yr}-12-31"

    # 12. Tahun ini (current year)
    if "tahun ini" in cleaned:
        return f"{today.year}-01-01", f"{today.year}-12-31"

    # 13. Tahun lalu (previous year)
    if "tahun lalu" in cleaned or "tahun kemarin" in cleaned:
        prev_yr = today.year - 1
        return f"{prev_yr}-01-01", f"{prev_yr}-12-31"

    return None, None


VENDOR_DELIMITERS = r"\b(?:bulan|hari|tanggal|minggu|tahun|status|sebesar|di|pada|yang|belum|sudah|lunas|semua|dibayar|terbayar|unpaid|paid|dan)\b"
VENDOR_NAME_STOPWORDS = ("semua", "yang", "belum", "sudah", "lunas", "ini", "itu", "ada", "draft", "posted")
VENDOR_PREFIXES = ("pt", "cv", "ud", "toko", "tb", "pd", "fa")


def is_vendor_name(name: Optional[str]) -> bool:
    """Check if candidate name matches vendor / corporate entity patterns."""
    if not name:
        return False
    cleaned = name.strip().lower()
    for p in VENDOR_PREFIXES:
        if cleaned == p or cleaned.startswith(p + " ") or cleaned.startswith(p + "."):
            return True
        if f" {p} " in f" {cleaned} " or f" {p}." in f" {cleaned}":
            return True
    return any(p in cleaned for p in ("vendor ", "supplier ", "pemasok "))


def is_account_code_or_name(name: Optional[str]) -> bool:
    """Check if candidate token is an account code or reserved financial keyword."""
    if not name:
        return False
    cleaned = name.strip().lower()
    if re.match(r"^\d{4,8}$", cleaned) and cleaned[0] in "123456":
        return True
    return cleaned in {"kas", "bank", "piutang", "hutang", "modal", "neraca", "biaya", "pendapatan", "saldo"}


def is_accounting_term(name: Optional[str]) -> bool:
    """Check if candidate token contains accounting / financial document keywords."""
    if not name:
        return False
    cleaned = name.strip().lower()
    return any(t in cleaned for t in ("faktur", "tagihan", "invoice", "jurnal", "saldo", "rekening", "neraca"))


ANAPHORA_TOKENS = {"itu", "tersebut", "tadi", "ini", "terkait", "dia", "nya", "sana", "situ"}
ANAPHORIC_CLITICS = (
    "stoknya", "harganya", "barangnya", "produknya", "saldonya",
    "tagihannya", "fakturnya", "invoicenya", "lokasinya", "detailnya",
    "transaksinya", "jurnalnya", "omzetnya", "omsetnya", "ruginya",
    "labanya", "neracanya", "kasnya", "sisanya",
)


def has_anaphoric_reference(text: str) -> bool:
    """Check if text contains Indonesian anaphoric clitics, pronouns, or elliptical clauses."""
    cleaned = (text or "").lower()
    if any(clitic in cleaned for clitic in ANAPHORIC_CLITICS):
        return True
    if any(re.search(rf"\b{re.escape(token)}\b", cleaned) for token in ANAPHORA_TOKENS):
        return True
    if re.search(r"\b(?:berapa|gimana|bagaimana|mana|ada berapa)\s+(?:stok|harga|saldo|tagihan|faktur|detail)\b", cleaned):
        return True
    return False


def _fallback_extract_vendor_invoice(
    query: str,
    ref_date: Optional[date] = None,
) -> Dict[str, Any]:
    """Deterministic heuristic extraction for vendor invoice inquiries."""
    params: Dict[str, Any] = {}
    cleaned = re.sub(r"[^\w\)]+$", "", (query or "").strip(), flags=re.UNICODE).strip()

    vendor_pattern = re.search(
        rf"(?:vendor|supplier|pemasok)\s+([A-Za-z0-9\.\s\-\&\']+?)(?:\s*[,?!;:/\-]+\s*{VENDOR_DELIMITERS}|\s+{VENDOR_DELIMITERS}|$)",
        cleaned,
        re.IGNORECASE,
    )
    if vendor_pattern:
        v_name = vendor_pattern.group(1).strip().rstrip("?!.,; ")
        if v_name.lower() not in VENDOR_NAME_STOPWORDS:
            params["vendor_name"] = v_name
    else:
        prefix_pattern = re.search(
            rf"\b((?:PT|CV|UD|Toko)\.?\s+[A-Za-z0-9\.\s\-\&\']+?)(?:\s*[,?!;:/\-]+\s*{VENDOR_DELIMITERS}|\s+{VENDOR_DELIMITERS}|$)",
            cleaned,
            re.IGNORECASE,
        )
        if prefix_pattern:
            v_name = prefix_pattern.group(1).strip().rstrip("?!.,; ")
            if v_name.lower() not in VENDOR_NAME_STOPWORDS:
                params["vendor_name"] = v_name

    # Document number detection (e.g. PC/2026/00002, BDG/JE/2026/00009, INV/2026/001)
    doc_match = re.search(r"\b((?:[A-Za-z0-9_\-]+/)+\d{4}/\d+)\b", cleaned)
    if doc_match:
        params["search"] = doc_match.group(1).upper()
        params["include_lines"] = True

    query_lower = (query or "").lower()
    if "belum lunas" in query_lower or "unpaid" in query_lower or "belum dibayar" in query_lower:
        params["status"] = "unpaid"
    elif "lunas" in query_lower or "paid" in query_lower or "sudah dibayar" in query_lower:
        params["status"] = "paid"

    date_from, date_to = resolve_relative_dates(query, ref_date)
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to

    return params


STOCK_STOPWORDS = (
    "ini", "itu", "barang", "produk", "product", "item", "sku", "kode", "semua", "ada", "tersedia",
    "stok", "stock", "sisa", "stocknya", "stoknya",
    "yang", "mana", "paling", "tersedikit", "sedikit", "terbanyak", "terkecil", "terendah", "tertinggi", "dimana",
    "di", "ke", "dari", "pada", "untuk", "terkait", "tentang", "atau", "dan", "serta", "juga",
    "gudang", "lokasi", "customer", "pelanggan", "masing-masing", "masing", "bandingkan", "coba",
    "segera", "kira-kira", "butuh", "order", "forecasting", "prediksi",
    "faktur", "tagihan", "invoice", "jurnal", "saldo",
)
STOCK_DELIMITERS = r"\b(?:dan|atau|serta|di|ke|pada|untuk|bulan|hari|berapa|status|faktur|tagihan|invoice|saldo|jurnal|gudang|lokasi)\b"

_NON_PRODUCT_PHRASE_RE = re.compile(
    r"^(?:yang|mana)\b|\b(?:tersedikit|paling\s+sedikit|stok\s+menipis|stock\s+menipis|habis|kritis|terbanyak|terkecil|terendah)\b",
    re.IGNORECASE,
)


def _is_valid_product_identifier(value: Optional[str]) -> bool:
    """Reject interrogative/superlative clause fragments masquerading as SKU or product names."""
    if not value or not isinstance(value, str):
        return False
    cleaned = value.strip()
    if not cleaned:
        return False
    if _NON_PRODUCT_PHRASE_RE.search(cleaned):
        return False
    remaining = [
        w for w in cleaned.split()
        if w.lower() not in STOCK_STOPWORDS and w.lower() not in PRODUCT_STOPWORDS
    ]
    return bool(remaining)


def _is_pure_branch_token(token: Optional[str]) -> bool:
    """Return True if token is solely a branch/store location phrase with no SKU/product info."""
    if not token:
        return False
    return bool(
        resolve_branch_company_id(token) is not None
        and (
            re.match(r"^(?:cabang|toko|kantor)\s+", token, re.IGNORECASE)
            or token.lower().strip() in {
                "bandung", "jakarta", "surabaya", "makassar", "makasar",
                "medan", "bekasi", "tangerang", "malang",
                "kantor pusat", "toko pusat", "head office", "main store", "mainstore",
            }
        )
    )


def _fallback_extract_stock_level(
    query: str,
    history: Optional[list] = None,
) -> Dict[str, Any]:
    """Deterministic heuristic extraction for stock level inquiries when LLM is unavailable."""
    params: Dict[str, Any] = {}
    cleaned = re.sub(r"[^\w\)]+$", "", (query or "").strip(), flags=re.UNICODE).strip()

    loc_match = re.search(
        r"(?:\bdi\s+(?:gudang\s+|lokasi\s+)?|\b(?:gudang|lokasi)\s+)(?:cabang\s+|toko\s+|kantor\s+)?([A-Za-z0-9\_\-]+(?:\s+[A-Za-z0-9\_\-]+)?)",
        cleaned,
        re.IGNORECASE,
    )
    if loc_match:
        raw_loc = loc_match.group(1).strip()
        loc_stop_tokens = {
            "mana", "sini", "situ", "kantor", "semua", "ini", "itu",
            "gudang", "lokasi", "cabang", "saat", "sekarang",
            "berapa", "status", "bulan", "hari", "minggu", "tahun", "kemarin",
            *STOCK_STOPWORDS,
        }
        loc_words: List[str] = []
        for w in raw_loc.split():
            if w.lower() in loc_stop_tokens:
                break
            loc_words.append(w)
        if loc_words:
            params["location_name"] = " ".join(loc_words)

    invalid_sku_tokens = {
        "setiap", "gudang", "lokasi", "distribusi", "semua", "cabang",
        "2024", "2025", "2026", "2027", "2028", "stok", "stock", "barang", "produk", "item",
    }
    extracted_multi_skus = [
        clean_product_identifier(t) for t in re.findall(r"\b([A-Z0-9]+(?:-[A-Z0-9]+)+|\d{4,14})\b", cleaned)
        if clean_product_identifier(t)
        and clean_product_identifier(t).lower() not in invalid_sku_tokens
        and not _is_pure_branch_token(clean_product_identifier(t))
    ]
    extracted_multi_skus = list(dict.fromkeys(extracted_multi_skus))
    if len(extracted_multi_skus) > 1:
        params["codes"] = extracted_multi_skus
        return params

    # Batch multi-product name extraction (e.g., 'Cek sisa stok kertas A4 dan cek stok tinta hitam')
    if re.search(r"\b(?:dan|serta)\b|,", cleaned, re.IGNORECASE) and not params.get("location_name"):
        clause_parts = re.split(r"\s*(?:\b(?:dan|serta)\b|,)\s*", cleaned, flags=re.IGNORECASE)
        if len(clause_parts) > 1:
            multi_names: List[str] = []
            for part in clause_parts:
                p_clean = part.strip()
                if not p_clean:
                    continue
                if any(kw in p_clean.lower() for kw in ("faktur", "tagihan", "invoice", "saldo", "jurnal", "omzet", "omset")):
                    continue
                m_sub = re.search(
                    r"(?:(?:stok|stock|sisa|barang|produk|product|item)\s+)+(?:untuk\s+|terkait\s+|dari\s+|tentang\s+)?"
                    r"(?!di\b|pada\b|ke\b|gudang\b|lokasi\b|faktur\b|tagihan\b|saldo\b|jurnal\b|customer\b|pelanggan\b)"
                    r"([A-Za-z0-9\-\_\.\+\/\&\(\)\%\s]+?)$",
                    p_clean,
                    re.IGNORECASE,
                )
                if m_sub:
                    tok = clean_product_identifier(m_sub.group(1).strip().rstrip("?!.,; '\""))
                    words = [w for w in tok.split() if w.lower() not in STOCK_STOPWORDS]
                    cand = " ".join(words)
                    if words and _is_valid_product_identifier(cand) and not _is_pure_branch_token(cand) and not is_vendor_name(cand):
                        if cand not in multi_names:
                            multi_names.append(cand)
            if len(multi_names) > 1:
                params["codes"] = multi_names
                return params

    pattern = re.compile(
        r"(?:(?:stok|stock|sisa|barang|produk|product|item)\s+)+(?:untuk\s+|terkait\s+|dari\s+|tentang\s+)?"
        r"(?!di\b|pada\b|ke\b|gudang\b|lokasi\b|faktur\b|tagihan\b|saldo\b|jurnal\b|customer\b|pelanggan\b)"
        r"([A-Za-z0-9\-\_\.\+\/\&\(\)\%\s]+?)"
        rf"(?:\s*[\.\?\!\;]\s+|\s+{STOCK_DELIMITERS}|$)",
        re.IGNORECASE,
    )
    match = pattern.search(cleaned)
    if not match:
        match = re.search(
            rf"(?:untuk|terkait|tentang)\s+(?!di\b|pada\b|ke\b|gudang\b|lokasi\b|faktur\b|tagihan\b|saldo\b|jurnal\b|customer\b|pelanggan\b)([A-Za-z0-9\-_\.\+\/\&\(\)\%\s]+?)(?:\s*[\.\?\!\;]\s+|\s+{STOCK_DELIMITERS}|$)",
            cleaned,
            re.IGNORECASE,
        )
    if match:
        token = match.group(1).strip().rstrip("?!.,; '\"")
        token = clean_product_identifier(token)
        words = [w for w in token.split() if w.lower() not in STOCK_STOPWORDS]
        candidate = " ".join(words)
        if words and _is_valid_product_identifier(candidate) and not _is_pure_branch_token(candidate):
            params["code"] = candidate
            return params

    tokens = re.findall(r"\b([A-Z0-9]+(?:-[A-Z0-9]+)+|\d{3,14})\b", cleaned)
    if tokens:
        clean_tokens = [
            clean_product_identifier(t) for t in tokens
            if clean_product_identifier(t)
            and clean_product_identifier(t).lower() not in invalid_sku_tokens
            and not _is_pure_branch_token(clean_product_identifier(t))
        ]
        clean_tokens = list(dict.fromkeys(clean_tokens))
        if len(clean_tokens) > 1:
            params["codes"] = clean_tokens
            return params
        elif clean_tokens:
            params["code"] = clean_tokens[0]
            return params

    if (
        not params.get("code")
        and not params.get("codes")
        and not params.get("product_id")
        and not params.get("location_name")
        and re.search(r"\bgudang\b", cleaned, re.IGNORECASE)
        and not has_anaphoric_reference(cleaned)
        and not re.search(r"\b(?:itu|tersebut|tadi)\b", cleaned, re.IGNORECASE)
        and resolve_branch_company_id(cleaned) is None
    ):
        params["allow_company_scan"] = True
        params["limit"] = 50
        return params

    is_referential = (
        has_anaphoric_reference(cleaned)
        or any(ref in cleaned.lower() for ref in ("itu", "tersebut", "tadi", "produk", "barang", "coba lagi", "cek lagi", "periksa lagi", "cari lagi", "ulang", "retry"))
    )
    # Check history if referential or if code is missing but history exists (follow-up slot filling)
    if (is_referential or not params.get("code")) and history:
        prior_msgs = history[:-1] if (history and cleaned and cleaned.lower() in str(getattr(history[-1], "content", "") or "").lower()) else history
        for prev_msg in reversed(prior_msgs):
            content = getattr(prev_msg, "content", "") or ""
            is_human = isinstance(prev_msg, HumanMessage) or getattr(prev_msg, "type", "") == "human"

            prev_tokens = re.findall(r"\b([A-Z0-9]+(?:-[A-Z0-9]+)+|\d{3,14})\b", content)
            if prev_tokens:
                sku_m = re.search(r"\b(?:SKU|kode)\s*[:\-]?\s*([A-Z0-9\-]+|\d{3,14})\b", content, re.IGNORECASE)
                if sku_m:
                    params["code"] = clean_product_identifier(sku_m.group(1).rstrip("."))
                else:
                    params["code"] = clean_product_identifier(prev_tokens[0].rstrip("."))
                break

            if is_human:
                prev_stock = _fallback_extract_stock_level(content)
                if prev_stock.get("code"):
                    params["code"] = prev_stock["code"]
                    break
                prev_info = _fallback_extract_product_info(content)
                if prev_info.get("search") and not is_vendor_name(prev_info["search"]) and not is_account_code_or_name(prev_info["search"]) and not is_accounting_term(prev_info["search"]):
                    params["code"] = prev_info["search"]
                    break

    return params


LOCATION_STOPWORDS = (
    "gudang", "lokasi", "warehouse",
    "apa", "saja", "semua", "yang", "ada", "aktif", "ini", "itu", "daftar", "list",
    "cek", "info", "mana", "cari",
)


def _fallback_extract_locations(query: str) -> Dict[str, Any]:
    """Deterministic heuristic extraction for inventory location inquiries."""
    params: Dict[str, Any] = {}
    cleaned = re.sub(r"[^\w\)]+$", "", (query or "").strip(), flags=re.UNICODE).strip()
    match = re.search(
        r"(?:(?:gudang|lokasi|warehouse)\s+)+([A-Za-z0-9\_\-\s]+?)(?:\s+\b(?:aktif|semua|yang|ini|itu|di|pada)\b|$)",
        cleaned,
        re.IGNORECASE,
    )
    if not match:
        match = re.search(r"(?:(?:gudang|lokasi|warehouse)\s+)+([A-Za-z0-9\_\-]+)", cleaned, re.IGNORECASE)
    if match:
        token = match.group(1).strip().rstrip("?!.,; ")
        words = [w for w in token.split() if w.lower() not in LOCATION_STOPWORDS]
        if words:
            params["search"] = " ".join(words)
    return params


PRODUCT_STOPWORDS = {
    "berapa", "harga", "beli", "jual", "belinya", "jualnya", "cek", "info",
    "detail", "spesifikasi", "stok", "stoknya", "stock", "stocknya",
    "produk", "product", "barang", "item", "sku", "kode", "ada",
    "yang", "mana", "paling", "tersedikit", "sedikit", "terbanyak", "terkecil", "terendah", "tertinggi", "dimana",
    "dan", "serta", "di", "ke", "dari", "pada", "untuk", "ini", "itu",
    "nya", "semua", "daftar", "list",
    # Anaphoric reference tokens
    "tersebut", "tadi", "terkait", "dimaksud", "dia", "tersebutnya",
}


def _fallback_extract_product_info(
    query: str,
    history: Optional[list] = None,
) -> Dict[str, Any]:
    """Deterministic heuristic extraction for product info inquiries."""
    params: Dict[str, Any] = {}
    cleaned = re.sub(r"[^\w\)]+$", "", (query or "").strip(), flags=re.UNICODE).strip()

    tokens = re.findall(r"\b([A-Z0-9]+(?:-[A-Z0-9]+)+|\d{4,8})\b", cleaned)
    if tokens:
        token = tokens[0]
        if not is_account_code_or_name(token) and not is_vendor_name(token):
            params["code"] = clean_product_identifier(token)
            return params

    m = re.search(
        r"(?:harga(?:\s+(?:beli|jual))?|detail\s+(?:produk|barang|product|item)?|info\s+(?:produk|barang|product|item)?|produk|barang|product|item|cari)\s+(?!dan\b|serta\b|di\b|ke\b|pada\b|untuk\b|faktur\b|tagihan\b|saldo\b|jurnal\b)([A-Za-z0-9\s\.\-\+\/\&\(\)\%]+?)(?:\s+\b(?:dan|serta|di|ke|pada|untuk|bulan|hari|berapa|status|faktur|tagihan|invoice|saldo|jurnal)\b|$)",
        cleaned,
        re.IGNORECASE,
    )
    if m:
        term = m.group(1).strip()
        term = clean_product_identifier(term)
        words = [w for w in term.split() if w.lower() not in PRODUCT_STOPWORDS]
        if words:
            candidate = " ".join(words)
            if (
                _is_valid_product_identifier(candidate)
                and not is_vendor_name(candidate)
                and not is_account_code_or_name(candidate)
                and not is_accounting_term(candidate)
                and not _is_pure_branch_token(candidate)
            ):
                params["search"] = candidate
                return params

    is_referential = (
        has_anaphoric_reference(cleaned)
        or any(ref in cleaned.lower() for ref in ("itu", "tersebut", "tadi", "produk", "barang", "coba lagi", "cek lagi", "periksa lagi", "cari lagi", "ulang", "retry"))
    )
    if is_referential and history:
        prior_msgs = history[:-1] if (history and cleaned and cleaned.lower() in str(getattr(history[-1], "content", "") or "").lower()) else history
        for prev_msg in reversed(prior_msgs):
            content = getattr(prev_msg, "content", "") or ""
            is_human = isinstance(prev_msg, HumanMessage) or getattr(prev_msg, "type", "") == "human"

            # 1. Check for explicit SKU pattern (e.g. SKU: 91030300) in AI or Human message
            sku_m = re.search(r"\b(?:SKU|kode)\s*[:\-]?\s*([A-Za-z0-9\-]+|\d{3,14})\b", content, re.IGNORECASE)
            if sku_m:
                cand = clean_product_identifier(sku_m.group(1).rstrip("."))
                if cand and not is_accounting_term(cand) and not is_account_code_or_name(cand):
                    params["code"] = cand
                    break

            # 2. Check for alphanumeric SKU / code tokens (excluding pure 4-digit years like 2024..2030)
            prev_tokens = re.findall(r"\b([A-Z0-9]+(?:-[A-Z0-9]+)+|\d{5,14})\b", content)
            if prev_tokens:
                token = prev_tokens[0].rstrip(".")
                if not is_account_code_or_name(token) and not is_vendor_name(token):
                    params["code"] = token
                    break

            # 3. For Human message: extract product from stock or product query
            if is_human:
                prev_stock = _fallback_extract_stock_level(content)
                if prev_stock.get("code"):
                    params["search"] = prev_stock["code"] if " " in str(prev_stock["code"]) else prev_stock["code"]
                    break

            # 4. Check product info heuristic extraction
            prev_info = _fallback_extract_product_info(content)
            if prev_info.get("code") and not is_vendor_name(prev_info["code"]) and not is_account_code_or_name(prev_info["code"]) and not is_accounting_term(prev_info["code"]):
                params["code"] = prev_info["code"]
                break
            if prev_info.get("search") and not is_vendor_name(prev_info["search"]) and not is_account_code_or_name(prev_info["search"]) and not is_accounting_term(prev_info["search"]):
                params["search"] = prev_info["search"]
                break

    return params


def _fallback_extract_account_balance(
    query: str,
    ref_date: Optional[date] = None,
) -> Dict[str, Any]:
    """Deterministic heuristic extraction for account balance inquiries."""
    params: Dict[str, Any] = {}
    cleaned = query.strip()
    q_low = cleaned.lower()

    acc_codes = re.findall(r"\b([1-6]\d{3,7})\b", cleaned)
    if acc_codes:
        params["account_code"] = acc_codes[0]

    has_bank_word = bool(re.search(r"\b(?:rekening\s*bank|bank(?:\s*(?:&|and|dan)\s*cash)?|rekening)\b", q_low))
    has_kas_word = bool(re.search(r"\b(?:kas|cash)\b", q_low))
    if (
        (has_bank_word and has_kas_word)
        or re.search(
            r"\b(?:rekening\s+bank|bank\s*(?:&|and|dan)\s*cash|kas\s*(?:&|dan)\s*(?:rekening\s+)?bank|semua\s+(?:bank|kas|rekening))\b",
            q_low,
        )
    ):
        params["account_type"] = "Bank and Cash"

    if re.search(r"\b(?:aktif|non[\s\-]?zero|tidak\s+nol|berisi)\b", q_low):
        params["non_zero_only"] = True

    m_bank_brand = re.search(
        r"\b(bca|mandiri|bri|bni|cimb(?:\s+niaga)?|permata|danamon|ocbc|bsi|btn|mega|panin|uob|maybank)\b",
        q_low,
    )
    if m_bank_brand and not params.get("account_code"):
        brand = m_bank_brand.group(1).strip()
        params["search"] = brand.upper() if len(brand) <= 4 else brand.title()

    date_from, date_to = resolve_relative_dates(cleaned, ref_date)
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to

    return params


def _fallback_extract_journal_entries(
    query: str,
    ref_date: Optional[date] = None,
) -> Dict[str, Any]:
    """Deterministic heuristic extraction for journal entries inquiries."""
    params: Dict[str, Any] = {}
    cleaned = query.strip()

    entry_match = re.search(r"\b((?:[A-Za-z0-9_\-]+/)+\d{4}/\d+)\b", cleaned, re.IGNORECASE)
    if entry_match:
        params["entry_number"] = entry_match.group(1).upper()
        params["include_lines"] = True

    query_lower = cleaned.lower()
    if "draft" in query_lower:
        params["status"] = "draft"
    elif "posted" in query_lower or "post" in query_lower:
        params["status"] = "posted"
    elif "cancel" in query_lower or "batal" in query_lower:
        params["status"] = "cancel"

    if not params.get("entry_number"):
        m_search = re.search(
            r"(?:jurnal|transaksi|catatan)\s+([A-Za-z0-9\s]+?)(?:\s+\b(?:bulan|hari|tanggal|status|sebesar|di|pada)\b|$)",
            cleaned,
            re.IGNORECASE,
        )
        if m_search:
            kw = m_search.group(1).strip()
            kw_words = [
                w for w in kw.split()
                if w.lower() not in (
                    "ini", "itu", "ada", "semua", "draft", "posted", "cancel",
                    "bulan", "hari", "tanggal", "minggu", "tahun",
                )
            ]
            if kw_words:
                cand = " ".join(kw_words)
                if not is_vendor_name(cand):
                    params["search"] = cand

    date_from, date_to = resolve_relative_dates(cleaned, ref_date)
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to

    return params


def _fallback_extract_stock_history(query: str, ref_date: Optional[date] = None) -> Dict[str, Any]:
    """Deterministic heuristic extraction for stock history inquiries."""
    params: Dict[str, Any] = {}
    cleaned = query.strip()
    tokens = re.findall(r"\b([A-Z0-9]+(?:-[A-Z0-9]+)+|\d{4,8})\b", cleaned)
    if tokens:
        params["code"] = tokens[0]
    else:
        info = _fallback_extract_product_info(cleaned)
        if info.get("code"):
            params["code"] = info["code"]
        elif info.get("search"):
            params["code"] = info["search"]

    date_from, date_to = resolve_relative_dates(cleaned, ref_date)
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to
    return params


def _fallback_extract_transfers(query: str, ref_date: Optional[date] = None) -> Dict[str, Any]:
    """Deterministic heuristic extraction for stock transfers/DO/receipt inquiries."""
    params: Dict[str, Any] = {}
    cleaned = query.strip()

    # Document reference e.g. WH/IN/0001, WH/OUT/0002
    ref_match = re.search(r"\b((?:[A-Za-z0-9_\-]+/)+\d{4}/\d+|WH/(?:IN|OUT)/(?:\d{4}/)?\d+)\b", cleaned, re.IGNORECASE)
    if ref_match:
        params["ref"] = ref_match.group(1).upper()

    # Source document PO / SO
    po_so_match = re.search(r"\b(PO/\d{4}/\d+|SO/\d{4}/\d+)\b", cleaned, re.IGNORECASE)
    if po_so_match:
        params["source_document"] = po_so_match.group(1).upper()

    # Status
    for st in ("draft", "waiting", "in progress", "done", "canceled"):
        if re.search(rf"\b{re.escape(st)}\b", cleaned, re.IGNORECASE):
            params["status"] = st.title()
            break

    date_from, date_to = resolve_relative_dates(cleaned, ref_date)
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to
    return params


def _fallback_extract_low_stock_alerts(query: str) -> Dict[str, Any]:
    """Deterministic heuristic extraction for low stock alert inquiries."""
    params: Dict[str, Any] = {}
    cleaned = query.strip()
    m_num = re.search(r"(?:threshold|ambang batas|minimum|di bawah)\s*(\d+(?:\.\d+)?)", cleaned, re.IGNORECASE)
    if m_num:
        try:
            params["threshold"] = float(m_num.group(1))
        except ValueError:
            pass
    if re.search(r"\b(?:tersedikit|paling\s+sedikit|terkecil|terendah)\b", cleaned, re.IGNORECASE):
        params["exclude_zero"] = True
        params["limit"] = 20
    return params


def _fallback_extract_stock_adjustments(query: str, ref_date: Optional[date] = None) -> Dict[str, Any]:
    """Deterministic heuristic extraction for stock adjustments/opname inquiries."""
    params: Dict[str, Any] = {}
    cleaned = query.strip()
    for st in ("accepted", "pending", "cancelled"):
        if re.search(rf"\b{re.escape(st)}\b", cleaned, re.IGNORECASE):
            params["status"] = st
            break

    date_from, date_to = resolve_relative_dates(cleaned, ref_date)
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to
    return params


def _fallback_extract_scrap_records(query: str, ref_date: Optional[date] = None) -> Dict[str, Any]:
    """Deterministic heuristic extraction for scrap inquiries."""
    params: Dict[str, Any] = {}
    cleaned = query.strip()
    date_from, date_to = resolve_relative_dates(cleaned, ref_date)
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to
    return params


def _fallback_extract_warehouses(query: str) -> Dict[str, Any]:
    """Deterministic heuristic extraction for warehouses inquiries."""
    params: Dict[str, Any] = {}
    cleaned = query.strip()
    term = re.sub(r"\b(?:daftar|list|semua|master|info|data)(?:\s+(?:daftar|list|semua|master|info|data))*\s+(?:gudang|warehouses?)\b", "", cleaned, flags=re.IGNORECASE).strip()
    if term and term.lower() not in ("apa saja", "yang ada", "aktif", "tersedia"):
        params["search"] = term
    return params


def _fallback_extract_product_categories(query: str) -> Dict[str, Any]:
    """Deterministic heuristic extraction for product category inquiries."""
    params: Dict[str, Any] = {}
    cleaned = query.strip()
    term = re.sub(r"\b(?:daftar|list|semua|master|info|data)(?:\s+(?:daftar|list|semua|master|info|data))*\s*kategori(?:\s+(?:produk|barang))?\b", "", cleaned, flags=re.IGNORECASE).strip()
    if term and term.lower() not in ("apa saja", "yang ada", "aktif", "tersedia"):
        params["search"] = term
    return params


def _fallback_extract_inventory_velocity_forecast(
    query: str,
    today: Optional[date] = None,
) -> Dict[str, Any]:
    """Deterministic heuristic extraction for inventory.get_inventory_velocity_forecast."""
    ref_day = today or get_current_erp_date()
    cleaned = (query or "").strip()
    params: Dict[str, Any] = {"lead_time_days": 7, "target_cover_days": 30, "limit": 10}

    df, dt = resolve_relative_dates(cleaned, ref_day)
    if df or dt:
        if df:
            params["date_from"] = df
        if dt:
            params["date_to"] = dt
    else:
        params["date_from"] = (ref_day - timedelta(days=90)).strftime("%Y-%m-%d")
        params["date_to"] = ref_day.strftime("%Y-%m-%d")

    m_lim = re.search(r"\b(?:top|peringkat|limit)\s*(\d+)\b|\b(\d+)\s+(?:produk|barang|item)\b", cleaned, re.IGNORECASE)
    if m_lim:
        raw_lim = m_lim.group(1) or m_lim.group(2)
        try:
            params["limit"] = max(1, min(50, int(raw_lim)))
        except ValueError:
            pass

    m_lt = re.search(r"\blead\s*time\s*(\d+)\s*hari\b", cleaned, re.IGNORECASE)
    if m_lt:
        params["lead_time_days"] = max(1, min(90, int(m_lt.group(1))))

    m_cov = re.search(r"\bcover\s*(\d+)\s*hari\b", cleaned, re.IGNORECASE)
    if m_cov:
        params["target_cover_days"] = max(1, min(365, int(m_cov.group(1))))

    m_sku = re.search(r"\bsku\s*[:#]?\s*([A-Za-z0-9._\-]{3,30})\b", cleaned, re.IGNORECASE)
    if m_sku:
        params["search"] = clean_product_identifier(m_sku.group(1))

    return params


def resolve_pnl_drilldown_from_history(
    query: str,
    messages: Optional[Sequence[Any]] = None,
) -> Optional[Dict[str, str]]:
    """Resolve an account code & name from prior P&L / Balance Sheet AI message for GL drill-down."""
    if not query or not messages:
        return None
    q_clean = re.sub(r"[?!.,;:'\"]+$", "", query.strip())
    m_req = re.match(
        r"^(?:(?:coba|tolong|mohon|bisa)\s+)*(?:berikan|tampilkan|lihat(?:kan)?|cek|kasih|minta)\s+"
        r"(?:detail|rincian|mutasi|transaksi|buku\s+besar)(?:\s+dari|\s+akun|\s+untuk)?\s+(.+)$",
        q_clean,
        re.IGNORECASE,
    )
    if not m_req:
        return None
    target_raw = re.sub(r"^(?:dari|akun|account)\s+", "", m_req.group(1).strip(), flags=re.IGNORECASE).strip()
    if not target_raw or len(target_raw) < 3:
        return None
    if re.match(r"^(?:kontak|contact|customer|pelanggan|so|sales\s*order|order|pesanan|produk|barang|sku)\b", target_raw, re.IGNORECASE):
        return None
    target_low = target_raw.lower()

    prior_msgs = list(messages)
    if prior_msgs and (getattr(prior_msgs[-1], "content", "") or "").strip().lower() == query.strip().lower():
        prior_msgs = prior_msgs[:-1]

    account_candidates: List[Tuple[str, str]] = []
    for msg in reversed(prior_msgs):
        msg_type = getattr(msg, "type", "")
        if msg_type != "ai" and msg.__class__.__name__ != "AIMessage":
            continue
        content = str(getattr(msg, "content", "") or "")
        if not content:
            continue
        for m_row in re.finditer(r"`?([1-6]\d{2,7}(?:[\.\-]\d+)*)`?\s*\|\s*([^|\n]+)", content):
            account_candidates.append((m_row.group(1).strip(), m_row.group(2).strip()))
        for m_dash in re.finditer(r"`?([1-6]\d{2,7}(?:[\.\-]\d+)*)`?\s*[-–:]\s*([^\n(|]+)", content):
            account_candidates.append((m_dash.group(1).strip(), m_dash.group(2).strip()))
        for m_paren in re.finditer(r"([^\n(|]+?)\s*\(\s*`?([1-6]\d{2,7}(?:[\.\-]\d+)*)`?\s*\)", content):
            account_candidates.append((m_paren.group(2).strip(), m_paren.group(1).strip()))
        if account_candidates:
            break

    if not account_candidates:
        return None

    for code, name in account_candidates:
        name_low = name.lower().strip()
        code_low = code.lower().strip()
        if target_low == code_low or target_low == name_low:
            return {"account_code": code, "account_name": name}
    target_tokens = [t for t in re.findall(r"[a-z0-9]+", target_low) if len(t) >= 2]
    for code, name in account_candidates:
        name_low = name.lower().strip()
        if target_low in name_low or name_low in target_low:
            return {"account_code": code, "account_name": name}
        if target_tokens and all(t in name_low for t in target_tokens):
            return {"account_code": code, "account_name": name}
    return None


def _fallback_extract_general_ledger(
    query: str,
    ref_date: Optional[date] = None,
    messages: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Deterministic heuristic extraction for general ledger inquiries."""
    params: Dict[str, Any] = {"limit": 50, "offset": 0}
    cleaned = (query or "").strip()
    acc_codes = re.findall(r"\b([1-6]\d{2,7}(?:[\.\-]\d+)*)\b", cleaned)
    if acc_codes:
        params["account_code"] = acc_codes[0]
    elif messages:
        drill = resolve_pnl_drilldown_from_history(cleaned, messages)
        if drill and drill.get("account_code"):
            params["account_code"] = drill["account_code"]

    date_from, date_to = resolve_relative_dates(cleaned, ref_date)
    if not date_from and not date_to and messages:
        prior_msgs = list(messages)
        if prior_msgs and (getattr(prior_msgs[-1], "content", "") or "").strip().lower() == cleaned.lower():
            prior_msgs = prior_msgs[:-1]
        for msg in reversed(prior_msgs):
            content = str(getattr(msg, "content", "") or "")
            m_per = re.search(r"Periode\s*:\s*(\d{4}-\d{2}-\d{2})\s*(?:s/d|-|to)\s*(\d{4}-\d{2}-\d{2})", content, re.IGNORECASE)
            if m_per:
                date_from, date_to = m_per.group(1), m_per.group(2)
                params["_inherited_dates"] = True
                break
            if getattr(msg, "type", "") == "human" or isinstance(msg, HumanMessage):
                h_df, h_dt = resolve_relative_dates(content, ref_date)
                if h_df or h_dt:
                    date_from, date_to = h_df, h_dt
                    params["_inherited_dates"] = True
                    break

    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to
    return params


def _fallback_extract_profit_and_loss(query: str, ref_date: Optional[date] = None) -> Dict[str, Any]:
    """Deterministic heuristic extraction for profit and loss inquiries (compact by default)."""
    params: Dict[str, Any] = {"include_breakdown": False}
    cleaned = query.strip().lower()
    if any(k in cleaned for k in ("rincian", "detail", "breakdown", "per akun", "daftar akun", "hierarki")):
        params["include_breakdown"] = True
    date_from, date_to = resolve_relative_dates(query, ref_date)
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to
    return params


def _fallback_extract_balance_sheet(query: str, ref_date: Optional[date] = None) -> Dict[str, Any]:
    """Deterministic heuristic extraction for balance sheet inquiries (compact by default)."""
    params: Dict[str, Any] = {"include_breakdown": False}
    cleaned = query.strip().lower()
    if any(k in cleaned for k in ("rincian", "detail", "breakdown", "per akun", "daftar akun", "hierarki")):
        params["include_breakdown"] = True
    date_from, date_to = resolve_relative_dates(query, ref_date)
    if date_to:
        params["as_of_date"] = date_to
    elif date_from:
        params["as_of_date"] = date_from
    return params


def _fallback_extract_vendor_bills(query: str, ref_date: Optional[date] = None) -> Dict[str, Any]:
    """Deterministic heuristic extraction for vendor bills and AP aging."""
    params: Dict[str, Any] = {"due_status": "all", "limit": 50, "offset": 0}
    cleaned = query.strip().lower()
    if any(k in cleaned for k in ("overdue", "lewat jatuh tempo", "jatuh tempo", "terlambat")):
        params["due_status"] = "overdue"
    elif any(k in cleaned for k in ("due soon", "segera jatuh tempo", "mendekati tempo", "akan tempo")):
        params["due_status"] = "due_soon"

    for st in ("draft", "post", "posted", "paid", "cancel"):
        if re.search(rf"\b{re.escape(st)}\b", cleaned):
            params["status"] = "post" if st == "posted" else st
            break

    inv_data = _fallback_extract_vendor_invoice(query, ref_date)
    if inv_data.get("vendor_name"):
        params["vendor_name"] = inv_data["vendor_name"]
    date_from, date_to = resolve_relative_dates(query, ref_date)
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to
    return params


def _fallback_extract_bank_reconciliation(query: str, ref_date: Optional[date] = None) -> Dict[str, Any]:
    """Deterministic heuristic extraction for bank reconciliation summary."""
    params: Dict[str, Any] = {}
    cleaned = query.strip()
    m_brand = re.search(
        r"\b((?:bank\s+)?(?:bca|mandiri|bri|bni|cimb(?:\s+niaga)?|permata|danamon|ocbc|bsi|btn|mega|panin|uob|maybank)|kas\s+(?:kecil|besar))\b",
        cleaned,
        re.IGNORECASE,
    )
    if m_brand:
        params["bank_name"] = m_brand.group(1).strip()
    else:
        m_generic_bank = re.search(r"\brekening\s+(?:bank\s+)?([A-Za-z0-9\-]{2,30})\b", cleaned, re.IGNORECASE)
        if m_generic_bank:
            cand = m_generic_bank.group(1).strip()
            if cand.lower() not in {"koran", "bank", "ini", "itu", "mana", "cabang", "pusat", "yang", "perusahaan"}:
                params["bank_name"] = cand
    date_from, date_to = resolve_relative_dates(cleaned, ref_date)
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to
    return params


def try_fast_path_stock_extraction(
    sub_query: str,
    user_query: str,
    company_id: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Fast-path (< 5ms) deterministic parameter extractor for single/batch product stock level inquiries.
    Matches explicit SKU/code patterns (e.g. 'SKU 91010600', 'kode 91010600', '91010600', 'FP-KARET-42').
    Bypasses expensive LLM calls when query is simple and unambiguous.
    """
    target_text = sub_query if sub_query else user_query
    if not target_text:
        return None
    cleaned = target_text.strip()

    # If query mentions temporal filters (date ranges, history), let LLM extract
    temporal_check = cleaned.lower()
    if any(re.search(rf"\b{kw}\b", temporal_check) for kw in ("tanggal", "bulan", "tahun", "riwayat", "history", "mutasi")):
        return None

    invalid_tokens = {"setiap", "gudang", "lokasi", "distribusi", "semua", "cabang", "2024", "2025", "2026", "2027", "2028"}
    candidate_codes: List[str] = []

    # 1. Match explicit SKU/kode prefixes first (e.g. 'SKU 91010600', 'kode 91030300')
    explicit_matches = re.findall(
        r"\b(?:sku|kode(?:\s+produk|\s+barang)?|item)\s*[:#]?\s*([A-Za-z0-9._\-\/]{4,35})\b",
        cleaned,
        re.IGNORECASE,
    )
    for raw in explicit_matches:
        c = clean_product_identifier(raw.strip())
        if c and c.lower() not in invalid_tokens and c not in candidate_codes and not _is_pure_branch_token(c):
            candidate_codes.append(c)

    # 2. Match hyphenated SKU codes (e.g. FP-KARET-42, AAEO-NPKG)
    hyphen_matches = re.findall(r"\b([A-Za-z0-9]+(?:-[A-Za-z0-9]+)+)\b", cleaned)
    for raw in hyphen_matches:
        c = clean_product_identifier(raw.strip())
        if c and c.lower() not in invalid_tokens and c not in candidate_codes and not _is_pure_branch_token(c):
            candidate_codes.append(c)

    # 3. Match 7-14 digit numeric SKUs / barcodes (e.g. 91010600, 91030300)
    numeric_matches = re.findall(r"\b([0-9]{7,14})\b", cleaned)
    for raw in numeric_matches:
        c = clean_product_identifier(raw.strip())
        if c and c.lower() not in invalid_tokens and c not in candidate_codes and not _is_pure_branch_token(c):
            candidate_codes.append(c)

    if candidate_codes:
        params: Dict[str, Any] = {
            "include_breakdown": True,
            "limit": 100,
            "offset": 0,
        }
        if len(candidate_codes) > 1:
            params["codes"] = candidate_codes
        else:
            params["code"] = candidate_codes[0]

        if company_id:
            params["company_id"] = company_id
        return params

    return None


_TOP_SALES_STOPWORDS = {
    "dengan", "pendapatan", "omzet", "omset", "penjualan", "tertinggi", "terbanyak",
    "terbesar", "terlaris", "teratas", "top", "paling", "laku", "produk", "product",
    "barang", "item", "sku", "kode", "yang", "mana", "dimana", "cabang", "perusahaan",
    "toko", "outlet", "kantor", "pusat", "semua", "seluruh", "tahun", "bulan", "hari",
    "minggu", "kemarin", "ini", "itu", "lalu", "terakhir", "berapa", "cek", "tampilkan",
    "daftar", "list", "peringkat", "dari", "di", "ke", "pada", "untuk", "dan", "serta", "atau",
    "unit", "qty", "revenue", "ytd", "mtd",
    "jenis", "apapun", "macam", "tipe", "segala", "aneka", "apa", "saja", "aja",
    "apakah", "ada", "terkait", "khusus", "semuanya", "tidak", "nya",
    "q1", "q2", "q3", "q4", "kuartal", "triwulan", "semester",
    "target", "owner", "minta", "bisa", "engga", "enggak", "gak", "menurutmu",
    "masing-masing", "masing", "per", "order", "segera", "forecasting", "prediksi",
    "estimasi", "buat", "buatin", "kenapa", "alasannya", "fastmoving", "slowmoving", "lowmoving",
    "wa", "whatsapp", "shopee", "tokopedia", "tokped", "tiktok", "lazada", "blibli",
    "website", "offline", "channel",
}

_SALES_CHANNEL_PATTERNS = [
    (r"\b(?:whatsapp|wa)\b", "WA"),
    (r"\bshopee\b", "Shopee"),
    (r"\btokopedia\b|\btokped\b", "Tokopedia"),
    (r"\btiktok(?:\s+shop)?\b", "TikTok"),
    (r"\blazada\b", "Lazada"),
    (r"\bblibli\b", "Blibli"),
    (r"\bwebsite\b|\bweb\b", "Website"),
    (r"\boffline\b|\bwalk[\s-]?in\b|\btoko\s+fisik\b", "Offline"),
]


def _extract_sales_channel(query: str) -> Optional[str]:
    """Extract canonical ERP Sales Channel name from natural language text."""
    q = (query or "").strip()
    if not q:
        return None
    for pat, canonical in _SALES_CHANNEL_PATTERNS:
        if re.search(pat, q, re.IGNORECASE):
            return canonical
    return None


def _extract_prior_sales_context(
    messages: Optional[Sequence[Any]],
    today: date,
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Recover product_search, date_from/date_to, and confirmed/active status hint
    from prior conversation turns or state for anaphoric follow-ups (e.g. 'di wa itu contactnya siapa saja').
    """
    ctx: Dict[str, Any] = {}
    if state:
        prior_params = state.get("extracted_params") or {}
        for p_key in ("product_search", "search", "code", "sku"):
            val = prior_params.get(p_key)
            if val and isinstance(val, str) and val.lower() not in _TOP_SALES_STOPWORDS:
                ctx["product_search"] = val.strip()
                break
        if prior_params.get("date_from"):
            ctx["date_from"] = prior_params["date_from"]
        if prior_params.get("date_to"):
            ctx["date_to"] = prior_params["date_to"]
        if prior_params.get("status"):
            ctx["prior_status"] = prior_params["status"]
        prior_tool = state.get("detected_tool") or ""
        if prior_tool in ("get_top_selling_products", "get_sales_summary"):
            ctx["prior_confirmed_sales"] = True

    if messages:
        # Inspect recent HumanMessages excluding current turn (last element)
        prior_msgs = list(messages[:-1]) if len(messages) > 1 else []
        for msg in reversed(prior_msgs):
            content = str(getattr(msg, "content", "") or "").strip()
            if not content:
                continue
            low = content.lower()
            if not ctx.get("prior_status"):
                if re.search(r"\b(?:dibatalkan|batal|cancel)\b|`cancel`", low):
                    ctx["prior_status"] = "cancel"
                elif re.search(r"\b(?:draft\s+quotation|status\s*:\s*`?quotation`?|hanya\s+quotation)\b", low):
                    ctx["prior_status"] = "quotation"
            if getattr(msg, "type", None) != "human" and not isinstance(msg, HumanMessage):
                continue
            if not ctx.get("product_search"):
                top_ext = _fallback_extract_top_selling_products(content, today)
                if top_ext.get("search"):
                    ctx["product_search"] = top_ext["search"]
                else:
                    prod_ext = _fallback_extract_product_info(content, None)
                    if prod_ext.get("search") or prod_ext.get("code"):
                        ctx["product_search"] = prod_ext.get("search") or prod_ext.get("code")
            if not ctx.get("date_from") and has_temporal_expression(content):
                df, dt = resolve_relative_dates(content, today)
                if df:
                    ctx["date_from"] = df
                if dt:
                    ctx["date_to"] = dt
            if not ctx.get("prior_confirmed_sales"):
                if re.search(r"\b(?:penjualan|terjual|omzet|omset)\b", low) and not re.search(r"\b(?:quotation|penawaran)\b", low):
                    ctx["prior_confirmed_sales"] = True
            if ctx.get("product_search") and ctx.get("date_from") and ctx.get("prior_status"):
                break

    return ctx


def _fallback_extract_top_selling_products(query: str, ref_date: Optional[date] = None) -> Dict[str, Any]:
    """Deterministic heuristic extraction for sales.get_top_selling_products."""
    today = ref_date or get_current_erp_date()
    params: Dict[str, Any] = {"limit": 10}
    cleaned = (query or "").strip()
    clean_q = cleaned.lower()

    # Determine ranking metric
    if (
        re.search(r"\b(?:terbanyak|unit|qty|kuantitas|terjual)\b", clean_q)
        and not re.search(r"\b(?:omzet|omset|pendapatan|revenue|laba)\b", clean_q)
    ):
        params["sort_by"] = "qty"
    else:
        params["sort_by"] = "revenue"

    ch = _extract_sales_channel(cleaned)
    if ch:
        params["channel_name"] = ch

    # Extract explicit SKU token first
    sku_m = re.search(r"\b(?:sku|kode(?:\s+produk|\s+barang)?)\s*[:#]?\s*([A-Za-z0-9._\-\/]{2,40})\b", cleaned, re.IGNORECASE)
    if sku_m:
        cand = clean_product_identifier(sku_m.group(1))
        if cand and cand.lower() not in _TOP_SALES_STOPWORDS and not _is_pure_branch_token(cand):
            params["search"] = cand

    if not params.get("search"):
        m_pre = re.search(
            r"^([A-Za-z0-9\-_]+(?:\s+[A-Za-z0-9\-_]+)?)\s+penjualan\s+(?:terbanyak|tertinggi|terbesar)\b",
            cleaned,
            re.IGNORECASE,
        )
        if m_pre:
            words = [
                w for w in m_pre.group(1).split()
                if w.lower() not in _TOP_SALES_STOPWORDS
                and not _is_pure_branch_token(w)
                and not re.match(r"^(?:20\d\d|q[1-4])$", w.lower())
            ]
            if words:
                params["search"] = " ".join(words)

    if not params.get("search"):
        m_post = re.search(
            r"\b(?:(?:omzet|omset)\s+)?(?:penjualan|omzet|omset|terjual)\s+(?:produk\s+|barang\s+)?([A-Za-z0-9\s\.\-\+\/]+?)(?:\s+\b(?:bulan|tahun|hari|minggu|kemarin|di|ke|pada|cabang|semua|terbanyak|tertinggi|terbesar|terlaris|dimana|berapa|jenis|apapun)\b|$)",
            cleaned,
            re.IGNORECASE,
        )
        if m_post:
            raw_phrase = strip_company_scope_keywords(m_post.group(1))
            words = [
                w for w in raw_phrase.split()
                if w.lower() not in _TOP_SALES_STOPWORDS
                and not _is_pure_branch_token(w)
                and not re.match(r"^(?:20\d\d|q[1-4])$", w.lower())
            ]
            if words:
                params["search"] = " ".join(words)

    # Extract limit (e.g. "10 produk", "top 5") — avoid mistaking numeric SKU as limit
    m_limit = re.search(r"\b(?:top|peringkat|daftar)\s*(\d+)\b|\b(\d+)\s+(?:produk|barang|item)\b", cleaned, re.IGNORECASE)
    if m_limit:
        raw_lim = m_limit.group(1) or m_limit.group(2)
        try:
            val = int(raw_lim)
            if 1 <= val <= 100:
                params["limit"] = val
        except ValueError:
            pass

    df, dt = resolve_relative_dates(query, today)
    if df or dt:
        if df:
            params["date_from"] = df
        if dt:
            params["date_to"] = dt
    else:
        params["date_from"] = f"{today.year}-01-01"
        params["date_to"] = f"{today.year}-12-31"

    return params


_CONTACT_STOPWORDS = {
    "detail", "detailnya", "info", "informasi", "profil", "profile", "data", "daftar", "list",
    "lengkap", "selengkapnya", "rincian", "rinciannya", "semuanya",
    "cek", "cari", "tampilkan", "lihat", "siapa", "saja", "aja", "apa", "mana",
    "dimana", "di", "ke", "dari", "pada", "untuk", "dengan", "yang", "yg", "itu", "ini", "tersebut",
    "tadi", "nya", "kontak", "contact", "contacts", "contactnya", "kontaknya",
    "pelanggan", "customer", "customers", "pembeli", "klien", "client", "vendor",
    "supplier", "pemasok", "toko", "no", "nomor", "hp", "telp", "telepon", "wa",
    "whatsapp", "alamat", "email", "kota", "shopee", "tokopedia", "tokped", "tiktok",
    "lazada", "blibli", "website", "offline", "channel", "penjualan", "pembelian", "belanja",
    "paling", "banyak", "terbanyak", "terbesar", "tertinggi", "teratas", "top",
    "peringkat", "ranking", "rangking", "frekuensi", "sering", "spender", "beli", "membeli", "terbaru",
    "kita", "kami", "maksutku", "maksudku", "nah", "sih", "buat", "buatin",
    "order", "transaksi", "bulan", "tahun", "hari", "minggu", "kemarin", "semua",
    "seluruh", "cabang", "perusahaan", "dong", "ya", "kak", "min", "tolong",
    "q1", "q2", "q3", "q4", "kuartal", "triwulan", "semester",
}


def _fallback_extract_contacts(
    query: str,
    today: date,
    messages: Optional[Sequence[Any]] = None,
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Deterministic heuristic extraction for sales.get_contacts."""
    params: Dict[str, Any] = {"limit": 20, "offset": 0}
    cleaned = (query or "").strip()
    clean_q = cleaned.lower()

    is_ranked_buyer = bool(
        re.search(
            r"\b(?:peringkat|ranking|rangking|top|teratas|terbanyak|terbesar|tertinggi|paling\s+(?:banyak|sering)\s+(?:beli|belanja|order))\b",
            clean_q,
        )
    )
    if is_ranked_buyer:
        params["sort_by"] = "orders_count" if re.search(r"\bpaling\s+sering\b", clean_q) else "total_spent"
        m_lim = re.search(
            r"\b(?:top|peringkat|ranking|rangking)\s*(\d+)\b|\b(\d+)\s+(?:pelanggan|customer|pembeli|buyer|contact|kontak)\b",
            clean_q,
        )
        if m_lim:
            raw_lim = m_lim.group(1) or m_lim.group(2)
            try:
                params["limit"] = max(1, min(100, int(raw_lim)))
            except ValueError:
                params["limit"] = 10
        else:
            params["limit"] = 10

    ch = _extract_sales_channel(cleaned)
    if ch:
        params["channel_name"] = ch

    # Direct explicit contact name / phone / email lookup (e.g. 'detail contact Budi Santoso', 'kontak 08123456')
    m_phone = re.search(r"\b(\+?62\d{7,14}|08\d{7,13})\b", cleaned)
    if m_phone:
        params["search"] = m_phone.group(1)
    elif "@" in cleaned:
        m_email = re.search(r"([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})", cleaned)
        if m_email:
            params["search"] = m_email.group(1)
    elif not is_ranked_buyer:
        # Strip structural prefixes and stopwords to isolate an entity name
        stripped = strip_company_scope_keywords(cleaned)
        # Preserve multi-word proper names after contact keyword if present
        m_named = re.search(
            r"\b(?:detail\s+(?:kontak|contact|customer|pelanggan)|kontak|contact|customer|pelanggan|pembeli)\s+([A-Za-z0-9\.\-\s]{2,60})$",
            stripped,
            re.IGNORECASE,
        )
        cand_src = m_named.group(1) if m_named else stripped
        words = [
            w.strip("?!.,:;\"'")
            for w in cand_src.split()
            if w.strip("?!.,:;\"'").lower() not in _CONTACT_STOPWORDS
            and w.strip("?!.,:;\"'").lower() not in _MONTH_NAME_MAP
            and not _is_pure_branch_token(w.strip("?!.,:;\"'"))
            and not re.match(r"^(?:20\d\d|q[1-4]|\d{1,3})$", w.strip("?!.,:;\"'").lower())
        ]
        if words:
            params["search"] = " ".join(words)

    df, dt = resolve_relative_dates(query, today)
    if df:
        params["date_from"] = df
    if dt:
        params["date_to"] = dt

    # Anaphoric / channel follow-up inheritance (e.g. "di wa itu contactnya siapa saja")
    is_followup = (
        has_anaphoric_reference(cleaned)
        or (bool(ch) and not params.get("search"))
        or bool(re.search(r"\b(?:siapa\s+(?:saja|aja)|contactnya|kontaknya|pembelinya|pelanggannya|customernya)\b", clean_q))
    )
    if is_ranked_buyer:
        if "quotation" in clean_q or "penawaran" in clean_q:
            params["status"] = "quotation"
        elif "cancel" in clean_q or "batal" in clean_q:
            params["status"] = "cancel"
        elif "semua" in clean_q and not re.search(r"\b(?:semua|seluruh)\s+(?:cabang|perusahaan|toko|outlet|pelanggan|customer)\b", clean_q):
            params["status"] = "all"
        else:
            params["status"] = "confirmed"
    elif is_followup:
        prior_ctx = _extract_prior_sales_context(messages, today, state=state)
        if not params.get("search") and prior_ctx.get("product_search"):
            params["product_search"] = prior_ctx["product_search"]
        if not params.get("date_from") and not params.get("date_to") and prior_ctx.get("date_from"):
            params["date_from"] = prior_ctx["date_from"]
            if prior_ctx.get("date_to"):
                params["date_to"] = prior_ctx["date_to"]
            params["_inherited_dates"] = True
        if prior_ctx.get("prior_confirmed_sales") and not re.search(r"\b(?:quotation|penawaran|semua|all)\b", clean_q):
            params["status"] = "confirmed"
        else:
            params["status"] = "active"
    else:
        params["status"] = "all" if params.get("search") else "active"

    # Vendor scoping: customer/channel/sales queries set is_vendor=False; direct name lookup searches all (None)
    if re.search(r"\b(?:vendor|supplier|pemasok)\b", clean_q):
        params["is_vendor"] = True
    elif is_ranked_buyer or params.get("channel_name") or params.get("product_search") or re.search(
        r"\b(?:pelanggan|customer|pembeli|penjualan|order|quotation)\b", clean_q
    ):
        params["is_vendor"] = False

    return params


def _fallback_extract_sales_orders(
    query: str,
    today: date,
    messages: Optional[Sequence[Any]] = None,
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Deterministic heuristic extraction for sales.get_sales_orders."""
    params: Dict[str, Any] = {"limit": 20, "offset": 0, "status": "confirmed"}

    clean_q = (query or "").lower()
    explicit_status = False
    if "cancel" in clean_q or "batal" in clean_q or "dibatalkan" in clean_q:
        params["status"] = "cancel"
        explicit_status = True
    elif re.search(r"\b(?:semua\s+(?:status|order|quotation)|all\s+status)\b", clean_q) or (
        ("semua" in clean_q or "all" in clean_q)
        and not re.search(r"\b(?:semua|seluruh)\s+(?:cabang|perusahaan|toko|outlet|data|datanya)\b", clean_q)
    ):
        params["status"] = "all"
        explicit_status = True
    elif "lunas" in clean_q or "paid" in clean_q:
        params["status"] = "paid"
        explicit_status = True
    elif "invoiced" in clean_q or "faktur" in clean_q:
        params["status"] = "invoiced"
        explicit_status = True
    elif re.search(r"\b(?:draft\s+quotation|status\s+quotation|masih\s+quotation|belum\s+(?:confirm|terkonfirmasi)|hanya\s+quotation|only\s+quotation|(?:daftar|list|tampilkan)\s+quotation)\b", clean_q):
        params["status"] = "quotation"
        explicit_status = True
    elif "quotation" in clean_q or "penawaran" in clean_q:
        params["status"] = "active"
        explicit_status = True

    if re.search(r"\b(?:ter\s*banyak|ter\s*besar|ter\s*tinggi|paling\s+(?:banyak|besar|tinggi)|top|nominal)\b", clean_q):
        params["sort_by"] = "amount_desc"

    ch = _extract_sales_channel(query)
    if ch:
        params["channel_name"] = ch

    is_breakdown_or_followup = has_anaphoric_reference(query) or bool(
        re.search(r"\b(?:breakdown|jabarkan|rincian(?:nya)?|tabel|table|datanya|detail(?:nya)?|semua\s+cabang)\b", clean_q)
    )
    prior_ctx = _extract_prior_sales_context(messages, today, state=state) if (is_breakdown_or_followup or ch) else {}
    if not explicit_status and is_breakdown_or_followup and prior_ctx.get("prior_status"):
        params["status"] = prior_ctx["prior_status"]
    if (has_anaphoric_reference(query) or ch) and not re.search(r"\b(?:terbanyak|terbesar|tertinggi)\b", clean_q):
        if prior_ctx.get("product_search"):
            params["product_search"] = prior_ctx["product_search"]

    m_limit = re.search(r"\b(?:limit|tampilkan|daftar)?\s*(\d+)\s*(?:so|pesanan|order|quotation)?\b", query, re.IGNORECASE)
    if m_limit:
        try:
            val = int(m_limit.group(1))
            if 1 <= val <= 100:
                params["limit"] = val
        except ValueError:
            pass
    elif re.search(r"\b(?:jabarkan|semua\s+datanya|dalam\s+table|dalam\s+tabel)\b", clean_q):
        params["limit"] = 50

    df, dt = resolve_relative_dates(query, today)
    if df or dt:
        if df:
            params["date_from"] = df
        if dt:
            params["date_to"] = dt
    elif is_breakdown_or_followup and prior_ctx.get("date_from"):
        params["date_from"] = prior_ctx["date_from"]
        if prior_ctx.get("date_to"):
            params["date_to"] = prior_ctx["date_to"]
        params["_inherited_dates"] = True

    return params


def _fallback_extract_sales_order_detail(query: str) -> Dict[str, Any]:
    """Deterministic heuristic extraction for sales.get_sales_order_detail."""
    params: Dict[str, Any] = {}

    m_branch_so = re.search(r"\b([A-Za-z]{2,5}/SO/\d+(?:/[A-Za-z0-9_-]+)?)\b", query, re.IGNORECASE)
    if m_branch_so:
        params["order_ref"] = m_branch_so.group(1).upper()
        return params

    m_so = re.search(r"\b(SO[/-][a-zA-Z0-9/_-]+)\b", query, re.IGNORECASE)
    if m_so:
        params["order_ref"] = m_so.group(1)
        return params

    m_inv = re.search(r"\b(INV[/-][a-zA-Z0-9/_-]+)\b", query, re.IGNORECASE)
    if m_inv:
        params["order_ref"] = m_inv.group(1)
        return params

    m_num = re.search(r"\b(?:so|quotation|penawaran|nomor|no\.?|order|pesanan|resi)\s*[:#]?\s*([a-zA-Z0-9/_-]+)\b", query, re.IGNORECASE)
    if m_num:
        cand = m_num.group(1).strip()
        if cand.lower() not in ("ini", "itu", "terbaru", "kemarin", "hari", "bulan", "tahun", "mana", "terbanyak"):
            params["order_ref"] = cand
            return params

    words = [w for w in query.strip().split() if any(c.isdigit() for c in w)]
    if words:
        params["order_ref"] = words[0]

    return params


def _fallback_extract_sales_summary(query: str, today: date) -> Dict[str, Any]:
    """Deterministic heuristic extraction for sales.get_sales_summary."""
    params: Dict[str, Any] = {"status": "confirmed"}

    clean_q = query.lower()
    if "quotation" in clean_q or "penawaran" in clean_q:
        params["status"] = "quotation"
    elif "cancel" in clean_q or "batal" in clean_q:
        params["status"] = "cancel"
    elif "semua" in clean_q and not re.search(r"\b(?:semua|seluruh)\s+(?:cabang|perusahaan|toko|outlet)\b", clean_q):
        params["status"] = "all"

    df, dt = resolve_relative_dates(query, today)
    if df:
        params["date_from"] = df
    if dt:
        params["date_to"] = dt

    return params


def _extract_mp_platform(query: str) -> Optional[str]:
    """Normalize marketplace platform mention into canonical ERP Marketplace.name."""
    cleaned = (query or "").lower()
    if re.search(r"\btiktok(?:\s*shop)?\b", cleaned):
        return "TikTok Shop"
    if re.search(r"\bshopee\b", cleaned):
        return "Shopee"
    if re.search(r"\btokopedia\b", cleaned):
        return "Tokopedia"
    if re.search(r"\blazada\b", cleaned):
        return "Lazada"
    return None


def _extract_mp_account_id(query: str) -> Optional[int]:
    """Extract explicit integer account_id when user mentions 'account_id 12' or 'akun id 12'."""
    m = re.search(r"\b(?:account_id|akun\s*id|id\s*akun)\s*[:=#]?\s*(\d+)\b", query or "", re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def _has_live_marketplace_intent(query: str) -> bool:
    """Return True only when user explicitly asks for live/real-time marketplace API reads."""
    return bool(
        re.search(
            r"\b(live|real[\s\-]?time|langsung\s+dari\s+(?:shopee|tiktok|seller\s*center)|api\s+(?:shopee|tiktok))\b",
            query or "",
            re.IGNORECASE,
        )
    )


def _extract_mp_sku_token(query: str) -> Optional[str]:
    """Extract explicit SKU code from marketplace queries."""
    cleaned = (query or "").strip()
    m_exp = re.search(
        r"\b(?:sku|kode(?:\s+produk|\s+barang)?)\s*[:#]?\s*([A-Za-z0-9._\-\/]{3,40})\b",
        cleaned,
        re.IGNORECASE,
    )
    if m_exp:
        cand = clean_product_identifier(m_exp.group(1))
        if cand and cand.lower() not in ("yang", "ini", "itu", "unmapped", "belum", "marketplace"):
            return cand
    m_hy = re.search(r"\b([A-Za-z0-9]+(?:-[A-Za-z0-9]+)+)\b", cleaned)
    if m_hy:
        return clean_product_identifier(m_hy.group(1))
    m_num = re.search(r"\b(\d{4,14})\b", cleaned)
    if m_num:
        cand_num = m_num.group(1)
        if not (len(cand_num) == 4 and 2020 <= int(cand_num) <= 2035):
            return cand_num
    return None


def _fallback_extract_account_health(
    query: str,
    today: Optional[date] = None,
) -> Dict[str, Any]:
    """Deterministic extraction for marketplace.get_account_health."""
    params: Dict[str, Any] = {"only_issues": False, "error_window_hours": 24}
    cleaned = (query or "").lower()
    mp = _extract_mp_platform(query)
    if mp:
        params["marketplace"] = mp
    acc_id = _extract_mp_account_id(query)
    if acc_id is not None:
        params["account_id"] = acc_id
    if re.search(r"\b(expired|expiring|bermasalah|error|gagal|issue|kendala|hanya\s+masalah|tidak\s+aktif|inactive)\b", cleaned):
        params["only_issues"] = True
    m_hr = re.search(r"\b(\d+)\s*jam\b", cleaned)
    if m_hr:
        hrs = max(1, min(168, int(m_hr.group(1))))
        params["error_window_hours"] = hrs
    else:
        m_day = re.search(r"\b(\d+)\s*hari\b", cleaned)
        if m_day:
            params["error_window_hours"] = max(1, min(168, int(m_day.group(1)) * 24))
    df, dt = resolve_relative_dates(query, today or get_current_erp_date())
    if df:
        params["date_from"] = df
    if dt:
        params["date_to"] = dt
    return params


def _fallback_extract_marketplace_orders(query: str, today: date) -> Dict[str, Any]:
    """Deterministic extraction for marketplace.get_marketplace_orders."""
    params: Dict[str, Any] = {"mp_status": "all", "sla_status": "all", "limit": 20, "offset": 0}
    cleaned = (query or "").lower()
    mp = _extract_mp_platform(query)
    if mp:
        params["marketplace"] = mp
    acc_id = _extract_mp_account_id(query)
    if acc_id is not None:
        params["account_id"] = acc_id

    if re.search(r"\b(overdue|lewat(?:i)?\s+batas\s+kirim|melewati\s+batas\s+kirim|telat\s+kirim)\b", cleaned):
        params["sla_status"] = "overdue_shipping"
    elif re.search(r"\b(due\s+soon|segera\s+kirim|mendekati\s+batas)\b", cleaned):
        params["sla_status"] = "due_soon"
    elif re.search(r"\b(missing\s+awb|belum\s+(?:punya|ada)\s+resi|tanpa\s+resi|resi\s+kosong)\b", cleaned):
        params["sla_status"] = "missing_awb"
    elif re.search(r"\b(status\s+mismatch|selisih\s+status)\b", cleaned):
        params["sla_status"] = "status_mismatch"

    if re.search(r"\b(cancel|cancelled|batal|dibatalkan)\b", cleaned):
        params["mp_status"] = "cancelled"
    elif re.search(r"\b(completed|selesai|done)\b", cleaned):
        params["mp_status"] = "completed"
    elif re.search(r"\b(processing|diproses|ready_to_ship)\b", cleaned):
        params["mp_status"] = "processing"

    m_search = re.search(r"\b(SO/\d{4}/\d+|2\d{12,}[A-Z0-9]*)\b", query or "", re.IGNORECASE)
    if m_search:
        params["search"] = m_search.group(1)

    df, dt = resolve_relative_dates(query, today)
    if df:
        params["date_from"] = df
    if dt:
        params["date_to"] = dt
    return params


def _fallback_extract_kilat_diagnostics(query: str, today: date) -> Dict[str, Any]:
    """Deterministic extraction for marketplace.get_kilat_order_diagnostics."""
    params: Dict[str, Any] = {"issue_type": "all", "limit": 20, "offset": 0}
    cleaned = (query or "").lower()
    acc_id = _extract_mp_account_id(query)
    if acc_id is not None:
        params["account_id"] = acc_id

    if re.search(r"\b(belum\s+update|unupdated|invoice_updated)\b", cleaned):
        params["issue_type"] = "unupdated_invoice"
    elif re.search(r"\b(duplikat|duplicate|collision|bentrok)\b", cleaned):
        params["issue_type"] = "duplicate_collision"
    elif re.search(r"\b(missing\s+so|tanpa\s+so|draft)\b", cleaned):
        params["issue_type"] = "missing_so_link"

    m_ref = re.search(r"\b(?:reservasi|no\.?\s*reservasi|invoice|referensi|ref)\s*[:#]?\s*\(?([A-Za-z0-9]{10,})\)?", query or "", re.IGNORECASE)
    if m_ref:
        params["reference"] = m_ref.group(1)
    else:
        m_tok = re.search(r"\b([0-9]{6}[A-Z0-9]{6,}|SO/\d{4}/\d+)\b", query or "")
        if m_tok:
            params["reference"] = m_tok.group(1)

    df, dt = resolve_relative_dates(query, today)
    if df:
        params["date_from"] = df
    if dt:
        params["date_to"] = dt
    return params


def _fallback_extract_product_mapping_status(query: str) -> Dict[str, Any]:
    """Deterministic extraction for marketplace.get_product_mapping_status."""
    params: Dict[str, Any] = {"mapping_status": "all", "include_deleted": False, "limit": 30, "offset": 0}
    cleaned = (query or "").lower()
    mp = _extract_mp_platform(query)
    if mp:
        params["marketplace"] = mp
    acc_id = _extract_mp_account_id(query)
    if acc_id is not None:
        params["account_id"] = acc_id

    if re.search(r"\b(exception|pengecualian)\b", cleaned):
        params["mapping_status"] = "exceptions"
    elif re.search(r"\b(unmapped_by_system|unmapped\s+(?:oleh\s+)?sistem)\b", cleaned):
        params["mapping_status"] = "unmapped_by_system"
    elif re.search(r"\b(unmapped_by_user|unmapped\s+(?:oleh\s+)?user)\b", cleaned):
        params["mapping_status"] = "unmapped_by_user"
    elif re.search(r"\b(unmapped|belum\s+(?:di|ter)-?mapping|belum\s+mapping)\b", cleaned):
        params["mapping_status"] = "unmapped"
    elif re.search(r"\b(qty_multiplier|multiplier|kelipatan|bundle)\b", cleaned):
        params["mapping_status"] = "multipliers_only"
    elif re.search(r"\b(duplikat|duplicate)\b", cleaned):
        params["mapping_status"] = "duplicates"
    elif re.search(r"\b(sudah\s+ter-?mapping|mapped)\b", cleaned):
        params["mapping_status"] = "mapped"

    if "deleted" in cleaned or "terhapus" in cleaned:
        params["include_deleted"] = True

    sku = _extract_mp_sku_token(query)
    if sku:
        params["search"] = sku
    return params


def _fallback_extract_stock_sync_simulation(query: str) -> Dict[str, Any]:
    """Deterministic extraction for marketplace.get_stock_sync_simulation."""
    params: Dict[str, Any] = {"only_discrepancies": False, "limit": 20, "offset": 0}
    cleaned = (query or "").lower()
    mp = _extract_mp_platform(query)
    if mp:
        params["marketplace"] = mp
    acc_id = _extract_mp_account_id(query)
    if acc_id is not None:
        params["account_id"] = acc_id
    if re.search(r"\b(selisih|discrepancy|discrepancies|beda|berbeda|anomali|peringatan|warning)\b", cleaned):
        params["only_discrepancies"] = True
    sku = _extract_mp_sku_token(query)
    if sku:
        params["sku"] = sku
    return params


def _fallback_extract_sync_execution_logs(query: str, today: date) -> Dict[str, Any]:
    """Deterministic extraction for marketplace.get_sync_execution_logs."""
    params: Dict[str, Any] = {"log_type": "all", "status": "FAILED", "include_summary": True, "limit": 20, "offset": 0}
    cleaned = (query or "").lower()
    mp = _extract_mp_platform(query)
    if mp:
        params["marketplace"] = mp
    acc_id = _extract_mp_account_id(query)
    if acc_id is not None:
        params["account_id"] = acc_id

    has_stock = bool(re.search(r"\b(stok|stock|stockupdatelog)\b", cleaned))
    has_price = bool(re.search(r"\b(harga|price|priceupdatelog)\b", cleaned))
    if "pcls" in cleaned or "inventoryupdatepricelog" in cleaned:
        params["log_type"] = "pcls_price"
    elif has_stock and not has_price:
        params["log_type"] = "stock"
    elif has_price and not has_stock:
        params["log_type"] = "price"

    if re.search(r"\b(semua\s+status|all)\b", cleaned):
        params["status"] = "all"
    elif re.search(r"\b(skipped|skip|dilewati)\b", cleaned):
        params["status"] = "SKIPPED"
    elif re.search(r"\b(success|sukses|berhasil)\b", cleaned) and not re.search(r"\b(gagal|failed|error)\b", cleaned):
        params["status"] = "SUCCESS"

    for ts in ("CELERY_BEAT", "MANUAL_BULK", "MANUAL_SELECTIVE", "MANUAL_API", "SYSTEM"):
        if ts.lower().replace("_", " ") in cleaned or ts.lower() in cleaned:
            params["trigger_source"] = ts
            break
    if "trigger_source" not in params and "celery" in cleaned:
        params["trigger_source"] = "CELERY_BEAT"

    sku = _extract_mp_sku_token(query)
    if sku:
        params["sku"] = sku

    df, dt = resolve_relative_dates(query, today)
    if df:
        params["date_from"] = df
    if dt:
        params["date_to"] = dt
    return params


def _fallback_extract_price_comparison(query: str) -> Dict[str, Any]:
    """Deterministic extraction for marketplace.get_price_comparison."""
    params: Dict[str, Any] = {
        "filter_status": "mismatch",
        "include_wholesales": False,
        "include_live": _has_live_marketplace_intent(query),
        "limit": 20,
        "offset": 0,
    }
    cleaned = (query or "").lower()
    acc_id = _extract_mp_account_id(query)
    if acc_id is not None:
        params["account_id"] = acc_id
    m_pl = re.search(r"\bpricelist[_\s]*id\s*[:=#]?\s*(\d+)\b", cleaned)
    if m_pl:
        params["pricelist_id"] = int(m_pl.group(1))

    if re.search(r"\b(grosir|wholesale|wholesales|tier)\b", cleaned):
        params["include_wholesales"] = True
    if re.search(r"\b(missing_mp|harga\s+mp\s+kosong|belum\s+ada\s+harga\s+(?:di\s+)?marketplace)\b", cleaned):
        params["filter_status"] = "missing_mp_price"
    elif re.search(r"\b(missing_erp|harga\s+erp\s+kosong|tidak\s+ada\s+di\s+pricelist)\b", cleaned):
        params["filter_status"] = "missing_erp_price"
    elif re.search(r"\b(sama|cocok|match)\b", cleaned) and "selisih" not in cleaned and "mismatch" not in cleaned:
        params["filter_status"] = "match"
    elif re.search(r"\b(semua|all)\b", cleaned) and "selisih" not in cleaned:
        params["filter_status"] = "all"

    m_pct = re.search(r"(?:di\s*atas|lebih\s+dari|>=?|>|minimal)\s*(\d+(?:\.\d+)?)\s*%", cleaned)
    if m_pct:
        params["min_diff_pct"] = float(m_pct.group(1))

    sku = _extract_mp_sku_token(query)
    if sku:
        params["search"] = sku
        if (
            params["filter_status"] == "mismatch"
            and not re.search(r"\b(selisih|mismatch|beda|lebih\s+mahal|lebih\s+murah)\b", cleaned)
        ):
            params["filter_status"] = "all"
    return params


def _fallback_extract_fee_anomalies(query: str, today: date) -> Dict[str, Any]:
    """Deterministic extraction for marketplace.get_fee_and_escrow_anomalies."""
    params: Dict[str, Any] = {
        "category": "all",
        "min_pct": 20.0,
        "exclude_cancelled": True,
        "order_by": "-fee_percentage",
        "limit": 20,
        "offset": 0,
    }
    cleaned = (query or "").lower()
    mp = _extract_mp_platform(query)
    if mp:
        params["marketplace"] = mp
    acc_id = _extract_mp_account_id(query)
    if acc_id is not None:
        params["account_id"] = acc_id

    if re.search(r"\b(negative_fee|fee\s+negatif|expected\s*>\s*mp_total)\b", cleaned):
        params["category"] = "negative_fee"
    elif re.search(r"\b(zero_income|nol\s+atau\s+minus|expected.*?(?:nol|0|minus))\b", cleaned):
        params["category"] = "zero_income"
    elif re.search(r"\b(erp_mp_mismatch|selisih\s+total\s+erp)\b", cleaned):
        params["category"] = "erp_mp_mismatch"
    elif re.search(r"\b(high_fee|di\s*atas|lebih\s+dari)\b", cleaned):
        params["category"] = "high_fee"

    m_pct = re.search(r"(?:di\s*atas|lebih\s+dari|>=?|>|minimal)\s*(\d+(?:\.\d+)?)\s*%", cleaned)
    if m_pct:
        params["min_pct"] = float(m_pct.group(1))
        if params["category"] == "all":
            params["category"] = "high_fee"

    if re.search(r"\b(termasuk\s+(?:yang\s+)?cancel|include\s+cancelled)\b", cleaned):
        params["exclude_cancelled"] = False

    df, dt = resolve_relative_dates(query, today)
    if df:
        params["date_from"] = df
    if dt:
        params["date_to"] = dt
    return params


def _fallback_extract_deduction_summary(query: str, today: date) -> Dict[str, Any]:
    """Deterministic extraction for marketplace.get_deduction_and_settlement_summary."""
    params: Dict[str, Any] = {
        "date_field": "received_date",
        "only_unposted_journals": False,
        "include_order_details": False,
        "limit": 20,
        "offset": 0,
    }
    cleaned = (query or "").lower()
    mp = _extract_mp_platform(query)
    if mp:
        params["marketplace"] = mp
    acc_id = _extract_mp_account_id(query)
    if acc_id is not None:
        params["account_id"] = acc_id

    if re.search(r"\b(belum\s+(?:ter-?)?posting|unposted)\b", cleaned):
        params["only_unposted_journals"] = True
        params["include_order_details"] = True
    if re.search(r"\b(detail|rincian\s+order|per\s+order|daftar\s+order)\b", cleaned):
        params["include_order_details"] = True
    if "created_at" in cleaned:
        params["date_field"] = "created_at"
    elif "order_date" in cleaned or "tanggal order" in cleaned:
        params["date_field"] = "order_date"

    for cat in ("Komisi Platform", "Biaya Layanan", "Ongkos Kirim", "Voucher", "Biaya Transaksi", "Pajak", "Penyesuaian"):
        if cat.lower() in cleaned:
            params["deduction_category"] = cat
            break

    df, dt = resolve_relative_dates(query, today)
    if df:
        params["date_from"] = df
    if dt:
        params["date_to"] = dt
    return params


def _fallback_extract_cashflow_balance(query: str, today: date) -> Dict[str, Any]:
    """Deterministic extraction for marketplace.get_cashflow_and_balance_summary."""
    params: Dict[str, Any] = {
        "include_cashflow_logs": False,
        "include_live": _has_live_marketplace_intent(query),
        "limit": 30,
        "offset": 0,
    }
    cleaned = (query or "").lower()
    mp = _extract_mp_platform(query)
    if mp:
        params["marketplace"] = mp
    acc_id = _extract_mp_account_id(query)
    if acc_id is not None:
        params["account_id"] = acc_id

    if re.search(r"\b(detail|rincian|mutasi|log\s+cashflow|riwayat\s+withdrawal|riwayat\s+settlement)\b", cleaned):
        params["include_cashflow_logs"] = True

    df, dt = resolve_relative_dates(query, today)
    if df:
        params["date_from"] = df
    if dt:
        params["date_to"] = dt
    return params


def _fallback_extract_competitor_insights(query: str, today: date) -> Dict[str, Any]:
    """Deterministic extraction for marketplace.get_competitor_insights."""
    params: Dict[str, Any] = {"focus": "overview", "stale_days": 7, "limit": 20, "offset": 0}
    cleaned = (query or "").lower()
    mp = _extract_mp_platform(query)
    if mp:
        params["marketplace"] = mp

    if re.search(r"\b(turun\s+harga|menurunkan\s+harga|price_drops|penurunan\s+harga)\b", cleaned):
        params["focus"] = "price_drops"
    elif re.search(r"\b(velocity|paling\s+tinggi|terlaris|penjualan(?:nya)?\s+(?:paling\s+)?tinggi|top_velocity)\b", cleaned):
        params["focus"] = "top_velocity"
    elif re.search(r"\b(stale|lama\s+tidak\s+di-?audit|belum\s+di-?audit)\b", cleaned):
        params["focus"] = "stale_audits"

    m_days = re.search(r"(?:>|lebih\s+dari)?\s*(\d+)\s*hari", cleaned)
    if m_days and params["focus"] == "stale_audits":
        params["stale_days"] = max(1, min(365, int(m_days.group(1))))

    sku = _extract_mp_sku_token(query)
    if sku:
        params["sku"] = sku

    df, dt = resolve_relative_dates(query, today)
    if df:
        params["date_from"] = df
    if dt:
        params["date_to"] = dt
    return params


def _extract_single_tool_params(
    plan: Dict[str, Any],
    all_plans: List[Dict[str, Any]],
    user_query: str,
    state: AgentState,
    model: Optional[BaseChatModel],
    today: date,
    retry_count: int,
    retry_instruction: Optional[str],
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
        else:
            scan_probe = _fallback_extract_stock_level(sub_query, state.get("messages"))
            if scan_probe.get("allow_company_scan"):
                extracted_dict = dict(scan_probe)
    elif tool_name == "get_inventory_velocity_forecast" and retry_count == 0:
        extracted_dict = _fallback_extract_inventory_velocity_forecast(sub_query, today)
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
        if (
            fast_params.get("status") in ("active", "cancel", "quotation", "paid", "invoiced")
            or fast_params.get("sort_by") == "amount_desc"
            or fast_params.get("channel_name")
            or fast_params.get("product_search")
            or fast_params.get("_inherited_dates")
        ):
            extracted_dict = dict(fast_params)

    # 1. LLM Structured Output Attempt
    if extracted_dict is None and model is not None:
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

            structured_model = model.with_structured_output(target_schema)
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
        except Exception:
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
            extracted_dict = _fallback_extract_general_ledger(sub_query, today, state.get("messages"))
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
        elif tool_name == "get_inventory_velocity_forecast":
            extracted_dict = _fallback_extract_inventory_velocity_forecast(sub_query, today)
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
            extracted_dict = _fallback_extract_account_health(sub_query, today)
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
        if not extracted_dict.get("code") and not extracted_dict.get("codes") and not extracted_dict.get("product_id") and not extracted_dict.get("allow_company_scan"):
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
    if supports_dates and not has_inherited_dates and not has_scope_temporal and tool_name != "get_inventory_velocity_forecast":
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
    except Exception:
        valid_keys = set(target_schema.model_fields.keys())
        clean_params = {k: v for k, v in extracted_dict.items() if k in valid_keys and v is not None}

    return clean_params


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
    from app.graph.nodes.router import find_previous_user_query, is_retry_query
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

    # Initialize model if not in test env
    model = llm
    is_test_env = "unittest" in sys.modules or os.environ.get("FASTPRINT_TEST_ENV") == "true"
    if model is None and not is_test_env:
        try:
            model = get_chat_model()
        except Exception:
            model = None

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
                    model,
                    today,
                    retry_count,
                    retry_instruction,
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
            model=model,
            today=today,
            retry_count=retry_count,
            retry_instruction=retry_instruction,
        )
        plan["params"] = extracted

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
        recovered_mc = resolve_prior_multi_company_scope(state.get("messages"), user_query)
        if recovered_mc and len(recovered_mc) > 1:
            result_update["is_multi_company"] = True
            result_update["company_ids"] = recovered_mc
            primary_cid = recovered_mc[0]
            result_update["company_id"] = primary_cid
            primary_params["company_id"] = primary_cid
            return result_update
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
