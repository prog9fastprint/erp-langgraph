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
from typing import Any, Dict, List, Optional, Tuple, Type
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

TEMPORAL_KEYWORDS = (
    "bulan", "hari", "minggu", "tahun", "kemarin", "today", "yesterday",
    "januari", "februari", "maret", "april", "mei", "juni",
    "juli", "agustus", "september", "oktober", "november", "desember",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "agu", "sep", "okt", "nov", "des",
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


def has_temporal_expression(text: Optional[str]) -> bool:
    """Check if query text contains explicit dates, date ranges, or relative temporal keywords."""
    if not text:
        return False
    cleaned = text.lower()
    if any(k in cleaned for k in TEMPORAL_KEYWORDS):
        return True
    if re.search(r"\b\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?\b", cleaned):
        return True
    if re.search(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", cleaned):
        return True
    return False

def resolve_relative_dates(
    text: str,
    ref_date: Optional[date] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve relative Indonesian temporal expressions into ISO YYYY-MM-DD date range.

    Supported phrases:
    - 'bulan ini': 1st day of current month to last day of current month
    - 'bulan lalu': 1st day of previous month to last day of previous month
    - 'hari ini': current date
    - 'kemarin': previous day
    - 'minggu ini': Monday to Sunday of the current week
    - 'N hari terakhir': N days window ending today
    - Explicit dates: YYYY-MM-DD format
    """
    today = ref_date or get_current_erp_date()
    cleaned = text.lower()

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


ANAPHORA_TOKENS = {"itu", "tersebut", "tadi", "ini", "terkait", "dia", "nya"}
ANAPHORIC_CLITICS = (
    "stoknya", "harganya", "barangnya", "produknya", "saldonya",
    "tagihannya", "fakturnya", "invoicenya", "lokasinya", "detailnya",
    "transaksinya", "jurnalnya",
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
    "stok", "stock", "sisa",
    "di", "ke", "dari", "pada", "untuk", "terkait", "tentang",
    "gudang", "lokasi", "customer", "pelanggan",
    "dan", "serta", "juga", "faktur", "tagihan", "invoice", "jurnal", "saldo",
)
STOCK_DELIMITERS = r"\b(?:dan|serta|di|ke|pada|untuk|bulan|hari|berapa|status|faktur|tagihan|invoice|saldo|jurnal|gudang|lokasi)\b"


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
        if words and not _is_pure_branch_token(token):
            params["code"] = token
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
    "detail", "spesifikasi", "stok", "stoknya", "produk", "product", "barang", "item", "sku", "kode", "ada",
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
    cleaned = query.strip()

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
                not is_vendor_name(candidate)
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

    acc_codes = re.findall(r"\b([1-6]\d{3,7})\b", cleaned)
    if acc_codes:
        params["account_code"] = acc_codes[0]

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


def _fallback_extract_general_ledger(query: str, ref_date: Optional[date] = None) -> Dict[str, Any]:
    """Deterministic heuristic extraction for general ledger inquiries."""
    params: Dict[str, Any] = {"limit": 50, "offset": 0}
    cleaned = query.strip()
    acc_codes = re.findall(r"\b([1-6]\d{3,7})\b", cleaned)
    if acc_codes:
        params["account_code"] = acc_codes[0]
    date_from, date_to = resolve_relative_dates(cleaned, ref_date)
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
        if not rel_date_from and not rel_date_to and sub_query != user_query:
            rel_date_from, rel_date_to = resolve_relative_dates(user_query, today)

    extracted_dict: Optional[Dict[str, Any]] = None

    # 0. Fast-Path Deterministic Extraction (Tier 1)
    if tool_name == "get_stock_level" and retry_count == 0:
        fast_params = try_fast_path_stock_extraction(sub_query, user_query, state.get("company_id"))
        if fast_params:
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
                f"- For 'code' (in get_stock_level & get_product_info): Extract the product SKU code, barcode, OR product name (e.g. 'AMPLOP BUBBLE 15x18 + 4CM LK', 'Kertas HVS A4', 'Tinta Epson 664', 'HVS-A4-70') into 'code'. If the user mentions a product by name, ALWAYS put that name into 'code'. Strip any leading filler words like 'product', 'produk', 'barang', 'item', 'sku', 'kode' (e.g. for 'product FP BROTHER...' extract 'FP BROTHER...').\n"
                f"- For 'codes' (in get_stock_level): If the user specifies multiple product SKU codes or barcodes (e.g. '91010600 dan 91030300', 'SKU A, B, dan C'), extract all of them as a list of strings into 'codes' rather than a single 'code'.\n"
                f"- For 'location_name' (in get_stock_level): Extract warehouse or location names (e.g. 'mainstore', 'gudang surabaya', 'pusat') into 'location_name'.\n"
                f"- STRICT DATE RULE: Do NOT populate 'date_from' or 'date_to' unless the user explicitly specifies a date, relative date (e.g. 'bulan ini', 'bulan lalu', 'kemarin'), or time window. If no time expression is mentioned, leave both None.\n"
                f"- STRICT STATUS RULE: Valid status values are only 'draft', 'posted', 'paid', 'unpaid', or 'cancel'. Never extract 'not_found' or conversational terms as status.\n"
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

            effective_query = user_query if has_anaphoric_reference(sub_query) else sub_query

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
        else:
            extracted_dict = {}
            if supports_dates and retry_count == 0:
                if rel_date_from:
                    extracted_dict["date_from"] = rel_date_from
                if rel_date_to:
                    extracted_dict["date_to"] = rel_date_to

    # 3. 3-Tier Entity Resolution & Anaphora Handling
    if tool_name == "get_stock_level":
        if not extracted_dict.get("code") and not extracted_dict.get("product_id"):
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
            # Tier 2: Global user_query inspection
            if not extracted_dict.get("code") and sub_query != user_query:
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
            # Tier 2: Global user_query inspection
            if not extracted_dict.get("code") and not extracted_dict.get("search") and sub_query != user_query:
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
            if not extracted_dict.get("vendor_name") and sub_query != user_query:
                v_ext = _fallback_extract_vendor_invoice(user_query, today)
                if v_ext.get("vendor_name") and is_vendor_name(v_ext["vendor_name"]):
                    extracted_dict["vendor_name"] = v_ext["vendor_name"]
                if not extracted_dict.get("status") and v_ext.get("status"):
                    extracted_dict["status"] = v_ext["status"]

    elif tool_name == "get_account_balance":
        if not extracted_dict.get("account_code") and not extracted_dict.get("account_id") and sub_query != user_query:
            b_ext = _fallback_extract_account_balance(user_query, today)
            if b_ext.get("account_code"):
                extracted_dict["account_code"] = b_ext["account_code"]

    elif tool_name == "get_journal_entries":
        if not extracted_dict.get("entry_number") and not extracted_dict.get("status") and not extracted_dict.get("search") and sub_query != user_query:
            j_ext = _fallback_extract_journal_entries(user_query, today)
            for k in ("entry_number", "status", "search"):
                if j_ext.get(k):
                    extracted_dict[k] = j_ext[k]

    elif tool_name == "get_locations":
        if not extracted_dict.get("search") and sub_query != user_query:
            l_ext = _fallback_extract_locations(user_query)
            if l_ext.get("search"):
                extracted_dict["search"] = l_ext["search"]

    # 4. Global Temporal Date Backfill (Accounting Domain Only)
    if supports_dates and retry_count == 0:
        if not extracted_dict.get("date_from") and rel_date_from:
            extracted_dict["date_from"] = rel_date_from
        if not extracted_dict.get("date_to") and rel_date_to:
            extracted_dict["date_to"] = rel_date_to

    # Strict date enforcement: if query mentions no explicit dates or temporal keywords, drop hallucinated dates
    if supports_dates and not has_temporal_expression(sub_query) and not has_temporal_expression(user_query):
        extracted_dict.pop("date_from", None)
        extracted_dict.pop("date_to", None)

    # Status sanitization against valid ERP whitelist
    if "status" in extracted_dict and extracted_dict["status"]:
        st_val = str(extracted_dict["status"]).lower().strip()
        if st_val in ALLOWED_JOURNAL_STATUSES:
            extracted_dict["status"] = st_val
        else:
            extracted_dict.pop("status", None)

    # Branch entity resolution into company_id (authoritative over default guess)
    detected_branch = resolve_branch_company_id(sub_query) or resolve_branch_company_id(user_query)
    if detected_branch:
        extracted_dict["company_id"] = detected_branch
    elif state.get("company_id"):
        extracted_dict["company_id"] = state.get("company_id")

    # Clean location_name stop-words (e.g. 'setiap', 'semua', 'gudang', 'rak')
    if "location_name" in extracted_dict and extracted_dict["location_name"]:
        loc_val = str(extracted_dict["location_name"]).lower().strip()
        if loc_val in LOCATION_STOP_WORDS:
            extracted_dict.pop("location_name", None)

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

    # For get_product_info: if code contains whitespace, it is a multi-word product name, NOT an exact SKU!
    # Django get_product_info requires exact default_code for 'code', but matches names via 'search'.
    if tool_name == "get_product_info" and extracted_dict.get("code"):
        code_str = str(extracted_dict["code"]).strip()
        if " " in code_str:
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

    # 5. Staged Retry Relaxation (Strict Identifier Guard: do not pop document number / code)
    has_doc_identifier = bool(
        extracted_dict.get("entry_number")
        or (extracted_dict.get("search") and re.search(r"\b(?:[A-Za-z0-9_\-]+/)+\d{4}/\d+\b", str(extracted_dict.get("search"))))
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

    # Clean product codes / search terms across inventory tools
    if "code" in extracted_dict and isinstance(extracted_dict["code"], str):
        extracted_dict["code"] = clean_product_identifier(extracted_dict["code"])
    if "codes" in extracted_dict and isinstance(extracted_dict["codes"], list):
        extracted_dict["codes"] = [
            clean_product_identifier(c) for c in extracted_dict["codes"] if c and isinstance(c, str)
        ]
    if "search" in extracted_dict and isinstance(extracted_dict["search"], str):
        if tool_name in ("get_product_info", "get_stock_level", "get_stock_history"):
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

    multi_cids, is_all = resolve_all_branch_company_ids(user_query)
    is_multi_co = bool(state.get("is_multi_company") or is_all or (len(multi_cids) > 1))

    if is_multi_co:
        effective_cids = (
            state.get("company_ids")
            if (state.get("company_ids") and len(state.get("company_ids")) > 1 and not is_all and not multi_cids)
            else (multi_cids if multi_cids else list(COMPANY_NAME_MAP.keys()))
        )
        result_update["is_multi_company"] = True
        result_update["company_ids"] = effective_cids
        primary_cid = effective_cids[0] if effective_cids else (state.get("company_id") or 1)
        result_update["company_id"] = primary_cid
        primary_params["company_id"] = primary_cid
    else:
        explicit_branch = resolve_branch_company_id(user_query)
        if explicit_branch is not None:
            result_update["company_id"] = explicit_branch
            for plan_item in working_active_tools:
                plan_params = plan_item.setdefault("params", {})
                if plan_params:
                    plan_params["company_id"] = explicit_branch
            if primary_params:
                primary_params["company_id"] = explicit_branch
        elif state.get("company_id") is not None:
            session_cid = state.get("company_id")
            result_update["company_id"] = session_cid
            for plan_item in working_active_tools:
                plan_params = plan_item.get("params")
                if isinstance(plan_params, dict) and "company_id" in plan_params:
                    plan_params["company_id"] = session_cid
            if "company_id" in primary_params:
                primary_params["company_id"] = session_cid
        elif primary_params.get("company_id") is not None:
            result_update["company_id"] = primary_params["company_id"]

    return result_update
