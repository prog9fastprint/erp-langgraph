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
    "makasar": 5,
    "makassar": 5,
    "fp makasar": 5,
    "fp makassar": 5,
    "bandung": 6,
    "fp bandung": 6,
    "surabaya": 7,
    "fp surabaya": 7,
    "surabaya mt": 7,
    "surabaya mth": 7,
    "jakarta": 8,
    "fp jakarta": 8,
    "toko fast print surabaya": 9,
    "toko surabaya": 9,
    "medan": 10,
    "fp medan": 10,
    "hr": 11,
    "sandy": 12,
    "bekasi": 13,
    "fp bekasi": 13,
    "tangerang": 14,
    "fp tangerang": 14,
    "malang": 15,
    "fp malang": 15,
    "fastmart": 17,
    "pusat": 1,
}

TEMPORAL_KEYWORDS = (
    "bulan", "hari", "minggu", "tahun", "kemarin", "today", "yesterday",
    "januari", "februari", "maret", "april", "mei", "juni",
    "juli", "agustus", "september", "oktober", "november", "desember",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "agu", "sep", "okt", "nov", "des",
    "lalu", "terakhir", "kebelakang", "sejak", "sampai", "hingga",
)


def resolve_branch_company_id(text: Optional[str]) -> Optional[int]:
    """
    Resolve Indonesian city or branch mentions into official FastPrint ERP company_id.
    Matches phrases like 'cabang bandung', 'dicabang bandung', 'di bandung', 'kantor pusat', etc.
    """
    if not text:
        return None
    cleaned = text.lower().strip()
    for branch_key in sorted(BRANCH_COMPANY_MAP.keys(), key=lambda k: -len(k)):
        pattern = rf"(?:\b(?:di\s*cabang|dicabang|cabang|di\s*toko|di\s*kantor|kantor|di)\s+)?\b{re.escape(branch_key)}\b"
        if re.search(pattern, cleaned):
            return BRANCH_COMPANY_MAP[branch_key]
    return None


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
    today = ref_date or date.today()
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
        params["code"] = clean_product_identifier(tokens[0])
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
            prev_tokens = re.findall(r"\b([A-Z0-9]+(?:-[A-Z0-9]+)+|\d{4,8})\b", content)
            if prev_tokens:
                params["code"] = prev_tokens[0].rstrip(".")
                break
            prev_info = _fallback_extract_product_info(content)
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

    # 1. LLM Structured Output Attempt
    if model is not None:
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
    elif not extracted_dict.get("company_id") and state.get("company_id"):
        extracted_dict["company_id"] = state.get("company_id")

    # For get_stock_level: prevent LLM from setting product_id to SKU or internal code when code/name is present
    if tool_name == "get_stock_level" and extracted_dict.get("product_id") and extracted_dict.get("code"):
        if not re.search(r"\b(?:id|product_id)\s*[:=]?\s*\d+\b", user_query, re.IGNORECASE):
            extracted_dict.pop("product_id", None)

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
        ref_date: Reference anchor date for relative date resolution (defaults to date.today()).

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
    today = ref_date or date.today()

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

    # Multi-tool extraction iteration
    for plan in working_active_tools:
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

    branch_co_id = (
        resolve_branch_company_id(user_query)
        or primary_params.get("company_id")
    )
    if branch_co_id is not None:
        result_update["company_id"] = branch_co_id
        if working_active_tools:
            working_active_tools[0].setdefault("params", {})["company_id"] = branch_co_id
            primary_params["company_id"] = branch_co_id
    elif state.get("company_id") is not None:
        result_update["company_id"] = state.get("company_id")

    return result_update
