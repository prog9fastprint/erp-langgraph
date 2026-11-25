"""
Domain parameter extraction helpers for FastPrint ERP (accounting).
"""
from datetime import date
import re
from typing import Any, Dict, Optional
from app.graph.nodes.extractors.common import (
    VENDOR_DELIMITERS,
    VENDOR_NAME_STOPWORDS,
    is_vendor_name,
    resolve_relative_dates,
)


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
