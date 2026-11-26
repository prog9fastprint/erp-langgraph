"""
Domain parameter extraction helpers for FastPrint ERP (accounting).
"""
from datetime import date
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
from langchain_core.messages import HumanMessage
from app.core.context import extract_period_from_messages, is_analytical_thinking_query
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
    q_low = cleaned.lower()

    acc_codes = re.findall(r"\b([1-6]\d{3,7})\b", cleaned)
    if acc_codes:
        params["account_code"] = acc_codes[0]

    has_bank_word = bool(re.search(r"\b(?:rekening\s*bank|akun\s*bank|saldo\s*bank|bank(?:\s*(?:&|and|dan)\s*cash)?|rekening)\b", q_low))
    has_kas_word = bool(re.search(r"\b(?:kas|cash)\b", q_low))
    if (
        (has_bank_word and has_kas_word)
        or re.search(
            r"\b(?:rekening\s+bank|akun\s+bank|saldo\s+bank|bank\s*(?:&|and|dan)\s*cash|kas\s*(?:&|dan)\s*(?:rekening\s+)?bank|semua\s+(?:bank|kas|rekening))\b",
            q_low,
        )
    ):
        params["account_type"] = "Bank and Cash"
    elif re.search(r"\b(?:akun\s+kas|saldo\s+kas)\b", q_low) and not params.get("account_code"):
        params["account_type"] = "Bank and Cash"
        if re.search(r"\bkas\s+kecil\b", q_low):
            params["search"] = "KAS KECIL"
        elif not has_bank_word:
            params["search"] = "KAS"

    if re.search(r"\b(?:aktif|non[\s\-]?zero|tidak\s+nol|berisi)\b", q_low):
        params["non_zero_only"] = True

    m_bank_brand = re.search(
        r"\b(bca|mandiri|bri|bni|cimb(?:\s+niaga)?|permata|danamon|ocbc|bsi|btn|mega|panin|uob|maybank|hsbc|dbs|icbc|seabank)\b",
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


_ACCOUNT_CODE_PATTERN = r"[1-6]\d{2,7}(?:[A-Za-z]{1,6})?(?:[\.\-][A-Za-z0-9]{1,8})*"


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
        for m_row in re.finditer(rf"`?({_ACCOUNT_CODE_PATTERN})`?\s*\|\s*([^|\n]+)", content):
            account_candidates.append((m_row.group(1).strip(), m_row.group(2).strip()))
        for m_dash in re.finditer(rf"(?<![\d\-])`?({_ACCOUNT_CODE_PATTERN})`?\s*[-–:]\s+([^\n(|]+)", content):
            account_candidates.append((m_dash.group(1).strip(), m_dash.group(2).strip()))
        for m_paren in re.finditer(rf"([^\n(|]+?)\s*\(\s*`?({_ACCOUNT_CODE_PATTERN})`?\s*\)", content):
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
    acc_codes = re.findall(rf"\b({_ACCOUNT_CODE_PATTERN})\b", cleaned)
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


def _fallback_extract_profit_and_loss(
    query: str,
    ref_date: Optional[date] = None,
    messages: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Deterministic heuristic extraction for profit and loss inquiries (compact by default)."""
    params: Dict[str, Any] = {"include_breakdown": False}
    cleaned = query.strip().lower()
    if any(k in cleaned for k in ("rincian", "detail", "breakdown", "per akun", "daftar akun", "hierarki")) or is_analytical_thinking_query(query):
        params["include_breakdown"] = True
    date_from, date_to = resolve_relative_dates(query, ref_date)
    if not date_from and not date_to and messages:
        prior_df, prior_dt, _ = extract_period_from_messages(messages)
        if prior_df or prior_dt:
            date_from, date_to = prior_df, prior_dt
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to
    return params


def _fallback_extract_balance_sheet(
    query: str,
    ref_date: Optional[date] = None,
    messages: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Deterministic heuristic extraction for balance sheet inquiries (compact by default)."""
    params: Dict[str, Any] = {"include_breakdown": False}
    cleaned = query.strip().lower()
    if any(k in cleaned for k in ("rincian", "detail", "breakdown", "per akun", "daftar akun", "hierarki")) or is_analytical_thinking_query(query):
        params["include_breakdown"] = True
    date_from, date_to = resolve_relative_dates(query, ref_date)
    if date_to:
        params["as_of_date"] = date_to
    elif date_from:
        params["as_of_date"] = date_from
    elif messages:
        _, prior_dt, prior_asof = extract_period_from_messages(messages)
        if prior_asof:
            params["as_of_date"] = prior_asof
        elif prior_dt:
            params["as_of_date"] = prior_dt
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


def _extract_gl_account_hint(query: Optional[str]) -> Optional[Dict[str, str]]:
    """
    Extract a generic accounting category or bank/cash keyword hint from a General Ledger
    query when no explicit account_code was provided (e.g. 'buku besar akun kas',
    'buku besar akun bank', 'buku besar bank bca', 'buku besar piutang usaha').
    """
    if not query:
        return None
    q_low = query.lower().strip()
    m_bank_brand = re.search(
        r"\b(bca|mandiri|bri|bni|cimb(?:\s+niaga)?|permata|danamon|ocbc|bsi|btn|mega|panin|uob|maybank|hsbc|dbs|icbc|seabank)\b",
        q_low,
    )
    if m_bank_brand:
        brand = m_bank_brand.group(1).strip()
        return {
            "account_type": "Bank and Cash",
            "search": brand.upper() if len(brand) <= 4 else brand.title(),
            "mode": "bank_brand",
        }
    if re.search(r"\bkas\s+kecil\b", q_low):
        return {"account_type": "Bank and Cash", "search": "KAS KECIL", "mode": "kas"}
    has_kas = bool(re.search(r"\b(?:kas|cash)\b", q_low))
    has_bank = bool(re.search(r"\b(?:bank|rekening)\b", q_low))
    if has_kas and has_bank:
        return {"account_type": "Bank and Cash", "mode": "bank_and_cash"}
    if has_kas:
        return {"account_type": "Bank and Cash", "search": "KAS", "mode": "kas"}
    if has_bank:
        return {"account_type": "Bank and Cash", "mode": "bank"}
    if re.search(r"\bpiutang\b", q_low):
        return {"account_type": "Receivable", "mode": "receivable"}
    if re.search(r"\b(?:hutang|utang)\b", q_low):
        return {"account_type": "Payable", "mode": "payable"}
    return None
