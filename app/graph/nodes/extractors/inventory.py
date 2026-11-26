"""
Domain parameter extraction helpers for FastPrint ERP (inventory).
"""
from datetime import date, timedelta
import re
from typing import Any, Dict, List, Optional
from langchain_core.messages import HumanMessage
from app.core.branches import resolve_branch_company_id
from app.core.context import clean_product_identifier, get_current_erp_date, has_anaphoric_reference
from app.graph.nodes.extractors.common import (
    LOCATION_STOPWORDS,
    PRODUCT_STOPWORDS,
    STOCK_DELIMITERS,
    STOCK_STOPWORDS,
    _is_pure_branch_token,
    _is_valid_product_identifier,
    is_account_code_or_name,
    is_accounting_term,
    is_vendor_name,
    resolve_relative_dates,
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
                    prev_code = str(prev_stock["code"])
                    params["search" if " " in prev_code else "code"] = prev_code
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


_PRICE_HISTORY_STOPWORDS = {
    "cek", "dicek", "lihat", "tampilkan", "cari", "tolong", "coba", "berapa", "bagaimana", "gimana",
    "perubahan", "riwayat", "histori", "history", "pergerakan", "fluktuasi", "kenaikan", "penurunan",
    "revisi", "update", "harga", "harganya", "price", "changes", "change", "changelog", "trend",
    "produk", "barang", "item", "sku", "kode", "dari", "untuk", "pada", "di", "ke", "yang", "yg",
    "dan", "atau", "serta", "juga", "ini", "itu", "tersebut", "tadi", "pernah", "berubah", "naik",
    "turun", "diubah", "diganti", "selama", "bulan", "tahun", "hari", "minggu", "kemarin", "terakhir",
    "shopee", "tiktok", "shop", "tokopedia", "tokped", "lazada", "marketplace", "pcls", "pricelist",
    "erp", "log", "gagal", "failed", "sukses", "success", "semua", "seluruh", "cabang", "perusahaan",
    "penjualan", "penjualannya", "omzet", "omzetnya", "omset", "omsetnya", "retur", "returnya",
    "komplain", "komplainnya", "stok", "stoknya", "fastpos", "jawa", "luar", "medan",
}


def _extract_product_target_for_metric_query(
    query: str,
    extra_stopwords: Optional[set] = None,
    history: Optional[list] = None,
) -> tuple[Optional[str], Optional[str]]:
    """
    Extract (sku, search) from product-centric queries including multi-metric lists such as
    'coba cek barang A penjualannya, returnya, perubahan harganya' or 'riwayat harga SKU-A01'.
    """
    cleaned = (query or "").strip()
    if not cleaned:
        return None, None
    stop = set(_PRICE_HISTORY_STOPWORDS)
    if extra_stopwords:
        stop.update(extra_stopwords)

    # 1. Explicit SKU / kode prefix (must be separated by whitespace or ':'/'#', not hyphenated into the SKU itself)
    m_sku = re.search(
        r"\b(?:sku|kode(?:\s+produk|\s+barang)?)(?:\s*[:#]\s*|\s+)([A-Za-z0-9._\-\/]{2,40})\b",
        cleaned,
        re.IGNORECASE,
    )
    if m_sku:
        cand = clean_product_identifier(m_sku.group(1))
        if cand and cand.lower() not in stop and not _is_pure_branch_token(cand):
            return cand, None

    # 2. Hyphenated SKU or numeric barcode/SKU (excluding 4-digit years 2020..2035)
    m_hy = re.search(r"\b([A-Za-z0-9]+(?:-[A-Za-z0-9]+)+)\b", cleaned)
    if m_hy:
        cand = clean_product_identifier(m_hy.group(1))
        if cand and cand.lower() not in stop and not is_account_code_or_name(cand) and not is_vendor_name(cand):
            return cand, None
    m_num = re.search(r"\b(\d{4,14})\b", cleaned)
    if m_num:
        cand_num = m_num.group(1)
        if not (len(cand_num) == 4 and 2020 <= int(cand_num) <= 2035) and not is_account_code_or_name(cand_num):
            return cand_num, None

    # 3. Product name after 'barang / produk / item' or after metric phrase ('perubahan harga <produk>')
    metric_boundary = (
        r"(?:\s*,|\s+\b(?:penjualan(?:nya)?|omzet(?:nya)?|omset(?:nya)?|retur(?:nya)?|komplain(?:nya)?|"
        r"perubahan|riwayat|histori|history|kenaikan|penurunan|pergerakan|harga(?:nya)?|stok(?:nya)?|"
        r"bulan|tahun|hari|minggu|kemarin|di|ke|pada|untuk|cabang|semua|seluruh|shopee|tiktok|tokopedia|lazada|pcls|pricelist|marketplace|fastpos)\b|$)"
    )
    patterns = [
        rf"\b(?:barang|produk|product|item)\s+([A-Za-z0-9\-\_\.\+\/\s]{{1,60}}?){metric_boundary}",
        rf"\b(?:perubahan|riwayat|histori|history|pergerakan|kenaikan|penurunan|revisi)\s+harga(?:nya)?\s+(?:untuk\s+|produk\s+|barang\s+)?([A-Za-z0-9\-\_\.\+\/\s]{{1,60}}?){metric_boundary}",
    ]
    for pat in patterns:
        m = re.search(pat, cleaned, re.IGNORECASE)
        if m:
            raw_phrase = clean_product_identifier(m.group(1).strip(" ,.;:?!'\""))
            words = [
                w for w in raw_phrase.split()
                if w.lower() not in stop
                and not _is_pure_branch_token(w)
                and not re.match(r"^(?:20\d\d|q[1-4])$", w.lower())
            ]
            if words:
                cand = " ".join(words)
                if not is_vendor_name(cand) and not is_account_code_or_name(cand) and not is_accounting_term(cand):
                    return None, cand

    # 4. History fallback for anaphoric references
    if history and (has_anaphoric_reference(cleaned) or re.search(r"\b(?:itu|tersebut|tadi)\b", cleaned, re.IGNORECASE)):
        info = _fallback_extract_product_info(cleaned, history=history)
        if info.get("code"):
            return info["code"], None
        if info.get("search"):
            return None, info["search"]

    return None, None


def _fallback_extract_price_history(
    query: str,
    today: Optional[date] = None,
    history: Optional[list] = None,
) -> Dict[str, Any]:
    """Deterministic heuristic extraction for inventory.get_price_history."""
    ref_day = today or get_current_erp_date()
    cleaned = (query or "").strip()
    clean_low = cleaned.lower()
    params: Dict[str, Any] = {
        "source": "all",
        "include_failed": False,
        "limit": 20,
        "offset": 0,
    }

    # Marketplace platform detection
    mp_plat: Optional[str] = None
    if re.search(r"\btiktok(?:\s*shop)?\b", clean_low):
        mp_plat = "TikTok Shop"
    elif re.search(r"\bshopee\b", clean_low):
        mp_plat = "Shopee"
    elif re.search(r"\btokopedia\b|\btokped\b", clean_low):
        mp_plat = "Tokopedia"
    elif re.search(r"\blazada\b", clean_low):
        mp_plat = "Lazada"
    if mp_plat:
        params["marketplace"] = mp_plat

    # Source disambiguation
    has_pcls = bool(re.search(r"\b(?:pcls|inventoryupdatepricelog)\b", clean_low))
    has_mp = bool(mp_plat or re.search(r"\b(?:marketplace|priceupdatelog)\b", clean_low))
    has_erp_pl = bool(re.search(r"\b(?:erp\s+pricelist|pricelist\s+erp|pricelist|master\s+harga)\b", clean_low))
    if has_pcls and not has_mp and not has_erp_pl:
        params["source"] = "pcls"
    elif has_mp and not has_pcls and not has_erp_pl:
        params["source"] = "marketplace"
    elif has_erp_pl and not has_pcls and not has_mp:
        params["source"] = "erp_pricelist"

    # Optional pricelist region hint
    if "pricelist" in clean_low:
        if re.search(r"\bluar\s+jawa\b", clean_low):
            params["pricelist_name"] = "Luar Jawa"
        elif re.search(r"\bjawa\b", clean_low):
            params["pricelist_name"] = "Jawa"
        elif re.search(r"\bmedan\b", clean_low):
            params["pricelist_name"] = "Medan"

    if re.search(r"\b(?:gagal|failed|error|termasuk\s+gagal|semua\s+status)\b", clean_low):
        params["include_failed"] = True

    sku_val, search_val = _extract_product_target_for_metric_query(cleaned, history=history)
    if sku_val:
        params["sku"] = sku_val
    elif search_val:
        params["search"] = search_val

    m_lim = re.search(r"\b(?:limit|top|tampilkan)\s*(\d+)\b", clean_low)
    if m_lim:
        try:
            params["limit"] = max(1, min(100, int(m_lim.group(1))))
        except ValueError:
            pass

    df, dt = resolve_relative_dates(cleaned, ref_day)
    if df or dt:
        if df:
            params["date_from"] = df
        if dt:
            params["date_to"] = dt
    else:
        params["date_from"] = (ref_day - timedelta(days=90)).strftime("%Y-%m-%d")
        params["date_to"] = ref_day.strftime("%Y-%m-%d")

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
