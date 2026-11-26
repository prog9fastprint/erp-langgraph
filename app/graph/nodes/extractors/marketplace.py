"""
Domain parameter extraction helpers for FastPrint ERP (marketplace).
"""
from datetime import date
import re
from typing import Any, Dict, Optional
from app.core.context import clean_product_identifier, get_current_erp_date
from app.graph.nodes.extractors.common import (
    resolve_relative_dates,
)


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

    if re.search(r"\b((?:belum|blm)\s+update|unupdated|invoice_updated)\b", cleaned):
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
