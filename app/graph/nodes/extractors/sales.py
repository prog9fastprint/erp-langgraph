"""
Domain parameter extraction helpers for FastPrint ERP (sales).
"""
from datetime import date
import re
from typing import Any, Dict, Optional, Sequence
from langchain_core.messages import HumanMessage
from app.core.branches import strip_company_scope_keywords
from app.core.context import clean_product_identifier, get_current_erp_date, has_anaphoric_reference
from app.graph.nodes.extractors.common import (
    _MONTH_NAME_MAP,
    _is_pure_branch_token,
    has_temporal_expression,
    resolve_relative_dates,
)
from app.graph.nodes.extractors.inventory import _fallback_extract_product_info


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
        # Inspect recent messages excluding current turn (last element)
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


def _apply_sales_date_filters(
    query: str,
    today: date,
    params: Dict[str, Any],
    default_current_year: bool = False,
) -> None:
    """Resolve explicit/relative date ranges (including quarters, months, and standalone years)."""
    df, dt = resolve_relative_dates(query, today)
    if df or dt:
        if df:
            params["date_from"] = df
        if dt:
            params["date_to"] = dt
        return

    if default_current_year:
        params["date_from"] = f"{today.year}-01-01"
        params["date_to"] = f"{today.year}-12-31"


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

    _apply_sales_date_filters(query, today, params, default_current_year=True)

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

    _apply_sales_date_filters(query, today, params, default_current_year=False)

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

    m_limit = re.search(
        r"\b(?:(?:limit|tampilkan|daftar|top)\s+(\d+)|(\d+)\s+(?:so|pesanan|order|quotation))\b",
        query,
        re.IGNORECASE,
    )
    if m_limit:
        try:
            val = int(m_limit.group(1) or m_limit.group(2))
            if 1 <= val <= 100:
                params["limit"] = val
        except ValueError:
            pass
    elif re.search(r"\b(?:jabarkan|semua\s+datanya|dalam\s+table|dalam\s+tabel)\b", clean_q):
        params["limit"] = 50

    _apply_sales_date_filters(query, today, params, default_current_year=False)
    if not params.get("date_from") and not params.get("date_to") and is_breakdown_or_followup and prior_ctx.get("date_from"):
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

    _apply_sales_date_filters(query, today, params, default_current_year=False)

    return params
