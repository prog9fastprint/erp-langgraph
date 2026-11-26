"""
Domain parameter extraction helpers for FastPrint ERP (common).
"""
import calendar
from datetime import date, timedelta
import re
from typing import Dict, Optional, Tuple
from app.core.branches import resolve_branch_company_id
from app.core.context import get_current_erp_date


ALLOWED_JOURNAL_STATUSES = {"draft", "posted", "post", "cancel", "unpaid", "paid"}

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


LOCATION_STOPWORDS = (
    "gudang", "lokasi", "warehouse",
    "apa", "saja", "semua", "yang", "ada", "aktif", "ini", "itu", "daftar", "list",
    "cek", "info", "mana", "cari",
)


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
