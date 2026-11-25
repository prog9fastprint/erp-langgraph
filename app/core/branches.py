"""
Canonical FastPrint ERP Branch / Company Registry and Resolution Helpers.

Single source of truth for branch-to-company_id mappings, display names,
multi-company rollup keywords, and branch scope resolution shared across
router, extractor, tool_node, synthesizer, and workflow.
"""
import re
from typing import Dict, List, Optional, Tuple

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

_PREFIX_ONLY_BRANCH_KEYS = {"hr", "sandy"}
_BRANCH_PREFIX = r"(?:di\s*cabang|dicabang|cabang|di\s*toko|toko|di\s*kantor|kantor|company|pt|di)\s+"
_NON_BRANCH_MS_FOLLOW_RE = re.compile(
    r"^\s+(?:word|excel|office|powerpoint|access|sql|teams|windows|dos|edge|outlook|project|visio)\b"
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
        esc_key = re.escape(branch_key)
        if branch_key in _PREFIX_ONLY_BRANCH_KEYS:
            pattern = rf"\b{_BRANCH_PREFIX}{esc_key}\b"
        else:
            pattern = rf"(?:\b{_BRANCH_PREFIX})?\b{esc_key}\b"
        for m in re.finditer(pattern, cleaned):
            if branch_key == "ms":
                if re.search(r"\d\s*$", cleaned[:m.start()]):
                    continue
                if _NON_BRANCH_MS_FOLLOW_RE.match(cleaned[m.end():]):
                    continue
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


__all__ = [
    "BRANCH_COMPANY_MAP",
    "COMPANY_NAME_MAP",
    "ALL_COMPANIES_KEYWORDS",
    "LOCATION_STOP_WORDS",
    "resolve_all_branch_company_ids",
    "resolve_branch_company_id",
    "strip_company_scope_keywords",
]
