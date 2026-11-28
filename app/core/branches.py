"""
Canonical FastPrint ERP Branch / Company Registry and Resolution Helpers.

Single source of truth for branch-to-company_id mappings, display names,
multi-company rollup keywords, and branch scope resolution shared across
router, extractor, tool_node, synthesizer, and workflow.
"""
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
try:
    from langchain_core.messages import HumanMessage
except ImportError:
    class HumanMessage:  # type: ignore
        """Fallback lightweight HumanMessage stub when langchain_core is not present."""
        pass

from app.core.context import has_anaphoric_reference

# Canonical mapping from Indonesian branch/city names to FastPrint ERP company_id
BRANCH_COMPANY_MAP: Dict[str, int] = {
    "kantor pusat": 1,
    "toko pusat": 2,
    "head office": 3,
    "head-office": 3,
    "headoffice": 3,
    "ho": 3,
    "main store": 4,
    "main-store": 4,
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
_CONTRASTIVE_NEGATED_BRANCH_RE = re.compile(
    r"(?:\b(?:kenapa|mengapa|kok)\s+(?:masih|tetap|malah)|"
    r"\b(?:bukan|jangan|selain)(?:\s+(?:di|cabang|toko|kantor))?)\s*$",
    re.IGNORECASE,
)

# Canonical branch suffix tokens embedded in FastPrint COA account codes (e.g. '11204HO', '40000MS')
BRANCH_ACCOUNT_SUFFIX_MAP: Dict[str, int] = {
    "HO": 3,
    "MS": 4,
    "MKS": 5,
    "BDG": 6,
    "SBY": 7,
    "JKT": 8,
    "MDN": 10,
    "BKS": 13,
    "TGR": 14,
    "MLG": 15,
}
_ACCOUNT_BRANCH_SUFFIX_RE = re.compile(r"\d([A-Za-z]{2,3})(?:$|[-\.\s])")
_AI_REPORT_BRANCH_HEADER_RE = re.compile(
    r"(?:\(\s*Cabang\s*:\s*([^,\)\n]+?)(?:\s*,|\s*\))|"
    r"###\s+Laporan\s+(?:Laba\s+Rugi|Neraca\s+Keuangan)[^\n\(]*\([^)]+\)\s*\(([^,\)\n]+?)\))",
    re.IGNORECASE,
)


class BranchRegistry:
    """
    Dynamic Registry for Multi-Tenant / Dynamic Branch and Company Mapping.
    Permits external systems (Odoo, ERPs, Retail POS) to register their branches dynamically.
    """
    def __init__(self) -> None:
        self._tenants: Dict[str, Dict[str, Any]] = {}
        self._default_branch_map = BRANCH_COMPANY_MAP
        self._default_name_map = COMPANY_NAME_MAP

    def register_tenant(
        self,
        tenant_id: str,
        branch_map: Dict[str, int],
        name_map: Dict[int, str],
    ) -> None:
        self._tenants[tenant_id] = {
            "branch_map": {k.lower().strip(): int(v) for k, v in branch_map.items()},
            "name_map": {int(k): str(v) for k, v in name_map.items()},
        }

    def get_branch_map(self, tenant_id: Optional[str] = None) -> Dict[str, int]:
        if tenant_id and tenant_id in self._tenants:
            return self._tenants[tenant_id]["branch_map"]
        return self._default_branch_map

    def get_name_map(self, tenant_id: Optional[str] = None) -> Dict[int, str]:
        if tenant_id and tenant_id in self._tenants:
            return self._tenants[tenant_id]["name_map"]
        return self._default_name_map

    def add_branch(
        self,
        branch_key: str,
        company_id: int,
        display_name: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> None:
        b_map = self.get_branch_map(tenant_id)
        n_map = self.get_name_map(tenant_id)
        b_map[branch_key.lower().strip()] = int(company_id)
        if display_name:
            n_map[int(company_id)] = str(display_name)


branch_registry = BranchRegistry()


def is_cross_branch_account_code(
    account_code: Optional[str],
    target_company_id: Optional[int],
) -> bool:
    """Return True if account_code carries an explicit branch suffix belonging to another company."""
    if not account_code or target_company_id is None:
        return False
    m = _ACCOUNT_BRANCH_SUFFIX_RE.search(str(account_code).strip())
    if not m:
        return False
    suffix = m.group(1).upper()
    owner_cid = BRANCH_ACCOUNT_SUFFIX_MAP.get(suffix)
    if owner_cid is None:
        return False
    try:
        return owner_cid != int(target_company_id)
    except (TypeError, ValueError):
        return False


def resolve_all_branch_company_ids(
    text: Optional[str],
    tenant_id: Optional[str] = None,
) -> Tuple[List[int], bool]:
    """
    Resolve multiple branch/city mentions or 'semua cabang' into a list of company_ids.
    Filters out contrastive/negated branches (e.g. 'kenapa masih ho bukanya aku minta mainstore')
    when at least one positive target branch is present in the same text.

    Returns:
        Tuple of (list_of_company_ids, is_all_companies_flag)
    """
    if not text:
        return [], False
    cleaned = text.lower().strip()

    active_name_map = branch_registry.get_name_map(tenant_id)
    active_branch_map = branch_registry.get_branch_map(tenant_id)

    # Check for all-companies / network-wide rollup keywords
    if any(re.search(rf"\b{re.escape(kw)}\b", cleaned) for kw in ALL_COMPANIES_KEYWORDS):
        return sorted(list(active_name_map.keys())), True

    matched_spans: List[Tuple[int, int, int, bool]] = []  # (start_idx, end_idx, company_id, is_negated)
    for branch_key in sorted(active_branch_map.keys(), key=lambda k: -len(k)):
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
            if not any(max(start, s) < min(end, e) for s, e, _, _ in matched_spans):
                prefix_window = cleaned[max(0, start - 32):start]
                is_negated = bool(_CONTRASTIVE_NEGATED_BRANCH_RE.search(prefix_window))
                matched_spans.append((start, end, active_branch_map[branch_key], is_negated))

    matched_spans.sort(key=lambda x: x[0])
    positive_spans = [item for item in matched_spans if not item[3]]
    effective_spans = positive_spans if positive_spans else matched_spans

    seen = set()
    ordered_ids: List[int] = []
    for _, _, cid, _ in effective_spans:
        if cid not in seen:
            seen.add(cid)
            ordered_ids.append(cid)

    return ordered_ids, False


def resolve_branch_company_id(
    text: Optional[str],
    tenant_id: Optional[str] = None,
) -> Optional[int]:
    """
    Resolve single Indonesian city or branch mention into official company_id.
    Matches phrases like 'cabang bandung', 'dicabang bandung', 'di bandung', 'kantor pusat', etc.
    """
    if not text:
        return None
    ids, is_all = resolve_all_branch_company_ids(text, tenant_id=tenant_id)
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

    prior_msgs = (
        list(messages[:-1])
        if (q_low and q_low in str(getattr(messages[-1], "content", "") or "").lower())
        else list(messages)
    )
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


def resolve_sticky_branch_from_history(
    prior_messages: Optional[Sequence[Any]],
) -> Optional[int]:
    """
    Recover the latest explicit single-branch override from conversation history
    during stateless hydration (e.g. after server reload or checkpoint eviction).
    Returns None if a newer assistant report explicitly rendered a different branch
    or a newer user turn requested multi-company scope.
    """
    if not prior_messages:
        return None

    latest_ai_cid: Optional[int] = None
    for msg in reversed(list(prior_messages)):
        content = str(getattr(msg, "content", "") or "").strip()
        if not content:
            continue
        msg_type = getattr(msg, "type", "")
        if isinstance(msg, HumanMessage) or msg_type in ("human", "user"):
            cids, is_all = resolve_all_branch_company_ids(content)
            if is_all or len(cids) > 1:
                return None
            if len(cids) == 1:
                human_cid = cids[0]
                if latest_ai_cid is not None and latest_ai_cid != human_cid:
                    return None
                return human_cid
        elif latest_ai_cid is None:
            m_hdr = _AI_REPORT_BRANCH_HEADER_RE.search(content)
            if m_hdr:
                branch_lbl = (m_hdr.group(1) or m_hdr.group(2) or "").strip()
                ai_cid = resolve_branch_company_id(branch_lbl)
                if ai_cid is not None:
                    latest_ai_cid = ai_cid
    return None


__all__ = [
    "BRANCH_COMPANY_MAP",
    "BRANCH_ACCOUNT_SUFFIX_MAP",
    "COMPANY_NAME_MAP",
    "ALL_COMPANIES_KEYWORDS",
    "LOCATION_STOP_WORDS",
    "BranchRegistry",
    "branch_registry",
    "is_cross_branch_account_code",
    "resolve_all_branch_company_ids",
    "resolve_branch_company_id",
    "resolve_prior_multi_company_scope",
    "resolve_sticky_branch_from_history",
    "strip_company_scope_keywords",
]
