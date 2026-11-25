"""
Multi-question clause decomposition and conversational retry helpers for Router.
"""
import re
from typing import List, Optional, Tuple
from langchain_core.messages import BaseMessage, HumanMessage

from app.graph.nodes.router_patterns import RETRY_PATTERNS, _match_any



_ABBREV_DOT_RE = re.compile(
    r"\b(PT|CV|UD|PD|TB|FA|No|Rp|Jl|Dr|Mr|Mrs|Ms|St|a\.n|s\.d|e\.g|i\.e)\.",
    re.IGNORECASE,
)
_CLAUSE_INTERROGATIVE_OR_VERB_RE = re.compile(
    r"^(?:(?:tolong|coba|mohon|silakan|boleh)\s+)*"
    r"(?:berapa(?:kah)?|apa(?:kah)?|siapa(?:kah)?|kenapa|mengapa|bagaimana|gimana|"
    r"dimana|di\s+mana|kapan|mana(?:kah)?|"
    r"cek|dicek|cari|dicari|tampilkan|lihat|lihatin|bandingkan|periksa|diperiksa|"
    r"hitung|rekap|jelaskan|sebutkan|kasih\s+tau)\b",
    re.IGNORECASE,
)
_PURE_GREETING_ONLY_RE = re.compile(
    r"^(?:halo|hai|hi|hey|hei|helo|hello|selamat\s+(?:pagi|siang|sore|malam|petang)|"
    r"assalamu\s*['’]?alaikum|permisi|pagi|siang|sore|malam)(?:\s+(?:min|kak|mas|mbak|gan|bro|pak|bu))?$",
    re.IGNORECASE,
)
_PROTECTED_CONJUNCTION_PAIR_RE = re.compile(
    r"(?:\b(?:laba|rugi)\s+(?:dan|&)\s+(?:rugi|laba)|"
    r"\bprofit\s+(?:and|&)\s+loss|"
    r"\baktiva\s+dan\s+pasiva|"
    r"\bsemua\s+cabang\s+dan\s+(?:ho|headoffice|head\s*office)|"
    r"\bharga\s+(?:jual|beli)\s+dan\s+(?:barcode|estimasi\s+hpp|hpp|beli|jual)|"
    r"\b(?:stok|stock)\s+(?:atau|dan)\s+(?:harga|price))\b",
    re.IGNORECASE,
)


def _split_conjunction_clauses(segment: str) -> List[str]:
    """
    Split a single sentence on coordinating conjunctions/commas only when the right side
    begins an independent question or imperative command.
    """
    if not segment or len(segment) < 6:
        return [segment] if segment else []

    splitter_re = re.compile(
        r"\s*(?:,\s*(?:dan|lalu|kemudian|serta|sekaligus|juga)\s+|"
        r"\b(?:dan|serta|sekaligus|lalu|kemudian)\b|"
        r"(?<!\d),(?!\d))\s*",
        re.IGNORECASE,
    )
    matches = list(splitter_re.finditer(segment))
    if not matches:
        return [segment]

    clauses: List[str] = []
    start_idx = 0
    for m in matches:
        left_part = segment[start_idx:m.start()].strip(" ,.;:-\t")
        right_part = segment[m.end():].strip(" ,.;:-\t")
        if not left_part or not right_part:
            continue
        # Never split inside protected domain phrases (e.g. 'laba dan rugi', 'semua cabang dan HO')
        window = segment[max(0, m.start() - 25):min(len(segment), m.end() + 25)]
        if _PROTECTED_CONJUNCTION_PAIR_RE.search(window):
            continue
        # Do not split off pure greetings ('Halo, cek stok...')
        if _PURE_GREETING_ONLY_RE.match(left_part):
            continue
        if len(left_part.split()) >= 2 and _CLAUSE_INTERROGATIVE_OR_VERB_RE.match(right_part):
            clauses.append(left_part)
            start_idx = m.end()

    tail = segment[start_idx:].strip()
    if tail:
        clauses.append(tail)
    return clauses or [segment]


def decompose_user_query(query: str) -> List[str]:
    """
    Deterministically decompose a multi-question prompt into ordered sub-questions.
    Splits on newlines, numbered/bulleted lines, sentence terminators (?, !, ;, non-abbrev .),
    and coordinating conjunctions introducing a new question or imperative verb.
    """
    raw = (query or "").strip()
    if not raw:
        return []

    # Protect abbreviations, decimals, and slash document numbers from period splitting
    placeholder = "\u0000"
    protected = _ABBREV_DOT_RE.sub(lambda m: f"{m.group(1)}{placeholder}", raw)
    protected = re.sub(r"(\d)\.(\d)", rf"\1{placeholder}\2", protected)

    # 1. Split on newlines and sentence terminators (? ! ; or non-numeric .)
    primary_segments: List[str] = []
    for line in re.split(r"[\r\n]+", protected):
        line_clean = line.strip()
        if not line_clean:
            continue
        # Strip leading list marker only when multiline or numbered list
        if "\n" in raw or re.match(r"^\d+[.)]\s+", line_clean):
            line_clean = re.sub(r"^(?:\d+[.)]\s+|[-*•]\s+)", "", line_clean).strip()
        # Split sentences inside the line
        sent_parts = re.split(r"(?<=[?!;])\s+|(?<!\d)\.(?!\d)(?=\s+[^\s])", line_clean)
        for sp in sent_parts:
            sp_restored = sp.replace(placeholder, ".").strip()
            # Strip dangling coordinating conjunction after '?' (e.g., "Apa SOP retur? dan berapa...")
            sp_restored = re.sub(
                r"^(?:dan|lalu|kemudian|serta|juga|terus)\s+(?=(?:berapa|apa|siapa|kenapa|mengapa|bagaimana|gimana|dimana|di\s+mana|kapan|mana|cek|cari|tampilkan|lihat|tolong|coba)\b)",
                "",
                sp_restored,
                flags=re.IGNORECASE,
            ).strip()
            if sp_restored:
                primary_segments.append(sp_restored)

    if not primary_segments:
        return [raw]

    # 2. Split coordinating clauses inside each sentence when right side is an independent question/command
    final_clauses: List[str] = []
    for seg in primary_segments:
        final_clauses.extend(_split_conjunction_clauses(seg))

    # Merge leading pure greeting into the first substantive clause if present
    if len(final_clauses) > 1 and _PURE_GREETING_ONLY_RE.match(final_clauses[0].rstrip("?!., ")):
        final_clauses[1] = f"{final_clauses[0]} {final_clauses[1]}".strip()
        final_clauses.pop(0)

    return final_clauses or [raw]


def _extract_sub_queries(query: str, candidates: List[Tuple[str, str, int]]) -> List[str]:
    """
    Split multi-intent query into focused sub-queries along sentence boundaries and Indonesian conjunctions.
    Falls back to the full query if clause count does not align with candidates count.
    """
    if len(candidates) <= 1:
        return [query] * len(candidates)

    decomposed = decompose_user_query(query)
    if len(decomposed) == len(candidates):
        return decomposed

    parts = re.split(r"\b(?:dan|serta|sekaligus|juga|lalu)\b|;|(?<!\d),(?!\d)|(?<=[?!])\s+", query, flags=re.IGNORECASE)
    cleaned_parts = [p.strip() for p in parts if p.strip()]

    if len(cleaned_parts) == len(candidates):
        return cleaned_parts

    return [query] * len(candidates)


def is_retry_query(text: str) -> bool:
    """Check if the text represents a conversational retry command."""
    cleaned = (text or "").strip().rstrip("?!.,;: '\"").lower()
    return _match_any(RETRY_PATTERNS, cleaned)


def find_previous_user_query(messages: Optional[List[BaseMessage]]) -> Optional[str]:
    """
    Find the most recent HumanMessage content prior to the current turn that is not a retry command.
    """
    if not messages:
        return None
    for msg in reversed(messages[:-1]):
        if isinstance(msg, HumanMessage):
            content = getattr(msg, "content", "") or ""
            content_str = str(content).strip()
            if content_str and not is_retry_query(content_str):
                return content_str
    return None

