"""
Shared Markdown formatting utilities for FastPrint ERP presentation formatters.
"""
from typing import Any, Dict, List, Optional


def _format_rupiah(value: Any) -> str:
    """Format numeric values into standard Indonesian Rupiah notation."""
    try:
        num = float(value)
        return f"Rp {num:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except (ValueError, TypeError):
        return str(value)


def _format_currency(value: Any, currency: Optional[str] = "IDR") -> str:
    """Format numeric monetary values respecting non-IDR marketplace currency codes."""
    curr = (currency or "IDR").strip().upper()
    if curr in ("IDR", "RP", ""):
        return _format_rupiah(value)
    try:
        num = float(value)
        return f"{curr} {num:,.2f}"
    except (ValueError, TypeError):
        return str(value)


def _format_rag_section(
    rag_context: Optional[List[Dict[str, Any]]],
    rag_sources: Optional[List[str]] = None,
) -> str:
    """Format retrieved RAG knowledge chunks and sources into structured Markdown."""
    if not rag_context:
        return ""
    lines = ["### Rujukan Pengetahuan & Dokumen SOP\n"]
    for i, chunk in enumerate(rag_context, 1):
        title = chunk.get("title") or (chunk.get("metadata") or {}).get("source") or f"Dokumen {i}"
        content = chunk.get("content", "").strip()
        score = chunk.get("similarity_score")
        score_str = f" (Relevansi: {score:.0%})" if score is not None else ""
        lines.append(f"#### {i}. {title}{score_str}")
        lines.append(f"{content}\n")

    sources = rag_sources or [c.get("title") for c in rag_context if c.get("title")]
    if sources:
        clean_sources = list(dict.fromkeys(sources))
        lines.append("**Sumber Rujukan Terverifikasi**:")
        for s in clean_sources:
            lines.append(f"- `{s}`")
    return "\n".join(lines)


def _extract_total_count(tool_result: Any, displayed_count: int) -> int:
    """Resolve pre-truncation record count from total_count or total_found."""
    if isinstance(tool_result, dict):
        if tool_result.get("total_count") is not None:
            return int(tool_result["total_count"])
        if tool_result.get("total_found") is not None:
            return int(tool_result["total_found"])
    return displayed_count


def _append_pagination_footer(
    lines: List[str],
    displayed_count: int,
    total_count: int,
    filter_hint: Optional[str] = None,
) -> None:
    """Append Markdown pagination notice when displayed rows are fewer than total records."""
    if displayed_count > 0 and total_count > displayed_count:
        hint = (
            filter_hint
            or "Gunakan filter periode tanggal atau kata kunci lebih spesifik untuk menelusuri data lainnya."
        )
        lines.append(f"\n*Menampilkan {displayed_count} dari {total_count} data. {hint}*")


__all__ = [
    "_format_rupiah",
    "_format_currency",
    "_format_rag_section",
    "_extract_total_count",
    "_append_pagination_footer",
]
