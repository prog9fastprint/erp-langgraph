"""
Shared Markdown formatting utilities for FastPrint ERP presentation formatters.
"""
from datetime import date, datetime
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from app.core.branches import COMPANY_NAME_MAP
from app.core.context import get_current_erp_date


def _get_formatter_today() -> date:
    """Resolve current ERP date honoring test patches on synthesizer.get_current_erp_date."""
    synth_mod = sys.modules.get("app.graph.nodes.synthesizer")
    date_fn = getattr(synth_mod, "get_current_erp_date", None) if synth_mod is not None else None
    if callable(date_fn):
        return date_fn()
    return get_current_erp_date()


_EVALUATIVE_QUERY_RE = re.compile(
    r"\b(?:menurutmu|bisa\s+(?:engga|enggak|gak|tidak|kah)|tercapai|tembus|target|proyeksi|"
    r"run[\s\-]?rate|impact|dampak|pengaruh|signifikan|alasannya|kenapa)\b",
    re.IGNORECASE,
)


def _parse_sales_target_idr(query: str) -> Optional[float]:
    """Extract numeric monetary target in IDR from Indonesian shorthand (e.g. '6M', '6 miliar', '500 jt')."""
    if not query:
        return None
    q_low = query.lower()
    m = re.search(
        r"(?:(?:target|tembus|mencapai|capaian|sebesar|senilai|rp)\s*)*(\d+(?:[.,]\d+)?)\s*(miliar|milyar|m\b|juta|jt\b|ribu|rb\b)",
        q_low,
    )
    if not m:
        return None
    raw_num, unit = m.group(1), m.group(2)
    if unit == "m" and not re.search(r"\b(?:target|tembus|mencapai|omzet|omset|penjualan|rp)\b", q_low):
        return None
    try:
        val = float(raw_num.replace(",", "."))
    except ValueError:
        return None
    if unit in ("miliar", "milyar", "m"):
        return val * 1_000_000_000.0
    if unit in ("juta", "jt"):
        return val * 1_000_000.0
    if unit in ("ribu", "rb"):
        return val * 1_000.0
    return None


def _parse_iso_date_safe(val: Any) -> Optional[date]:
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, str) and re.match(r"^\d{4}-\d{2}-\d{2}", val.strip()):
        try:
            return date.fromisoformat(val.strip()[:10])
        except ValueError:
            return None
    return None


def _build_evaluative_verdict(
    user_query: str,
    tool_entries: List[Tuple[str, Any, Optional[Dict[str, Any]]]],
) -> str:
    """Compute deterministic run-rate projection & marketplace impact verdicts when user asks an evaluative question."""
    if not user_query or not _EVALUATIVE_QUERY_RE.search(user_query):
        return ""

    sales_entry: Optional[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]] = None
    ded_res: Optional[Dict[str, Any]] = None
    ah_res: Optional[Dict[str, Any]] = None

    for t_name, res, params in tool_entries:
        if not isinstance(res, dict):
            continue
        if t_name == "get_sales_summary":
            sales_entry = (res, params)
        elif t_name == "get_deduction_and_settlement_summary":
            ded_res = res
        elif t_name == "get_account_health":
            ah_res = res

    verdict_blocks: List[str] = []
    q_low = user_query.lower()

    # 1. Target / Run-rate / Pace Evaluation
    target_val = _parse_sales_target_idr(user_query)
    if sales_entry and (target_val is not None or re.search(r"\b(?:target|proyeksi|run[\s\-]?rate|tembus|tercapai|bisa\s+(?:engga|enggak|gak))\b", q_low)):
        s_res, s_params = sales_entry
        conf = s_res.get("confirmed_metrics") or {}
        real_rev = float(conf.get("realized_revenue") or 0.0)
        period_blk = s_res.get("period") if isinstance(s_res.get("period"), dict) else {}
        df_s = period_blk.get("date_from") or (s_params.get("date_from") if isinstance(s_params, dict) else None)
        dt_s = period_blk.get("date_to") or (s_params.get("date_to") if isinstance(s_params, dict) else None)
        p_start = _parse_iso_date_safe(df_s)
        p_end = _parse_iso_date_safe(dt_s)
        today = _get_formatter_today()

        if p_start and p_end and p_end >= p_start:
            obs_end = min(today, p_end) if today >= p_start else p_start
            elapsed_days = max(1, (obs_end - p_start).days + 1) if today >= p_start else 0
            remaining_days = max(0, (p_end - obs_end).days) if today >= p_start else (p_end - p_start).days + 1
            daily_rate = (real_rev / elapsed_days) if elapsed_days > 0 else 0.0
            linear_proj = real_rev + (daily_rate * remaining_days)

            b_lines = ["#### Analisis Proyeksi Target & Run-Rate\n"]
            b_lines.append(
                f"- **Periode Evaluasi**: `{p_start.isoformat()}` s/d `{p_end.isoformat()}` "
                f"({elapsed_days} hari berjalan s/d `{obs_end.isoformat()}`, sisa {remaining_days} hari)"
            )
            b_lines.append(f"- **Realisasi Berjalan**: **{_format_rupiah(real_rev)}**")
            b_lines.append(f"- **Run-Rate Harian Berjalan**: **{_format_rupiah(daily_rate)}** / hari")
            b_lines.append(f"- **Proyeksi Linear Akhir Periode**: **{_format_rupiah(linear_proj)}**")
            if target_val and target_val > 0:
                gap = max(0.0, target_val - real_rev)
                pct = (real_rev / target_val) * 100.0
                req_pace = (gap / remaining_days) if remaining_days > 0 and gap > 0 else 0.0
                if real_rev >= target_val:
                    verdict_str = "Tercapai / Melebihi Target"
                elif linear_proj >= target_val * 0.95:
                    verdict_str = "Berpotensi Tercapai (On-Track)"
                else:
                    verdict_str = "Berisiko Tidak Tercapai pada Laju Saat Ini"
                b_lines.append(f"- **Target**: **{_format_rupiah(target_val)}** (Tercapai **{pct:.1f}%**, Gap **{_format_rupiah(gap)}**)")
                if gap > 0 and remaining_days > 0:
                    b_lines.append(f"- **Laju Harian Dibutuhkan (Sisa {remaining_days} Hari)**: **{_format_rupiah(req_pace)}** / hari")
                b_lines.append(f"- **Kesimpulan**: **{verdict_str}**")
            verdict_blocks.append("\n".join(b_lines))

    # 2. Marketplace Pull Failure & Deduction Impact Evaluation
    if (ded_res or ah_res) and re.search(r"\b(?:impact|dampak|pengaruh|signifikan|gagal|potongan|deduction|omzet|omset)\b", q_low):
        i_lines = ["#### Ringkasan Evaluasi Dampak ke Omzet & Operasional Marketplace\n"]
        has_content = False
        if ah_res:
            ah_sum = ah_res.get("summary") or {}
            err_cnt = int(ah_sum.get("recent_fetch_errors_count") or 0)
            pulled_cnt = ah_sum.get("pulled_orders_count")
            err_ratio = ah_sum.get("error_to_pulled_order_ratio")
            ratio_note = ""
            if pulled_cnt is not None:
                pulled_int = int(pulled_cnt)
                pct_val = (err_cnt / pulled_int * 100.0) if pulled_int > 0 else (float(err_ratio or 0.0) * 100.0)
                ratio_note = f" vs **{pulled_int:,} order** berhasil ditarik (rasio **{pct_val:.2f}%**)"
            by_type = ah_res.get("error_breakdown_by_type") or []
            top_type = f" — dominan `{by_type[0].get('error_type')}` ({by_type[0].get('count')}x)" if by_type else ""
            i_lines.append(f"- **Dampak Penarikan Order Marketplace**: Tercatat **{err_cnt:,} event error fetch**{ratio_note}{top_type}.")
            has_content = True
        if ded_res:
            d_sum = ded_res.get("summary") or {}
            tot_gross = float(d_sum.get("total_gross") or 0.0)
            tot_ded = abs(float(d_sum.get("total_deduction") or 0.0))
            tot_nett = float(d_sum.get("total_nett") or 0.0)
            eff_pct = float(d_sum.get("effective_deduction_pct") or 0.0)
            if eff_pct >= 10.0:
                severity = "signifikan (tinggi, >= 10%)"
            elif eff_pct >= 5.0:
                severity = "moderat (5% - 10%)"
            else:
                severity = "rendah (< 5%)"
            sales_share_note = ""
            if sales_entry:
                s_rev = float((sales_entry[0].get("confirmed_metrics") or {}).get("realized_revenue") or 0.0)
                if s_rev > 0:
                    share_pct = (tot_ded / s_rev) * 100.0
                    sales_share_note = f" (setara **{share_pct:.2f}%** dari total omzet ERP {_format_rupiah(s_rev)})"
            i_lines.append(
                f"- **Dampak Potongan Fee Marketplace**: Total potongan **{_format_rupiah(tot_ded)}** dari Gross **{_format_rupiah(tot_gross)}** "
                f"(efektif **{eff_pct:.2f}%** — tingkat **{severity}**, Nett **{_format_rupiah(tot_nett)}**){sales_share_note}."
            )
            has_content = True
        if has_content:
            verdict_blocks.append("\n".join(i_lines))

    return "\n\n".join(verdict_blocks)


def _is_zero_metric(*values: Any) -> bool:
    """Return True when every provided metric value is effectively zero."""
    return all(abs(float(v or 0.0)) < 0.005 for v in values)


def _append_zero_branch_footnote(
    lines: List[str],
    zero_branches: List[str],
    label: str = "Cabang tanpa aktivitas / saldo Rp 0,00 (tidak ditampilkan di tabel)",
) -> None:
    """Append one compact footnote listing omitted zero-activity branches."""
    if not zero_branches:
        return
    clean_names = list(dict.fromkeys(str(b).strip() for b in zero_branches if str(b).strip()))
    if clean_names:
        lines.append(f"\n*{label}: {', '.join(clean_names)}.*")


def _resolve_company_display_label(
    tool_result: Any,
    extracted_params: Optional[Dict[str, Any]] = None,
    default_company_id: Optional[int] = None,
) -> Optional[str]:
    """Resolve readable branch/company label from tool response or scoped company_id."""
    if isinstance(tool_result, dict) and tool_result.get("company_name"):
        return str(tool_result["company_name"]).strip()
    cid = (
        (tool_result.get("company_id") if isinstance(tool_result, dict) else None)
        or (extracted_params.get("company_id") if isinstance(extracted_params, dict) else None)
        or default_company_id
    )
    if cid is not None:
        try:
            return COMPANY_NAME_MAP.get(int(cid), f"Company #{int(cid)}")
        except (TypeError, ValueError):
            pass
    return None


def _resolve_period_bounds(
    tool_result: Any,
    extracted_params: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Extract period date_from/date_to from tool result envelope or extracted_params."""
    period = tool_result.get("period") if isinstance(tool_result, dict) and isinstance(tool_result.get("period"), dict) else {}
    df = (
        (tool_result.get("date_from") if isinstance(tool_result, dict) else None)
        or period.get("date_from")
        or (extracted_params.get("date_from") if isinstance(extracted_params, dict) else None)
    )
    dt = (
        (tool_result.get("date_to") if isinstance(tool_result, dict) else None)
        or period.get("date_to")
        or (extracted_params.get("date_to") if isinstance(extracted_params, dict) else None)
    )
    return (str(df) if df else None), (str(dt) if dt else None)


def _resolve_mc_period_note(mc_results: Dict[int, Any], extracted_params: Optional[Dict[str, Any]]) -> str:
    """Resolve optional period suffix for multi-company report headers."""
    for res in mc_results.values():
        if isinstance(res, dict):
            df, dt = _resolve_period_bounds(res, extracted_params)
            if df or dt:
                return f" (Periode: {df or '-'} s/d {dt or '-'})"
    df, dt = _resolve_period_bounds(None, extracted_params)
    return f" (Periode: {df or '-'} s/d {dt or '-'})" if (df or dt) else ""


def _resolve_product_display_price(prod: Dict[str, Any]) -> float:
    """Resolve best sales price prioritizing positive effective pricelist over nominal fallbacks."""
    for key in (
        "effective_sales_price",
        "default_sales_price",
        "sales_price",
        "list_price",
        "price",
        "last_unit_price",
    ):
        val = prod.get(key)
        if val is not None:
            try:
                fval = float(val)
                if fval > 0:
                    return fval
            except (TypeError, ValueError):
                pass
    return float(
        prod.get("default_sales_price")
        or prod.get("list_price")
        or prod.get("last_unit_price")
        or 0.0
    )


def _format_contact_table_rows(
    contacts: List[Dict[str, Any]],
    max_rows: Optional[int] = None,
) -> List[str]:
    """Render standardized Markdown rows for contact/buyer tables with list-field fallbacks."""
    rows: List[str] = []
    subset = contacts[:max_rows] if max_rows is not None else contacts
    for idx, c in enumerate(subset, start=1):
        c_name = c.get("name") or "-"
        hp = c.get("phone") or c.get("mobile") or "-"
        city = c.get("city") or c.get("state") or "-"
        ch_list = c.get("channels") or []
        ch_src = ", ".join(str(x) for x in ch_list if x) if isinstance(ch_list, list) else str(ch_list or "")
        if not ch_src:
            ch_src = c.get("first_time_source") or "-"
        sp_list = c.get("sales_phones") or []
        s_hp = ", ".join(str(x) for x in sp_list if x) if isinstance(sp_list, list) else str(sp_list or "")
        if not s_hp:
            s_hp = c.get("sales_phone") or "-"
        o_cnt = int(c.get("orders_count") or 0)
        t_spent = float(c.get("total_spent") or 0.0)
        spent_str = f"{o_cnt} order ({_format_rupiah(t_spent)})" if o_cnt > 0 or t_spent > 0 else "-"
        recent = c.get("recent_orders") or []
        recent_str = ", ".join(
            f"`{ro.get('so_number')}` ({ro.get('company_name') or '-'}, `{ro.get('status') or '-'}`)"
            for ro in recent[:2]
            if isinstance(ro, dict) and ro.get("so_number")
        ) or "-"
        rows.append(f"| {idx} | **{c_name}** | {hp} | {city} | {ch_src} | {s_hp} | {spent_str} | {recent_str} |")
    return rows


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
    "_EVALUATIVE_QUERY_RE",
    "_parse_sales_target_idr",
    "_parse_iso_date_safe",
    "_build_evaluative_verdict",
    "_is_zero_metric",
    "_append_zero_branch_footnote",
    "_resolve_company_display_label",
    "_resolve_period_bounds",
    "_resolve_mc_period_note",
    "_resolve_product_display_price",
    "_format_contact_table_rows",
    "_format_rupiah",
    "_format_currency",
    "_format_rag_section",
    "_extract_total_count",
    "_append_pagination_footer",
]
