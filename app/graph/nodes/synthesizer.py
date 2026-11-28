"""
Synthesizer Node for FastPrint ERP LangGraph Agent Workflow (TASK-4.5).
Generates structured, professional final answers formatted in Indonesian Markdown
based on state verification status, tool execution data, calculations, and error messages.
"""
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple
from langchain_core.callbacks import adispatch_custom_event
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.branches import COMPANY_NAME_MAP, resolve_all_branch_company_ids
from app.core.config import settings
from app.core.context import (
    get_current_erp_date,
    is_analytical_thinking_query,
    trim_conversation_messages,
)
from app.graph.nodes.formatters import (
    _append_pagination_footer,
    _append_zero_branch_footnote,
    _build_evaluative_verdict,
    _extract_total_count,
    _format_contact_table_rows,
    _format_currency,
    _format_multi_company_presentation,
    _format_rag_section,
    _format_rupiah,
    _format_single_tool_presentation,
    _is_zero_metric,
    _parse_iso_date_safe,
    _parse_sales_target_idr,
    _resolve_company_display_label,
    _resolve_period_bounds,
    _resolve_product_display_price,
)
from app.graph.nodes.router_patterns import (
    is_general_help_inquiry,
    is_meta_question,
    is_pure_greeting,
)
from app.graph.state import AgentState
from app.services.llm import get_chat_model

logger = logging.getLogger("synthesizer_node")


def _format_meta_question_answer(state: AgentState) -> Optional[str]:
    """Answer deterministic session meta-questions about period, company scope, or latest ERP date."""
    user_query = (state.get("user_query") or "").strip()
    if not user_query or not is_meta_question(user_query):
        return None

    q_low = user_query.lower()
    today = get_current_erp_date()
    messages = state.get("messages") or []

    # 1. Active Period Meta-Question
    if re.search(r"\b(?:periode|rentang\s+(?:waktu|tanggal)|tanggal\s+(?:data\s+)?(?:yang|yg))\b", q_low):
        found_period: Optional[Tuple[str, str]] = None
        ext_p = state.get("extracted_params") or {}
        if ext_p.get("date_from") or ext_p.get("date_to"):
            found_period = (str(ext_p.get("date_from") or "-"), str(ext_p.get("date_to") or "-"))
        if not found_period:
            for msg in reversed(messages):
                content = str(getattr(msg, "content", "") or "")
                m_per = re.search(
                    r"Periode(?:\s+Evaluasi)?\s*[:\s]\s*`?(\d{4}-\d{2}-\d{2})`?\s*(?:s/d|-|to)\s*`?(\d{4}-\d{2}-\d{2})`?",
                    content,
                    re.IGNORECASE,
                )
                if m_per:
                    found_period = (m_per.group(1), m_per.group(2))
                    break
        if found_period:
            return (
                f"### Periode Data Aktif\n\n"
                f"Periode data yang dipakai pada penelusuran terakhir adalah **{found_period[0]} s/d {found_period[1]}** "
                f"(tanggal acuan sistem saat ini: `{today.isoformat()}`)."
            )
        return (
            f"### Periode Data Aktif\n\n"
            f"Belum ada rentang tanggal spesifik yang diterapkan pada sesi ini (mengikuti filter default masing-masing modul). "
            f"Tanggal acuan ERP saat ini adalah **`{today.isoformat()}`**."
        )

    # 2. Active Company Scope Meta-Question
    if re.search(r"\b(?:company|perusahaan|cabang)\b", q_low):
        cids = [int(c) for c in (state.get("company_ids") or []) if c]
        active_cid = state.get("company_id")
        if not cids and active_cid:
            cids = [int(active_cid)]
        auth_cids = [int(c) for c in (state.get("authorized_company_ids") or []) if c]

        lines = ["### Cakupan Perusahaan / Cabang (Company Scope)\n"]
        if cids:
            names = [f"{COMPANY_NAME_MAP.get(cid, f'Company #{cid}')} (`ID {cid}`)" for cid in cids]
            lines.append(f"- **Perusahaan Aktif Dipakai**: {', '.join(names)}")
        if auth_cids:
            lines.append("\n| Company ID | Nama Cabang / Entitas |")
            lines.append("|:---:|:---|")
            for cid in sorted(dict.fromkeys(auth_cids)):
                lines.append(f"| `{cid}` | {COMPANY_NAME_MAP.get(cid, f'Company #{cid}')} |")
        else:
            lines.append("\n| Company ID | Nama Cabang / Entitas |")
            lines.append("|:---:|:---|")
            for cid, cname in sorted(COMPANY_NAME_MAP.items()):
                lines.append(f"| `{cid}` | {cname} |")
        return "\n".join(lines)

    # 3. Latest ERP Date Meta-Question
    latest_seen: Optional[str] = None
    all_dates: List[str] = []
    for msg in reversed(messages):
        content = str(getattr(msg, "content", "") or "")
        for d_str in re.findall(r"\b(20\d{2}-\d{2}-\d{2})\b", content):
            if d_str <= today.isoformat():
                all_dates.append(d_str)
        if all_dates:
            break
    if all_dates:
        latest_seen = max(all_dates)

    lines = [
        "### Informasi Tanggal Acuan & Data Terbaru\n",
        f"- **Tanggal Acuan ERP Saat Ini**: **`{today.isoformat()}`**",
    ]
    if latest_seen:
        lines.append(f"- **Tanggal Transaksi Terbaru pada Hasil Penelusuran Terakhir**: **`{latest_seen}`**")
    return "\n".join(lines)


def _plan_dedup_key(plan: Dict[str, Any]) -> Tuple[Any, ...]:
    """Build normalized signature to skip duplicate tool plans rendering identical tables."""
    t_name = str(plan.get("tool") or "")
    params = plan.get("params") if isinstance(plan.get("params"), dict) else {}
    norm_params = json.dumps(
        {k: v for k, v in params.items() if k != "company_id"},
        sort_keys=True,
        default=str,
    )
    cids = tuple(int(c) for c in (plan.get("company_ids") or []) if c)
    if not cids and plan.get("company_id") is not None:
        cids = (int(plan["company_id"]),)
    elif not cids and params.get("company_id") is not None:
        cids = (int(params["company_id"]),)
    has_mc = bool(plan.get("multi_company_results"))
    return (t_name, norm_params, cids, has_mc, str(plan.get("status") or ""))



def _format_empty_plan_notice(
    plan: Dict[str, Any],
    default_company_id: Optional[int] = None,
) -> str:
    """Format a concise per-question empty-data notice for multi-question responses."""

    t_name = plan.get("tool") or "erp_tool"
    params = plan.get("params") or {}
    cid = plan.get("company_id") or params.get("company_id") or default_company_id
    c_label = COMPANY_NAME_MAP.get(cid, f"Cabang {cid}") if cid else None

    hints: List[str] = []
    if params.get("code"):
        hints.append(f"Produk/SKU: `{params['code']}`")
    elif params.get("codes"):
        hints.append(f"Produk/SKU: {', '.join(f'`{c}`' for c in params['codes'][:5])}")
    if params.get("account_code"):
        hints.append(f"Kode Akun: `{params['account_code']}`")
    if params.get("vendor_name"):
        hints.append(f"Vendor: **{params['vendor_name']}**")
    if params.get("customer_name"):
        hints.append(f"Pelanggan: **{params['customer_name']}**")
    if params.get("order_ref"):
        hints.append(f"No. SO: `{params['order_ref']}`")
    if params.get("location_name"):
        hints.append(f"Lokasi: **{params['location_name']}**")
    if params.get("date_from") or params.get("date_to"):
        hints.append(f"Periode: **{params.get('date_from', '-')} s/d {params.get('date_to', '-')}**")
    if c_label:
        hints.append(f"Cabang: **{c_label}**")

    hint_str = f" ({', '.join(hints)})" if hints else ""
    return (
        f"### Data Tidak Ditemukan (`{t_name}`)\n\n"
        f"Tidak ada catatan atau transaksi ERP yang sesuai untuk permintaan ini{hint_str}."
    )


def _format_non_tool_item_notice(
    item: Dict[str, Any],
    llm_general_answers: Optional[Dict[int, str]] = None,
) -> str:
    """Format a single general / out_of_scope / guardrail sub-question response block."""
    cat = item.get("category") or "general"
    q_idx = int(item.get("question_index") or 0)

    if llm_general_answers and q_idx in llm_general_answers and llm_general_answers[q_idx].strip():
        return llm_general_answers[q_idx].strip()

    if cat == "customer_billing":
        return (
            "### Informasi Tagihan Pelanggan (Customer) & Panduan Faktur\n\n"
            "Sistem AI Orchestrator saat ini dikhususkan untuk **tagihan vendor / pemasok** (Hutang Usaha), "
            "buku jurnal, saldo akun kas, dan inventaris barang.\n\n"
            "- **Tagihan / Piutang Pelanggan**: Silakan periksa modul **Penjualan (Sales / Invoicing)** atau POS di FastPrint ERP.\n"
            "- **Jika Rekanan adalah Vendor**: Sertakan kata `'vendor'` pada permintaan Anda."
        )
    if cat == "marketplace_write":
        return (
            "### Operasi Tulis Marketplace Dibatasi (Read-Only)\n\n"
            "Asisten AI hanya mendukung penelusuran data dan simulasi read-only untuk modul Marketplace. "
            "Perubahan stok/harga langsung atau penggabungan Pesanan Kilat wajib dilakukan melalui antarmuka resmi ERP (`/marketplace/`)."
        )
    if cat == "out_of_scope":
        return (
            "### Permintaan di Luar Cakupan (Out of Scope)\n\n"
            "Maaf, pertanyaan ini berada di luar cakupan dan kemampuan Asisten AI FastPrint ERP. "
            "Asisten ini hanya melayani data operasional FastPrint ERP (Akuntansi, Inventaris, Penjualan, Marketplace read-only, dan SOP Internal)."
        )
    return (
        "Saya dapat membantu memeriksa data Akuntansi, Inventaris/Stok Gudang, Penjualan, "
        "Marketplace, serta SOP dan spesifikasi produk FastPrint ERP."
    )


def _format_rag_question_section(
    q_text: str,
    rag_entry: Dict[str, Any],
) -> str:
    """Format a single RAG sub-question result section."""
    chunks = rag_entry.get("chunks") or []
    sources = rag_entry.get("sources") or []
    err = rag_entry.get("error")
    if err and not chunks:
        return (
            f"### Kendala Pencarian Pengetahuan\n\n"
            f"> [!WARNING]\n"
            f"> Gagal mengambil dokumen SOP/spesifikasi: {err}"
        )
    if chunks:
        return _format_rag_section(chunks, sources)
    return (
        "### Informasi Pengetahuan Tidak Ditemukan\n\n"
        f"Dokumen SOP, kebijakan, atau spesifikasi terkait pertanyaan *\"{q_text}\"* belum terdaftar dalam basis data pengetahuan FastPrint ERP."
    )


def _render_single_plan_section(
    plan: Dict[str, Any],
    key: str,
    multi_results: Dict[str, Any],
    multi_status: Dict[str, Any],
    user_query: str,
    state: AgentState,
    use_empty_notice_fallback: bool = False,
) -> Optional[str]:
    """Render one tool execution plan (single or multi-company) into a Markdown section."""
    t_name = plan.get("tool")
    if not t_name:
        return None

    plan_mc_results = plan.get("multi_company_results")
    if isinstance(plan_mc_results, dict) and plan_mc_results:
        plan_unauth = plan.get("unauthorized_companies") or []
        if use_empty_notice_fallback and len(plan_unauth) == len(plan_mc_results) and len(plan_unauth) > 0:
            return (
                f"### Kendala Akses Multi-Cabang (`{t_name}`)\n\n"
                f"> [!WARNING]\n"
                f"> Akses ditolak ke seluruh cabang yang diminta (HTTP 403): {plan_unauth}"
            )
        plan_state: Dict[str, Any] = {
            **state,
            "detected_tool": t_name,
            "detected_domain": plan.get("domain") or state.get("detected_domain"),
            "multi_company_results": plan_mc_results,
            "unauthorized_companies": plan_unauth,
            "extracted_params": plan.get("params") or {},
            "user_query": plan.get("sub_query") or user_query,
        }
        return _format_multi_company_presentation(plan_state)  # type: ignore[arg-type]

    res = plan.get("result")
    if res is None and key in multi_results:
        res = multi_results[key]
    elif res is None and t_name in multi_results:
        res = multi_results[t_name]

    plan_err = plan.get("error") or (res.get("error") if isinstance(res, dict) else None)
    plan_ver_st = multi_status.get(key)

    if plan.get("status") == "error" or plan_err:
        if use_empty_notice_fallback:
            domain_label = "ERP"
        elif "stock" in t_name or "product" in t_name or "location" in t_name:
            domain_label = "Inventaris / Stok"
        elif "balance" in t_name or "invoice" in t_name or "journal" in t_name:
            domain_label = "Akuntansi / Keuangan"
        elif "sales" in t_name or "selling" in t_name:
            domain_label = "Penjualan (Sales)"
        else:
            domain_label = "Modul ERP"
        return (
            f"### Kendala Modul {domain_label} (`{t_name}`)\n\n"
            f"> [!WARNING]\n"
            f"> Gagal memuat data modul ini: {plan_err or 'Terjadi kendala pada layanan ERP.'}"
        )

    if plan_ver_st == "EMPTY":
        return _format_empty_plan_notice(plan, default_company_id=state.get("company_id"))

    if res:
        plan_default_cid = (
            plan.get("company_id")
            or (plan.get("params") or {}).get("company_id")
            or state.get("company_id")
        )
        return _format_single_tool_presentation(
            t_name,
            res,
            plan.get("sub_query") or user_query,
            plan.get("params"),
            default_company_id=plan_default_cid,
        )

    if use_empty_notice_fallback:
        return _format_empty_plan_notice(plan, default_company_id=state.get("company_id"))

    domain_label = "Modul ERP"
    if "stock" in t_name or "product" in t_name or "location" in t_name:
        domain_label = "Inventaris / Stok"
    elif "balance" in t_name or "invoice" in t_name or "journal" in t_name:
        domain_label = "Akuntansi / Keuangan"
    elif "sales" in t_name or "selling" in t_name:
        domain_label = "Penjualan (Sales)"
    return (
        f"### Informasi {domain_label} (`{t_name}`)\n\n"
        f"Tidak ditemukan data atau terjadi kendala saat memuat informasi untuk modul ini."
    )


def _build_question_ordered_answer(
    state: AgentState,
    llm_general_answers: Optional[Dict[int, str]] = None,
) -> Optional[str]:
    """
    Assemble multi-question responses in original user question order across
    ERP plans, RAG sub-queries, and general/guardrail sub-questions.
    Returns None when standard single-tool or legacy multi-tool layout applies.
    """
    sub_questions = state.get("sub_questions") or []
    non_tool_questions = state.get("non_tool_questions") or []
    rag_queries = state.get("rag_queries") or []
    rag_results_by_query = state.get("rag_results_by_query") or {}
    active_tools = state.get("active_tools") or []
    multi_results = state.get("multi_tool_results") or {}
    multi_status = state.get("multi_verification_status") or {}
    user_query = state.get("user_query") or ""

    is_multi_q = (
        len(sub_questions) > 1
        or bool(non_tool_questions)
        or len(rag_queries) > 1
        or (bool(rag_queries) and len(active_tools) >= 1)
    )
    if not is_multi_q:
        return None

    tool_counts: Dict[str, int] = {}
    for p in active_tools:
        if isinstance(p, dict):
            tn = p.get("tool", "unnamed_tool")
            tool_counts[tn] = tool_counts.get(tn, 0) + 1

    # Map question_index -> rendered sections
    buckets: Dict[int, List[str]] = {}
    q_labels: Dict[int, str] = {}
    seen_plan_keys = set()
    seen_rendered_blocks = set()

    for i, sq in enumerate(sub_questions):
        if isinstance(sq, dict) and "question_index" in sq:
            idx = int(sq["question_index"])
            if sq.get("text"):
                q_labels[idx] = sq["text"]
        elif isinstance(sq, str) and sq.strip():
            q_labels.setdefault(i, sq.strip())

    for idx, plan in enumerate(active_tools):
        if not isinstance(plan, dict):
            continue
        t_name = plan.get("tool")
        if not t_name:
            continue
        p_key = _plan_dedup_key(plan)
        if p_key in seen_plan_keys:
            continue
        seen_plan_keys.add(p_key)

        q_idx = int(plan.get("question_index") if plan.get("question_index") is not None else idx)
        if plan.get("question_text") and q_idx not in q_labels:
            q_labels[q_idx] = plan["question_text"]
        elif plan.get("sub_query") and q_idx not in q_labels:
            q_labels[q_idx] = plan["sub_query"]

        key = f"{t_name}_{idx}" if tool_counts.get(t_name, 0) > 1 else t_name
        rendered = _render_single_plan_section(
            plan=plan,
            key=key,
            multi_results=multi_results,
            multi_status=multi_status,
            user_query=user_query,
            state=state,
            use_empty_notice_fallback=True,
        )
        if rendered:
            norm_rendered = rendered.strip()
            if norm_rendered in seen_rendered_blocks:
                continue
            seen_rendered_blocks.add(norm_rendered)
            buckets.setdefault(q_idx, []).append(rendered)

    # Map RAG queries by matching text in sub_questions when available
    rag_q_to_idx: Dict[str, int] = {}
    for i, sq in enumerate(sub_questions):
        if isinstance(sq, dict) and sq.get("query_mode") == "vector_rag" and sq.get("text"):
            rag_q_to_idx[sq["text"]] = int(sq.get("question_index") or 0)
        elif isinstance(sq, str) and sq.strip():
            rag_q_to_idx.setdefault(sq.strip(), i)

    next_fallback_idx = (max(q_labels.keys()) + 1) if q_labels else 0
    for r_idx, rq in enumerate(rag_queries):
        q_idx = rag_q_to_idx.get(rq)
        if q_idx is None:
            q_idx = next_fallback_idx + r_idx
        if rq not in q_labels.values() and q_idx not in q_labels:
            q_labels[q_idx] = rq
        entry = rag_results_by_query.get(rq)
        if entry is None:
            entry = {
                "chunks": state.get("rag_context") or [],
                "sources": state.get("rag_sources") or [],
                "error": state.get("error_message") if state.get("verification_status") == "ERROR" else None,
            }
        buckets.setdefault(q_idx, []).append(_format_rag_question_section(rq, entry))

    for nt in non_tool_questions:
        if not isinstance(nt, dict):
            continue
        q_idx = int(nt.get("question_index") or 0)
        if nt.get("question_text") and q_idx not in q_labels:
            q_labels[q_idx] = nt["question_text"]
        buckets.setdefault(q_idx, []).append(
            _format_non_tool_item_notice(nt, llm_general_answers=llm_general_answers)
        )

    if not buckets:
        return None

    ordered_indices = sorted(buckets.keys())
    eval_entries: List[Tuple[str, Any, Optional[Dict[str, Any]]]] = []
    for idx_p, plan in enumerate(active_tools):
        if isinstance(plan, dict) and plan.get("tool"):
            tn = plan["tool"]
            ekey = f"{tn}_{idx_p}" if tool_counts.get(tn, 0) > 1 else tn
            eres = plan.get("result") or multi_results.get(ekey) or multi_results.get(tn)
            if isinstance(eres, dict):
                eval_entries.append((tn, eres, plan.get("params")))
    multi_verdict = _build_evaluative_verdict(user_query, eval_entries)

    if len(ordered_indices) == 1:
        body_one = "\n\n---\n\n".join(buckets[ordered_indices[0]])
        return f"{multi_verdict}\n\n---\n\n{body_one}" if multi_verdict else body_one

    final_sections: List[str] = []
    for display_num, q_idx in enumerate(ordered_indices, start=1):
        label = (q_labels.get(q_idx) or "").strip()
        body = "\n\n".join(buckets[q_idx])
        if label:
            final_sections.append(f"#### Pertanyaan {display_num}: {label}\n\n{body}")
        else:
            final_sections.append(body)
    joined = "\n\n---\n\n".join(final_sections)
    return f"{multi_verdict}\n\n---\n\n{joined}" if multi_verdict else joined


def format_deterministic_answer(state: AgentState) -> str:
    """
    Deterministic template formatter for ERP results when offline or LLM is unavailable.
    Guarantees consistent, professional markdown output.
    """
    status = state.get("verification_status")
    domain = state.get("detected_domain")
    tool = state.get("detected_tool")
    tool_result = state.get("tool_result") or {}
    user_query = state.get("user_query") or ""
    error_message = state.get("error_message")
    multi_results = state.get("multi_tool_results") or {}
    active_tools = state.get("active_tools") or []
    sub_questions = state.get("sub_questions") or []
    non_tool_questions = state.get("non_tool_questions") or []
    rag_queries = state.get("rag_queries") or []
    is_single_tool_mc = (
        not state.get("is_multi_tool")
        and len(active_tools) <= 1
        and len(sub_questions) <= 1
        and not non_tool_questions
        and len(rag_queries) <= 1
        and bool(state.get("is_multi_company"))
        and bool(state.get("multi_company_results"))
    )
    is_multi = (
        state.get("is_multi_tool")
        or (len(active_tools) > 1)
        or (len(multi_results) > 1 and not is_single_tool_mc)
        or (len(sub_questions) > 1)
        or bool(non_tool_questions)
        or (len(rag_queries) > 1)
    )

    # 0. Multi-Company Execution Unified Formatting (Single-Tool Only)
    if is_single_tool_mc:
        return _format_multi_company_presentation(state)

    # 0. Empty History Retry Handling
    if error_message == "NO_PREVIOUS_QUERY_TO_RETRY":
        return (
            "### Tidak Ada Permintaan Sebelumnya\n\n"
            "Belum ada riwayat transaksi atau pertanyaan sebelumnya pada sesi percakapan ini untuk diulangi.\n\n"
            "Silakan sampaikan pertanyaan spesifik Anda seputar data FastPrint ERP:\n"
            "- **Tagihan Vendor / Pemasok** (Hutang usaha & status verifikasi faktur)\n"
            "- **Saldo Akun Buku Kas & Neraca** (Accounting Balance & Kas Kecil)\n"
            "- **Entri Jurnal Umum** (General Journal Entries)\n"
            "- **Stok Barang & Master Produk** (Inventory & Kuantitas Gudang)\n"
            "- **Penjualan & Marketplace** (Sales Order, Kesehatan Akun, Sinkronisasi Stok/Harga, Potongan Fee, Saldo Dompet)\n"
            "- **Pengetahuan & SOP Perusahaan** (SOP Retur, Kebijakan, dan Spesifikasi Teknis)"
        )

    # 1. Error Status Handling
    if status == "ERROR":
        msg = error_message or "Terjadi kesalahan internal pada layanan ERP."
        if "MISSING_PRICELIST_OR_ACCOUNT" in msg or "MISSING_PRICELIST_ID" in msg:
            return (
                "### Akun Marketplace atau Pricelist Belum Teridentifikasi\n\n"
                f"{msg}\n\n"
                "**Saran Tindakan**:\n"
                "- Sebutkan nama toko/platform (contoh: *\"Bandingkan harga ERP vs Shopee\"*) atau sertakan `account_id` / `pricelist_id`.\n"
                "- Periksa apakah akun marketplace terkait sudah memiliki konfigurasi `AccountOrder` dan `Pricelist` di ERP."
            )

        if "ACCOUNT_NOT_FOUND" in msg or "PRICELIST_NOT_FOUND" in msg:
            return (
                "### Akun Marketplace atau Pricelist Tidak Ditemukan\n\n"
                f"{msg}\n\n"
                "**Saran Tindakan**:\n"
                "- Jalankan pengecekan kesehatan akun marketplace untuk melihat daftar `account_id` aktif pada cabang Anda."
            )

        if "PRODUCT_NOT_FOUND" in msg or "Product variant not found" in msg:
            params = state.get("extracted_params") or {}
            identifier = params.get("code") or params.get("product_id") or "yang diminta"
            return (
                f"### Produk Tidak Ditemukan\n\n"
                f"Produk **{identifier}** tidak ditemukan pada master data ERP FastPrint.\n\n"
                f"**Saran Tindakan**:\n"
                f"- Pastikan kode SKU, barcode, atau nama produk yang dimasukkan sudah sesuai.\n"
                f"- Periksa apakah produk tersebut aktif dan tidak dalam status diarsipkan (archive) di ERP."
            )

        if "MISSING_PRODUCT_IDENTIFIER" in msg:
            return (
                "### Informasi Produk atau Gudang Belum Spesifik\n\n"
                "Untuk memeriksa ketersediaan stok barang pada FastPrint ERP, mohon sertakan salah satu informasi berikut:\n"
                "- **Kode SKU / Barcode / Nama Produk** (contoh: `HVS-A4-70`, `Tinta Epson 664`, `Kertas Art Paper`)\n"
                "- **Nama Gudang / Lokasi Rak** jika ingin memindai seluruh stok gudang (contoh: `Gudang Surabaya`, `Gudang Cabang Bandung`)\n\n"
                "**Contoh pertanyaan yang dapat diajukan**:\n"
                "- *\"Berapa stok Kertas HVS A4 saat ini?\"*\n"
                "- *\"Cek stok SKU HVS-A4-70 di Gudang Surabaya\"*\n"
                "- *\"Tampilkan stok barang di Gudang Cabang Jakarta\"*"
            )

        return (
            f"### Kendala Akses Data ERP\n\n"
            f"Permintaan Anda tidak dapat diproses karena terjadi kendala teknis:\n"
            f"> **Pesan Kesalahan**: {msg}\n\n"
            f"**Saran Tindakan**:\n"
            f"- Pastikan token autentikasi (JWT) masih aktif atau lakukan login ulang.\n"
            f"- Pastikan pengguna Anda memiliki hak akses ke modul dan perusahaan (Company ID) yang diminta."
        )

    # 2.0 Multi-Question Unified Rendering (Ordered by question_index)
    if is_multi:
        ordered_answer = _build_question_ordered_answer(state)
        if ordered_answer:
            return ordered_answer

    # 2. Out of Scope Guardrail
    if domain == "out_of_scope":
        from app.graph.nodes.router_patterns import MARKETPLACE_WRITE_PATTERNS, _match_any
        if _match_any(MARKETPLACE_WRITE_PATTERNS, user_query.lower()):
            return (
                "### Operasi Tulis Marketplace Dibatasi (Read-Only)\n\n"
                "Asisten AI hanya mendukung penelusuran data dan simulasi read-only untuk modul Marketplace. "
                "Perubahan stok/harga langsung ke Shopee, TikTok Shop, Tokopedia, atau penggabungan Pesanan Kilat "
                "wajib dilakukan melalui antarmuka resmi ERP:\n"
                "- Kelola Sinkronisasi Stok & Harga: `/marketplace/`\n"
                "- Penanganan Pesanan Kilat: `/marketplace/tools/handling-order-kilat/`\n\n"
                "Anda tetap dapat meminta AI menjalankan **simulasi stok marketplace**, **perbandingan harga**, atau **diagnosa Pesanan Kilat** secara read-only."
            )
        return (
            "### Permintaan di Luar Cakupan (Out of Scope)\n\n"
            "Maaf, permintaan ini berada di luar cakupan dan kemampuan Asisten AI FastPrint ERP.\n\n"
            "Asisten ini hanya dapat membantu penelusuran data operasional dan pembukuan internal FastPrint ERP:\n"
            "- **Akuntansi & Keuangan** (Laba Rugi, Neraca, Buku Besar, Rekonsiliasi Bank, Saldo Kas/Bank, Jurnal Umum, Tagihan Vendor)\n"
            "- **Inventaris & Gudang** (Stok Fisik, Kartu Stok, Mutasi Transfer, Alert Stok Menipis, Stock Opname, Scrap, Master Produk & Gudang)\n"
            "- **Penjualan (Sales)** (Sales Order, Quotation, Omzet/Summary, Produk Terlaris, Kontak Pelanggan)\n"
            "- **Marketplace (Read-Only)** (Kesehatan Akun, SLA Order, Pesanan Kilat, Mapping SKU, Simulasi & Log Stok/Harga, Perbandingan Harga, Anomali & Rekap Fee, Cashflow, Kompetitor)\n"
            "- **Pengetahuan & SOP Perusahaan** (SOP Retur, Kebijakan, dan Spesifikasi Teknis Produk)\n\n"
            "Silakan ajukan pertanyaan yang berkaitan dengan modul operasional di atas."
        )

    query_mode = state.get("query_mode")
    rag_context = state.get("rag_context")
    rag_sources = state.get("rag_sources")

    # 2.1 Pure RAG Knowledge Handling
    if domain == "knowledge" or query_mode == "vector_rag":
        if status == "EMPTY" or not rag_context:
            return (
                "### Informasi Pengetahuan Tidak Ditemukan\n\n"
                "Dokumen SOP, kebijakan, atau spesifikasi terkait pertanyaan Anda belum terdaftar "
                "dalam basis data pengetahuan FastPrint ERP.\n\n"
                "**Saran Tindakan**:\n"
                "- Pastikan kata kunci menggunakan istilah baku operasional (misal: 'SOP Retur', 'Spesifikasi Art Paper').\n"
                "- Hubungi administrator sistem untuk mendaftarkan dokumen SOP ini ke basis data pengetahuan."
            )
        return _format_rag_section(rag_context, rag_sources)

    # 2.5 Contextual Analytical Follow-up (prior report data without new tool execution)
    if query_mode == "contextual_analysis" and not is_multi:
        return _build_deterministic_analytical_fallback(state)

    # 3. General Domain / Guardrail Inquiries / Session Meta-Questions
    if (domain == "general" or not tool) and not is_multi:
        meta_answer = _format_meta_question_answer(state)
        if meta_answer:
            return meta_answer
        query_lower = user_query.lower()
        billing_terms = (
            "pelanggan", "customer", "piutang", "tagihan",
            "faktur", "invoice", "bill", "kuitansi", "bon"
        )
        if any(term in query_lower for term in billing_terms):
            return (
                "### Informasi Tagihan Pelanggan (Customer) & Panduan Faktur\n\n"
                "Sistem AI Orchestrator saat ini dikhususkan untuk **tagihan vendor / pemasok** (Hutang Usaha), "
                "buku jurnal, saldo akun kas, dan inventaris barang.\n\n"
                "**Panduan & Solusi**:\n"
                "- **Tagihan / Piutang Pelanggan**: Untuk memeriksa transaksi atau tagihan pelanggan (seperti penjualan toko/mainstore), "
                "silakan akses modul **Penjualan (Sales / Invoicing)** atau POS secara langsung di aplikasi FastPrint ERP.\n"
                "- **Jika Rekanan adalah Vendor**: Jika pihak yang Anda maksud adalah pemasok/vendor barang, sertakan kata `'vendor'` "
                "(contoh: `coba cek tagihan vendor [Nama Vendor]`)."
            )
        return (
            "Halo! Saya adalah Asisten AI FastPrint ERP. Saya dapat membantu Anda memeriksa:\n"
            "- **Tagihan Vendor / Pemasok** (Hutang usaha dan status pembayaran)\n"
            "- **Saldo Akun Buku Kas & Neraca** (Accounting Balance)\n"
            "- **Entri Jurnal Umum** (General Journal Entries)\n"
            "- **Stok Barang & Master Produk** (Inventory & Warehouse Stock)\n"
            "- **Penjualan & Marketplace** (Sales Order, Kesehatan Akun, Sinkronisasi Stok/Harga, Anomali Fee, Cashflow & Kompetitor)\n\n"
            "Silakan sampaikan pertanyaan spesifik Anda!"
        )

    # 3. Empty Status (No Data Found after Max Retries)
    if status == "EMPTY":
        all_params: List[Dict[str, Any]] = []
        if state.get("extracted_params"):
            all_params.append(state["extracted_params"])
        for plan in state.get("active_tools") or []:
            if isinstance(plan, dict) and plan.get("params"):
                all_params.append(plan["params"])

        filter_hints: List[str] = []
        seen_hints = set()

        def _add_hint(hint: str) -> None:
            if hint not in seen_hints:
                seen_hints.add(hint)
                filter_hints.append(hint)

        for p in all_params:
            if not isinstance(p, dict):
                continue
            if p.get("vendor_name"):
                _add_hint(f"Vendor: **{p['vendor_name']}**")
            if p.get("customer_name"):
                _add_hint(f"Pelanggan: **{p['customer_name']}**")
            if p.get("channel_name"):
                _add_hint(f"Channel: **{p['channel_name']}**")
            if p.get("product_search"):
                _add_hint(f"Produk: **{p['product_search']}**")
            if p.get("search"):
                _add_hint(f"Kata Kunci: **{p['search']}**")
            if p.get("order_ref"):
                _add_hint(f"No. SO/Quotation: **{p['order_ref']}**")
            if p.get("date_from") or p.get("date_to"):
                _add_hint(f"Periode: **{p.get('date_from', '-')} s/d {p.get('date_to', '-')}**")
            if p.get("account_code"):
                _add_hint(f"Kode Akun: **{p['account_code']}**")
            if p.get("code"):
                _add_hint(f"Produk / SKU: **{p['code']}**")
            elif p.get("product_id"):
                _add_hint(f"Product ID: **{p['product_id']}**")
            if p.get("location_name"):
                _add_hint(f"Lokasi: **{p['location_name']}**")

        if not state.get("is_multi_company"):
            eff_cid = state.get("company_id")
            if eff_cid is None:
                for p in all_params:
                    if isinstance(p, dict) and p.get("company_id") is not None:
                        eff_cid = p.get("company_id")
                        break
            if eff_cid is not None:
                try:
                    c_label = COMPANY_NAME_MAP.get(int(eff_cid))
                    if c_label:
                        _add_hint(f"Cabang: **{c_label}**")
                except (TypeError, ValueError):
                    pass

        hint_str = f" ({', '.join(filter_hints)})" if filter_hints else ""
        if domain in ("sales", "inventory", "marketplace") or tool in (
            "get_contacts", "get_sales_orders", "get_sales_order_detail", "get_top_selling_products", "get_sales_summary"
        ):
            advice = (
                "- Silakan periksa kembali nomor SO/quotation, nama kontak, kata kunci produk, "
                "atau pastikan transaksi telah tercatat di cabang terkait pada periode tersebut."
            )
        else:
            advice = (
                "- Silakan periksa kembali ketepatan nomor dokumen, nama vendor, "
                "atau pastikan periode transaksi telah tercatat di ERP."
            )
        return (
            f"### Data Tidak Ditemukan\n\n"
            f"Setelah dilakukan penelusuran pada database ERP FastPrint{hint_str}, "
            f"tidak ditemukan data transaksi yang sesuai.\n\n"
            f"**Catatan Verifikasi**:\n"
            f"- Sistem telah mencoba relaksasi batasan filter atau pengecekan cabang secara otomatis (maksimal 2x).\n"
            f"{advice}"
        )

    multi_results = state.get("multi_tool_results") or {}
    active_tools = state.get("active_tools") or []
    multi_status = state.get("multi_verification_status") or {}
    is_multi = state.get("is_multi_tool") or (len(multi_results) > 1) or (len(active_tools) > 1)

    # 4. Multi-Tool Execution Unified Formatting (Milestone 4)
    if is_multi and (multi_results or active_tools):
        sections: List[str] = []

        if active_tools:
            tool_counts: Dict[str, int] = {}
            for p in active_tools:
                tn = p.get("tool", "unnamed_tool")
                tool_counts[tn] = tool_counts.get(tn, 0) + 1

            seen_mt_keys = set()
            seen_mt_sections = set()
            for idx, plan in enumerate(active_tools):
                t_name = plan.get("tool")
                if not t_name:
                    continue
                p_dedup = _plan_dedup_key(plan)
                if p_dedup in seen_mt_keys:
                    continue
                seen_mt_keys.add(p_dedup)
                key = f"{t_name}_{idx}" if tool_counts.get(t_name, 0) > 1 else t_name
                rendered = _render_single_plan_section(
                    plan=plan,
                    key=key,
                    multi_results=multi_results,
                    multi_status=multi_status,
                    user_query=user_query,
                    state=state,
                    use_empty_notice_fallback=False,
                )
                if rendered:
                    norm_sec = rendered.strip()
                    if norm_sec not in seen_mt_sections:
                        seen_mt_sections.add(norm_sec)
                        sections.append(rendered)
        else:
            for key, res in multi_results.items():
                base_tool = key.rsplit("_", 1)[0] if ("_" in key and key.rsplit("_", 1)[1].isdigit()) else key
                if res:
                    formatted = _format_single_tool_presentation(
                        base_tool,
                        res,
                        user_query,
                        state.get("extracted_params"),
                        default_company_id=state.get("company_id"),
                    )
                    if formatted:
                        sections.append(formatted)
                else:
                    domain_label = "Modul ERP"
                    if "stock" in base_tool or "product" in base_tool or "location" in base_tool:
                        domain_label = "Inventaris / Stok"
                    elif "balance" in base_tool or "invoice" in base_tool or "journal" in base_tool:
                        domain_label = "Akuntansi / Keuangan"
                    sections.append(
                        f"### Informasi {domain_label} (`{base_tool}`)\n\n"
                        f"Tidak ditemukan data atau terjadi kendala saat memuat informasi untuk modul ini."
                    )

        if sections:
            eval_entries_legacy: List[Tuple[str, Any, Optional[Dict[str, Any]]]] = []
            for idx_p, plan in enumerate(active_tools):
                if isinstance(plan, dict) and plan.get("tool"):
                    tn = plan["tool"]
                    ekey = f"{tn}_{idx_p}" if len(active_tools) > 1 else tn
                    eres = plan.get("result") or multi_results.get(ekey) or multi_results.get(tn)
                    if isinstance(eres, dict):
                        eval_entries_legacy.append((tn, eres, plan.get("params")))
            legacy_verdict = _build_evaluative_verdict(user_query, eval_entries_legacy)
            joined_legacy = "\n\n---\n\n".join(sections)
            return f"{legacy_verdict}\n\n---\n\n{joined_legacy}" if legacy_verdict else joined_legacy

    # 5. Single-Tool Valid Data Presentation (Backward Compatibility)
    return _format_single_tool_presentation(
        tool,
        tool_result,
        user_query,
        state.get("extracted_params"),
        default_company_id=state.get("company_id"),
    )


def _clean_label_emojis(label: str) -> str:
    """Strip emojis and decorative symbols from follow-up chip label."""
    if not label:
        return ""
    cleaned = re.sub(r"[\U00010000-\U0010ffff\u2600-\u27bf\u2300-\u23ff\u2b50-\u2b55\u200d\ufe0f]", "", label)
    cleaned = re.sub(r"^[\s\-_:•>→↪]+", "", cleaned)
    return cleaned.strip()[:35]


def _clean_ai_slop(text: str) -> str:
    """
    Sanitize LLM output by removing conversational AI slop, throat-clearing openers,
    filler sign-offs, and empty corporate buzzwords.
    Preserves tables, numeric calculations, markdown headers, and codeblocks intact.
    """
    if not text or not isinstance(text, str):
        return ""

    cleaned = text.strip()

    # 1. Strip conversational throat-clearing openers from the beginning of text
    opener_re = re.compile(
        r"^(?:(?:Tentu(?: saja)?|Baik(?:lah)?|Halo)\s*[,!.:-]?\s*)+",
        re.IGNORECASE,
    )
    cleaned = opener_re.sub("", cleaned).strip()

    # Strip pretentious assistant intros: "Sebagai Asisten AI FastPrint ERP, ..."
    asisten_re = re.compile(
        r"^(?:Sebagai\s+Asisten\s+AI\s+FastPrint\s+ERP[^\n,.:]*[,.:]\s*)",
        re.IGNORECASE,
    )
    cleaned = asisten_re.sub("", cleaned).strip()

    # Strip redundant preamble: "Berdasarkan data yang (?:ada|tercatat|ditemukan)..."
    preamble_re = re.compile(
        r"^(?:Berdasarkan\s+data\s+(?:yang\s+)?(?:ada|tercatat|ditemukan)[^\n,.:]*[,.:]\s*)",
        re.IGNORECASE,
    )
    cleaned = preamble_re.sub("", cleaned).strip()

    # 2. Strip conversational closing sign-offs at the end of the text
    closing_patterns = [
        re.compile(r"(?:\n\s*)*\n\s*(?:Semoga\s+(?:informasi\s+ini\s+)?(?:bermanfaat|membantu)[^\n]*)[.!]?\s*$", re.IGNORECASE),
        re.compile(r"(?:\n\s*)*\n\s*(?:Jangan\s+ragu\s+untuk\s+bertanya[^\n]*)[.!]?\s*$", re.IGNORECASE),
        re.compile(r"(?:\n\s*)*\n\s*(?:Jika\s+(?:ada|Anda\s+memiliki)\s+pertanyaan[^\n]*)[.!]?\s*$", re.IGNORECASE),
        re.compile(r"(?:\n\s*)*\n\s*(?:Silakan\s+tanyakan\s+(?:kembali|lagi)[^\n]*)[.!]?\s*$", re.IGNORECASE),
        re.compile(r"(?:\n\s*)*\n\s*(?:Ada\s+(?:hal\s+lain|yang)\s+(?:bisa|dapat)\s+saya\s+bantu[^\n]*)[.?]?\s*$", re.IGNORECASE),
    ]
    for cp in closing_patterns:
        cleaned = cp.sub("", cleaned).strip()

    # 3. Strip empty metadiscourse phrases mid-text
    cleaned = re.sub(
        r"\b(?:Penting\s+untuk\s+dicatat\s+bahwa|Perlu\s+diperhatikan\s+bahwa|Dapat\s+disimpulkan\s+bahwa)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    # 4. Strip unnecessary decorative emojis and emoticons
    emoji_pattern = re.compile(
        r"[\U00010000-\U0010ffff\u2600-\u27bf\u2300-\u23ff\u2b50-\u2b55\u200d\ufe0f]"
    )
    cleaned = emoji_pattern.sub("", cleaned)

    emoticon_pattern = re.compile(
        r"(?:\s|^)(?::[-~]?[)D(pP|/\\oO]|;[-~]?[)D]|<3|\^\^)(?=\s|$|[.,!?])"
    )
    cleaned = emoticon_pattern.sub("", cleaned)

    # Fix extra spaces around markdown headers (#  Heading -> # Heading)
    cleaned = re.sub(r"^(#{1,6})[ \t]{2,}", r"\1 ", cleaned, flags=re.MULTILINE)
    # Fix extra spaces after bullet markers (-  Item -> - Item)
    cleaned = re.sub(r"^(\s*[-*+])[ \t]{2,}", r"\1 ", cleaned, flags=re.MULTILINE)
    # Trim trailing whitespace on each line
    lines = [line.rstrip() for line in cleaned.splitlines()]
    cleaned = "\n".join(lines).strip()

    # 5. Capitalize first character if lowercase after trimming
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]

    return cleaned.strip()


def _clean_trailing_followup_headers(text: str) -> str:
    """Strip preceding headers like 'Rekomendasi lanjutan (READ-ONLY)' from markdown answer."""
    header_pat = re.compile(
        r"(?:\n\s*)*\n\s*(?:#{1,4}\s*)?(?:\*\*|__)?\s*Rekomendasi\s+(?:Lanjutan|Tindak\s+Lanjut)[^\n]*?(?:\*\*|__)?[:\s]*\s*$",
        re.IGNORECASE,
    )
    return header_pat.sub("", text).strip()


def _extract_llm_followups(llm_output: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Extract structured follow-up recommendations from LLM output block.
    Supports ```json:followups ... ```, ```json ... ```, [FOLLOWUPS] ... [/FOLLOWUPS],
    or raw trailing JSON objects containing 'followups'.
    Normalizes icons to clean monochrome 'fa-arrow-right' and strips emoji residue.

    Returns:
        Tuple of (clean_markdown_answer, list_of_followup_dicts)
    """
    if not llm_output or not isinstance(llm_output, str):
        return ("", [])

    raw = llm_output.strip()
    followups: List[Dict[str, str]] = []
    clean_text = raw

    # Candidate extraction patterns:
    # 1. Codeblocks: ```(?:json:followups|json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```
    # 2. Tagged: \[(?:FOLLOWUPS|SUGGESTED_FOLLOWUPS)\]\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*\[/(?:FOLLOWUPS|SUGGESTED_FOLLOWUPS)\]
    # 3. Trailing raw JSON object or array starting with { "followups" ... } or [ { "label" ... } ]
    patterns = [
        re.compile(r"```(?:json:followups|json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", re.IGNORECASE),
        re.compile(r"\[(?:FOLLOWUPS|SUGGESTED_FOLLOWUPS)\]\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*\[/(?:FOLLOWUPS|SUGGESTED_FOLLOWUPS)\]", re.IGNORECASE),
        re.compile(r"(\{\s*\"(?:followups|suggested_followups|recommendations)\"\s*:\s*\[[\s\S]*?\]\s*\})", re.IGNORECASE),
    ]

    for pat in patterns:
        match = pat.search(raw)
        if match:
            json_str = match.group(1).strip()
            try:
                parsed = json.loads(json_str)
                items_list = []
                if isinstance(parsed, list):
                    items_list = parsed
                elif isinstance(parsed, dict):
                    for k in ("followups", "suggested_followups", "recommendations", "items"):
                        if isinstance(parsed.get(k), list):
                            items_list = parsed[k]
                            break

                for item in items_list:
                    if isinstance(item, dict) and item.get("label") and item.get("query"):
                        raw_label = str(item["label"]).strip()
                        clean_label = _clean_label_emojis(raw_label) or raw_label[:35]
                        followups.append({
                            "label": clean_label,
                            "query": str(item["query"]).strip(),
                            "icon": "fa-arrow-right",
                        })

                if followups:
                    prefix = raw[:match.start()]
                    suffix = raw[match.end():]
                    clean_prefix = _clean_trailing_followup_headers(prefix)
                    clean_text = f"{clean_prefix}\n\n{suffix}".strip() if suffix.strip() else clean_prefix.strip()
                    break
            except Exception as exc:
                logger.debug("Failed parsing LLM followups pattern: %s", exc)

    return (clean_text, followups[:3])


def generate_deterministic_followups(state: AgentState) -> List[Dict[str, str]]:
    """
    Generate context-aware follow-up prompt suggestions deterministically from state entities.
    Acts as a robust fallback when LLM is offline or during testing.
    """
    suggestions: List[Dict[str, str]] = []
    status = state.get("verification_status")
    domain = state.get("detected_domain") or state.get("prior_detected_domain") or ""
    tool = state.get("detected_tool") or state.get("prior_detected_tool") or ""
    params = state.get("extracted_params") or state.get("prior_extracted_params") or {}

    # Inspect parameters in active_tools if empty in extracted_params
    if not params:
        for plan in state.get("active_tools") or []:
            if isinstance(plan, dict) and plan.get("params"):
                params = plan["params"]
                break

    # 0. ERROR Status Guard: never interpolate unvalidated/failed entity params into follow-up chips
    if status == "ERROR":
        return [
            {
                "label": "Cek Stok Menipis",
                "query": "Tampilkan daftar produk yang stoknya menipis atau habis",
                "icon": "fa-arrow-right",
            },
            {
                "label": "Saldo Kas & Bank",
                "query": "Berapa total saldo kas dan rekening bank aktif saat ini?",
                "icon": "fa-arrow-right",
            },
            {
                "label": "Panduan Fitur AI",
                "query": "/scope",
                "icon": "fa-arrow-right",
            },
        ]

    # 1. EMPTY Status Fallback
    if status == "EMPTY":
        if params.get("date_from") or params.get("date_to"):
            suggestions.append({
                "label": "Perluas Tanggal",
                "query": "Coba cari transaksi tanpa batasan tanggal untuk periode lebih luas",
                "icon": "fa-calendar-days",
            })
        if params.get("warehouse_name") or params.get("location_name") or params.get("company_id"):
            suggestions.append({
                "label": "Cek Semua Cabang",
                "query": "Cek ketersediaan di seluruh cabang dan gudang perusahaan",
                "icon": "fa-globe",
            })
        if params.get("code") or params.get("product_id"):
            identifier = params.get("code") or params.get("product_id")
            suggestions.append({
                "label": f"Master {str(identifier)[:15]}",
                "query": f"Tampilkan informasi master katalog untuk produk {identifier}",
                "icon": "fa-circle-info",
            })
        if not suggestions:
            suggestions.append({
                "label": "Panduan Fitur AI",
                "query": "/scope",
                "icon": "fa-compass",
            })
        for s in suggestions:
            s["icon"] = "fa-arrow-right"
        return suggestions[:3]

    # 2. Inventory Domain
    if domain == "inventory" or any(k in tool for k in ("stock", "product", "location", "warehouse", "scrap", "transfer")):
        codes = params.get("codes") or []
        code = params.get("code") or (codes[0] if codes and isinstance(codes, list) else None) or params.get("product_id")
        if not code and isinstance(state.get("tool_result"), dict):
            products = state["tool_result"].get("products")
            if isinstance(products, list) and len(products) > 0 and isinstance(products[0], dict):
                code = products[0].get("code") or products[0].get("sku")
        loc_name = params.get("warehouse_name") or params.get("location_name")

        if code:
            suggestions.append({
                "label": f"Mutasi Stok {str(code)[:15]}",
                "query": f"Cek riwayat pergerakan kartu stok untuk produk {code}",
                "icon": "fa-clock-rotate-left",
            })
            suggestions.append({
                "label": f"Detail Master {str(code)[:15]}",
                "query": f"Tampilkan detail master produk dan harga {code}",
                "icon": "fa-circle-info",
            })
        if loc_name:
            suggestions.append({
                "label": f"Stok {str(loc_name)[:15]}",
                "query": f"Tampilkan seluruh stok barang di {loc_name}",
                "icon": "fa-warehouse",
            })
        elif not code:
            suggestions.append({
                "label": "Peringatan Stok Menipis",
                "query": "Tampilkan daftar produk dengan stok menipis atau di bawah batas minimum",
                "icon": "fa-triangle-exclamation",
            })
            suggestions.append({
                "label": "Daftar Gudang Aktif",
                "query": "Tampilkan daftar master gudang dan lokasi penyimpanan",
                "icon": "fa-warehouse",
            })

    # 3. Accounting Domain
    elif domain == "accounting" or any(k in tool for k in ("balance", "bill", "invoice", "journal", "profit", "ledger", "bank")):
        vendor = params.get("vendor_name")
        acc = params.get("account_code")

        # Vendor bills / invoices
        if "bill" in tool or "invoice" in tool or vendor:
            v_label = f" {str(vendor)[:12]}" if vendor else ""
            suggestions.append({
                "label": f"Tagihan Overdue{v_label}",
                "query": f"Tampilkan tagihan vendor{f' {vendor}' if vendor else ''} yang telah lewat jatuh tempo (overdue)",
                "icon": "fa-clock",
            })
            suggestions.append({
                "label": f"Jurnal Transaksi{v_label}",
                "query": f"Cek riwayat entri jurnal transaksi{f' terkait vendor {vendor}' if vendor else ' bulan ini'}",
                "icon": "fa-book",
            })
            suggestions.append({
                "label": "Saldo Kas & Bank",
                "query": "Berapa total saldo kas dan rekening bank aktif saat ini?",
                "icon": "fa-wallet",
            })
        # Bank recon
        elif "bank" in tool or "recon" in tool:
            if tool == "reconcile_bank_statement":
                res = state.get("tool_result") or {}
                erp_accs = res.get("erp_accounts_used") or []
                primary_code = (
                    erp_accs[0].get("account_code")
                    if erp_accs and isinstance(erp_accs[0], dict)
                    else None
                )
                if not params.get("include_cross_check"):
                    suggestions.append({
                        "label": "Cek Salah Kamar",
                        "query": "Cek mutasi yang diduga salah kamar di rekening bank lain dari berkas rekening koran ini",
                        "icon": "fa-magnifying-glass",
                    })
                if primary_code:
                    suggestions.append({
                        "label": f"Buku Besar {str(primary_code)[:15]}",
                        "query": f"Tampilkan mutasi buku besar akun {primary_code} bulan ini",
                        "icon": "fa-book-open",
                    })
                suggestions.append({
                    "label": "Posisi Neraca Keuangan",
                    "query": "Cek laporan neraca keuangan dan verifikasi persamaan aset vs liabilitas + ekuitas",
                    "icon": "fa-scale-balanced",
                })
            else:
                j_name = params.get("journal_name") or "Bank Utama"
                suggestions.append({
                    "label": f"Buku Besar {str(j_name)[:15]}",
                    "query": f"Tampilkan mutasi buku besar untuk {j_name} bulan ini",
                    "icon": "fa-book-open",
                })
                suggestions.append({
                    "label": "Posisi Neraca Keuangan",
                    "query": "Cek laporan neraca keuangan dan verifikasi persamaan aset vs liabilitas + ekuitas",
                    "icon": "fa-scale-balanced",
                })
        # Profit and loss
        elif "profit" in tool:
            suggestions.append({
                "label": "Neraca Keuangan",
                "query": "Cek laporan neraca keuangan dan verifikasi persamaan aset vs liabilitas + ekuitas",
                "icon": "fa-scale-balanced",
            })
            suggestions.append({
                "label": "Saldo Kas & Bank",
                "query": "Berapa total saldo kas dan rekening bank aktif saat ini?",
                "icon": "fa-wallet",
            })
            suggestions.append({
                "label": "Tagihan Vendor Overdue",
                "query": "Tampilkan daftar tagihan vendor yang overdue dan belum lunas",
                "icon": "fa-file-invoice-dollar",
            })
        # Balance sheet
        elif "balance_sheet" in tool:
            suggestions.append({
                "label": "Laba Rugi YTD",
                "query": "Tampilkan ringkasan Laba Rugi tahun berjalan dan persentase gross margin",
                "icon": "fa-chart-pie",
            })
            suggestions.append({
                "label": "Rekonsiliasi Bank",
                "query": "Cek ringkasan rekonsiliasi bank dan selisih buku besar kas/bank",
                "icon": "fa-building-columns",
            })
            suggestions.append({
                "label": "Saldo Kas & Bank",
                "query": "Berapa total saldo kas dan rekening bank aktif saat ini?",
                "icon": "fa-wallet",
            })
        # General ledger / Account balance
        else:
            if acc:
                suggestions.append({
                    "label": f"Buku Besar {acc}",
                    "query": f"Tampilkan rincian transaksi buku besar akun {acc} periode ini",
                    "icon": "fa-book-open",
                })
            suggestions.append({
                "label": "Posisi Kas & Bank",
                "query": "Berapa total saldo kas dan rekening bank aktif saat ini?",
                "icon": "fa-wallet",
            })
            suggestions.append({
                "label": "Cek Neraca Keuangan",
                "query": "Cek laporan neraca keuangan dan verifikasi persamaan aset vs liabilitas + ekuitas",
                "icon": "fa-scale-balanced",
            })

    # 4. Sales Domain
    elif domain == "sales" or "selling" in tool or "sales" in tool:
        res = state.get("tool_result") or {}
        p_list = res.get("products") or []
        first_sku = p_list[0].get("code") if p_list and isinstance(p_list[0], dict) else None
        if first_sku:
            suggestions.append({
                "label": f"Cek Stok {str(first_sku)[:12]}",
                "query": f"Cek ketersediaan stok produk {first_sku} di semua cabang",
                "icon": "fa-boxes-stacked",
            })
            suggestions.append({
                "label": f"Mutasi {str(first_sku)[:12]}",
                "query": f"Cek riwayat kartu stok barang untuk SKU {first_sku}",
                "icon": "fa-clock-rotate-left",
            })
        suggestions.append({
            "label": "Laba Rugi Cabang",
            "query": "Tampilkan laporan laba rugi cabang untuk melihat perbandingan total pendapatan",
            "icon": "fa-chart-line",
        })

    # 4.5 Marketplace Domain
    elif domain == "marketplace" or tool in (
        "get_account_health",
        "get_marketplace_orders",
        "get_kilat_order_diagnostics",
        "get_product_mapping_status",
        "get_stock_sync_simulation",
        "get_sync_execution_logs",
        "get_price_comparison",
        "get_fee_and_escrow_anomalies",
        "get_deduction_and_settlement_summary",
        "get_cashflow_and_balance_summary",
        "get_competitor_insights",
    ):
        if tool == "get_account_health":
            suggestions.append({
                "label": "Log Sync Gagal",
                "query": "Kenapa update stok atau harga marketplace gagal 7 hari terakhir?",
                "icon": "fa-arrow-right",
            })
            suggestions.append({
                "label": "Order Overdue SLA",
                "query": "Cek order marketplace yang melewati batas kirim (overdue SLA)",
                "icon": "fa-arrow-right",
            })
        elif tool in ("get_marketplace_orders", "get_kilat_order_diagnostics"):
            suggestions.append({
                "label": "Anomali Potongan Fee",
                "query": "Cek anomali potongan marketplace di atas 20% bulan ini",
                "icon": "fa-arrow-right",
            })
            suggestions.append({
                "label": "Pesanan Kilat Stuck",
                "query": "Cek Pesanan Kilat yang nomor invoicenya belum update",
                "icon": "fa-arrow-right",
            })
        elif tool in ("get_product_mapping_status", "get_stock_sync_simulation", "get_sync_execution_logs"):
            suggestions.append({
                "label": "Simulasi Stok MP",
                "query": "Cek perbedaan kalkulasi stok ERP vs log update stok terakhir",
                "icon": "fa-arrow-right",
            })
            suggestions.append({
                "label": "SKU Belum Mapping",
                "query": "Cek produk marketplace yang belum ter-mapping (unmapped)",
                "icon": "fa-arrow-right",
            })
        elif tool in ("get_price_comparison", "get_competitor_insights"):
            suggestions.append({
                "label": "Selisih Harga Shopee",
                "query": "Bandingkan harga ERP vs harga Shopee yang selisih",
                "icon": "fa-arrow-right",
            })
            suggestions.append({
                "label": "Kompetitor Turun Harga",
                "query": "Kompetitor mana yang menurunkan harga minggu ini?",
                "icon": "fa-arrow-right",
            })
        else:
            suggestions.append({
                "label": "Rekap Potongan Bulan Ini",
                "query": "Ringkasan potongan fee marketplace bulan ini",
                "icon": "fa-arrow-right",
            })
            suggestions.append({
                "label": "Saldo Dompet & Escrow",
                "query": "Berapa saldo dompet dan pending escrow marketplace saat ini?",
                "icon": "fa-arrow-right",
            })

    # 5. General / Knowledge / Guardrail Fallback
    if not suggestions:
        suggestions.append({
            "label": "Cek Stok Barang",
            "query": "Cek stok produk di gudang",
            "icon": "fa-boxes-stacked",
        })
        suggestions.append({
            "label": "Saldo Kas & Bank",
            "query": "Berapa total saldo kas dan rekening bank aktif saat ini?",
            "icon": "fa-wallet",
        })
        suggestions.append({
            "label": "Panduan Fitur AI",
            "query": "/scope",
            "icon": "fa-compass",
        })

    for s in suggestions:
        s["icon"] = "fa-arrow-right"
    return suggestions[:3]


def _is_routable_followup_query(query_str: str) -> bool:
    """Return True if a follow-up query routes to a concrete ERP tool, RAG knowledge, or /scope."""
    q = (query_str or "").strip()
    if not q:
        return False
    if q.lower().startswith("/scope"):
        return True
    try:
        from app.graph.nodes.router import classify_query_rule_based
        res = classify_query_rule_based(q)
        if not isinstance(res, dict):
            return False
        return res.get("detected_domain") not in (None, "general", "out_of_scope")
    except Exception:
        return True


def _embed_active_branch_in_chip_query(query: str, state: AgentState) -> str:
    """
    Append the active branch scope to a chip query when the thread is operating under
    an in-chat branch override (company_scope_source == 'text'), unless the query already
    names a branch, targets all branches, or is a slash command.
    """
    q = (query or "").strip()
    if not q or q.startswith("/"):
        return q
    if state.get("company_scope_source") != "text" or state.get("is_multi_company"):
        return q
    cid = state.get("company_id")
    if cid is None:
        return q
    try:
        branch_name = COMPANY_NAME_MAP.get(int(cid))
    except (TypeError, ValueError):
        return q
    if not branch_name:
        return q
    existing_cids, is_all = resolve_all_branch_company_ids(q)
    if existing_cids or is_all:
        return q
    if q.endswith("?"):
        return f"{q[:-1].rstrip()} di {branch_name}?"
    return f"{q} di {branch_name}"


def _finalize_followups(
    followups: List[Dict[str, str]],
    user_query: Optional[str],
    state: AgentState,
) -> List[Dict[str, str]]:
    """
    Deduplicate, strip self-repeating chips matching current user_query,
    validate routability via classify_query_rule_based, embed active in-chat branch scope,
    and backfill cleanly.
    """
    norm_user = re.sub(r"\s+", " ", (user_query or "").strip().rstrip("?!.,;:'\"> ")).lower()
    seen_queries = set()
    valid_chips: List[Dict[str, str]] = []

    def _accept(chip: Dict[str, str], check_route: bool = True) -> None:
        if not isinstance(chip, dict):
            return
        q_raw = str(chip.get("query") or "").strip()
        lbl_raw = str(chip.get("label") or "").strip()
        if not q_raw or not lbl_raw:
            return
        norm_raw = re.sub(r"\s+", " ", q_raw.rstrip("?!.,;:'\"> ")).lower()
        if norm_raw == norm_user:
            return
        if check_route and not _is_routable_followup_query(q_raw):
            return
        q_scoped = _embed_active_branch_in_chip_query(q_raw, state)
        norm_q = re.sub(r"\s+", " ", q_scoped.rstrip("?!.,;:'\"> ")).lower()
        if norm_q == norm_user or norm_q in seen_queries:
            return
        seen_queries.add(norm_q)
        valid_chips.append({
            "label": _clean_label_emojis(lbl_raw) or lbl_raw[:35],
            "query": q_scoped,
            "icon": "fa-arrow-right",
        })

    for item in followups or []:
        _accept(item, check_route=True)

    if not valid_chips:
        for det_item in generate_deterministic_followups(state):
            _accept(det_item, check_route=False)
            if len(valid_chips) >= 3:
                break

    return valid_chips[:3]


_PAGINATED_TABLE_TOOLS_META: Dict[str, str] = {
    "get_low_stock_alerts": "alerts",
    "get_product_info": "products",
    "get_locations": "locations",
    "get_stock_history": "history",
    "get_transfers": "transfers",
    "get_stock_adjustments": "adjustments",
    "get_scrap_records": "scraps",
    "get_warehouses": "warehouses",
    "get_product_categories": "categories",
    "get_account_balance": "accounts",
    "get_journal_entries": "entries",
    "get_general_ledger": "accounts",
    "get_vendor_bills": "bills",
    "get_sales_orders": "orders",
    "get_contacts": "contacts",
    "get_marketplace_orders": "orders",
}

_NATIVE_MC_TABLE_TOOLS = {
    "get_sales_orders",
    "get_contacts",
    "get_marketplace_orders",
    "get_kilat_order_diagnostics",
    "get_product_mapping_status",
    "get_stock_sync_simulation",
    "get_price_comparison",
    "get_fee_and_escrow_anomalies",
    "get_competitor_insights",
}

_EXPLICIT_TABULAR_BREAKDOWN_RE = re.compile(
    r"\b(?:breakdown|jabarkan|tabel|table|per\s+akun|per\s+cabang|rincian\s+akun|detail\s+akun)\b",
    re.IGNORECASE,
)


def _should_keep_verbatim_deterministic(
    deterministic_answer: str,
    user_query: Optional[str],
    state: AgentState,
    is_multi_q: bool,
) -> bool:
    """Return True when Markdown tables must not be collapsed into short LLM prose summaries."""
    if state.get("query_mode") == "contextual_analysis":
        return False
    if re.search(
        r"#### (?:Analisis Proyeksi Target|Ringkasan Evaluasi Dampak|Rekomendasi Segera Di-Order)",
        deterministic_answer,
    ):
        return True
    if _EXPLICIT_TABULAR_BREAKDOWN_RE.search(user_query or ""):
        return True
    if is_analytical_thinking_query(user_query):
        return False
    if state.get("detected_tool") == "reconcile_bank_statement" and "\n|" in deterministic_answer:
        return True
    if state.get("is_multi_company", False) or is_multi_q or len(deterministic_answer) > 1500:
        return True
    if "\n|" not in deterministic_answer:
        return False
    ext_params = state.get("extracted_params") or {}
    return bool(isinstance(ext_params, dict) and ext_params.get("include_breakdown") and len(deterministic_answer) > 600)


def _build_table_query_meta(state: AgentState) -> Optional[Dict[str, Any]]:
    """Build lightweight query metadata for remote table pagination/export when rows are truncated."""
    if state.get("verification_status") != "VALID":
        return None
    if state.get("is_multi_tool"):
        return None
    active_tools = state.get("active_tools") or []
    if len(active_tools) > 1:
        return None

    tool = state.get("detected_tool") or (
        active_tools[0].get("tool")
        if active_tools and isinstance(active_tools[0], dict)
        else None
    )
    if not tool:
        return None

    tool_result = state.get("tool_result")
    if not isinstance(tool_result, dict):
        return None

    is_native_mc_single_call = bool(
        state.get("is_multi_company")
        and not (state.get("multi_company_results") or {})
        and (tool in _NATIVE_MC_TABLE_TOOLS or isinstance(tool_result.get("company_ids"), list))
    )
    if state.get("is_multi_company") and not is_native_mc_single_call:
        return None

    list_key = _PAGINATED_TABLE_TOOLS_META.get(tool)
    if tool == "get_stock_level" and tool_result.get("mode") in ("location_scan", "company_scan"):
        list_key = "items"
    if not list_key:
        return None

    rows = tool_result.get(list_key)
    if not isinstance(rows, list):
        return None

    displayed_count = len(rows)
    total_count = _extract_total_count(tool_result, displayed_count)
    if displayed_count <= 0 or total_count <= displayed_count:
        return None

    from app.core.context import sanitize_tool_params
    from app.schemas.tools import TOOL_DOMAIN_MAP

    raw_params = (
        (active_tools[0].get("params") if active_tools and isinstance(active_tools[0], dict) else None)
        or state.get("extracted_params")
        or {}
    )
    clean_params = sanitize_tool_params(raw_params)
    clean_params.pop("offset", None)
    clean_params.pop("limit", None)
    if is_native_mc_single_call and not clean_params.get("company_ids"):
        eff_cids = [int(c) for c in (state.get("company_ids") or tool_result.get("company_ids") or []) if c]
        if len(eff_cids) > 1:
            clean_params["company_ids"] = eff_cids
    if tool == "get_stock_level":
        if tool_result.get("location_id") is not None and not clean_params.get("location_id"):
            clean_params["location_id"] = tool_result["location_id"]
        if tool_result.get("warehouse_id") is not None and not clean_params.get("warehouse_id"):
            clean_params["warehouse_id"] = tool_result["warehouse_id"]

    domain = (
        state.get("detected_domain")
        or (active_tools[0].get("domain") if active_tools and isinstance(active_tools[0], dict) else None)
        or TOOL_DOMAIN_MAP.get(tool)
        or "inventory"
    )
    resolved_cid = (
        tool_result.get("company_id")
        or clean_params.get("company_id")
        or state.get("company_id")
    )

    return {
        "domain": domain,
        "tool": tool,
        "company_id": int(resolved_cid) if resolved_cid is not None else None,
        "params": clean_params,
        "total_count": total_count,
        "displayed_count": displayed_count,
    }


def _apply_retry_header(text: str, retry_header: str) -> str:
    """Prepend retry banner only once when not already present."""
    if retry_header and not text.startswith("> 🔄"):
        return f"{retry_header}{text}"
    return text


def _extract_resolved_model_name(response: Any) -> Optional[str]:
    """Read concrete model identifier from LLM response metadata if present."""
    resp_meta = getattr(response, "response_metadata", {}) or {}
    resolved = resp_meta.get("model_name") or resp_meta.get("model")
    return str(resolved) if resolved else None


def _prepare_synthesis_context(
    state: AgentState,
    llm: Optional[BaseChatModel],
) -> Tuple[Any, Optional[str], str, str, Optional[str], List[Dict[str, Any]], bool, Optional[BaseChatModel]]:
    """Compute shared synthesis inputs, hybrid RAG merge, multi-question flags, and optional LLM."""
    user_query = state.get("user_query")
    status = state.get("verification_status")
    query_mode = state.get("query_mode")
    retry_instruction = state.get("retry_instruction") or ""
    retry_header = (
        f"> 🔄 **{retry_instruction}**\n\n"
        if retry_instruction.startswith("Mengulang permintaan:")
        else ""
    )
    deterministic_answer = format_deterministic_answer(state)
    actual_model = state.get("active_model")

    sub_questions = state.get("sub_questions") or []
    non_tool_questions = state.get("non_tool_questions") or []
    rag_queries = state.get("rag_queries") or []
    active_tools_list = state.get("active_tools") or []
    is_multi_q = (
        len(sub_questions) > 1
        or len(active_tools_list) > 1
        or len(rag_queries) > 1
        or bool(non_tool_questions)
        or bool(state.get("is_multi_tool"))
    )

    if query_mode == "hybrid" and not is_multi_q:
        rag_section = _format_rag_section(state.get("rag_context"), state.get("rag_sources"))
        if rag_section:
            deterministic_answer = f"{deterministic_answer}\n\n---\n\n{rag_section}"
        else:
            deterministic_answer = (
                f"{deterministic_answer}\n\n"
                f"> [!NOTE]\n"
                f"> Dokumen SOP atau spesifikasi kebijakan terkait belum terdaftar di basis data pengetahuan perusahaan."
            )

    model = llm
    is_test_env = "unittest" in sys.modules or os.environ.get("FASTPRINT_TEST_ENV") == "true"
    if model is None and not is_test_env:
        try:
            model = get_chat_model()
        except Exception:
            model = None

    return (
        user_query,
        status,
        retry_header,
        deterministic_answer,
        actual_model,
        non_tool_questions,
        is_multi_q,
        model,
    )


def _parse_indonesian_rupiah(raw: str) -> Optional[float]:
    """Parse Indonesian formatted Rupiah string (e.g. 'Rp -2.667.690.147,11') to float."""
    if not raw or not isinstance(raw, str):
        return None
    cleaned = raw.replace("Rp", "").replace("**", "").replace("*", "").strip()
    m = re.search(r"-?[\d\.]+(?:,\d+)?", cleaned)
    if not m:
        return None
    num_str = m.group(0).replace(".", "").replace(",", ".")
    try:
        return float(num_str)
    except ValueError:
        return None


def _parse_pnl_from_markdown(text: str) -> Optional[Dict[str, Any]]:
    """Reconstruct P&L summary and account breakdown rows from a prior Markdown P&L report."""
    if not text or ("Laporan Laba Rugi" not in text and "Pendapatan Usaha" not in text):
        return None

    summary: Dict[str, Any] = {}
    metric_patterns = {
        "total_revenue": r"Pendapatan Usaha \(Revenue\)\**\s*\|\s*([^|\n]+)",
        "cogs": r"Beban Pokok Penjualan[^|\n]*\|\s*([^|\n]+)",
        "gross_profit": r"Laba Kotor \(Gross Profit\)\**\s*\|\s*([^|\n]+)",
        "operating_expenses": r"Beban Operasional\**\s*\|\s*([^|\n]+)",
        "operating_profit": r"Laba Operasional\**\s*\|\s*([^|\n]+)",
        "other_net": r"Pendapatan / Beban Lain-Lain Bersih\**\s*\|\s*([^|\n]+)",
        "net_profit": r"Laba Bersih Periode Berjalan[^|\n]*\|\s*([^|\n]+)",
    }
    for key, pat in metric_patterns.items():
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = _parse_indonesian_rupiah(m.group(1))
            if val is not None:
                summary[key] = val

    m_gm = re.search(r"Gross Profit Margin\*?\s*\|\s*\*?(-?[\d.,]+)\s*%", text, re.IGNORECASE)
    if m_gm:
        try:
            summary["gross_margin_percent"] = float(m_gm.group(1).replace(",", "."))
        except ValueError:
            pass

    if not summary:
        return None

    m_per = re.search(
        r"Periode\s*:\s*`?(\d{4}-\d{2}-\d{2})`?\s*(?:s/d|-|to)\s*`?(\d{4}-\d{2}-\d{2})`?",
        text,
        re.IGNORECASE,
    )
    groups_map: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) != 5 or parts[0].startswith(":") or parts[0].lower() == "grup akun":
            continue
        grp_name, tp_name, raw_code, acc_name, raw_bal = parts
        bal = _parse_indonesian_rupiah(raw_bal)
        if bal is None:
            continue
        code = raw_code.strip("`").strip()
        groups_map.setdefault((grp_name, tp_name), []).append({
            "account_code": code,
            "account_name": acc_name,
            "balance": bal,
        })

    breakdown: List[Dict[str, Any]] = []
    if groups_map:
        by_group: Dict[str, List[Dict[str, Any]]] = {}
        for (g_n, t_n), accs in groups_map.items():
            by_group.setdefault(g_n, []).append({"type_name": t_n, "accounts": accs})
        for g_n, t_list in by_group.items():
            breakdown.append({"group_name": g_n, "types": t_list})

    return {
        "date_from": m_per.group(1) if m_per else None,
        "date_to": m_per.group(2) if m_per else None,
        "summary": summary,
        "breakdown": breakdown,
    }


def _collect_pnl_accounts(pnl_data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
    """Flatten non-zero P&L breakdown accounts and count zero-balance entries."""
    accounts: List[Dict[str, Any]] = []
    zero_count = 0
    for grp in pnl_data.get("breakdown") or []:
        grp_name = str(grp.get("group_name") or grp.get("name") or "-")
        for tp in grp.get("types", []):
            tp_name = str(tp.get("type_name") or tp.get("name") or "-")
            for acc in tp.get("accounts", []):
                bal = float(acc.get("balance", 0.0) or 0.0)
                if abs(bal) <= 0.01:
                    zero_count += 1
                    continue
                accounts.append({
                    "group": grp_name,
                    "type": tp_name,
                    "code": str(acc.get("account_code") or "-"),
                    "name": str(acc.get("account_name") or "-"),
                    "balance": bal,
                })
    return accounts, zero_count


def _build_pnl_analytical_digest(pnl_data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Compute structured P&L analytical metrics, top cost drivers, and anomaly flags."""
    summary = pnl_data.get("summary") or {}
    rev = float(summary.get("total_revenue", 0.0) or 0.0)
    cogs = float(summary.get("cogs", 0.0) or 0.0)
    gross = float(summary.get("gross_profit", 0.0) or 0.0)
    gm = float(summary.get("gross_margin_percent", 0.0) or 0.0)
    opex = float(summary.get("operating_expenses", 0.0) or 0.0)
    op_profit = float(summary.get("operating_profit", 0.0) or 0.0)
    other_net = float(summary.get("other_net", 0.0) or 0.0)
    net_profit = float(summary.get("net_profit", 0.0) or 0.0)

    accounts, zero_count = _collect_pnl_accounts(pnl_data)
    opex_accounts = [
        a for a in accounts
        if a["group"].lower() == "expense" and "cost of revenue" not in a["type"].lower()
    ]
    positive_opex = sorted([a for a in opex_accounts if a["balance"] > 0], key=lambda x: x["balance"], reverse=True)
    negative_opex = sorted([a for a in opex_accounts if a["balance"] < 0], key=lambda x: x["balance"])
    negative_income = [
        a for a in accounts
        if a["group"].lower() == "income" and "other income" not in a["type"].lower() and a["balance"] < 0
    ]
    cogs_accounts = [
        a for a in accounts
        if "cost of revenue" in a["type"].lower()
    ]
    other_income_accounts = sorted(
        [a for a in accounts if "other" in a["type"].lower() or a["code"].startswith("41") or a["code"].startswith("52")],
        key=lambda x: abs(x["balance"]),
        reverse=True,
    )

    shown_opex_sum = sum(a["balance"] for a in opex_accounts)
    opex_gap = opex - shown_opex_sum if opex_accounts else 0.0

    d_from = pnl_data.get("date_from") or "-"
    d_to = pnl_data.get("date_to") or "-"
    digest_lines = [
        f"Periode Laporan: {d_from} s/d {d_to}",
        f"- Pendapatan Usaha (Revenue): {_format_rupiah(rev)}",
        f"- HPP (COGS): {_format_rupiah(cogs)}",
        f"- Laba Kotor (Gross Profit): {_format_rupiah(gross)} (Gross Margin: {gm:.1f}%)",
        f"- Beban Operasional: {_format_rupiah(opex)}",
        f"- Laba / (Rugi) Operasional: {_format_rupiah(op_profit)}",
        f"- Pendapatan / Beban Lain-Lain Bersih: {_format_rupiah(other_net)}",
        f"- Laba / (Rugi) Bersih: {_format_rupiah(net_profit)}",
    ]

    if positive_opex:
        digest_lines.append("\nPos Beban Operasional Terbesar:")
        for a in positive_opex[:10]:
            pct_rev = f" ({(a['balance'] / rev * 100.0):.1f}% dari revenue)" if rev > 0 else ""
            digest_lines.append(f"- `{a['code']}` {a['name']}: {_format_rupiah(a['balance'])}{pct_rev}")

    if negative_income or negative_opex:
        digest_lines.append("\nPos Pengurang Pendapatan / Penyesuaian Negatif:")
        for a in negative_income:
            digest_lines.append(f"- [Contra/Negatif Income] `{a['code']}` {a['name']}: {_format_rupiah(a['balance'])}")
        for a in negative_opex:
            digest_lines.append(f"- [Kredit/Offset Beban] `{a['code']}` {a['name']}: {_format_rupiah(a['balance'])}")

    if cogs_accounts:
        digest_lines.append("\nRincian HPP (COGS):")
        for a in sorted(cogs_accounts, key=lambda x: abs(x["balance"]), reverse=True)[:5]:
            digest_lines.append(f"- `{a['code']}` {a['name']}: {_format_rupiah(a['balance'])}")

    if other_income_accounts:
        digest_lines.append("\nPos Pendapatan / Beban Lain-Lain Terbesar:")
        for a in other_income_accounts[:5]:
            digest_lines.append(f"- `{a['code']}` {a['name']}: {_format_rupiah(a['balance'])}")

    if opex_accounts and abs(opex_gap) > 1000.0:
        digest_lines.append(
            f"\nCatatan Kelengkapan Data: Jumlah pos beban operasional yang tampil ({_format_rupiah(shown_opex_sum)}) "
            f"memiliki selisih {_format_rupiah(opex_gap)} terhadap total Beban Operasional ({_format_rupiah(opex)}), "
            f"menandakan sebagian akun di bawah baris terakhir tidak tercantum pada rincian."
        )
    if zero_count > 0:
        digest_lines.append(f"Catatan: {zero_count} akun bersaldo Rp 0,00 diabaikan dari ringkasan analisis.")

    meta = {
        "rev": rev,
        "cogs": cogs,
        "gross": gross,
        "gm": gm,
        "opex": opex,
        "op_profit": op_profit,
        "other_net": other_net,
        "net_profit": net_profit,
        "positive_opex": positive_opex,
        "negative_income": negative_income,
        "negative_opex": negative_opex,
        "opex_accounts": opex_accounts,
        "shown_opex_sum": shown_opex_sum,
        "opex_gap": opex_gap,
        "d_from": d_from,
        "d_to": d_to,
    }
    return "\n".join(digest_lines), meta


def _format_pnl_analytical_fallback(pnl_data: Dict[str, Any]) -> str:
    """Deterministic causal P&L explanation with compact summary and top drivers when LLM is unavailable."""
    _, meta = _build_pnl_analytical_digest(pnl_data)
    rev = meta["rev"]
    cogs = meta["cogs"]
    gross = meta["gross"]
    gm = meta["gm"]
    opex = meta["opex"]
    op_profit = meta["op_profit"]
    other_net = meta["other_net"]
    net_profit = meta["net_profit"]
    d_from = meta["d_from"]
    d_to = meta["d_to"]

    lines: List[str] = []
    if net_profit < 0:
        if gross > 0 and opex > gross:
            excess = opex - gross
            lead = (
                f"Laba bersih periode **{d_from} s/d {d_to}** tercatat minus **{_format_rupiah(net_profit)}** "
                f"karena Beban Operasional (**{_format_rupiah(opex)}**) lebih besar **{_format_rupiah(excess)}** "
                f"dibandingkan Laba Kotor (**{_format_rupiah(gross)}**, margin **{gm:.1f}%**), sehingga "
                f"menimbulkan Rugi Operasional **{_format_rupiah(op_profit)}**."
            )
            if other_net > 0:
                lead += f" Pendapatan/beban lain-lain bersih (**{_format_rupiah(other_net)}**) hanya menutup sebagian rugi operasional tersebut."
        elif gross <= 0:
            lead = (
                f"Laba bersih periode **{d_from} s/d {d_to}** minus **{_format_rupiah(net_profit)}** "
                f"karena HPP (**{_format_rupiah(cogs)}**) sudah melebihi Pendapatan Usaha (**{_format_rupiah(rev)}**)."
            )
        else:
            lead = (
                f"Laba bersih periode **{d_from} s/d {d_to}** tercatat minus **{_format_rupiah(net_profit)}** "
                f"meskipun Laba Operasional sebesar **{_format_rupiah(op_profit)}**, karena tergerus pos lain-lain bersih "
                f"(**{_format_rupiah(other_net)}**)."
            )
    else:
        lead = (
            f"Laba bersih periode **{d_from} s/d {d_to}** positif sebesar **{_format_rupiah(net_profit)}** "
            f"dari Laba Kotor **{_format_rupiah(gross)}** (margin **{gm:.1f}%**) setelah dikurangi Beban Operasional **{_format_rupiah(opex)}**."
        )
    lines.append(lead)
    lines.append("")
    lines.append("| Komponen Laba Rugi | Nominal |")
    lines.append("|:---|:---|")
    lines.append(f"| Pendapatan Usaha (Revenue) | {_format_rupiah(rev)} |")
    lines.append(f"| Beban Pokok Penjualan (HPP) | {_format_rupiah(cogs)} |")
    lines.append(f"| Laba Kotor (Margin {gm:.1f}%) | **{_format_rupiah(gross)}** |")
    lines.append(f"| Beban Operasional | {_format_rupiah(opex)} |")
    lines.append(f"| Laba / (Rugi) Operasional | **{_format_rupiah(op_profit)}** |")
    lines.append(f"| Pendapatan / Beban Lain-Lain Bersih | {_format_rupiah(other_net)} |")
    lines.append(f"| **Laba / (Rugi) Bersih** | **{_format_rupiah(net_profit)}** |")

    pos_opex = meta["positive_opex"]
    if pos_opex:
        lines.append("\n#### Pemicu Beban Operasional Terbesar:")
        for a in pos_opex[:5]:
            share_opex = f" ({(a['balance'] / opex * 100.0):.1f}% dari beban operasional)" if opex > 0 else ""
            lines.append(f"- `{a['code']}` **{a['name']}**: **{_format_rupiah(a['balance'])}**{share_opex}")

    anomalies: List[str] = []
    for a in meta["negative_income"]:
        anomalies.append(f"Pos pengurang/negatif pada pendapatan: `{a['code']}` {a['name']} ({_format_rupiah(a['balance'])}).")
    for a in meta["negative_opex"]:
        anomalies.append(f"Pos kredit/offset pada beban: `{a['code']}` {a['name']} ({_format_rupiah(a['balance'])}).")
    if meta["opex_accounts"] and abs(meta["opex_gap"]) > 1000.0:
        anomalies.append(
            f"Selisih **{_format_rupiah(meta['opex_gap'])}** antara total Beban Operasional ({_format_rupiah(opex)}) "
            f"dan penjumlahan pos beban yang tercantum ({_format_rupiah(meta['shown_opex_sum'])})."
        )
    if anomalies:
        lines.append("\n#### Catatan & Anomali Data:")
        for note in anomalies[:5]:
            lines.append(f"- {note}")

    return "\n".join(lines)


def _extract_latest_erp_report_message(messages: Optional[Sequence[Any]]) -> str:
    """Return content of the most recent substantive ERP AIMessage from conversation history."""
    if not messages:
        return ""
    prior_msgs = list(messages)
    if prior_msgs and (getattr(prior_msgs[-1], "type", "") in ("human", "user") or isinstance(prior_msgs[-1], HumanMessage)):
        prior_msgs = prior_msgs[:-1]

    for msg in reversed(prior_msgs):
        msg_type = getattr(msg, "type", "")
        if msg_type != "ai" and not isinstance(msg, AIMessage):
            continue
        content = str(getattr(msg, "content", "") or "").strip()
        if not content:
            continue
        if any(
            marker in content
            for marker in (
                "Halo! Saya adalah Asisten AI FastPrint ERP",
                "Permintaan di Luar Cakupan",
                "Data Tidak Ditemukan",
                "Kendala Akses Data ERP",
            )
        ):
            continue
        return content
    return ""


def _build_deterministic_analytical_fallback(state: AgentState) -> str:
    """Provide grounded analytical fallback without LLM for contextual_analysis or analytical tool turns."""
    tool_result = state.get("tool_result") or state.get("prior_tool_result")
    tool = state.get("detected_tool") or state.get("prior_detected_tool")
    if tool == "get_profit_and_loss" and isinstance(tool_result, dict) and tool_result.get("summary"):
        return _format_pnl_analytical_fallback(tool_result)

    latest_ai = _extract_latest_erp_report_message(state.get("messages"))
    parsed_pnl = _parse_pnl_from_markdown(latest_ai)
    if parsed_pnl:
        return _format_pnl_analytical_fallback(parsed_pnl)

    if latest_ai:
        # Return concise non-empty slice without repeating giant full tables
        non_zero_lines = [
            ln for ln in latest_ai.splitlines()
            if ln.strip() and "| Rp 0,00 |" not in ln
        ]
        if len(non_zero_lines) > 25:
            return "\n".join(non_zero_lines[:25]) + "\n\n*(Ringkasan dipotong dari laporan sebelumnya.)*"
        return "\n".join(non_zero_lines)

    return "Tidak terdapat data laporan ERP sebelumnya yang cukup untuk dianalisis lebih lanjut."


def _build_analytical_data_context(
    state: AgentState,
    deterministic_answer: str,
    max_chars: int = 12000,
) -> str:
    """Build structured, zero-clutter ERP data context for analytical LLM reasoning."""
    query_mode = state.get("query_mode")
    tool_result = state.get("tool_result") or state.get("prior_tool_result")
    tool = state.get("detected_tool") or state.get("prior_detected_tool")
    latest_ai = _extract_latest_erp_report_message(state.get("messages"))

    if tool == "get_profit_and_loss" and isinstance(tool_result, dict) and tool_result.get("summary"):
        digest, _ = _build_pnl_analytical_digest(tool_result)
        raw_table = deterministic_answer if query_mode != "contextual_analysis" else latest_ai
        if not raw_table and tool_result:
            raw_table = format_deterministic_answer({**state, "tool_result": tool_result, "detected_tool": "get_profit_and_loss", "verification_status": "VALID"})
        filtered_table = "\n".join(ln for ln in (raw_table or "").splitlines() if "| Rp 0,00 |" not in ln)
        combined = f"{digest}\n\nRincian Laporan:\n{filtered_table}" if filtered_table else digest
        return combined[:max_chars]

    if query_mode == "contextual_analysis":
        parsed_pnl = _parse_pnl_from_markdown(latest_ai)
        if parsed_pnl:
            digest, _ = _build_pnl_analytical_digest(parsed_pnl)
            filtered_ai = "\n".join(ln for ln in latest_ai.splitlines() if "| Rp 0,00 |" not in ln)
            return f"{digest}\n\nLaporan Sebelumnya:\n{filtered_ai}"[:max_chars]
        source_text = latest_ai or deterministic_answer
        filtered = "\n".join(ln for ln in source_text.splitlines() if "| Rp 0,00 |" not in ln)
        return filtered[:max_chars]

    filtered_det = "\n".join(ln for ln in (deterministic_answer or "").splitlines() if "| Rp 0,00 |" not in ln)
    return filtered_det[:max_chars]


def _build_analytical_prompt(
    user_query: Any,
    data_context: str,
    history_excerpt: str = "",
) -> str:
    """Build analytical reasoning prompt grounded strictly in ERP data."""
    hist_block = f"Konteks Percakapan Sebelumnya:\n{history_excerpt}\n\n" if history_excerpt else ""
    return (
        "Anda adalah Analis Keuangan & Operasional FastPrint ERP yang presisi, lugas, dan bebas basa-basi (ZERO AI SLOP).\n"
        "Tugas Anda adalah MENJAWAB pertanyaan analitis/berpikir pengguna secara langsung berdasarkan data ERP terlampir — BUKAN mengulang tabel laporan mentah.\n\n"
        "ATURAN KETAT ANALISIS:\n"
        "1. LANGSUNG KE POKOK JAWABAN (Lead with the answer): Pada kalimat pertama, langsung jawab penyebab utama, temuan inti, atau jawaban dari pertanyaan pengguna.\n"
        "2. KUTIP ANGKA & KODE AKUN/SKU SECARA PERSIS: Wajib hanya menggunakan angka, kode akun, kode SKU, nama cabang, dan nominal yang tertera pada data ERP terlampir. DILARANG mengarang angka atau berspekulasi tentang penyebab eksternal di luar data.\n"
        "3. RINGKASAN PEMICU UTAMA (Compact Top Drivers): Sajikan maksimal 4-7 pos/faktor pemicu terbesar beserta nominal (dan persentase bila relevan). DILARANG mengulang tabel rincian panjang (80+ baris) kecuali pengguna meminta tabel.\n"
        "4. SOROT ANOMALI DATA JIKA ADA: Jika terlihat pos tidak lazim pada rincian (misal pendapatan bersaldo negatif/contra, akun beban bersaldo negatif/offset penyusutan, atau selisih antara penjumlahan pos yang tampil dengan total ringkasan), jelaskan secara faktual.\n"
        "5. ZERO AI SLOP & TANPA EMOJI: Dilarang basa-basi pembuka/penutup ('Tentu!', 'Berdasarkan data...', 'Semoga membantu!') dan dilarang karakter emoji apapun.\n\n"
        "Sertakan 2 hingga 3 rekomendasi pertanyaan lanjutan READ-ONLY yang spesifik terhadap pos/entitas pada temuan di atas dalam blok JSON:\n"
        "```json:followups\n"
        "[\n"
        '  {"label": "Label Singkat (1-3 kata)", "query": "Prompt lengkap READ-ONLY", "icon": "fa-arrow-right"}\n'
        "]\n"
        "```\n\n"
        f"{hist_block}"
        f"Pertanyaan Pengguna: {user_query}\n\n"
        f"Data ERP Terstruktur:\n{data_context}"
    )


def _build_followup_only_prompt(user_query: Any, deterministic_answer: str) -> str:
    """Build prompt that requests ONLY json:followups suggestions without repeating report tables."""
    return (
        "Anda adalah Asisten AI FastPrint ERP.\n"
        "Berdasarkan laporan di bawah ini, susunlah 2 hingga 3 rekomendasi pertanyaan/prompt lanjutan (READ-ONLY) yang paling logis dan relevan untuk diajukan pengguna berikutnya.\n\n"
        "ATURAN KETAT REKOMENDASI:\n"
        "1. Wajib hanya menyarankan pertanyaan penelusuran data (READ-ONLY) pada modul FastPrint ERP yang didukung: Akuntansi (Saldo Akun, Jurnal, Buku Besar, Laba Rugi, Neraca, Tagihan Vendor, Rekonsiliasi Bank), Inventaris (Stok Fisik Gudang, Master Katalog & Harga, Kartu Stok, Mutasi Transfer, Alert Stok Rendah), Penjualan (Sales Order, Omzet, Produk Terlaris), dan Marketplace (Kesehatan Akun & Token, Order SLA, Pesanan Kilat, Mapping SKU, Simulasi & Log Stok/Harga, Perbandingan Harga, Anomali Fee, Potongan & Settlement, Cashflow & Saldo Dompet, Kompetitor). DILARANG menyarankan aksi membuat/mengubah transaksi atau modul di luar daftar ini.\n"
        "2. DILARANG KERAS menggunakan karakter emoji pada label, query, maupun icon.\n"
        "3. Formatkan rekomendasi HANYA dalam blok JSON berikut tanpa teks lain di luar blok kode:\n"
        "```json:followups\n"
        "[\n"
        '  {"label": "Label Singkat (1-3 kata)", "query": "Prompt lengkap yang akan dieksekusi", "icon": "fa-arrow-right"}\n'
        "]\n"
        "```\n\n"
        f"Pertanyaan Pengguna: {user_query}\n"
        f"Ringkasan Laporan:\n{deterministic_answer[:1000]}"
    )


def _build_polish_prompt(user_query: Any, deterministic_answer: str) -> str:
    """Build prompt that polishes compact deterministic tables while strictly preserving values."""
    return (
        "Anda adalah Asisten AI FastPrint ERP yang profesional, presisi, dan lugas.\n"
        "Sajikan data laporan berikut dalam Bahasa Indonesia yang ringkas, jelas, dan bebas basa-basi (ZERO AI SLOP):\n"
        "1. LANGSUNG KE POKOK JAWABAN (Lead with the answer): Tuliskan angka total utama, saldo, atau status di kalimat paling pertama tanpa pengantar bertele-tele.\n"
        "2. DILARANG KERAS menggunakan basa-basi pembuka ('Tentu!', 'Berikut rinciannya...', 'Berdasarkan data...') maupun penutup ('Semoga membantu!', 'Jika ada pertanyaan lain...'). Hentikan respons langsung setelah data dan rekomendasi selesai disajikan.\n"
        "3. DILARANG KERAS menggunakan karakter emoji, emoticon (seperti 😊, :), 📊, 💰, 📦, 🏢, ✅, ⚠️, ❌), atau simbol dekoratif visual apapun pada judul, heading, poin peluru, tabel, maupun kalimat penjelas.\n"
        "4. DILARANG menggunakan kata klise korporat: mengoptimalkan, memfasilitasi, komprehensif, signifikan, ekosistem, holistik, menyeluruh, memberdayakan, penting untuk dicatat bahwa.\n"
        "5. PERTAHANKAN seluruh angka transaksi, kode, dan rincian tabel Markdown dengan akurat tanpa mengubah nilai.\n"
        "6. DILARANG KERAS menghitung ulang atau mengestimasi nominal keuangan. Seluruh angka wajib dikutip persis dari data laporan terstruktur.\n\n"
        "Selain menyajikan data laporan di atas, susunlah 2 hingga 3 rekomendasi pertanyaan/prompt lanjutan (READ-ONLY) yang paling logis, relevan, dan kontekstual untuk diajukan pengguna berikutnya berdasarkan data riil yang ditemukan di atas (misal: rincian transaksi terkait, mutasi kartu stok produk ini, cek tagihan overdue, atau rekonsiliasi bank).\n\n"
        "ATURAN KETAT REKOMENDASI:\n"
        "1. Wajib hanya menyarankan pertanyaan penelusuran data (READ-ONLY) pada modul FastPrint ERP yang didukung: Akuntansi (Saldo Akun, Jurnal, Buku Besar, Laba Rugi, Neraca, Tagihan Vendor, Rekonsiliasi Bank), Inventaris (Stok Fisik Gudang, Master Katalog & Harga, Kartu Stok, Mutasi Transfer, Alert Stok Rendah), Penjualan (Sales Order, Omzet, Produk Terlaris), dan Marketplace (Kesehatan Akun & Token, Order SLA, Pesanan Kilat, Mapping SKU, Simulasi & Log Stok/Harga, Perbandingan Harga, Anomali Fee, Potongan & Settlement, Cashflow & Saldo Dompet, Kompetitor). DILARANG menyarankan aksi membuat/mengubah transaksi atau modul di luar daftar ini.\n"
        "2. Wajib menggunakan entitas riil (kode SKU, nama vendor, kode akun, nama gudang) yang terdapat pada data di atas.\n"
        "3. DILARANG KERAS menggunakan karakter emoji pada label, query, maupun icon.\n"
        "4. Panjang teks query rekomendasi WAJIB SINGKAT dan FOKUS SATU INTENSI (maksimal 10-12 kata, contoh: 'Cek mutasi kartu stok SKU ...', 'Cek stok SKU ... di semua cabang'). DILARANG membuat kalimat majemuk bertingkat yang berbelit-belit.\n"
        "5. Formatkan rekomendasi di akhir respons dalam blok JSON khusus berikut:\n"
        "```json:followups\n"
        "[\n"
        '  {"label": "Label Singkat (1-3 kata)", "query": "Prompt lengkap yang akan dieksekusi", "icon": "fa-arrow-right"}\n'
        "]\n"
        "```\n"
        "DILARANG menuliskan judul pengantar seperti 'Rekomendasi lanjutan' atau teks apapun di luar blok kode json:followups.\n\n"
        f"Pertanyaan Pengguna: {user_query}\n"
        f"Data Laporan Terstruktur:\n{deterministic_answer}"
    )


def _finalize_synthesis_output(
    final_text: str,
    deterministic_answer: str,
    followups: List[Dict[str, str]],
    actual_model: Optional[str],
    user_query: Any,
    retry_header: str,
    state: AgentState,
) -> Dict[str, Any]:
    """Normalize followups, apply retry header, and build final synthesizer state delta."""
    resolved_followups = followups if followups else generate_deterministic_followups(state)
    resolved_followups = _finalize_followups(resolved_followups, user_query, state)
    resolved_final_text = _apply_retry_header(final_text, retry_header)

    from app.graph.nodes.formatters.entity_links import build_entity_links

    out_dict: Dict[str, Any] = {
        "final_answer": resolved_final_text,
        "deterministic_answer": deterministic_answer,
        "suggested_followups": resolved_followups,
        "table_query_meta": _build_table_query_meta(state),
        "entity_links": build_entity_links(state),
        "messages": [AIMessage(content=resolved_final_text)],
    }
    if actual_model:
        out_dict["active_model"] = actual_model
    return out_dict


def _build_synthesis_plan(
    state: AgentState,
    user_query: Any,
    status: Optional[str],
    deterministic_answer: str,
    is_multi_q: bool,
    model: Optional[BaseChatModel],
) -> Tuple[bool, bool, bool, bool, Optional[str]]:
    """Determine synthesis mode (verbatim, polish, or analytical) and resolve prompt if LLM runs."""
    query_mode = state.get("query_mode")
    is_contextual_analysis = query_mode == "contextual_analysis"
    is_large_deterministic = _should_keep_verbatim_deterministic(
        deterministic_answer, user_query, state, is_multi_q
    )
    is_analytical = is_contextual_analysis or (
        status == "VALID"
        and is_analytical_thinking_query(str(user_query or ""))
        and not is_large_deterministic
    )
    is_rag_synthesis = query_mode in ("vector_rag", "hybrid")
    skip_tabular_llm = (
        bool(getattr(settings, "SYNTHESIZER_FAST_DETERMINISTIC", False))
        and not is_analytical
        and not is_rag_synthesis
    )
    should_llm_synthesize = (
        model is not None
        and not skip_tabular_llm
        and (is_contextual_analysis or status == "VALID")
    )
    should_polish = should_llm_synthesize and (not is_large_deterministic or is_analytical)

    if not should_llm_synthesize:
        return is_large_deterministic, is_contextual_analysis, is_analytical, should_polish, None

    if is_large_deterministic and not is_analytical:
        prompt = _build_followup_only_prompt(user_query, deterministic_answer)
    elif is_analytical:
        data_context = _build_analytical_data_context(state, deterministic_answer)
        raw_hist = (state.get("messages") or [])[:-1]
        trimmed_hist = trim_conversation_messages(raw_hist, max_messages=4)
        hist_lines = []
        for m in trimmed_hist:
            if isinstance(m, HumanMessage) or getattr(m, "type", "") in ("human", "user"):
                hist_lines.append(f"User: {str(getattr(m, 'content', '') or '').strip()}")
        prompt = _build_analytical_prompt(
            user_query=user_query,
            data_context=data_context,
            history_excerpt="\n".join(hist_lines),
        )
    else:
        prompt = _build_polish_prompt(user_query, deterministic_answer)

    return is_large_deterministic, is_contextual_analysis, is_analytical, should_polish, prompt


def _resolve_fallback_text_on_missing_or_failed_llm(
    state: AgentState,
    deterministic_answer: str,
    is_analytical: bool,
) -> str:
    """Use grounded analytical fallback for analytical queries instead of dumping giant raw tables."""
    if is_analytical:
        tool = state.get("detected_tool") or state.get("prior_detected_tool")
        if state.get("query_mode") == "contextual_analysis" or tool in ("get_profit_and_loss", "get_balance_sheet"):
            return _build_deterministic_analytical_fallback(state)
    return deterministic_answer


def _sanitize_clarification_options(
    options: Optional[List[Dict[str, Any]]],
    max_items: int = 4,
) -> List[Dict[str, str]]:
    """Sanitize clarification option chips (no emojis, valid icon, deduplicated, capped at max_items)."""
    cleaned: List[Dict[str, str]] = []
    seen_queries: set[str] = set()
    for item in options or []:
        if not isinstance(item, dict):
            continue
        raw_label = str(item.get("label") or "").strip()
        raw_query = str(item.get("query") or "").strip()
        if not raw_label or not raw_query:
            continue
        norm_q = re.sub(r"\s+", " ", raw_query.rstrip("?!.,;:'\"> ")).lower()
        if norm_q in seen_queries:
            continue
        seen_queries.add(norm_q)

        clean_label = _clean_label_emojis(raw_label) or raw_label[:40]
        raw_icon = str(item.get("icon") or "fa-arrow-right").strip()
        clean_icon = raw_icon if re.fullmatch(r"[A-Za-z0-9_\- ]+", raw_icon) else "fa-arrow-right"
        cleaned.append({
            "label": clean_label,
            "query": raw_query,
            "icon": clean_icon or "fa-arrow-right",
        })
        if len(cleaned) >= max_items:
            break
    return cleaned


def _build_pdf_inspect_clarification_output(state: AgentState) -> Optional[Dict[str, Any]]:
    """
    When reconcile_bank_statement ran in 'inspect' mode (user uploaded PDF without clear instructions),
    convert the inspection payload directly into a proactive CLARIFICATION turn with clickable chips.
    """
    if state.get("detected_tool") != "reconcile_bank_statement":
        return None
    if state.get("verification_status") == "ERROR":
        return None
    res = state.get("tool_result")
    if not isinstance(res, dict) or res.get("mode") != "inspect":
        return None

    meta = res.get("statement_meta") or {}
    checksum = res.get("checksum_gate") or {}
    det = res.get("account_detection") or {}

    bank_code = meta.get("bank_code") or "Bank"
    acc_no = meta.get("account_number") or "-"
    acc_holder = meta.get("account_holder") or "-"
    period = meta.get("statement_period") or "-"
    cr_count = meta.get("cr_count", 0)
    db_count = meta.get("db_count", 0)
    saldo_awal = _format_rupiah(meta.get("saldo_awal", 0.0))
    saldo_akhir = _format_rupiah(meta.get("saldo_akhir", 0.0))
    checksum_str = "PASS (Valid 100%)" if checksum.get("passed") else "FAIL (Ada selisih pembacaan baris)"

    codes = det.get("candidate_account_codes") or []
    codes_str = ", ".join(f"`{c}`" for c in codes) if codes else "Belum terdeteksi otomatis"
    shared_note = (
        " *(rekening bersama lintas cabang)*"
        if det.get("is_shared_multi_branch")
        else ""
    )

    question = (
        f"### Berkas Rekening Koran Terdeteksi: {bank_code} `{acc_no}`\n\n"
        f"- **Pemilik Rekening**: {acc_holder}\n"
        f"- **Periode**: {period}\n"
        f"- **Mutasi Terbaca**: {cr_count} uang masuk (CR) & {db_count} uang keluar (DB)\n"
        f"- **Saldo Awal / Akhir**: {saldo_awal} &rarr; **{saldo_akhir}**\n"
        f"- **Golden Checksum Gate**: **{checksum_str}**\n"
        f"- **Akun GL ERP Terkait**: {codes_str}{shared_note}\n\n"
        f"Apa yang ingin saya lakukan dengan berkas rekening koran ini?"
    )

    chips = _sanitize_clarification_options(res.get("suggested_clarifications"), max_items=4)
    out_dict: Dict[str, Any] = {
        "final_answer": question,
        "deterministic_answer": question,
        "suggested_followups": chips,
        "verification_status": "CLARIFICATION",
        "needs_clarification": True,
        "query_mode": "clarification",
        "table_query_meta": None,
        "entity_links": [],
        "messages": [AIMessage(content=question)],
    }
    if state.get("active_model"):
        out_dict["active_model"] = state["active_model"]
    return out_dict


def _build_clarification_synthesis_output(state: AgentState) -> Optional[Dict[str, Any]]:
    """
    Fast-path synthesis output for conservative disambiguation (query_mode='clarification').
    Produces final_answer + sanitized suggested_followups chips without invoking an extra LLM call.
    """
    pdf_inspect_out = _build_pdf_inspect_clarification_output(state)
    if pdf_inspect_out is not None:
        return pdf_inspect_out

    if state.get("query_mode") != "clarification" and not state.get("needs_clarification"):
        return None

    raw_question = str(state.get("clarification_question") or "").strip()
    question = _clean_ai_slop(raw_question) if raw_question else ""
    if not question:
        question = (
            "Mohon perjelas maksud pertanyaan Anda dengan memilih salah satu opsi di bawah ini "
            "agar saya dapat mengambil data FastPrint ERP yang tepat."
        )

    chips = _sanitize_clarification_options(state.get("clarification_options"))
    for chip in chips:
        chip["query"] = _embed_active_branch_in_chip_query(chip["query"], state)
    out_dict: Dict[str, Any] = {
        "final_answer": question,
        "deterministic_answer": question,
        "suggested_followups": chips,
        "verification_status": "CLARIFICATION",
        "needs_clarification": True,
        "query_mode": "clarification",
        "table_query_meta": None,
        "entity_links": [],
        "messages": [AIMessage(content=question)],
    }
    if state.get("active_model"):
        out_dict["active_model"] = state["active_model"]
    return out_dict


async def asynthesizer_node(
    state: AgentState,
    llm: Optional[BaseChatModel] = None,
) -> Dict[str, Any]:
    """
    Asynchronous LangGraph pipeline node that synthesizes the final conversational response
    with True Streaming and hybrid presentation support.
    """
    clarification_out = _build_clarification_synthesis_output(state)
    if clarification_out is not None:
        if clarification_out.get("final_answer"):
            try:
                await adispatch_custom_event("token", {"content": clarification_out["final_answer"]})
            except Exception as dispatch_err:
                logger.debug(f"Could not dispatch clarification chunk: {dispatch_err}")
        return clarification_out

    (
        user_query,
        status,
        retry_header,
        deterministic_answer,
        actual_model,
        non_tool_questions,
        is_multi_q,
        model,
    ) = _prepare_synthesis_context(state, llm)

    final_text = deterministic_answer
    followups: List[Dict[str, str]] = []

    general_nt_items = [
        nt for nt in non_tool_questions
        if isinstance(nt, dict)
        and (nt.get("category") or "general") == "general"
        and nt.get("question_text")
        and (is_pure_greeting(nt["question_text"]) or is_general_help_inquiry(nt["question_text"]))
    ]
    if model is not None and general_nt_items:
        llm_general_answers: Dict[int, str] = {}
        for nt in general_nt_items:
            try:
                g_resp = await model.ainvoke(
                    [
                        SystemMessage(content="Anda adalah Asisten AI FastPrint ERP yang ringkas, jelas, dan bebas AI slop."),
                        HumanMessage(
                            content=(
                                "Balas sapaan atau jelaskan secara singkat kemampuan Asisten AI FastPrint ERP "
                                "(1-3 kalimat, Bahasa Indonesia). Jangan menjawab topik di luar FastPrint ERP:\n"
                                f"{nt['question_text']}"
                            )
                        ),
                    ],
                    config={"tags": ["internal_no_stream"]},
                )
                g_content = getattr(g_resp, "content", None)
                if g_content and isinstance(g_content, str) and g_content.strip():
                    llm_general_answers[int(nt.get("question_index") or 0)] = _clean_ai_slop(g_content.strip())
            except Exception as exc:
                logger.debug("LLM general sub-question synthesis failed: %s", exc)
        if llm_general_answers:
            updated_det = _build_question_ordered_answer(state, llm_general_answers=llm_general_answers)
            if updated_det:
                deterministic_answer = updated_det
                final_text = deterministic_answer

    (
        is_large_deterministic,
        is_contextual_analysis,
        is_analytical,
        should_polish,
        prompt,
    ) = _build_synthesis_plan(state, user_query, status, deterministic_answer, is_multi_q, model)

    if is_analytical and model is None:
        final_text = _resolve_fallback_text_on_missing_or_failed_llm(state, deterministic_answer, is_analytical)

    # Immediate <10ms token dispatch for non-polished deterministic/multi-company/multi-question paths
    if not should_polish and final_text:
        try:
            await adispatch_custom_event("token", {"content": _apply_retry_header(final_text, retry_header)})
        except Exception as dispatch_err:
            logger.debug(f"Could not dispatch deterministic_chunk: {dispatch_err}")

    if prompt is not None and model is not None:
        try:
            response = await model.ainvoke(
                [
                    SystemMessage(content="Anda adalah Asisten AI FastPrint ERP yang presisi, lugas, dan bebas AI slop."),
                    HumanMessage(content=prompt),
                ],
                config={"tags": ["internal_no_stream"]},
            )
            resolved_model = _extract_resolved_model_name(response)
            if resolved_model:
                actual_model = resolved_model
            content = getattr(response, "content", None)
            if content and isinstance(content, str) and content.strip():
                clean_ans, llm_followups = _extract_llm_followups(content.strip())
                if not is_large_deterministic or is_analytical:
                    clean_ans = _clean_ai_slop(clean_ans)
                    if clean_ans:
                        final_text = clean_ans
                if llm_followups:
                    followups = llm_followups
        except Exception as exc:
            logger.warning(f"LLM synthesis failed, falling back to deterministic output: {exc}")
            final_text = _resolve_fallback_text_on_missing_or_failed_llm(state, deterministic_answer, is_analytical)

    out_dict = _finalize_synthesis_output(
        final_text=final_text,
        deterministic_answer=deterministic_answer,
        followups=followups,
        actual_model=actual_model,
        user_query=user_query,
        retry_header=retry_header,
        state=state,
    )
    if is_contextual_analysis and not out_dict.get("verification_status"):
        out_dict["verification_status"] = state.get("verification_status") or "VALID"
    if should_polish and out_dict["final_answer"]:
        try:
            await adispatch_custom_event("token", {"content": out_dict["final_answer"]})
        except Exception as dispatch_err:
            logger.debug(f"Could not dispatch polished answer: {dispatch_err}")
    return out_dict


def synthesizer_node(
    state: AgentState,
    llm: Optional[BaseChatModel] = None,
) -> Dict[str, Any]:
    """
    LangGraph pipeline node that synthesizes the final conversational response (synchronous wrapper).

    Args:
        state: Current AgentState.
        llm: Optional BaseChatModel instance for natural language synthesis.

    Returns:
        Partial AgentState update dictionary containing:
            - final_answer: Markdown formatted response string
            - deterministic_answer: Raw formatted ERP table/data string
            - suggested_followups: Contextual follow-up suggestions generated by LLM or deterministic fallback
            - messages: [AIMessage(content=final_answer)]
    """
    clarification_out = _build_clarification_synthesis_output(state)
    if clarification_out is not None:
        return clarification_out

    (
        user_query,
        status,
        retry_header,
        deterministic_answer,
        actual_model,
        non_tool_questions,
        is_multi_q,
        model,
    ) = _prepare_synthesis_context(state, llm)

    final_text = deterministic_answer
    followups: List[Dict[str, str]] = []

    general_nt_items = [
        nt for nt in non_tool_questions
        if isinstance(nt, dict)
        and (nt.get("category") or "general") == "general"
        and nt.get("question_text")
        and (is_pure_greeting(nt["question_text"]) or is_general_help_inquiry(nt["question_text"]))
    ]
    if model is not None and general_nt_items:
        llm_general_answers: Dict[int, str] = {}
        for nt in general_nt_items:
            try:
                g_resp = model.invoke([
                    SystemMessage(content="Anda adalah Asisten AI FastPrint ERP yang ringkas, jelas, dan bebas AI slop."),
                    HumanMessage(
                        content=(
                            "Balas sapaan atau jelaskan secara singkat kemampuan Asisten AI FastPrint ERP "
                            "(1-3 kalimat, Bahasa Indonesia). Jangan menjawab topik di luar FastPrint ERP:\n"
                            f"{nt['question_text']}"
                        )
                    ),
                ])
                g_content = getattr(g_resp, "content", None)
                if g_content and isinstance(g_content, str) and g_content.strip():
                    llm_general_answers[int(nt.get("question_index") or 0)] = _clean_ai_slop(g_content.strip())
            except Exception as exc:
                logger.debug("LLM sync general sub-question synthesis failed: %s", exc)
        if llm_general_answers:
            updated_det = _build_question_ordered_answer(state, llm_general_answers=llm_general_answers)
            if updated_det:
                deterministic_answer = updated_det
                final_text = deterministic_answer

    (
        is_large_deterministic,
        is_contextual_analysis,
        is_analytical,
        _,
        prompt,
    ) = _build_synthesis_plan(state, user_query, status, deterministic_answer, is_multi_q, model)

    if is_analytical and model is None:
        final_text = _resolve_fallback_text_on_missing_or_failed_llm(state, deterministic_answer, is_analytical)

    if prompt is not None and model is not None:
        try:
            response = model.invoke([
                SystemMessage(content="Anda adalah Asisten AI FastPrint ERP yang presisi, lugas, dan bebas AI slop."),
                HumanMessage(content=prompt),
            ])
            resolved_model = _extract_resolved_model_name(response)
            if resolved_model:
                actual_model = resolved_model
            content = getattr(response, "content", None)
            if content and isinstance(content, str) and content.strip():
                clean_ans, llm_followups = _extract_llm_followups(content.strip())
                if not is_large_deterministic or is_analytical:
                    clean_ans = _clean_ai_slop(clean_ans)
                    if clean_ans:
                        final_text = clean_ans
                if llm_followups:
                    followups = llm_followups
        except Exception as exc:
            logger.warning(f"LLM synthesis failed, falling back to deterministic output: {exc}")
            final_text = _resolve_fallback_text_on_missing_or_failed_llm(state, deterministic_answer, is_analytical)

    out_dict = _finalize_synthesis_output(
        final_text=final_text,
        deterministic_answer=deterministic_answer,
        followups=followups,
        actual_model=actual_model,
        user_query=user_query,
        retry_header=retry_header,
        state=state,
    )
    if is_contextual_analysis and not out_dict.get("verification_status"):
        out_dict["verification_status"] = state.get("verification_status") or "VALID"
    return out_dict

__all__ = [
    "_build_evaluative_verdict",
    "_format_meta_question_answer",
    "_plan_dedup_key",
    "_should_keep_verbatim_deterministic",
    "_parse_sales_target_idr",
    "_parse_iso_date_safe",
    "_is_zero_metric",
    "_append_zero_branch_footnote",
    "_resolve_company_display_label",
    "_resolve_period_bounds",
    "_resolve_product_display_price",
    "_format_contact_table_rows",
    "_format_rupiah",
    "_format_currency",
    "_format_rag_section",
    "_extract_total_count",
    "_append_pagination_footer",
    "_format_single_tool_presentation",
    "_format_multi_company_presentation",
    "_format_empty_plan_notice",
    "_format_non_tool_item_notice",
    "_format_rag_question_section",
    "_render_single_plan_section",
    "_build_question_ordered_answer",
    "format_deterministic_answer",
    "_clean_label_emojis",
    "_clean_ai_slop",
    "_extract_llm_followups",
    "generate_deterministic_followups",
    "_is_routable_followup_query",
    "_finalize_followups",
    "_build_table_query_meta",
    "asynthesizer_node",
    "synthesizer_node",
]
