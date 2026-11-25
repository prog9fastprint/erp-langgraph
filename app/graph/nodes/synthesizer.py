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
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.callbacks import adispatch_custom_event
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.graph.state import AgentState
from app.services.llm import get_chat_model

logger = logging.getLogger("synthesizer_node")


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
        title = chunk.get("title") or chunk.get("metadata", {}).get("source") or f"Dokumen {i}"
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


def _append_pagination_footer(lines: List[str], displayed_count: int, total_count: int) -> None:
    """Append Markdown pagination notice when displayed rows are fewer than total records."""
    if displayed_count > 0 and total_count > displayed_count:
        lines.append(
            f"\n*Menampilkan {displayed_count} dari {total_count} data. "
            f"Gunakan filter periode tanggal atau kata kunci lebih spesifik untuk menelusuri data lainnya.*"
        )


def _format_single_tool_presentation(
    tool: Optional[str],
    tool_result: Any,
    user_query: Optional[str] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
) -> str:
    """Format an individual tool's response payload into Markdown."""
    if not tool or not tool_result:
        return ""

    lines: List[str] = []

    # 1. Vendor Invoices / Journal Entries
    if tool in ("fetch_vendor_invoices", "get_journal_entries"):
        entries = tool_result.get("entries") or []
        total_count = _extract_total_count(tool_result, len(entries))
        calculations = tool_result.get("calculations") or {}

        # Single document detail presentation with line items
        if len(entries) == 1 and entries[0].get("lines"):
            single_entry = entries[0]
            doc_lines = single_entry.get("lines") or []
            num = single_entry.get("number") or single_entry.get("entry_number") or "-"
            tgl = single_entry.get("date") or "-"
            vendor = single_entry.get("vendor_name") or single_entry.get("journal_name") or "-"
            st = single_entry.get("status") or single_entry.get("state") or "posted"
            desc = single_entry.get("description") or single_entry.get("reference") or "-"

            lines.append(f"### Rincian Dokumen Transaksi: `{num}`\n")
            lines.append(f"- **Nomor Dokumen**: `{num}`")
            lines.append(f"- **Tanggal**: {tgl}")
            lines.append(f"- **Rekanan / Vendor**: {vendor}")
            lines.append(f"- **Status**: **{st.upper()}**")
            if desc and desc != "-":
                lines.append(f"- **Keterangan**: {desc}")
            lines.append("")

            is_bill = any(l.get("type") == "bill_line" or "product_name" in l for l in doc_lines)
            if is_bill:
                lines.append("#### Rincian Item Tagihan:")
                lines.append("| No | Nama Produk / Item | Kuantitas | Harga Satuan | Subtotal |")
                lines.append("|:---|:---|:---|:---|:---|")
                for l_idx, l_item in enumerate(doc_lines, start=1):
                    p_name = l_item.get("product_name") or l_item.get("label") or "-"
                    qty = l_item.get("quantity", 1.0)
                    price = _format_rupiah(l_item.get("price", 0.0))
                    subtot = _format_rupiah(l_item.get("subtotal", 0.0))
                    lines.append(f"| {l_idx} | {p_name} | {qty:g} | {price} | {subtot} |")
            else:
                lines.append("#### Rincian Baris Jurnal:")
                lines.append("| No | Kode Akun | Nama Akun | Keterangan | Debit | Kredit |")
                lines.append("|:---|:---|:---|:---|:---|:---|")
                for l_idx, l_item in enumerate(doc_lines, start=1):
                    acc_code = l_item.get("account_code") or "-"
                    acc_name = l_item.get("account_name") or "-"
                    l_desc = l_item.get("description") or "-"
                    deb = _format_rupiah(l_item.get("debit", 0.0))
                    kred = _format_rupiah(l_item.get("credit", 0.0))
                    lines.append(f"| {l_idx} | `{acc_code}` | {acc_name} | {l_desc} | {deb} | {kred} |")

            total_val = single_entry.get("total_amount") or single_entry.get("total_debit", 0.0)
            lines.append("\n#### Ringkasan Kalkulasi:")
            lines.append(f"- **Total Nilai Transaksi**: **{_format_rupiah(total_val)}**")
            return "\n".join(lines)

        lines.append(f"### Laporan Transaksi Tagihan / Jurnal ({total_count} Catatan Ditemukan)\n")
        if entries:
            lines.append("| No | Nomor Dokumen | Tanggal | Rekanan / Vendor | Total Debit | Status |")
            lines.append("|:---|:---|:---|:---|:---|:---|")
            for idx, item in enumerate(entries, start=1):
                num = item.get("number") or item.get("entry_number") or "-"
                tgl = item.get("date") or "-"
                vendor = item.get("vendor_name") or item.get("journal_name") or "-"
                debit = _format_rupiah(item.get("total_debit", 0.0))
                st = item.get("status") or item.get("state") or "posted"
                lines.append(f"| {idx} | `{num}` | {tgl} | {vendor} | {debit} | {st.upper()} |")
            _append_pagination_footer(lines, len(entries), total_count)
        else:
            lines.append("Tidak ada transaksi tercatat pada periode ini.")

        # Summary calculations block (DoD TASK-4.3 preservation)
        if calculations:
            lines.append("\n#### Ringkasan Kalkulasi Total:")
            if "total_amount" in calculations:
                lines.append(f"- **Total Nilai Transaksi**: {_format_rupiah(calculations['total_amount'])}")
            if "total_unpaid" in calculations:
                lines.append(f"- **Total Belum Dibayar**: {_format_rupiah(calculations['total_unpaid'])}")
            if "total_paid" in calculations:
                lines.append(f"- **Total Sudah Dibayar**: {_format_rupiah(calculations['total_paid'])}")
            if "grand_total" in calculations:
                lines.append(f"- **Grand Total Tagihan**: {_format_rupiah(calculations['grand_total'])}")
        return "\n".join(lines)

    # 2. Account Balance
    if tool == "get_account_balance":
        accounts = tool_result.get("accounts") or []
        total_count = _extract_total_count(tool_result, len(accounts))
        totals = tool_result.get("totals") or {}
        lines.append("### Informasi Saldo Chart of Account (Accounting Balance)\n")

        # Tier 1: Single account presentation (Direct KPI + Bullet List)
        if len(accounts) == 1:
            acc = accounts[0]
            code = acc.get("account_code") or "-"
            name = acc.get("account_name") or "-"
            op = _format_rupiah(acc.get("opening_balance", 0.0)) if acc.get("opening_balance") is not None else "-"
            deb = _format_rupiah(acc.get("period_debit", acc.get("total_debit", 0.0)))
            kred = _format_rupiah(acc.get("period_credit", acc.get("total_credit", 0.0)))
            net = _format_rupiah(acc.get("ending_balance", acc.get("net_balance", 0.0)))
            pos = acc.get("balance_position")
            pos_str = f" (Posisi: {pos})" if pos else ""

            lines.append(f"Saldo akhir untuk akun **{name}** (`{code}`) adalah **{net}**{pos_str}.\n")
            lines.append(f"- **Kode Akun**: `{code}`")
            lines.append(f"- **Nama Akun**: {name}")
            if op != "-":
                lines.append(f"- **Saldo Awal**: {op}")
            lines.append(f"- **Mutasi Debit**: {deb}")
            lines.append(f"- **Mutasi Kredit**: {kred}")
            lines.append(f"- **Saldo Akhir**: **{net}**")
            return "\n".join(lines)

        if totals:
            op = _format_rupiah(totals.get("opening_balance", 0.0))
            deb = _format_rupiah(totals.get("period_debit", 0.0))
            kred = _format_rupiah(totals.get("period_credit", 0.0))
            end = _format_rupiah(totals.get("ending_balance", 0.0))
            lines.append("#### Ringkasan Saldo Periode:")
            lines.append(f"- **Saldo Awal**: {op}")
            lines.append(f"- **Mutasi Debit**: {deb}")
            lines.append(f"- **Mutasi Kredit**: {kred}")
            lines.append(f"- **Saldo Akhir**: **{end}**\n")

        if accounts:
            lines.append("| Kode Akun | Nama Akun | Saldo Awal | Mutasi Debit | Mutasi Kredit | Saldo Akhir |")
            lines.append("|:---|:---|:---|:---|:---|:---|")
            for acc in accounts:
                code = acc.get("account_code") or "-"
                name = acc.get("account_name") or "-"
                op = _format_rupiah(acc.get("opening_balance", 0.0)) if acc.get("opening_balance") is not None else "-"
                deb = _format_rupiah(acc.get("period_debit", acc.get("total_debit", 0.0)))
                kred = _format_rupiah(acc.get("period_credit", acc.get("total_credit", 0.0)))
                net = _format_rupiah(acc.get("ending_balance", acc.get("net_balance", 0.0)))
                lines.append(f"| `{code}` | {name} | {op} | {deb} | {kred} | **{net}** |")
            _append_pagination_footer(lines, len(accounts), total_count)
        elif not totals:
            net = _format_rupiah(tool_result.get("net_balance", 0.0))
            deb = _format_rupiah(tool_result.get("total_debit", 0.0))
            kred = _format_rupiah(tool_result.get("total_credit", 0.0))
            lines.append(f"- **Total Debit**: {deb}")
            lines.append(f"- **Total Kredit**: {kred}")
            lines.append(f"- **Saldo Bersih**: **{net}**")
        return "\n".join(lines)

    # 2.1 General Ledger (Buku Besar)
    if tool == "get_general_ledger":
        accounts = tool_result.get("accounts") or []
        d_from = tool_result.get("date_from") or "-"
        d_to = tool_result.get("date_to") or "-"
        lines.append(f"### Buku Besar Akun (General Ledger) — Periode: {d_from} s/d {d_to}\n")
        if not accounts:
            lines.append("Tidak ada transaksi buku besar yang tercatat pada filter ini.")
            return "\n".join(lines)

        for acc in accounts:
            acc_c = acc.get("account_code") or "-"
            acc_n = acc.get("account_name") or "-"
            op_bal = _format_rupiah(acc.get("opening_balance", 0.0))
            cl_bal = _format_rupiah(acc.get("closing_balance", 0.0))
            p_deb = _format_rupiah(acc.get("period_debit", 0.0))
            p_kred = _format_rupiah(acc.get("period_credit", 0.0))

            lines.append(f"#### Akun: `{acc_c}` — {acc_n}")
            lines.append(f"- Saldo Awal: **{op_bal}** | Mutasi Debit: {p_deb} | Mutasi Kredit: {p_kred} | Saldo Akhir: **{cl_bal}**\n")

            items = acc.get("items") or []
            if items:
                lines.append("| Tanggal | No. Entri / Ref | Rekanan | Keterangan | Debit | Kredit | Saldo Berjalan |")
                lines.append("|:---|:---|:---|:---|:---|:---|:---|")
                for it in items:
                    tgl = it.get("date") or "-"
                    num = it.get("entry_number") or "-"
                    ref = it.get("ref") or "-"
                    doc_ref = f"`{num}` ({ref})" if ref and ref != "-" and ref != num else f"`{num}`"
                    partner = it.get("partner_name") or "-"
                    desc = it.get("description") or "-"
                    deb = _format_rupiah(it.get("debit", 0.0))
                    kred = _format_rupiah(it.get("credit", 0.0))
                    run_bal = _format_rupiah(it.get("running_balance", 0.0))
                    lines.append(f"| {tgl} | {doc_ref} | {partner} | {desc} | {deb} | {kred} | **{run_bal}** |")
            else:
                lines.append("*(Tidak ada mutasi transaksi pada periode ini)*")
            lines.append("")
        return "\n".join(lines)

    # 2.2 Profit & Loss (Laba Rugi)
    if tool == "get_profit_and_loss":
        summary = tool_result.get("summary") or {}
        d_from = tool_result.get("date_from") or "-"
        d_to = tool_result.get("date_to") or "-"
        lines.append(f"### Laporan Laba Rugi (Profit & Loss) — Periode: {d_from} s/d {d_to}\n")
        lines.append("| Metrik Keuangan | Nilai Nominal |")
        lines.append("|:---|:---|")
        lines.append(f"| **Pendapatan Usaha (Revenue)** | {_format_rupiah(summary.get('total_revenue', 0.0))} |")
        lines.append(f"| Beban Pokok Penjualan (HPP / COGS) | {_format_rupiah(summary.get('cogs', 0.0))} |")
        lines.append(f"| **Laba Kotor (Gross Profit)** | **{_format_rupiah(summary.get('gross_profit', 0.0))}** |")
        gm_pct = summary.get("gross_margin_percent", 0.0)
        lines.append(f"| *Gross Profit Margin* | *{gm_pct:.1f}%* |")
        lines.append(f"| Beban Operasional | {_format_rupiah(summary.get('operating_expenses', 0.0))} |")
        lines.append(f"| Laba Operasional | {_format_rupiah(summary.get('operating_profit', 0.0))} |")
        lines.append(f"| Pendapatan / Beban Lain-Lain Bersih | {_format_rupiah(summary.get('other_net', 0.0))} |")
        lines.append(f"| **Laba Bersih Periode Berjalan (Net Profit)** | **{_format_rupiah(summary.get('net_profit', 0.0))}** |")

        breakdown = tool_result.get("breakdown") or []
        if breakdown:
            lines.append("\n#### Rincian Hierarki Laba Rugi:")
            lines.append("| Grup Akun | Tipe / Sub-Tipe | Kode Akun | Nama Akun | Saldo |")
            lines.append("|:---|:---|:---|:---|:---|")
            for grp in breakdown:
                grp_name = grp.get("group_name") or grp.get("name") or "-"
                for tp in grp.get("types", []):
                    tp_name = tp.get("type_name") or tp.get("name") or "-"
                    for acc in tp.get("accounts", []):
                        acc_c = acc.get("account_code") or "-"
                        acc_n = acc.get("account_name") or "-"
                        bal = _format_rupiah(acc.get("balance", 0.0))
                        lines.append(f"| {grp_name} | {tp_name} | `{acc_c}` | {acc_n} | {bal} |")
        return "\n".join(lines)

    # 2.3 Balance Sheet (Neraca Keuangan)
    if tool == "get_balance_sheet":
        summary = tool_result.get("summary") or {}
        as_of = tool_result.get("as_of_date") or "-"
        lines.append(f"### Laporan Neraca Keuangan (Balance Sheet) — Posisi per: {as_of}\n")
        lines.append("| Posisi Keuangan | Nilai Nominal |")
        lines.append("|:---|:---|")
        lines.append(f"| **Total Aset (Assets)** | **{_format_rupiah(summary.get('total_assets', 0.0))}** |")
        lines.append(f"| Total Kewajiban (Liabilities) | {_format_rupiah(summary.get('total_liabilities', 0.0))} |")
        lines.append(f"| Total Ekuitas (Equity) | {_format_rupiah(summary.get('total_equity', 0.0))} |")
        lines.append(f"| **Total Kewajiban & Ekuitas** | **{_format_rupiah(summary.get('total_liabilities_and_equity', 0.0))}** |")

        check = summary.get("net_balance_check") or {}
        is_bal = check.get("is_balanced", True)
        diff = check.get("difference", 0.0)
        lines.append("\n#### Status Keseimbangan Neraca (Assets = Liabilities + Equity):")
        if is_bal and abs(diff) < 0.01:
            lines.append("- **Status**: **SEIMBANG (BALANCED)** (Selisih: Rp 0,00)")
        else:
            lines.append(f"- **Status**: **TIDAK SEIMBANG** (Selisih: {_format_rupiah(diff)})")

        breakdown = tool_result.get("breakdown") or []
        if breakdown:
            lines.append("\n#### Rincian Akun Neraca:")
            lines.append("| Kategori | Tipe Aset/Kewajiban | Kode Akun | Nama Akun | Saldo |")
            lines.append("|:---|:---|:---|:---|:---|")
            for grp in breakdown:
                grp_name = grp.get("group_name") or grp.get("name") or "-"
                for tp in grp.get("types", []):
                    tp_name = tp.get("type_name") or tp.get("name") or "-"
                    for acc in tp.get("accounts", []):
                        acc_c = acc.get("account_code") or "-"
                        acc_n = acc.get("account_name") or "-"
                        bal = _format_rupiah(acc.get("balance", 0.0))
                        lines.append(f"| {grp_name} | {tp_name} | `{acc_c}` | {acc_n} | {bal} |")
        return "\n".join(lines)

    # 2.4 Vendor Bills (Tagihan Vendor & AP Aging)
    if tool == "get_vendor_bills":
        bills = tool_result.get("bills") or []
        total_count = _extract_total_count(tool_result, len(bills))
        calculations = tool_result.get("calculations") or {}
        lines.append(f"### Daftar Tagihan Faktur Vendor (Vendor Bills) — ({total_count} Tagihan Ditemukan)\n")
        if bills:
            lines.append("| No | No. Faktur | Tanggal | Jatuh Tempo | Vendor / Rekanan | Total Nilai | Terbayar | Sisa Hutang | Status Bayar | Tempo |")
            lines.append("|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|")
            total_amount_sum = 0.0
            total_residual_sum = 0.0
            overdue_count = 0
            for idx, b in enumerate(bills, start=1):
                num = b.get("number") or "-"
                b_date = b.get("bill_date") or "-"
                due_d = b.get("due_date") or "-"
                v_name = b.get("vendor_name") or "-"
                amt = float(b.get("total_amount", 0.0) or 0.0)
                paid = float(b.get("paid_amount", 0.0) or 0.0)
                res = float(b.get("residual_balance", 0.0) or 0.0)
                p_st = (b.get("payment_status") or "-").upper()
                d_st = (b.get("due_status") or "all").upper()
                if b.get("due_status") == "overdue":
                    overdue_count += 1
                total_amount_sum += amt
                total_residual_sum += res
                lines.append(f"| {idx} | `{num}` | {b_date} | {due_d} | {v_name} | {_format_rupiah(amt)} | {_format_rupiah(paid)} | **{_format_rupiah(res)}** | {p_st} | {d_st} |")

            # Prefer pre-truncated server/sanitization aggregates across all rows over visible-page sum
            if calculations:
                if calculations.get("total_amount") is not None:
                    total_amount_sum = float(calculations["total_amount"])
                if calculations.get("total_residual") is not None:
                    total_residual_sum = float(calculations["total_residual"])
                if calculations.get("overdue_count") is not None:
                    overdue_count = int(calculations["overdue_count"])

            _append_pagination_footer(lines, len(bills), total_count)

            lines.append("\n#### Ringkasan Hutang Dagang (AP):")
            lines.append(f"- **Total Tagihan Faktur**: {_format_rupiah(total_amount_sum)}")
            lines.append(f"- **Total Sisa Hutang Belum Lunas**: **{_format_rupiah(total_residual_sum)}**")
            if overdue_count > 0:
                lines.append(f"- **Tagihan Lewat Jatuh Tempo (Overdue)**: **{overdue_count} faktur**")
        else:
            lines.append("Tidak ada tagihan faktur vendor pada kriteria ini.")
        return "\n".join(lines)

    # 2.5 Bank Reconciliation Summary
    if tool == "get_bank_reconciliation_summary":
        summary = tool_result.get("summary") or {}
        j_name = tool_result.get("journal_name") or "Bank Utama"
        acc_c = tool_result.get("bank_account_code") or "-"
        acc_n = tool_result.get("bank_account_name") or "-"
        d_from = tool_result.get("date_from") or "-"
        d_to = tool_result.get("date_to") or "-"

        lines.append(f"### Ringkasan Rekonsiliasi Bank — {j_name}\n")
        lines.append(f"- **Buku Jurnal**: {j_name} (`{acc_c}` — {acc_n})")
        lines.append(f"- **Periode**: {d_from} s/d {d_to}\n")

        stmt_bal = summary.get("bank_statement_balance", 0.0)
        book_bal = summary.get("erp_book_balance", 0.0)
        diff = summary.get("difference", 0.0)
        unrec_stmt = summary.get("unreconciled_statement_lines_count", 0)
        unrec_gl = summary.get("unreconciled_gl_lines_count", 0)

        lines.append("| Indikator Rekonsiliasi | Nilai Saldo |")
        lines.append("|:---|:---|")
        lines.append(f"| Saldo Rekening Koran Bank | {_format_rupiah(stmt_bal)} |")
        lines.append(f"| Saldo Buku Besar ERP (GL Book) | {_format_rupiah(book_bal)} |")
        lines.append(f"| **Selisih Rekonsiliasi (Difference)** | **{_format_rupiah(diff)}** |")

        lines.append("\n#### Status Transaksi Belum Terekonsiliasi:")
        lines.append(f"- Transaksi Rekening Koran Belum Cocok: **{unrec_stmt} baris**")
        lines.append(f"- Transaksi Jurnal ERP Belum Cocok: **{unrec_gl} baris**")
        if abs(diff) < 0.01 and unrec_stmt == 0 and unrec_gl == 0:
            lines.append("- **Status**: **Terekonsiliasi Sempurna (Fully Reconciled)**")
        else:
            lines.append("- **Status**: **Memerlukan Tindak Lanjut Pencocokan Transaksi**")
        return "\n".join(lines)

    # 3. Stock Level
    if tool == "get_stock_level":
        lines.append("### Informasi Ketersediaan Stok Barang (Inventory)\n")
        if isinstance(tool_result, dict) and "products" in tool_result and isinstance(tool_result["products"], list):
            items = tool_result["products"]
        elif isinstance(tool_result, list):
            items = tool_result
        else:
            items = [tool_result]
        total_count = _extract_total_count(tool_result, len(items))

        # Tier 1: Single item stock presentation (Direct KPI + Bullet List)
        if len(items) == 1:
            st_item = items[0]
            code = st_item.get("code") or st_item.get("sku") or "-"
            name = st_item.get("name") or st_item.get("product_name") or "-"
            comp = (
                st_item.get("company_name")
                or (tool_result.get("company_name") if isinstance(tool_result, dict) else None)
                or f"Company #{st_item.get('company_id', tool_result.get('company_id', 1) if isinstance(tool_result, dict) else 1)}"
            )
            loc = (
                st_item.get("location_name")
                or (tool_result.get("location_name") if isinstance(tool_result, dict) else None)
                or (f"Location #{st_item.get('location_id')}" if st_item.get("location_id") else "Semua Lokasi")
            )
            qty = st_item.get("qty_on_hand") or st_item.get("quantity") or 0.0

            lines.append(f"Total stok fisik untuk **{name}** (SKU `{code}`) saat ini adalah **{qty} unit** ({comp} - {loc}).\n")
            lines.append(f"- **Nama Produk**: {name}")
            lines.append(f"- **Kode / SKU**: `{code}`")
            lines.append(f"- **Perusahaan**: {comp}")
            lines.append(f"- **Lokasi**: {loc}")
            lines.append(f"- **Kuantitas Fisik On Hand**: **{qty} unit**")

            # Render breakdown if available
            single_bd = (
                (tool_result.get("breakdown") if isinstance(tool_result, dict) else None)
                or (st_item.get("breakdown") if isinstance(st_item, dict) else None)
            )
            if single_bd:
                lines.append("\n#### Rincian Stok per Lokasi / Rak:")
                lines.append("| Gudang | Lokasi / Rak | Kuantitas |")
                lines.append("|:---|:---|:---|")
                for bd in single_bd:
                    wh = bd.get("warehouse_name") or f"Gudang #{bd.get('warehouse_id', '-')}"
                    loc_n = bd.get("location_name") or f"Rak #{bd.get('location_id', '-')}"
                    b_qty = bd.get("qty", 0.0)
                    lines.append(f"| {wh} | {loc_n} | **{b_qty:g} unit** |")

            # Missing requested SKU warning callout
            if extracted_params and isinstance(extracted_params, dict):
                req_codes = extracted_params.get("codes")
                if req_codes and isinstance(req_codes, list):
                    found_codes = {
                        str(it.get("code") or it.get("sku") or "").strip().lower()
                        for it in items
                        if isinstance(it, dict)
                    }
                    missing_codes = [c for c in req_codes if str(c).strip().lower() not in found_codes]
                    if missing_codes:
                        lines.append("")
                        for mc in missing_codes:
                            lines.append(f"> **Catatan:** Kode SKU `{mc}` tidak ditemukan pada master data produk FastPrint ERP.")
            return "\n".join(lines)

        lines.append("| SKU / Kode | Nama Produk | Perusahaan | Lokasi / Gudang | Qty Fisik |")
        lines.append("|:---|:---|:---|:---|:---|")
        for st_item in items:
            code = st_item.get("code") or st_item.get("sku") or "-"
            name = st_item.get("name") or st_item.get("product_name") or "-"
            comp = (
                st_item.get("company_name")
                or (tool_result.get("company_name") if isinstance(tool_result, dict) else None)
                or f"Company #{st_item.get('company_id', tool_result.get('company_id', 1) if isinstance(tool_result, dict) else 1)}"
            )
            loc = (
                st_item.get("location_name")
                or (tool_result.get("location_name") if isinstance(tool_result, dict) else None)
                or (f"Location #{st_item.get('location_id')}" if st_item.get("location_id") else "Semua Lokasi")
            )
            qty = st_item.get("qty_on_hand") or st_item.get("quantity") or 0.0
            lines.append(f"| `{code}` | {name} | {comp} | {loc} | **{qty} unit** |")

        _append_pagination_footer(lines, len(items), total_count)

        # Render breakdown if available (single product multi-location breakdown)
        if isinstance(tool_result, dict) and tool_result.get("breakdown"):
            lines.append("\n#### Rincian Stok per Lokasi / Rak:")
            lines.append("| Gudang | Lokasi / Rak | Kuantitas |")
            lines.append("|:---|:---|:---|")
            for bd in tool_result["breakdown"]:
                wh = bd.get("warehouse_name") or f"Gudang #{bd.get('warehouse_id', '-')}"
                loc_n = bd.get("location_name") or f"Rak #{bd.get('location_id', '-')}"
                b_qty = bd.get("qty", 0.0)
                lines.append(f"| {wh} | {loc_n} | **{b_qty:g} unit** |")
        elif any(isinstance(it, dict) and it.get("breakdown") for it in items):
            lines.append("\n#### Rincian Stok per Lokasi / Rak:")
            for it in items:
                it_bd = it.get("breakdown") if isinstance(it, dict) else None
                if it_bd:
                    p_code = it.get("code") or it.get("sku") or "-"
                    p_name = it.get("name") or "-"
                    lines.append(f"\n**Produk: `{p_code}` — {p_name}**")
                    lines.append("| Gudang | Lokasi / Rak | Kuantitas |")
                    lines.append("|:---|:---|:---|")
                    for bd in it_bd:
                        wh = bd.get("warehouse_name") or f"Gudang #{bd.get('warehouse_id', '-')}"
                        loc_n = bd.get("location_name") or f"Rak #{bd.get('location_id', '-')}"
                        b_qty = bd.get("qty", 0.0)
                        lines.append(f"| {wh} | {loc_n} | **{b_qty:g} unit** |")

        # Missing requested SKU warning callout
        if extracted_params and isinstance(extracted_params, dict):
            req_codes = extracted_params.get("codes")
            if req_codes and isinstance(req_codes, list):
                found_codes = {
                    str(it.get("code") or it.get("sku") or "").strip().lower()
                    for it in items
                    if isinstance(it, dict)
                }
                missing_codes = [c for c in req_codes if str(c).strip().lower() not in found_codes]
                if missing_codes:
                    lines.append("")
                    for mc in missing_codes:
                        lines.append(f"> **Catatan:** Kode SKU `{mc}` tidak ditemukan pada master data produk FastPrint ERP.")

        return "\n".join(lines)

    # 4. Product Info
    if tool == "get_product_info":
        def _resolve_product_display_price(prod: Dict[str, Any]) -> float:
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

        products = tool_result.get("products") if isinstance(tool_result, dict) else (
            tool_result if isinstance(tool_result, list) else [tool_result]
        )
        total_count = _extract_total_count(tool_result, len(products))
        company_label = tool_result.get("company_name") if isinstance(tool_result, dict) else None
        q_lower = (user_query or "").lower()
        is_price_query = any(k in q_lower for k in ("harga", "harganya", "pricelist", "biaya", "tarip", "tarif", "berapa"))

        if products:
            if is_price_query:
                # Opsi 1: Kartu/Tabel Ringkas Khusus Harga (Disepakati di /grill-me)
                if len(products) == 1:
                    p = products[0]
                    p_name = p.get("name") or p.get("display_name") or "-"
                    sku = p.get("code") or p.get("sku") or p.get("default_code") or "-"
                    cat = p.get("category")
                    cat_name = cat.get("name", "-") if isinstance(cat, dict) else (cat or "-")
                    uom = p.get("uom")
                    uom_name = uom.get("name", "Unit") if isinstance(uom, dict) else (uom or "Unit")
                    price_val = _resolve_product_display_price(p)
                    cost_val = p.get("default_cost") or p.get("cost")
                    pl_prices = p.get("pricelist_prices") or []

                    lines.append("### Informasi Harga Produk\n")
                    lines.append("| Atribut | Rincian |")
                    lines.append("|:---|:---|")
                    lines.append(f"| **Nama Produk** | {p_name} |")
                    lines.append(f"| **Kode / SKU** | `{sku}` |")
                    if company_label:
                        lines.append(f"| **Cabang** | {company_label} |")
                    lines.append(f"| **Harga Jual** | **{_format_rupiah(price_val)}** / {uom_name} |")
                    if cost_val is not None:
                        lines.append(f"| **Estimasi HPP** | {_format_rupiah(cost_val)} |")
                    lines.append(f"| **Kategori** | {cat_name} |")
                    if pl_prices:
                        lines.append("\n#### Daftar Harga Pricelist Cabang:")
                        lines.append("| Pricelist | Harga | Min. Qty |")
                        lines.append("|:---|:---|:---:|")
                        for pl in pl_prices:
                            lines.append(f"| {pl.get('pricelist_name', '-')} | **{_format_rupiah(pl.get('price', 0))}** | {pl.get('min_qty', 1)} |")
                    scope_note = (
                        f" di **{company_label}**"
                        if company_label and pl_prices
                        else (f" (harga master default; belum ada pricelist khusus {company_label})" if company_label else "")
                    )
                    lines.append(f"\nHarga jual resmi untuk produk **{p_name}** (SKU `{sku}`){scope_note} tercatat sebesar **{_format_rupiah(price_val)}** per {uom_name}.")
                else:
                    lines.append("### Informasi Perbandingan Harga Produk\n")
                    lines.append("| No | Nama Produk | Kode / SKU | Satuan (UoM) | Harga Jual | Estimasi HPP |")
                    lines.append("|:---:|:---|:---|:---:|:---:|:---:|")
                    for idx, p in enumerate(products, 1):
                        p_name = p.get("name") or p.get("display_name") or "-"
                        sku = p.get("code") or p.get("sku") or p.get("default_code") or "-"
                        uom = p.get("uom")
                        uom_name = uom.get("name", "Unit") if isinstance(uom, dict) else (uom or "Unit")
                        price_val = _resolve_product_display_price(p)
                        cost_val = p.get("default_cost") or p.get("cost")
                        cost_str = _format_rupiah(cost_val) if cost_val is not None else "-"
                        lines.append(f"| {idx} | {p_name} | `{sku}` | {uom_name} | **{_format_rupiah(price_val)}** | {cost_str} |")
                    _append_pagination_footer(lines, len(products), total_count)
            else:
                lines.append("### Informasi Master Produk (Catalog)\n")
                for p in products:
                    lines.append(f"- **Nama Produk**: {p.get('name', '-')}")
                    lines.append(f"  - Kode / SKU: `{p.get('code') or p.get('sku', '-')}`")
                    cat = p.get("category")
                    if isinstance(cat, dict):
                        lines.append(f"  - Kategori: {cat.get('name', '-')}")
                    elif cat:
                        lines.append(f"  - Kategori: {cat}")
                    uom = p.get("uom")
                    if isinstance(uom, dict):
                        lines.append(f"  - Satuan (UoM): {uom.get('name', '-')}")
                    elif uom:
                        lines.append(f"  - Satuan (UoM): {uom}")
                    if any(k in p for k in ("effective_sales_price", "default_sales_price", "last_unit_price", "price", "sales_price", "list_price")):
                        price_val = _resolve_product_display_price(p)
                        lines.append(f"  - Harga Jual: {_format_rupiah(price_val)}")
                    if p.get("default_cost") is not None:
                        lines.append(f"  - Estimasi HPP: {_format_rupiah(p.get('default_cost'))}")
                _append_pagination_footer(lines, len(products), total_count)
        return "\n".join(lines)

    # 5. Locations
    if tool == "get_locations":
        locs = tool_result.get("locations") if isinstance(tool_result, dict) else (
            tool_result if isinstance(tool_result, list) else [tool_result]
        )
        total_found = _extract_total_count(tool_result, len(locs))
        lines.append(f"### Informasi Daftar Lokasi & Gudang ({total_found} Lokasi Ditemukan)\n")
        if locs:
            lines.append("| ID | Nama Lokasi | Gudang Induk | Tipe | Kategori |")
            lines.append("|:---|:---|:---|:---|:---|")
            for l in locs:
                lid = l.get("id", "-")
                lname = l.get("name", "-")
                wh = l.get("warehouse_name") or "-"
                ltype = (l.get("type") or "-").capitalize()
                lcat = (l.get("category") or "-").capitalize()
                lines.append(f"| `{lid}` | {lname} | {wh} | {ltype} | {lcat} |")
            _append_pagination_footer(lines, len(locs), total_found)
        return "\n".join(lines)

    # 6. Stock History (Kartu Stok / Buku Besar Persediaan)
    if tool == "get_stock_history":
        hist = tool_result.get("history") if isinstance(tool_result, dict) else (
            tool_result if isinstance(tool_result, list) else [tool_result]
        )
        total_count = _extract_total_count(tool_result, len(hist))
        lines.append(f"### Riwayat Kartu Stok / Pergerakan Barang ({total_count} Catatan Ditemukan)\n")
        if hist:
            lines.append("| No | Tanggal | Referensi | Produk / SKU | Operasi | Lokasi / Rak | Mutasi Qty | Saldo Akhir |")
            lines.append("|:---|:---|:---|:---|:---|:---|:---|:---|")
            for idx, h in enumerate(hist, start=1):
                tgl = h.get("date") or "-"
                ref = h.get("reference") or "-"
                p_label = h.get("code") or h.get("product_name") or "-"
                op = h.get("operation_type") or "-"
                loc = h.get("location_name") or "-"
                qty_done = h.get("qty_done", 0.0)
                bal = h.get("balance")
                bal_str = f"{bal:g}" if bal is not None else "-"
                lines.append(f"| {idx} | {tgl} | `{ref}` | {p_label} | {op} | {loc} | **{qty_done:+g}** | {bal_str} |")
            _append_pagination_footer(lines, len(hist), total_count)
        else:
            lines.append("Tidak ada riwayat pergerakan stok tercatat pada periode ini.")
        return "\n".join(lines)

    # 7. Transfers (Surat Jalan / Penerimaan / Mutasi Gudang)
    if tool == "get_transfers":
        transfers = tool_result.get("transfers") if isinstance(tool_result, dict) else (
            tool_result if isinstance(tool_result, list) else [tool_result]
        )
        total_count = _extract_total_count(tool_result, len(transfers))
        lines.append(f"### Dokumen Transfer & Mutasi Persediaan ({total_count} Dokumen Ditemukan)\n")
        if transfers:
            lines.append("| No | No. Dokumen | Tanggal | Tipe Operasi | Dokumen Asal | Rekanan | Status |")
            lines.append("|:---|:---|:---|:---|:---|:---|:---|")
            for idx, t in enumerate(transfers, start=1):
                doc_num = t.get("ref") or t.get("name") or "-"
                tgl = t.get("date") or "-"
                op = t.get("operation_type") or "-"
                src = t.get("source_document") or "-"
                contact = t.get("contact_name") or "-"
                st = (t.get("status") or "-").upper()
                lines.append(f"| {idx} | `{doc_num}` | {tgl} | {op} | {src} | {contact} | **{st}** |")

            _append_pagination_footer(lines, len(transfers), total_count)

            # If single transfer with item lines
            if len(transfers) == 1 and transfers[0].get("lines"):
                lines.append("\n#### Rincian Baris Barang:")
                lines.append("| No | Kode Produk | Nama Produk | Kuantitas | Satuan |")
                lines.append("|:---|:---|:---|:---|:---|")
                for l_idx, line in enumerate(transfers[0]["lines"], start=1):
                    p_code = line.get("product_code") or "-"
                    p_name = line.get("product_name") or "-"
                    l_qty = line.get("quantity", 0.0)
                    uom_str = line.get("uom") or "unit"
                    lines.append(f"| {l_idx} | `{p_code}` | {p_name} | {l_qty:g} | {uom_str} |")
        else:
            lines.append("Tidak ada dokumen transfer yang ditemukan.")
        return "\n".join(lines)

    # 8. Low Stock Alerts
    if tool == "get_low_stock_alerts":
        alerts = tool_result.get("alerts") if isinstance(tool_result, dict) else (
            tool_result if isinstance(tool_result, list) else [tool_result]
        )
        total_found = _extract_total_count(tool_result, len(alerts))
        exclude_zero = isinstance(tool_result, dict) and bool(tool_result.get("exclude_zero"))
        oos_count = int(tool_result.get("out_of_stock_count") or 0) if isinstance(tool_result, dict) else 0

        if exclude_zero:
            lines.append(f"### Produk dengan Stok Tersedikit (Stok Positif Terendah — {total_found} Item)\n")
            if oos_count > 0:
                lines.append(f"Terdapat **{oos_count} SKU** lain berstatus habis/minus (<= 0 unit) pada cabang ini.\n")
        else:
            lines.append(f"### Peringatan Stok Menipis / Kritis ({total_found} Produk Perlu Perhatian)\n")

        if alerts:
            lines.append("| No | SKU | Nama Produk | Gudang | Stok Saat Ini | Batas Minimum | Status |")
            lines.append("|:---|:---|:---|:---|:---|:---|:---|")
            for idx, a in enumerate(alerts, start=1):
                code = a.get("code") or "-"
                name = a.get("name") or "-"
                wh = a.get("warehouse_name") or "-"
                qty = a.get("qty_on_hand", 0.0)
                thresh = a.get("threshold", 0.0)
                status_label = "HABIS" if qty <= 0 else "MENIPIS"
                lines.append(f"| {idx} | `{code}` | {name} | {wh} | **{qty:g} unit** | {thresh:g} | **{status_label}** |")
            _append_pagination_footer(lines, len(alerts), total_found)
        elif exclude_zero:
            lines.append("Tidak ditemukan produk dengan stok positif (> 0 unit) yang tersisa pada cabang ini.")
        else:
            lines.append("Semua stok produk terpantau dalam batas aman.")
        return "\n".join(lines)

    # 9. Stock Adjustments (Stock Opname)
    if tool == "get_stock_adjustments":
        adjustments = tool_result.get("adjustments") if isinstance(tool_result, dict) else (
            tool_result if isinstance(tool_result, list) else [tool_result]
        )
        total_count = _extract_total_count(tool_result, len(adjustments))
        lines.append(f"### Catatan Penyesuaian Stock Opname ({total_count} Catatan Ditemukan)\n")
        if adjustments:
            lines.append("| No | Referensi | Produk | Lokasi / Rak | Qty Sistem | Qty Fisik | Selisih | Status |")
            lines.append("|:---|:---|:---|:---|:---|:---|:---|:---|")
            for idx, adj in enumerate(adjustments, start=1):
                ref = adj.get("reference") or "-"
                p_name = adj.get("product_name") or "-"
                loc = adj.get("location_name") or "-"
                theo = adj.get("theoretical_qty", 0.0)
                real = adj.get("real_qty", 0.0)
                diff = adj.get("difference", 0.0)
                st = (adj.get("status") or "-").upper()
                lines.append(f"| {idx} | `{ref}` | {p_name} | {loc} | {theo:g} | {real:g} | **{diff:+g}** | {st} |")
            _append_pagination_footer(lines, len(adjustments), total_count)
        else:
            lines.append("Tidak ada catatan penyesuaian stock opname pada filter ini.")
        return "\n".join(lines)

    # 10. Scrap Records
    if tool == "get_scrap_records":
        scraps = tool_result.get("scraps") if isinstance(tool_result, dict) else (
            tool_result if isinstance(tool_result, list) else [tool_result]
        )
        total_count = _extract_total_count(tool_result, len(scraps))
        lines.append(f"### Catatan Barang Rusak / Scrap Order ({total_count} Catatan Ditemukan)\n")
        if scraps:
            lines.append("| No | Referensi | Produk | Lokasi Asal | Kuantitas Cacat | Alasan / Keterangan | Tanggal |")
            lines.append("|:---|:---|:---|:---|:---|:---|:---|")
            for idx, sc in enumerate(scraps, start=1):
                ref = sc.get("reference") or "-"
                p_name = sc.get("product_name") or "-"
                loc = sc.get("source_location_name") or "-"
                qty = sc.get("quantity", 0.0)
                reason = sc.get("reason") or "-"
                tgl = sc.get("date") or "-"
                lines.append(f"| {idx} | `{ref}` | {p_name} | {loc} | **{qty:g} unit** | {reason} | {tgl} |")
            _append_pagination_footer(lines, len(scraps), total_count)
        else:
            lines.append("Tidak ada catatan barang rusak / scrap pada filter ini.")
        return "\n".join(lines)

    # 11. Warehouses
    if tool == "get_warehouses":
        whs = tool_result.get("warehouses") if isinstance(tool_result, dict) else (
            tool_result if isinstance(tool_result, list) else [tool_result]
        )
        total_found = _extract_total_count(tool_result, len(whs))
        lines.append(f"### Daftar Master Gudang Perusahaan ({total_found} Gudang Ditemukan)\n")
        if whs:
            lines.append("| ID | Nama Gudang | Kode Referensi | Perusahaan | Lokasi Internal Terkait |")
            lines.append("|:---|:---|:---|:---|:---|")
            for w in whs:
                wid = w.get("id", "-")
                wname = w.get("name", "-")
                code = w.get("code") or "-"
                comp = f"Company #{w.get('company_id', 1)}"
                locs_count = len(w.get("locations") or [])
                lines.append(f"| `{wid}` | {wname} | `{code}` | {comp} | {locs_count} lokasi |")
            _append_pagination_footer(lines, len(whs), total_found)
        return "\n".join(lines)

    # 12. Product Categories
    if tool == "get_product_categories":
        cats = tool_result.get("categories") if isinstance(tool_result, dict) else (
            tool_result if isinstance(tool_result, list) else [tool_result]
        )
        total_found = _extract_total_count(tool_result, len(cats))
        lines.append(f"### Daftar Master Kategori Produk ({total_found} Kategori Ditemukan)\n")
        if cats:
            lines.append("| ID | Nama Kategori | Metode Costing | UoM Standar |")
            lines.append("|:---|:---|:---|:---|")
            for c in cats:
                cid = c.get("id", "-")
                cname = c.get("name", "-")
                cm = c.get("costing_method") or "-"
                uom_n = c.get("standard_uom") or "-"
                lines.append(f"| `{cid}` | {cname} | {cm} | {uom_n} |")
            _append_pagination_footer(lines, len(cats), total_found)
        return "\n".join(lines)

    # 13. Sales: Top Selling Products
    if tool == "get_top_selling_products":
        products = tool_result.get("products") if isinstance(tool_result, dict) else (
            tool_result if isinstance(tool_result, list) else []
        )
        summary = tool_result.get("summary") or {}
        period = tool_result.get("period") or {}
        period_str = ""
        if period.get("date_from") or period.get("date_to"):
            df = period.get("date_from") or ""
            dt = period.get("date_to") or ""
            period_str = f" (Periode {df} s/d {dt})".strip()

        lines.append(f"### Laporan Produk Penjualan Tertinggi ({len(products)} Produk Ditemukan{period_str})\n")
        if products:
            lines.append("| No | Kode / SKU | Nama Produk | Kategori | Kuantitas Terjual | Total Pendapatan |")
            lines.append("|:---:|:---|:---|:---|:---:|:---|")
            for p in products:
                idx = p.get("rank") or "-"
                sku = p.get("code") or "-"
                p_name = p.get("name") or "-"
                cat = p.get("category") or "-"
                qty = float(p.get("total_qty") or 0)
                rev = float(p.get("total_revenue") or 0.0)
                lines.append(f"| {idx} | `{sku}` | {p_name} | {cat} | {qty:g} unit | **{_format_rupiah(rev)}** |")

            tot_rev = float(summary.get("total_revenue") or sum(float(p.get("total_revenue") or 0.0) for p in products))
            tot_qty = float(summary.get("total_qty") or sum(float(p.get("total_qty") or 0) for p in products))
            lines.append(f"| | **TOTAL** | | | **{tot_qty:g} unit** | **{_format_rupiah(tot_rev)}** |")
        channel_breakdown = tool_result.get("channel_breakdown") if isinstance(tool_result, dict) else None
        if channel_breakdown:
            lines.append("\n#### Ringkasan per Channel Penjualan:\n")
            lines.append("| No | Channel | Jumlah Order | Kuantitas Terjual | Total Omzet |")
            lines.append("|:---:|:---|:---:|:---:|:---|")
            for c_idx, ch in enumerate(channel_breakdown, start=1):
                ch_name = ch.get("channel_name") or "Direct / Manual"
                ch_orders = int(ch.get("order_count") or 0)
                ch_qty = float(ch.get("total_qty") or 0)
                ch_rev = float(ch.get("total_revenue") or 0.0)
                lines.append(
                    f"| {c_idx} | **{ch_name}** | {ch_orders:,} | {ch_qty:g} unit | **{_format_rupiah(ch_rev)}** |"
                )
        return "\n".join(lines)

    # 14. Sales: Sales Orders List
    if tool == "get_sales_orders":
        orders = tool_result.get("orders") if isinstance(tool_result, dict) else (
            tool_result if isinstance(tool_result, list) else []
        )
        total_cnt = tool_result.get("total_count", len(orders)) if isinstance(tool_result, dict) else len(orders)
        tot_sum = float(tool_result.get("total_amount_sum") or sum(float(o.get("total_amount") or 0.0) for o in orders))
        st_filter = tool_result.get("status_filter", "confirmed") if isinstance(tool_result, dict) else "confirmed"

        lines.append(f"### Daftar Pesanan Penjualan ({len(orders)} dari {total_cnt} Dokumen, Status: `{st_filter}`)\n")
        if orders:
            lines.append("| No | No. SO | Tanggal | Pelanggan | Channel | Status | Total Nominal |")
            lines.append("|:---:|:---|:---:|:---|:---|:---:|:---|")
            for idx, o in enumerate(orders, start=1):
                so_num = o.get("so_number") or "-"
                dt = o.get("quotation_date") or "-"
                cust = o.get("customer_name") or "-"
                ch = o.get("channel_name") or "Direct"
                st = o.get("status") or "-"
                tot = float(o.get("total_amount") or 0.0)
                lines.append(f"| {idx} | `{so_num}` | {dt} | {cust} | {ch} | `{st}` | **{_format_rupiah(tot)}** |")
            lines.append(f"| | **TOTAL** | | | | | **{_format_rupiah(tot_sum)}** |")
        else:
            lines.append("*Tidak ada dokumen sales order yang sesuai dengan kriteria filter.*")
        return "\n".join(lines)

    # 15. Sales: Sales Order Detail
    if tool == "get_sales_order_detail":
        if not isinstance(tool_result, dict) or not tool_result.get("id"):
            return "Informasi detail sales order tidak ditemukan."

        so_num = tool_result.get("so_number") or "-"
        alias = tool_result.get("so_number_alias")
        alias_str = f" (`{alias}`)" if alias else ""
        c_name = tool_result.get("company_name") or "-"
        status = tool_result.get("status") or "-"
        q_date = tool_result.get("quotation_date") or "-"
        pay_term = tool_result.get("payment_terms") or "-"
        channel = tool_result.get("channel") or "Direct"
        deliv = tool_result.get("delivery_method") or "-"
        resi = tool_result.get("manual_resi")
        packing = "Selesai" if tool_result.get("status_packing") else "Belum Packing"

        cust = tool_result.get("customer") or {}
        cust_name = cust.get("name") or "-"
        cust_phone = cust.get("sales_phone") or "-"

        fin = tool_result.get("financial_summary") or {}
        tot_amount = float(fin.get("total_amount") or 0.0)
        disc_nom = float(fin.get("discount_nominal") or 0.0)

        lines.append(f"### Detail Sales Order: `{so_num}`{alias_str}\n")
        lines.append(f"- **Cabang / Perusahaan**: {c_name}")
        lines.append(f"- **Status Dokumen**: `{status}` | **Status Packing**: `{packing}`")
        lines.append(f"- **Tanggal Order**: {q_date} | **Termin Pembayaran**: {pay_term}")
        lines.append(f"- **Pelanggan**: **{cust_name}** (Telp: {cust_phone})")
        lines.append(f"- **Channel**: {channel} | **Metode Pengiriman**: {deliv}")
        if resi:
            lines.append(f"- **No. Resi Kurir**: `{resi}`")

        # Marketplace block if applicable
        mp = tool_result.get("marketplace")
        if mp:
            mp_acc = mp.get("marketplace_account") or "-"
            mp_inv = mp.get("mp_invoice_number") or "-"
            awb = mp.get("awb_number") or "-"
            buyer = mp.get("buyer_name") or "-"
            carrier = mp.get("carrier_name") or "-"
            lines.append(f"\n> **Data Marketplace ({mp_acc})**:")
            lines.append(f"> - No. Invoice MP: `{mp_inv}` | Resi AWB: `{awb}`")
            lines.append(f"> - Pembeli: {buyer} | Kurir: {carrier}")

        # Items Table
        lines_data = tool_result.get("lines") or []
        lines.append(f"\n#### Rincian Produk ({len(lines_data)} Item):\n")
        if lines_data:
            lines.append("| No | SKU | Nama Produk | Qty | Harga Satuan | Diskon | Subtotal |")
            lines.append("|:---:|:---|:---|:---:|:---|:---|:---|")
            for idx, li in enumerate(lines_data, start=1):
                sku = li.get("sku") or "-"
                p_name = li.get("product_name") or "-"
                qty = li.get("qty") or 0
                uom = li.get("uom") or "Pcs"
                up = float(li.get("unit_price") or 0.0)
                disc = float(li.get("disc_nominal") or 0.0)
                sub = float(li.get("sub_total") or 0.0)
                lines.append(f"| {idx} | `{sku}` | {p_name} | {qty} {uom} | {_format_rupiah(up)} | {_format_rupiah(disc)} | **{_format_rupiah(sub)}** |")

            lines.append(f"\n- **Total Diskon**: {_format_rupiah(disc_nom)}")
            lines.append(f"- **Total Tagihan**: **{_format_rupiah(tot_amount)}**")
        return "\n".join(lines)

    # 16. Sales: Sales Summary & KPI
    if tool == "get_sales_summary":
        if not isinstance(tool_result, dict):
            return "Ringkasan data penjualan tidak ditemukan."

        period = tool_result.get("period") or {}
        df = period.get("date_from") or ""
        dt = period.get("date_to") or ""
        p_str = f" (Periode {df} s/d {dt})" if (df or dt) else ""

        conf = tool_result.get("confirmed_metrics") or {}
        real_rev = float(conf.get("realized_revenue") or 0.0)
        orders_cnt = int(conf.get("orders_count") or 0)
        aov = float(conf.get("average_order_value") or 0.0)

        pipe = tool_result.get("pipeline_metrics") or {}
        pipe_cnt = int(pipe.get("quotation_orders_count") or 0)
        pipe_val = float(pipe.get("quotation_potential_value") or 0.0)

        canc = tool_result.get("cancelled_metrics") or {}
        canc_cnt = int(canc.get("cancelled_orders_count") or 0)
        canc_val = float(canc.get("cancelled_value") or 0.0)

        lines.append(f"### Ringkasan Eksekutif Penjualan & Omzet{p_str}\n")
        lines.append(f"- **Total Omzet Terealisasi**: **{_format_rupiah(real_rev)}**")
        lines.append(f"- **Volume Pesanan Dikonfirmasi**: **{orders_cnt:,} pesanan**")
        lines.append(f"- **Average Order Value (AOV)**: **{_format_rupiah(aov)}** / order")
        lines.append(f"- **Pipeline Penawaran (Quotation)**: {pipe_cnt} draft ({_format_rupiah(pipe_val)})")
        lines.append(f"- **Pesanan Dibatalkan (Cancel)**: {canc_cnt} order ({_format_rupiah(canc_val)})\n")

        channels = tool_result.get("channels_breakdown") or []
        if channels:
            lines.append("#### Distribusi Performa Channel Penjualan:\n")
            lines.append("| No | Channel Penjualan | Jumlah Pesanan | Total Omzet | Kontribusi |")
            lines.append("|:---:|:---|:---:|:---|:---:|")
            for idx, ch in enumerate(channels, start=1):
                ch_name = ch.get("channel_name") or "-"
                cnt = ch.get("order_count") or 0
                ch_rev = float(ch.get("total_revenue") or 0.0)
                contrib = (ch_rev / real_rev * 100) if real_rev > 0 else 0.0
                lines.append(f"| {idx} | **{ch_name}** | {cnt} order | {_format_rupiah(ch_rev)} | {contrib:.1f}% |")

        return "\n".join(lines)

    # 17. Marketplace: Account Health
    if tool == "get_account_health":
        summary = tool_result.get("summary") or {}
        accounts = tool_result.get("accounts") or []
        win_h = tool_result.get("error_window_hours", 24)
        lines.append(f"### Kesehatan Akun & Token Marketplace (Window Error: {win_h} Jam)\n")
        lines.append(
            f"- **Total Akun**: {summary.get('total_accounts', len(accounts))} | "
            f"**Aktif**: {summary.get('active_accounts', 0)} | "
            f"**Sehat**: {summary.get('healthy_accounts', 0)} | "
            f"**Bermasalah**: **{summary.get('accounts_with_issues', 0)}**"
        )
        lines.append(
            f"- **Token Expired**: {summary.get('expired_tokens', 0)} | "
            f"**Akan Expired (<24j)**: {summary.get('expiring_soon_tokens', 0)} | "
            f"**Konfigurasi Kurang**: {summary.get('missing_configs_count', 0)} | "
            f"**Error Fetch Order**: {summary.get('recent_fetch_errors_count', 0)}\n"
        )
        if accounts:
            lines.append("| No | Akun | Platform | Cabang | State | Token | Sisa Jam | Status Toko | Issues |")
            lines.append("|:---:|:---|:---|:---|:---|:---|:---:|:---|:---|")
            for idx, a in enumerate(accounts, start=1):
                acc_n = a.get("account_name") or "-"
                mp = a.get("marketplace") or "-"
                comp = a.get("company_name") or "-"
                st = a.get("state") or "-"
                tok = a.get("token_status") or "-"
                hrs = a.get("hours_until_expiry")
                hrs_str = f"{hrs:g} j" if hrs is not None else "-"
                shop_st = a.get("shop_status") or "-"
                iss = ", ".join(a.get("issues") or []) or "OK"
                lines.append(f"| {idx} | **{acc_n}** | {mp} | {comp} | `{st}` | `{tok}` | {hrs_str} | `{shop_st}` | {iss} |")
        else:
            lines.append("Semua akun marketplace terpantau sehat dan tidak ada kendala pada filter ini.")
        unassigned = tool_result.get("unassigned_company_errors") or []
        if unassigned:
            lines.append("\n#### Error Fetch Marketplace Tingkat Perusahaan:")
            for ue in unassigned[:5]:
                lines.append(f"- `{ue.get('timestamp', '-')}` [{ue.get('error_type', '-')}] {ue.get('message', '-')}")
        return "\n".join(lines)

    # 18. Marketplace: Orders & SLA
    if tool == "get_marketplace_orders":
        orders = tool_result.get("orders") or []
        total_cnt = _extract_total_count(tool_result, len(orders))
        sla_f = tool_result.get("sla_status_filter") or "all"
        mp_f = tool_result.get("mp_status_filter") or "all"
        lines.append(f"### Daftar Order Marketplace ({total_cnt} Order Ditemukan, SLA: `{sla_f}`, Status: `{mp_f}`)\n")
        if orders:
            lines.append("| No | Invoice MP | No. SO | Platform / Akun | Status MP | Status ERP | Resi AWB | Batas Kirim | Flag SLA | Nominal MP |")
            lines.append("|:---:|:---|:---|:---|:---|:---|:---|:---|:---|:---|")
            for idx, o in enumerate(orders, start=1):
                inv = o.get("mp_invoice_number") or "-"
                so = o.get("so_number") or "-"
                mp_acc = f"{o.get('marketplace', '-')} ({o.get('account_name', '-')})"
                st_mp = o.get("shopee_order_status") or o.get("mp_order_status") or "-"
                st_erp = o.get("erp_status") or "-"
                awb = o.get("awb_number") or "-"
                max_ship = o.get("max_shipping_date") or "-"
                flags = ", ".join(o.get("sla_flags") or []) or "-"
                mp_tot = _format_rupiah(o.get("mp_total") or 0.0)
                lines.append(f"| {idx} | `{inv}` | `{so}` | {mp_acc} | `{st_mp}` | `{st_erp}` | `{awb}` | {max_ship} | {flags} | {mp_tot} |")
            _append_pagination_footer(lines, len(orders), total_cnt)
        else:
            lines.append("Tidak ada order marketplace yang sesuai dengan kriteria SLA/status ini.")
        return "\n".join(lines)

    # 19. Marketplace: Kilat Order Diagnostics
    if tool == "get_kilat_order_diagnostics":
        summary = tool_result.get("summary") or {}
        records = tool_result.get("records") or []
        total_cnt = _extract_total_count(tool_result, len(records))
        lines.append("### Diagnosa Pesanan Kilat Shopee (Read-Only)\n")
        lines.append(
            f"- **Total Kilat Dipindai**: {summary.get('total_kilat_scanned', 0)} | "
            f"**Invoice Belum Update**: **{summary.get('unupdated_invoice_count', 0)}** | "
            f"**Duplikat/Bentrok**: {summary.get('duplicate_collision_count', 0)} | "
            f"**Tanpa SO Confirmed**: {summary.get('missing_so_link_count', 0)}\n"
        )
        if records:
            lines.append("| No | Invoice MP | No. Reservasi | Invoice Updated | No. SO | Status ERP | Akun | Temuan |")
            lines.append("|:---:|:---|:---|:---:|:---|:---|:---|:---|")
            for idx, r in enumerate(records, start=1):
                inv = r.get("mp_invoice_number") or "-"
                res = r.get("no_reservasi") or "-"
                inv_upd = "Ya" if r.get("invoice_updated") else "Tidak"
                so = r.get("so_number") or "-"
                st_erp = r.get("erp_status") or "-"
                acc = r.get("account_name") or "-"
                iss = ", ".join(r.get("issues") or []) or "-"
                lines.append(f"| {idx} | `{inv}` | `{res}` | {inv_upd} | `{so}` | `{st_erp}` | {acc} | {iss} |")
            _append_pagination_footer(lines, len(records), total_cnt)
        else:
            lines.append("Tidak ditemukan masalah Pesanan Kilat pada filter ini.")
        log_cands = tool_result.get("log_entry_candidates") or []
        note_cands = tool_result.get("quotation_note_candidates") or []
        if log_cands or note_cands:
            lines.append("\n#### Kandidat Referensi SO dari LogEntry / Catatan Quotation:")
            for c in (log_cands + note_cands)[:10]:
                lines.append(f"- `{c.get('so_number', '-')}` (ID: `{c.get('quotation_id', '-')}`, Status: `{c.get('status', '-')}`, Tanggal: {c.get('quotation_date', '-')})")
        lines.append("\nUntuk eksekusi perbaikan atau penggabungan Pesanan Kilat, buka `/marketplace/tools/handling-order-kilat/`.")
        return "\n".join(lines)

    # 20. Marketplace: Product Mapping Status
    if tool == "get_product_mapping_status":
        m_filter = tool_result.get("mapping_status_filter") or "all"
        if "exceptions" in tool_result:
            excs = tool_result.get("exceptions") or []
            total_cnt = _extract_total_count(tool_result, len(excs))
            lines.append(f"### Daftar Pengecualian Mapping Produk (ExceptionProduct: {total_cnt} Item)\n")
            if excs:
                lines.append("| No | Akun | Platform | SKU ERP | Nama Produk ERP | Parent SKU | Child SKU |")
                lines.append("|:---:|:---|:---|:---|:---|:---|:---|")
                for idx, e in enumerate(excs, start=1):
                    lines.append(
                        f"| {idx} | {e.get('account_name', '-')} | {e.get('marketplace', '-')} | "
                        f"`{e.get('erp_sku') or '-'}` | {e.get('erp_name') or '-'} | `{e.get('parent_sku') or '-'}` | `{e.get('child_sku') or '-'}` |"
                    )
                _append_pagination_footer(lines, len(excs), total_cnt)
            else:
                lines.append("Tidak ada data ExceptionProduct pada filter ini.")
            return "\n".join(lines)

        summary = tool_result.get("summary") or {}
        mappings = tool_result.get("mappings") or []
        total_cnt = _extract_total_count(tool_result, len(mappings))
        lines.append(f"### Status Mapping Produk Marketplace (Filter: `{m_filter}`)\n")
        lines.append(
            f"- **Total Baris**: {summary.get('total_lines', 0)} | "
            f"**Mapped**: {summary.get('mapped_count', 0)} ({summary.get('mapped_percentage', 0):.1f}%) | "
            f"**Unmapped**: **{summary.get('unmapped_count', 0)}** | "
            f"**Qty Multiplier > 1**: {summary.get('multiplier_gt_1_count', 0)}\n"
        )
        if mappings:
            lines.append("| No | Akun | Platform | SKU ERP | SKU MP | Status Mapping | Multiplier | Peringatan |")
            lines.append("|:---:|:---|:---|:---|:---|:---|:---:|:---|")
            for idx, m in enumerate(mappings, start=1):
                erp_p = m.get("erp_product") or {}
                mp_p = m.get("mp_product") or {}
                warn_str = ", ".join(m.get("warnings") or []) or "-"
                lines.append(
                    f"| {idx} | {m.get('account_name', '-')} | {m.get('marketplace', '-')} | "
                    f"`{erp_p.get('sku') or '-'}` | `{mp_p.get('sku_mp') or '-'}` | `{m.get('status', '-')}` | "
                    f"{m.get('qty_multiplier', 1)}x | {warn_str} |"
                )
            _append_pagination_footer(lines, len(mappings), total_cnt)
        else:
            lines.append("Tidak ada baris mapping produk yang cocok dengan filter ini.")
        return "\n".join(lines)

    # 21. Marketplace: Stock Sync Simulation
    if tool == "get_stock_sync_simulation":
        summary = tool_result.get("summary") or {}
        sims = tool_result.get("simulations") or []
        total_cnt = _extract_total_count(tool_result, len(sims))
        lines.append("### Simulasi Sinkronisasi Stok Marketplace (Dry-Run)\n")
        lines.append(
            f"- **Total SKU Dievaluasi**: {summary.get('total_evaluated', 0)} | "
            f"**Selisih / Peringatan**: **{summary.get('discrepancy_or_warning_count', 0)}**\n"
        )
        if sims:
            lines.append("| No | SKU ERP | SKU MP | Akun | Setting | Stok Mentah ERP | Mult | Simulasi Push | Log Terakhir | Selisih | Peringatan |")
            lines.append("|:---:|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|")
            for idx, s in enumerate(sims, start=1):
                last_log = s.get("last_stock_log") or {}
                last_val = last_log.get("new_stock")
                last_str = f"{last_val} ({last_log.get('status', '-')})" if last_log else "-"
                diff = s.get("diff_vs_last_log")
                diff_str = f"{diff:+d}" if isinstance(diff, int) else (str(diff) if diff is not None else "-")
                warns = list(s.get("warnings") or [])
                masked_cnt = int(s.get("masked_shared_branches_count") or 0)
                if masked_cnt > 0:
                    warns.append(f"shared_branches_masked:{masked_cnt}")
                raw_st = s.get("raw_erp_stock")
                sim_st = s.get("simulated_push_stock")
                lines.append(
                    f"| {idx} | `{s.get('erp_sku') or '-'}` | `{s.get('mp_sku') or '-'}` | "
                    f"{s.get('account_name', '-')} ({s.get('marketplace', '-')}) | {s.get('stock_setting') or '-'} | "
                    f"{raw_st if raw_st is not None else '-'} | {s.get('qty_multiplier', 1)}x | "
                    f"**{sim_st if sim_st is not None else '-'}** | {last_str} | {diff_str} | {', '.join(warns) or '-'} |"
                )
            _append_pagination_footer(lines, len(sims), total_cnt)
        else:
            lines.append("Tidak ditemukan selisih simulasi stok atau peringatan pada filter ini.")
        return "\n".join(lines)

    # 22. Marketplace: Sync Execution Logs
    if tool == "get_sync_execution_logs":
        dw = tool_result.get("date_window") or {}
        st_f = tool_result.get("status_filter") or "FAILED"
        lines.append(f"### Log Eksekusi Sinkronisasi Marketplace (Status: `{st_f}`, Periode: {dw.get('date_from', '-')} s/d {dw.get('date_to', '-')})\n")

        s_sum = tool_result.get("stock_summary")
        if isinstance(s_sum, dict):
            lines.append(
                f"- **Ringkasan Log Stok**: Total {s_sum.get('total', 0)} | "
                f"Sukses {s_sum.get('success', 0)} | Gagal **{s_sum.get('failed', 0)}** | "
                f"Skip {s_sum.get('skipped', 0)} | Rata-rata {s_sum.get('avg_duration_ms', 0)} ms"
            )
        p_sum = tool_result.get("price_summary")
        if isinstance(p_sum, dict):
            lines.append(
                f"- **Ringkasan Log Harga**: Total {p_sum.get('total', 0)} | "
                f"Sukses {p_sum.get('success', 0)} | Gagal **{p_sum.get('failed', 0)}** | "
                f"Rata-rata {p_sum.get('avg_duration_ms', 0)} ms"
            )

        stock_logs = tool_result.get("stock_logs") or []
        if stock_logs:
            lines.append(f"\n#### Log Update Stok ({tool_result.get('stock_logs_total', len(stock_logs))} Total):\n")
            lines.append("| Waktu | Akun | SKU | Trigger | Stok Lama -> Baru | Status | Error |")
            lines.append("|:---|:---|:---|:---|:---:|:---:|:---|")
            for lg in stock_logs:
                ne = lg.get("normalized_error") or {}
                err_txt = ne.get("message") or lg.get("error_message") or "-"
                lines.append(
                    f"| {lg.get('created_at', '-')} | {lg.get('account_name', '-')} | `{lg.get('sku') or '-'}` | "
                    f"`{lg.get('trigger_source') or '-'}` | {lg.get('old_stock', '-')} -> {lg.get('new_stock', '-')} | "
                    f"`{lg.get('status', '-')}` | {err_txt} |"
                )

        price_logs = tool_result.get("price_logs") or []
        if price_logs:
            lines.append(f"\n#### Log Update Harga ({tool_result.get('price_logs_total', len(price_logs))} Total):\n")
            lines.append("| Waktu | Akun | SKU ERP | Tier Qty | Harga Lama -> Baru | Status | Kode/Pesan Error |")
            lines.append("|:---|:---|:---|:---|:---:|:---:|:---|")
            for lg in price_logs:
                ne = lg.get("normalized_error") or {}
                err_txt = f"`{ne.get('code', '')}` {ne.get('message', '')}".strip() or "-"
                lines.append(
                    f"| {lg.get('created_at', '-')} | {lg.get('account_name', '-')} | `{lg.get('erp_sku') or '-'}` | "
                    f"{lg.get('qty_tier', '1')} | {_format_rupiah(lg.get('old_price', 0))} -> {_format_rupiah(lg.get('new_price', 0))} | "
                    f"`{lg.get('status', '-')}` | {err_txt} |"
                )

        pcls_logs = tool_result.get("pcls_logs") or []
        if pcls_logs:
            lines.append(f"\n#### Log Sinkronisasi Harga PCLS ({tool_result.get('pcls_logs_total', len(pcls_logs))} Total):\n")
            lines.append("| Waktu | Service | SKU | Status | HTTP | Error |")
            lines.append("|:---|:---|:---|:---:|:---:|:---|")
            for lg in pcls_logs:
                lines.append(
                    f"| {lg.get('created_at', '-')} | {lg.get('service_name', '-')} | `{lg.get('sku') or '-'}` | "
                    f"`{lg.get('status', '-')}` | {lg.get('response_code', '-')} | {lg.get('error_message') or '-'} |"
                )

        if not stock_logs and not price_logs and not pcls_logs:
            lines.append("\nTidak ada entri log sinkronisasi yang cocok dengan filter status/periode ini.")
        return "\n".join(lines)

    # 23. Marketplace: Price Comparison
    if tool == "get_price_comparison":
        summary = tool_result.get("summary") or {}
        items = tool_result.get("items") or []
        total_cnt = _extract_total_count(tool_result, len(items))
        pl = tool_result.get("pricelist") or {}
        pl_label = f"{pl.get('name', '-')} (ID: `{pl.get('id', '-')}`)" if pl else "-"
        lines.append(f"### Perbandingan Harga ERP vs Marketplace (Pricelist: {pl_label}, Live: `{tool_result.get('include_live', False)}`)\n")
        lines.append(
            f"- **Dievaluasi**: {summary.get('total_evaluated', 0)} | "
            f"**Cocok**: {summary.get('match_count', 0)} | "
            f"**Selisih**: **{summary.get('mismatch_count', 0)}** | "
            f"**Harga MP Kosong**: {summary.get('missing_mp_price', 0)} | "
            f"**Harga ERP Kosong**: {summary.get('missing_erp_price', 0)}\n"
        )
        if items:
            lines.append("| No | SKU ERP | Nama Produk | Akun | Harga ERP (Expected) | Harga Marketplace | Sumber | Selisih (%) | Status |")
            lines.append("|:---:|:---|:---|:---|:---|:---|:---:|:---|:---|")
            for idx, it in enumerate(items, start=1):
                erp_p = _format_rupiah(it["expected_erp_bundle_price"]) if it.get("expected_erp_bundle_price") is not None else "-"
                mp_p = _format_rupiah(it["marketplace_price"]) if it.get("marketplace_price") is not None else "-"
                diff = _format_rupiah(it["difference"]) if it.get("difference") is not None else "-"
                dpct = f" ({it['difference_pct']:.1f}%)" if it.get("difference_pct") is not None else ""
                lines.append(
                    f"| {idx} | `{it.get('erp_sku') or '-'}` | {it.get('erp_name') or '-'} | "
                    f"{it.get('account_name', '-')} | {erp_p} | {mp_p} | `{it.get('price_source', 'db_cache')}` | "
                    f"{diff}{dpct} | `{it.get('comparison_status', '-')}` |"
                )
            _append_pagination_footer(lines, len(items), total_cnt)
        else:
            lines.append("Tidak ditemukan selisih harga atau item pada filter perbandingan ini.")
        live_errs = tool_result.get("live_fallback_errors")
        if isinstance(live_errs, dict) and live_errs:
            lines.append("\n> **Catatan Live Fallback**: Beberapa akun memakai DB cache karena panggilan live API gagal/tidak didukung:")
            for acc_k, err_v in live_errs.items():
                lines.append(f"> - Akun `{acc_k}`: {err_v}")
        return "\n".join(lines)

    # 24. Marketplace: Fee & Escrow Anomalies
    if tool == "get_fee_and_escrow_anomalies":
        summary = tool_result.get("summary") or {}
        anoms = tool_result.get("anomalies") or []
        total_cnt = _extract_total_count(tool_result, len(anoms))
        dw = tool_result.get("date_window") or {}
        lines.append(f"### Deteksi Anomali Potongan Fee & Escrow Marketplace ({dw.get('date_from', '-')} s/d {dw.get('date_to', '-')})\n")
        lines.append(
            f"- **Total Anomali**: **{summary.get('total_anomalies', 0)}** | "
            f"**High Fee**: {summary.get('high_fee_count', 0)} | "
            f"**Negative Fee**: {summary.get('negative_fee_count', 0)} | "
            f"**Zero Income**: {summary.get('zero_income_count', 0)} | "
            f"**ERP vs MP Mismatch**: {summary.get('erp_mp_mismatch_count', 0)}"
        )
        lines.append(f"- **Total Selisih Potongan**: **{_format_rupiah(summary.get('total_fee_difference_idr', 0.0))}**\n")
        if anoms:
            lines.append("| No | Invoice MP | No. SO | Akun | MP Total | Expected Income | Potongan | % Fee | Jenis Anomali |")
            lines.append("|:---:|:---|:---|:---|:---|:---|:---|:---:|:---|")
            for idx, a in enumerate(anoms, start=1):
                a_types = ", ".join(a.get("anomaly_types") or [])
                lines.append(
                    f"| {idx} | `{a.get('mp_invoice_number') or '-'}` | `{a.get('so_number') or '-'}` | "
                    f"{a.get('account_name', '-')} ({a.get('marketplace', '-')}) | {_format_rupiah(a.get('mp_total', 0))} | "
                    f"{_format_rupiah(a.get('seller_expected_income', 0))} | {_format_rupiah(a.get('fee_difference', 0))} | "
                    f"**{a.get('fee_percentage', 0):.2f}%** | {a_types} |"
                )
            _append_pagination_footer(lines, len(anoms), total_cnt)
        else:
            lines.append("Tidak ditemukan order dengan anomali potongan fee pada periode ini.")
        return "\n".join(lines)

    # 25. Marketplace: Deduction & Settlement Summary
    if tool == "get_deduction_and_settlement_summary":
        summary = tool_result.get("summary") or {}
        dw = tool_result.get("date_window") or {}
        lines.append(
            f"### Rekapitulasi Potongan Fee & Settlement Marketplace "
            f"({tool_result.get('date_field', 'received_date')}: {dw.get('date_from', '-')} s/d {dw.get('date_to', '-')})\n"
        )
        lines.append(f"- **Total Order Settled**: **{summary.get('total_orders', 0):,}**")
        lines.append(f"- **Total Gross**: {_format_rupiah(summary.get('total_gross', 0.0))}")
        lines.append(f"- **Total Potongan Fee**: {_format_rupiah(summary.get('total_deduction', 0.0))} (**{summary.get('effective_deduction_pct', 0.0):.2f}%**)")
        lines.append(f"- **Total Pajak**: {_format_rupiah(summary.get('total_tax', 0.0))}")
        lines.append(f"- **Total Nett Diterima**: **{_format_rupiah(summary.get('total_nett', 0.0))}**")
        lines.append(
            f"- **Settlement Belum Posting Jurnal**: **{summary.get('unposted_journal_orders_count', 0)} order** "
            f"({_format_rupiah(summary.get('unposted_journal_nett_amount', 0.0))})"
        )
        if summary.get("arithmetic_mismatch_count"):
            lines.append(f"- **Selisih Aritmatika Gross-Nett**: **{summary.get('arithmetic_mismatch_count')} record**")

        by_cat = tool_result.get("by_deduction_category") or []
        if by_cat:
            lines.append("\n#### Rincian Potongan per Kategori:\n")
            lines.append("| Kategori Potongan | Jumlah Item | Total Nominal |")
            lines.append("|:---|:---:|:---|")
            for c in by_cat:
                lines.append(f"| {c.get('category', '-')} | {c.get('line_count', 0):,} | {_format_rupiah(c.get('total_amount', 0.0))} |")

        by_store = tool_result.get("by_store") or []
        if by_store:
            lines.append("\n#### Rincian per Toko:\n")
            lines.append("| Akun Toko | Platform | Order | Gross | Potongan | Pajak | Nett |")
            lines.append("|:---|:---|:---:|:---|:---|:---|:---|")
            for st in by_store:
                lines.append(
                    f"| {st.get('account_name', '-')} | {st.get('marketplace', '-')} | {st.get('orders', 0)} | "
                    f"{_format_rupiah(st.get('amount_gross', 0))} | {_format_rupiah(st.get('deduction_fee', 0))} | "
                    f"{_format_rupiah(st.get('tax', 0))} | **{_format_rupiah(st.get('amount_nett', 0))}** |"
                )

        det_orders = tool_result.get("orders") or []
        if det_orders:
            total_cnt = _extract_total_count(tool_result, len(det_orders))
            lines.append("\n#### Daftar Settlement Order:\n")
            lines.append("| Invoice | No. SO | Akun | Tgl Terima | Gross | Potongan | Nett | Status Jurnal |")
            lines.append("|:---|:---|:---|:---|:---|:---|:---|:---|")
            for o in det_orders:
                j_st = f"Posted (`{o.get('journal_number')}`)" if o.get("journal_posted") else "Belum Posting"
                lines.append(
                    f"| `{o.get('invoice') or '-'}` | `{o.get('so_number') or '-'}` | {o.get('account_name', '-')} | "
                    f"{o.get('received_date') or '-'} | {_format_rupiah(o.get('amount_gross', 0))} | "
                    f"{_format_rupiah(o.get('deduction_fee', 0))} | {_format_rupiah(o.get('amount_nett', 0))} | {j_st} |"
                )
            _append_pagination_footer(lines, len(det_orders), total_cnt)
        return "\n".join(lines)

    # 26. Marketplace: Cashflow & Balance Summary
    if tool == "get_cashflow_and_balance_summary":
        summary = tool_result.get("summary") or {}
        accounts = tool_result.get("accounts") or []
        dw = tool_result.get("date_window") or {}
        lines.append(f"### Ringkasan Saldo Dompet & Cashflow Marketplace ({dw.get('date_from', '-')} s/d {dw.get('date_to', '-')})\n")
        lines.append(f"- **Total Available Balance**: **{_format_rupiah(summary.get('total_available_balance', 0.0))}**")
        lines.append(f"- **Total Pending Escrow**: **{_format_rupiah(summary.get('total_pending_balance', 0.0))}**")
        lines.append(f"- **Total Saldo Dompet**: **{_format_rupiah(summary.get('total_balance', 0.0))}**")
        lines.append(
            f"- **Mutasi Periode**: Settlement Inflow {_format_rupiah(summary.get('period_settlement_inflow', 0.0))} | "
            f"Withdrawal Outflow {_format_rupiah(summary.get('period_withdrawal_outflow', 0.0))} | "
            f"Adjustment {_format_rupiah(summary.get('period_adjustment_net', 0.0))}"
        )
        if summary.get("stale_or_failed_snapshots"):
            lines.append(f"- **Snapshot Usang/Gagal**: **{summary.get('stale_or_failed_snapshots')} akun**\n")
        else:
            lines.append("")

        if accounts:
            lines.append("| Akun | Platform | Available | Pending Escrow | Total | Sumber | Tgl Snapshot | Peringatan |")
            lines.append("|:---|:---|:---|:---|:---|:---:|:---|:---|")
            for a in accounts:
                curr = a.get("currency") or "IDR"
                warns = list(a.get("warnings") or [])
                if a.get("live_error"):
                    warns.append(f"live_err: {a['live_error']}")
                lines.append(
                    f"| {a.get('account_name', '-')} | {a.get('marketplace', '-')} | "
                    f"{_format_currency(a.get('available_balance', 0), curr)} | "
                    f"{_format_currency(a.get('pending_balance', 0), curr)} | "
                    f"**{_format_currency(a.get('total_balance', 0), curr)}** | "
                    f"`{a.get('balance_source', 'db_snapshot')}` | {a.get('snapshot_date') or '-'} | "
                    f"{', '.join(warns) or '-'} |"
                )

        cf_logs = tool_result.get("cashflow_logs") or []
        if cf_logs:
            cf_tot = tool_result.get("cashflow_logs_total", len(cf_logs))
            lines.append(f"\n#### Rincian Mutasi Cashflow ({cf_tot} Total):\n")
            lines.append("| Waktu | Akun | Tipe | Referensi | Nominal | Saldo Setelah |")
            lines.append("|:---|:---|:---|:---|:---|:---|")
            for lg in cf_logs:
                bal_after = _format_rupiah(lg["balance_after"]) if lg.get("balance_after") is not None else "-"
                lines.append(
                    f"| {lg.get('transaction_time', '-')} | {lg.get('account_name', '-')} | "
                    f"`{lg.get('transaction_type', '-')}` | `{lg.get('reference_id') or '-'}` | "
                    f"{_format_rupiah(lg.get('amount', 0))} | {bal_after} |"
                )
            _append_pagination_footer(lines, len(cf_logs), cf_tot)
        return "\n".join(lines)

    # 27. Marketplace: Competitor Insights
    if tool == "get_competitor_insights":
        summary = tool_result.get("summary") or {}
        comps = tool_result.get("competitors") or []
        total_cnt = _extract_total_count(tool_result, len(comps))
        dw = tool_result.get("date_window") or {}
        lines.append(
            f"### Insight Kompetitor Marketplace (Focus: `{tool_result.get('focus', 'overview')}`, "
            f"Periode: {dw.get('date_from', '-')} s/d {dw.get('date_to', '-')})\n"
        )
        lines.append(
            f"- **Kompetitor Aktif**: {summary.get('active_competitors', 0)} | "
            f"**Audit Stale (>{tool_result.get('stale_days', 7)} hari)**: **{summary.get('stale_competitors_count', 0)}** | "
            f"**Turun Harga**: {summary.get('competitors_with_price_drops', 0)} | "
            f"**Total Velocity Terjual**: {summary.get('total_period_sold_velocity', 0):g} unit\n"
        )
        if comps:
            lines.append("| No | Toko Kompetitor | Platform | Grup / SKU ERP | Harga Saat Ini | Delta Harga | Velocity Terjual | Status Audit |")
            lines.append("|:---:|:---|:---|:---|:---|:---|:---:|:---|")
            for idx, c in enumerate(comps, start=1):
                cur_p = _format_rupiah(c["current_price"]) if c.get("current_price") is not None else "-"
                d_p = _format_rupiah(c.get("net_price_delta", 0))
                grp_sku = f"{c.get('group_name', '-')}" + (f" (`{c['erp_sku']}`)" if c.get("erp_sku") else "")
                lines.append(
                    f"| {idx} | **{c.get('store_name', '-')}** | {c.get('marketplace', '-')} | {grp_sku} | "
                    f"{cur_p} | {d_p} ({c.get('price_drop_events', 0)}x turun) | "
                    f"**{c.get('period_sold_velocity', 0):g} unit** | `{c.get('audit_status', '-')}` |"
                )
            _append_pagination_footer(lines, len(comps), total_cnt)
        else:
            lines.append("Tidak ada data kompetitor yang cocok dengan filter ini.")
        return "\n".join(lines)

    # Default fallback
    return f"Hasil eksekusi data ERP: {str(tool_result)}"


def _format_multi_company_presentation(state: AgentState) -> str:
    """
    Format multi-company comparison and network-wide rollups into structured Markdown.
    Includes per-company breakdown, grand total rollup summary, and partial permission warnings.
    """
    from app.graph.nodes.extractor import COMPANY_NAME_MAP

    tool = state.get("detected_tool") or "get_account_balance"
    mc_results = state.get("multi_company_results") or {}
    unauth = state.get("unauthorized_companies") or []
    lines: List[str] = []

    # 1. Account Balance Multi-Company Presentation
    if tool == "get_account_balance":
        lines.append("### Laporan Saldo Akun Multi-Perusahaan (Combined Accounting Balance)\n")
        lines.append("| No | Perusahaan / Cabang | Saldo Awal | Mutasi Debit | Mutasi Kredit | Saldo Akhir |")
        lines.append("|:---|:---|:---|:---|:---|:---|")
        tot_op = 0.0
        tot_deb = 0.0
        tot_kred = 0.0
        tot_end = 0.0
        row_idx = 1

        for cid in sorted(mc_results.keys()):
            c_name = COMPANY_NAME_MAP.get(cid, f"Cabang {cid}")
            res = mc_results.get(cid)
            if not res:
                st_text = "Akses Dibatasi (HTTP 403)" if cid in unauth else "Tidak Ada Data"
                lines.append(f"| {row_idx} | {c_name} | - | - | - | *{st_text}* |")
                row_idx += 1
                continue

            totals = res.get("totals") or {}
            op = float(totals.get("opening_balance", res.get("opening_balance", 0.0)) or 0.0)
            deb = float(totals.get("period_debit", res.get("total_debit", 0.0)) or 0.0)
            kred = float(totals.get("period_credit", res.get("total_credit", 0.0)) or 0.0)
            end = float(totals.get("ending_balance", res.get("net_balance", 0.0)) or 0.0)

            tot_op += op
            tot_deb += deb
            tot_kred += kred
            tot_end += end

            lines.append(
                f"| {row_idx} | {c_name} | {_format_rupiah(op)} | {_format_rupiah(deb)} | "
                f"{_format_rupiah(kred)} | **{_format_rupiah(end)}** |"
            )
            row_idx += 1

        lines.append(
            f"| | **TOTAL GABUNGAN** | **{_format_rupiah(tot_op)}** | **{_format_rupiah(tot_deb)}** | "
            f"**{_format_rupiah(tot_kred)}** | **{_format_rupiah(tot_end)}** |"
        )

    # 2. Profit & Loss Multi-Company Presentation
    elif tool == "get_profit_and_loss":
        lines.append("### Laporan Laba Rugi Multi-Perusahaan (Combined Profit & Loss)\n")
        lines.append("| No | Perusahaan / Cabang | Pendapatan Usaha | HPP (COGS) | Laba Kotor | Laba / (Rugi) Bersih |")
        lines.append("|:---|:---|:---|:---|:---|:---|")
        tot_rev = 0.0
        tot_cogs = 0.0
        tot_gross = 0.0
        tot_net = 0.0
        row_idx = 1

        for cid in sorted(mc_results.keys()):
            c_name = COMPANY_NAME_MAP.get(cid, f"Cabang {cid}")
            res = mc_results.get(cid)
            if not res:
                st_text = "Akses Dibatasi (HTTP 403)" if cid in unauth else "Tidak Ada Data"
                lines.append(f"| {row_idx} | {c_name} | - | - | - | *{st_text}* |")
                row_idx += 1
                continue

            summary = res.get("summary") or res
            rev = float(summary.get("total_revenue", 0.0) or 0.0)
            cogs = float(summary.get("cogs", 0.0) or 0.0)
            gross = float(summary.get("gross_profit", 0.0) or 0.0)
            net = float(summary.get("net_profit", 0.0) or 0.0)

            tot_rev += rev
            tot_cogs += cogs
            tot_gross += gross
            tot_net += net

            lines.append(
                f"| {row_idx} | {c_name} | {_format_rupiah(rev)} | {_format_rupiah(cogs)} | "
                f"{_format_rupiah(gross)} | **{_format_rupiah(net)}** |"
            )
            row_idx += 1

        lines.append(
            f"| | **TOTAL GABUNGAN** | **{_format_rupiah(tot_rev)}** | **{_format_rupiah(tot_cogs)}** | "
            f"**{_format_rupiah(tot_gross)}** | **{_format_rupiah(tot_net)}** |"
        )

        # Rincian akun per cabang jika terdapat breakdown aktif
        has_breakdown = any(
            bool(res and isinstance(res, dict) and res.get("breakdown"))
            for res in mc_results.values()
        )
        if has_breakdown:
            lines.append("\n#### Rincian Akun Laba Rugi per Cabang / Perusahaan:\n")
            for cid in sorted(mc_results.keys()):
                c_name = COMPANY_NAME_MAP.get(cid, f"Cabang {cid}")
                res = mc_results.get(cid)
                if not res or not isinstance(res, dict):
                    continue
                breakdown = res.get("breakdown") or []
                active_accounts = []
                for grp in breakdown:
                    grp_name = grp.get("group_name") or grp.get("name") or "-"
                    for tp in grp.get("types", []):
                        tp_name = tp.get("type_name") or tp.get("name") or "-"
                        for acc in tp.get("accounts", []):
                            bal = float(acc.get("balance", 0.0) or 0.0)
                            if abs(bal) > 0.01:
                                active_accounts.append((grp_name, tp_name, acc.get("account_code", "-"), acc.get("account_name", "-"), bal))

                if active_accounts:
                    lines.append(f"**{c_name}**:")
                    lines.append("| Grup | Tipe Akun | Kode | Nama Akun | Saldo |")
                    lines.append("|:---|:---|:---|:---|:---|")
                    for g_n, t_n, a_c, a_n, bal in active_accounts:
                        lines.append(f"| {g_n} | {t_n} | `{a_c}` | {a_n} | {_format_rupiah(bal)} |")
                    lines.append("")

    # 3. Balance Sheet Multi-Company Presentation
    elif tool == "get_balance_sheet":
        lines.append("### Laporan Neraca Multi-Perusahaan (Combined Balance Sheet)\n")
        lines.append("| No | Perusahaan / Cabang | Total Aset | Total Liabilitas | Total Ekuitas |")
        lines.append("|:---|:---|:---|:---|:---|")
        tot_ast = 0.0
        tot_lia = 0.0
        tot_equ = 0.0
        row_idx = 1

        for cid in sorted(mc_results.keys()):
            c_name = COMPANY_NAME_MAP.get(cid, f"Cabang {cid}")
            res = mc_results.get(cid)
            if not res:
                st_text = "Akses Dibatasi (HTTP 403)" if cid in unauth else "Tidak Ada Data"
                lines.append(f"| {row_idx} | {c_name} | - | - | *{st_text}* |")
                row_idx += 1
                continue

            summary = res.get("summary") or res
            ast = float(summary.get("total_assets", 0.0) or 0.0)
            lia = float(summary.get("total_liabilities", 0.0) or 0.0)
            equ = float(summary.get("total_equity", 0.0) or 0.0)

            tot_ast += ast
            tot_lia += lia
            tot_equ += equ

            lines.append(
                f"| {row_idx} | {c_name} | {_format_rupiah(ast)} | {_format_rupiah(lia)} | "
                f"**{_format_rupiah(equ)}** |"
            )
            row_idx += 1

        lines.append(
            f"| | **TOTAL GABUNGAN** | **{_format_rupiah(tot_ast)}** | **{_format_rupiah(tot_lia)}** | "
            f"**{_format_rupiah(tot_equ)}** |"
        )

    # 4. Stock Level Multi-Company Presentation
    elif tool == "get_stock_level":
        lines.append("### Laporan Ketersediaan Stok Multi-Cabang (Combined Stock Level)\n")
        lines.append("| No | Perusahaan / Cabang | SKU / Kode | Nama Produk | Gudang | Lokasi / Rak | Qty Fisik | Qty Siap Jual |")
        lines.append("|:---|:---|:---|:---|:---|:---|:---|:---|")
        tot_qty = 0.0
        tot_avail = 0.0
        row_idx = 1

        for cid in sorted(mc_results.keys()):
            c_name = COMPANY_NAME_MAP.get(cid, f"Cabang {cid}")
            res = mc_results.get(cid)
            if not res:
                st_text = "Akses Dibatasi (HTTP 403)" if cid in unauth else "Tidak Ada Data"
                lines.append(f"| {row_idx} | {c_name} | - | - | - | - | - | *{st_text}* |")
                row_idx += 1
                continue

            items = res.get("stocks") or res.get("items") or [res]
            for it in items:
                sku = it.get("code") or it.get("sku") or "-"
                p_name = it.get("product_name") or it.get("name") or "-"
                qty = float(it.get("quantity", it.get("qty_on_hand", 0.0)) or 0.0)
                avail = float(it.get("available_quantity", qty) or 0.0)
                breakdown = it.get("breakdown") or []

                tot_qty += qty
                tot_avail += avail

                if breakdown:
                    wh_list = [bd.get("warehouse_name") or f"WH #{bd.get('warehouse_id')}" for bd in breakdown if bd.get("warehouse_name") or bd.get("warehouse_id")]
                    loc_list = [bd.get("location_name") or f"Rak #{bd.get('location_id')}" for bd in breakdown if bd.get("location_name") or bd.get("location_id")]
                    wh_str = ", ".join(dict.fromkeys(wh_list)) if wh_list else "-"
                    loc_str = ", ".join(dict.fromkeys(loc_list)) if loc_list else "-"
                else:
                    wh_str = it.get("warehouse_name") or "-"
                    loc_str = it.get("location_name") or "-"

                avail_str = f"**{avail:g} unit**" if avail > 0 else "0 unit"
                qty_str = f"{qty:g} unit" if qty > 0 else "0 unit"

                lines.append(
                    f"| {row_idx} | {c_name} | `{sku}` | {p_name} | {wh_str} | {loc_str} | {qty_str} | {avail_str} |"
                )
                row_idx += 1

        lines.append(
            f"| | **TOTAL NASIONAL** | | | | | **{tot_qty:g} unit** | **{tot_avail:g} unit** |"
        )

    # 5. Vendor Invoices / Vendor Bills Multi-Company Presentation
    elif tool in ("fetch_vendor_invoices", "get_vendor_bills"):
        lines.append("### Ringkasan Tagihan Vendor Multi-Perusahaan\n")
        lines.append("| No | Perusahaan / Cabang | Total Tagihan | Belum Dibayar | Sudah Dibayar | Status |")
        lines.append("|:---|:---|:---|:---|:---|:---|")
        tot_amt = 0.0
        tot_unp = 0.0
        tot_paid = 0.0
        row_idx = 1

        for cid in sorted(mc_results.keys()):
            c_name = COMPANY_NAME_MAP.get(cid, f"Cabang {cid}")
            res = mc_results.get(cid)
            if not res:
                st_text = "Akses Dibatasi (HTTP 403)" if cid in unauth else "Tidak Ada Data"
                lines.append(f"| {row_idx} | {c_name} | - | - | - | *{st_text}* |")
                row_idx += 1
                continue

            calc = res.get("calculations") or res.get("summary") or res
            amt = float(calc.get("total_amount", calc.get("total_bill_amount", 0.0)) or 0.0)
            unp = float(calc.get("total_unpaid", 0.0) or 0.0)
            paid = float(calc.get("total_paid", 0.0) or 0.0)

            tot_amt += amt
            tot_unp += unp
            tot_paid += paid

            lines.append(
                f"| {row_idx} | {c_name} | {_format_rupiah(amt)} | {_format_rupiah(unp)} | "
                f"{_format_rupiah(paid)} | Selesai |"
            )
            row_idx += 1

        lines.append(
            f"| | **TOTAL GABUNGAN** | **{_format_rupiah(tot_amt)}** | **{_format_rupiah(tot_unp)}** | "
            f"**{_format_rupiah(tot_paid)}** | |"
        )

    # 6. Top Selling Products Multi-Company Presentation (Consolidated National Ranking)
    elif tool == "get_top_selling_products":
        params = state.get("extracted_params") or {}
        search_term = params.get("search")
        sort_by = params.get("sort_by") or "revenue"
        user_q_low = (state.get("user_query") or "").lower()
        is_branch_ranking_mode = bool(search_term) or bool(
            re.search(r"\b(?:di\s*mana|cabang\s+mana|toko\s+mana|outlet\s+mana)\b", user_q_low)
        )

        if is_branch_ranking_mode:
            branch_rows: List[Dict[str, Any]] = []
            agg_channels: Dict[str, Dict[str, float]] = {}

            for cid, res in mc_results.items():
                c_name = COMPANY_NAME_MAP.get(cid, f"Cabang {cid}")
                if not res or not isinstance(res, dict):
                    continue
                p_list = res.get("products") or []
                ch_list = res.get("channel_breakdown") or []
                b_qty = sum(float(p.get("total_qty") or 0.0) for p in p_list)
                b_rev = sum(float(p.get("total_revenue") or 0.0) for p in p_list)
                top_prod_label = ", ".join(
                    f"`{p.get('code') or '-'}` {p.get('name') or ''}".strip()
                    for p in p_list[:2]
                ) or "-"
                top_ch_label = (
                    f"{ch_list[0].get('channel_name', '-')} ({float(ch_list[0].get('total_qty') or 0):g} unit)"
                    if ch_list
                    else "-"
                )
                for ch in ch_list:
                    ch_n = ch.get("channel_name") or "Direct / Manual"
                    if ch_n not in agg_channels:
                        agg_channels[ch_n] = {"order_count": 0.0, "total_qty": 0.0, "total_revenue": 0.0}
                    agg_channels[ch_n]["order_count"] += float(ch.get("order_count") or 0)
                    agg_channels[ch_n]["total_qty"] += float(ch.get("total_qty") or 0)
                    agg_channels[ch_n]["total_revenue"] += float(ch.get("total_revenue") or 0)

                if b_qty > 0 or b_rev > 0:
                    branch_rows.append({
                        "company_id": cid,
                        "company_name": c_name,
                        "products_label": top_prod_label,
                        "total_qty": b_qty,
                        "total_revenue": b_rev,
                        "top_channel": top_ch_label,
                    })

            sort_key = (
                (lambda r: (r["total_qty"], r["total_revenue"]))
                if sort_by == "qty"
                else (lambda r: (r["total_revenue"], r["total_qty"]))
            )
            branch_rows.sort(key=sort_key, reverse=True)

            target_label = f" untuk `{search_term}`" if search_term else ""
            if branch_rows:
                win = branch_rows[0]
                lines.append(
                    f"Penjualan terbanyak{target_label} tercatat di cabang **{win['company_name']}** "
                    f"(**{win['total_qty']:g} unit**, **{_format_rupiah(win['total_revenue'])}**).\n"
                )
            lines.append(f"### Peringkat Penjualan per Cabang{target_label}\n")
            if branch_rows:
                lines.append("| Peringkat | Cabang | Produk / SKU | Kuantitas Terjual | Total Omzet | Channel Teratas |")
                lines.append("|:---:|:---|:---|:---:|:---|:---|")
                tot_qty = 0.0
                tot_rev = 0.0
                for idx, br in enumerate(branch_rows, start=1):
                    tot_qty += br["total_qty"]
                    tot_rev += br["total_revenue"]
                    lines.append(
                        f"| {idx} | **{br['company_name']}** | {br['products_label']} | "
                        f"**{br['total_qty']:g} unit** | **{_format_rupiah(br['total_revenue'])}** | {br['top_channel']} |"
                    )
                lines.append(f"| | **TOTAL** | | **{tot_qty:g} unit** | **{_format_rupiah(tot_rev)}** | |")
            else:
                lines.append(f"Tidak ditemukan transaksi penjualan terkonfirmasi{target_label} pada periode ini.")

            if agg_channels:
                sorted_ch = sorted(
                    agg_channels.items(),
                    key=lambda kv: (kv[1]["total_qty"], kv[1]["total_revenue"]) if sort_by == "qty" else (kv[1]["total_revenue"], kv[1]["total_qty"]),
                    reverse=True,
                )
                lines.append(f"\n#### Ringkasan per Channel Penjualan{target_label}:\n")
                lines.append("| No | Channel | Jumlah Order | Kuantitas Terjual | Total Omzet |")
                lines.append("|:---:|:---|:---:|:---:|:---|")
                for c_idx, (ch_n, c_vals) in enumerate(sorted_ch, start=1):
                    lines.append(
                        f"| {c_idx} | **{ch_n}** | {int(c_vals['order_count']):,} | "
                        f"{c_vals['total_qty']:g} unit | **{_format_rupiah(c_vals['total_revenue'])}** |"
                    )
        else:
            # Merge products across branches
            merged: Dict[str, Dict[str, Any]] = {}
            for cid, res in mc_results.items():
                if not res or not isinstance(res, dict):
                    continue
                p_list = res.get("products") or []
                for p in p_list:
                    key = p.get("code") or str(p.get("product_id") or p.get("name"))
                    if key not in merged:
                        merged[key] = {
                            "code": p.get("code") or "-",
                            "name": p.get("name") or "-",
                            "category": p.get("category") or "-",
                            "branch_rev": {},
                            "branch_qty": {},
                            "total_rev": 0.0,
                            "total_qty": 0.0,
                        }
                    rev = float(p.get("total_revenue") or 0.0)
                    qty = float(p.get("total_qty") or 0.0)
                    merged[key]["branch_rev"][cid] = rev
                    merged[key]["branch_qty"][cid] = qty
                    merged[key]["total_rev"] += rev
                    merged[key]["total_qty"] += qty

            sorted_merged = sorted(merged.values(), key=lambda x: x["total_rev"], reverse=True)
            limit = int(params.get("limit") or 10)
            top_list = sorted_merged[:limit]

            lines.append(f"### Laporan Produk Pendapatan Tertinggi Multi-Cabang (Top {len(top_list)} Nasional)\n")
            headers = ["No", "Kode / SKU", "Nama Produk"]
            for cid in sorted(mc_results.keys()):
                c_name = COMPANY_NAME_MAP.get(cid, f"Cabang {cid}")
                headers.append(f"Omzet {c_name}")
            headers.append("Total Omzet Gabungan")

            lines.append("| " + " | ".join(headers) + " |")
            lines.append("|:" + ":|:".join(["---"] * len(headers)) + ":|")

            branch_totals = {cid: 0.0 for cid in sorted(mc_results.keys())}
            grand_total = 0.0

            for idx, p in enumerate(top_list, start=1):
                row = [str(idx), f"`{p['code']}`", p["name"]]
                for cid in sorted(mc_results.keys()):
                    b_rev = p["branch_rev"].get(cid, 0.0)
                    branch_totals[cid] += b_rev
                    row.append(_format_rupiah(b_rev) if b_rev > 0 else "-")
                grand_total += p["total_rev"]
                row.append(f"**{_format_rupiah(p['total_rev'])}**")
                lines.append("| " + " | ".join(row) + " |")

            tot_row = ["", "**TOTAL GABUNGAN**", ""]
            for cid in sorted(mc_results.keys()):
                tot_row.append(f"**{_format_rupiah(branch_totals[cid])}**")
            tot_row.append(f"**{_format_rupiah(grand_total)}**")
            lines.append("| " + " | ".join(tot_row) + " |")

    elif tool == "get_sales_summary":
        lines.append("### Perbandingan Performa Penjualan Multi-Cabang\n")
        lines.append("| Cabang | Omzet Terealisasi | Jumlah Order | Rata-Rata Order (AOV) | Pipeline Quotation |")
        lines.append("|:---|:---|:---:|:---|:---|")
        tot_rev = 0.0
        tot_orders = 0
        tot_pipe = 0.0
        for cid in sorted(mc_results.keys()):
            res = mc_results[cid]
            c_name = COMPANY_NAME_MAP.get(cid, f"Cabang {cid}")
            if not res or not isinstance(res, dict):
                lines.append(f"| **{c_name}** | *Tidak ada data* | - | - | - |")
                continue
            conf = res.get("confirmed_metrics") or {}
            rev = float(conf.get("realized_revenue") or 0.0)
            cnt = int(conf.get("orders_count") or 0)
            aov = float(conf.get("average_order_value") or 0.0)
            pipe = res.get("pipeline_metrics") or {}
            pval = float(pipe.get("quotation_potential_value") or 0.0)
            tot_rev += rev
            tot_orders += cnt
            tot_pipe += pval
            lines.append(f"| **{c_name}** | **{_format_rupiah(rev)}** | {cnt:,} | {_format_rupiah(aov)} | {_format_rupiah(pval)} |")
        aov_grand = (tot_rev / tot_orders) if tot_orders > 0 else 0.0
        lines.append(f"| **TOTAL GABUNGAN** | **{_format_rupiah(tot_rev)}** | **{tot_orders:,}** | **{_format_rupiah(aov_grand)}** | **{_format_rupiah(tot_pipe)}** |")

    elif tool == "get_sales_orders":
        lines.append("### Ringkasan Pesanan Penjualan Multi-Cabang\n")
        lines.append("| Cabang | Jumlah Pesanan | Total Nominal | Status Filter |")
        lines.append("|:---|:---:|:---|:---:|")
        grand_cnt = 0
        grand_sum = 0.0
        for cid in sorted(mc_results.keys()):
            res = mc_results[cid]
            c_name = COMPANY_NAME_MAP.get(cid, f"Cabang {cid}")
            if not res or not isinstance(res, dict):
                lines.append(f"| **{c_name}** | - | *Tidak ada data* | - |")
                continue
            cnt = int(res.get("total_count") or 0)
            tot = float(res.get("total_amount_sum") or 0.0)
            st = res.get("status_filter") or "confirmed"
            grand_cnt += cnt
            grand_sum += tot
            lines.append(f"| **{c_name}** | {cnt:,} | **{_format_rupiah(tot)}** | `{st}` |")
        lines.append(f"| **TOTAL GABUNGAN** | **{grand_cnt:,}** | **{_format_rupiah(grand_sum)}** | |")

    # Generic Fallback for other tools
    else:
        lines.append(f"### Laporan Multi-Perusahaan (`{tool}`)\n")
        for cid in sorted(mc_results.keys()):
            c_name = COMPANY_NAME_MAP.get(cid, f"Cabang {cid}")
            res = mc_results.get(cid)
            if not res:
                st_text = "Akses Dibatasi (HTTP 403)" if cid in unauth else "Tidak Ada Data"
                lines.append(f"- **{c_name}**: *{st_text}*")
            else:
                cnt = _extract_total_count(res, 1)
                lines.append(f"- **{c_name}**: Ditemukan {cnt} data master / catatan.")

    # Partial Permission Alert if unauthorized companies exist
    if unauth:
        unauth_names = [COMPANY_NAME_MAP.get(c, f"Cabang {c}") for c in unauth]
        lines.append(
            f"\n> [!WARNING]\n"
            f"> **Akses Dibatasi**: Data tidak mencakup entitas **{', '.join(unauth_names)}** "
            f"karena akun Anda belum memiliki izin akses untuk entitas tersebut (HTTP 403 Forbidden)."
        )

    return "\n".join(lines)


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
    is_multi = state.get("is_multi_tool") or (len(multi_results) > 1)

    # 0. Multi-Company Execution Unified Formatting
    if state.get("is_multi_company") and state.get("multi_company_results"):
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

    # 2. Out of Scope Guardrail
    if domain == "out_of_scope":
        from app.graph.nodes.router import MARKETPLACE_WRITE_PATTERNS, _match_any
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
            "Permintaan ini berada di luar cakupan operasional Asisten AI FastPrint ERP.\n\n"
            "Sistem ini dikhususkan untuk melayani penelusuran data operasional dan pembukuan internal perusahaan:\n"
            "- **Tagihan Vendor / Pemasok** (Hutang usaha & status verifikasi faktur)\n"
            "- **Saldo Akun Buku Kas & Neraca** (Accounting Balance & Kas Kecil)\n"
            "- **Entri Jurnal Umum** (General Journal Entries)\n"
            "- **Stok Barang & Master Produk** (Inventory & Kuantitas Gudang)\n"
            "- **Penjualan & Marketplace** (Sales Order, Kesehatan Akun, Sinkronisasi Stok/Harga, Potongan Fee, Cashflow)\n"
            "- **Pengetahuan & SOP Perusahaan** (SOP Retur, Kebijakan, dan Spesifikasi Teknis)\n\n"
            "Silakan sampaikan pertanyaan spesifik Anda seputar data dan modul operasional di atas."
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

    # 3. General Domain / Guardrail Inquiries
    if (domain == "general" or not tool) and not is_multi:
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

        hint_str = f" ({', '.join(filter_hints)})" if filter_hints else ""
        return (
            f"### Data Tidak Ditemukan\n\n"
            f"Setelah dilakukan penelusuran pada database ERP FastPrint{hint_str}, "
            f"tidak ditemukan data transaksi yang sesuai.\n\n"
            f"**Catatan Verifikasi**:\n"
            f"- Sistem telah mencoba relaksasi batasan filter tanggal dan kata kunci pencarian secara otomatis (maksimal 2x).\n"
            f"- Silakan periksa kembali ketepatan nomor dokumen, nama vendor, atau pastikan periode transaksi telah tercatat di ERP."
        )

    multi_results = state.get("multi_tool_results") or {}
    active_tools = state.get("active_tools") or []
    is_multi = state.get("is_multi_tool") or (len(multi_results) > 1) or (len(active_tools) > 1)

    # 4. Multi-Tool Execution Unified Formatting (Milestone 4)
    if is_multi and (multi_results or active_tools):
        sections: List[str] = []

        if active_tools:
            tool_counts: Dict[str, int] = {}
            for p in active_tools:
                tn = p.get("tool", "unnamed_tool")
                tool_counts[tn] = tool_counts.get(tn, 0) + 1

            for idx, plan in enumerate(active_tools):
                t_name = plan.get("tool")
                if not t_name:
                    continue
                key = f"{t_name}_{idx}" if tool_counts.get(t_name, 0) > 1 else t_name
                # Check plan result first, then composite key, then base tool name
                res = plan.get("result")
                if res is None and key in multi_results:
                    res = multi_results[key]
                elif res is None and t_name in multi_results:
                    res = multi_results[t_name]

                if res and plan.get("status") != "error":
                    formatted = _format_single_tool_presentation(t_name, res, user_query, plan.get("params"))
                    if formatted:
                        sections.append(formatted)
                else:
                    domain_label = "Modul ERP"
                    if "stock" in t_name or "product" in t_name or "location" in t_name:
                        domain_label = "Inventaris / Stok"
                    elif "balance" in t_name or "invoice" in t_name or "journal" in t_name:
                        domain_label = "Akuntansi / Keuangan"
                    elif "sales" in t_name or "selling" in t_name:
                        domain_label = "Penjualan (Sales)"
                    plan_err = plan.get("error") or (res.get("error") if isinstance(res, dict) else None)
                    if plan.get("status") == "error" or plan_err:
                        sections.append(
                            f"### Kendala Modul {domain_label} (`{t_name}`)\n\n"
                            f"> [!WARNING]\n"
                            f"> Gagal memuat data modul ini: {plan_err or 'Terjadi kendala pada layanan ERP.'}"
                        )
                    else:
                        sections.append(
                            f"### Informasi {domain_label} (`{t_name}`)\n\n"
                            f"Tidak ditemukan data atau terjadi kendala saat memuat informasi untuk modul ini."
                        )
        else:
            for key, res in multi_results.items():
                base_tool = key.rsplit("_", 1)[0] if ("_" in key and key.rsplit("_", 1)[1].isdigit()) else key
                if res:
                    formatted = _format_single_tool_presentation(base_tool, res, user_query, state.get("extracted_params"))
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
            return "\n\n---\n\n".join(sections)

    # 5. Single-Tool Valid Data Presentation (Backward Compatibility)
    return _format_single_tool_presentation(tool, tool_result, user_query, state.get("extracted_params"))


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
    domain = state.get("detected_domain") or ""
    tool = state.get("detected_tool") or ""
    params = state.get("extracted_params") or {}

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


def _finalize_followups(
    followups: List[Dict[str, str]],
    user_query: Optional[str],
    state: AgentState,
) -> List[Dict[str, str]]:
    """
    Deduplicate, strip self-repeating chips matching current user_query,
    validate routability via classify_query_rule_based, and backfill cleanly.
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
        norm_q = re.sub(r"\s+", " ", q_raw.rstrip("?!.,;:'\"> ")).lower()
        if norm_q == norm_user or norm_q in seen_queries:
            return
        if check_route and not _is_routable_followup_query(q_raw):
            return
        seen_queries.add(norm_q)
        valid_chips.append({
            "label": _clean_label_emojis(lbl_raw) or lbl_raw[:35],
            "query": q_raw,
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


async def asynthesizer_node(
    state: AgentState,
    llm: Optional[BaseChatModel] = None,
) -> Dict[str, Any]:
    """
    Asynchronous LangGraph pipeline node that synthesizes the final conversational response
    with True Streaming and hybrid presentation support.
    """
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

    # For hybrid inquiries, combine live ERP results with semantic RAG findings
    if query_mode == "hybrid":
        rag_context = state.get("rag_context")
        rag_sources = state.get("rag_sources")
        rag_section = _format_rag_section(rag_context, rag_sources)
        if rag_section:
            deterministic_answer = f"{deterministic_answer}\n\n---\n\n{rag_section}"
        else:
            deterministic_answer = (
                f"{deterministic_answer}\n\n"
                f"> [!NOTE]\n"
                f"> Dokumen SOP atau spesifikasi kebijakan terkait belum terdaftar di basis data pengetahuan perusahaan."
            )

    final_text = deterministic_answer
    followups: List[Dict[str, str]] = []

    # Emit deterministic table immediately via custom event for instant TTFT (<10ms)
    if deterministic_answer:
        try:
            await adispatch_custom_event("token", {"content": deterministic_answer})
        except Exception as dispatch_err:
            logger.debug(f"Could not dispatch deterministic_chunk: {dispatch_err}")

    # Attempt LLM natural language polish and follow-up generation if LLM is provided/available
    model = llm
    is_test_env = "unittest" in sys.modules or os.environ.get("FASTPRINT_TEST_ENV") == "true"
    if model is None and not is_test_env:
        try:
            model = get_chat_model()
        except Exception:
            model = None

    is_multi_co = state.get("is_multi_company", False)
    is_large_deterministic = is_multi_co or len(deterministic_answer) > 1500

    if model is not None and status == "VALID":
        try:
            if is_large_deterministic:
                # For large tabular or multi-company reports, keep deterministic tables intact
                # to prevent hallucination, truncation, and slow token generation.
                final_text = deterministic_answer
                prompt = (
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
                response = await model.ainvoke([
                    SystemMessage(content="Anda adalah Asisten AI FastPrint ERP yang presisi, lugas, dan bebas AI slop."),
                    HumanMessage(content=prompt),
                ])
                resp_meta = getattr(response, "response_metadata", {}) or {}
                resolved_model = resp_meta.get("model_name") or resp_meta.get("model")
                if resolved_model:
                    actual_model = str(resolved_model)
                content = getattr(response, "content", None)
                if content and isinstance(content, str) and content.strip():
                    _, llm_followups = _extract_llm_followups(content.strip())
                    if llm_followups:
                        followups = llm_followups
            else:
                prompt = (
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
                response = await model.ainvoke([
                    SystemMessage(content="Anda adalah Asisten AI FastPrint ERP yang presisi, lugas, dan bebas AI slop."),
                    HumanMessage(content=prompt),
                ])
                resp_meta = getattr(response, "response_metadata", {}) or {}
                resolved_model = resp_meta.get("model_name") or resp_meta.get("model")
                if resolved_model:
                    actual_model = str(resolved_model)
                content = getattr(response, "content", None)
                if content and isinstance(content, str) and content.strip():
                    clean_ans, llm_followups = _extract_llm_followups(content.strip())
                    clean_ans = _clean_ai_slop(clean_ans)
                    if clean_ans:
                        final_text = clean_ans
                    if llm_followups:
                        followups = llm_followups
        except Exception as exc:
            logger.warning(f"LLM synthesis failed, falling back to deterministic output: {exc}")
            final_text = deterministic_answer

    # Fallback to deterministic follow-up generation if LLM didn't provide suggestions
    if not followups:
        followups = generate_deterministic_followups(state)
    followups = _finalize_followups(followups, user_query, state)

    if retry_header and not final_text.startswith("> 🔄"):
        final_text = f"{retry_header}{final_text}"

    out_dict: Dict[str, Any] = {
        "final_answer": final_text,
        "deterministic_answer": deterministic_answer,
        "suggested_followups": followups,
        "messages": [AIMessage(content=final_text)],
    }
    if actual_model:
        out_dict["active_model"] = actual_model
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

    # For hybrid inquiries, combine live ERP results with semantic RAG findings
    if query_mode == "hybrid":
        rag_context = state.get("rag_context")
        rag_sources = state.get("rag_sources")
        rag_section = _format_rag_section(rag_context, rag_sources)
        if rag_section:
            deterministic_answer = f"{deterministic_answer}\n\n---\n\n{rag_section}"
        else:
            deterministic_answer = (
                f"{deterministic_answer}\n\n"
                f"> [!NOTE]\n"
                f"> Dokumen SOP atau spesifikasi kebijakan terkait belum terdaftar di basis data pengetahuan perusahaan."
            )

    final_text = deterministic_answer
    followups: List[Dict[str, str]] = []

    # Attempt LLM natural language polish and follow-up generation if LLM is provided/available
    model = llm
    is_test_env = "unittest" in sys.modules or os.environ.get("FASTPRINT_TEST_ENV") == "true"
    if model is None and not is_test_env:
        try:
            model = get_chat_model()
        except Exception:
            model = None

    is_multi_co = state.get("is_multi_company", False)
    is_large_deterministic = is_multi_co or len(deterministic_answer) > 1500

    if model is not None and status == "VALID":
        try:
            if is_large_deterministic:
                final_text = deterministic_answer
                prompt = (
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
                response = model.invoke([
                    SystemMessage(content="Anda adalah Asisten AI FastPrint ERP yang presisi, lugas, dan bebas AI slop."),
                    HumanMessage(content=prompt),
                ])
                resp_meta = getattr(response, "response_metadata", {}) or {}
                resolved_model = resp_meta.get("model_name") or resp_meta.get("model")
                if resolved_model:
                    actual_model = str(resolved_model)
                content = getattr(response, "content", None)
                if content and isinstance(content, str) and content.strip():
                    _, llm_followups = _extract_llm_followups(content.strip())
                    if llm_followups:
                        followups = llm_followups
            else:
                prompt = (
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
                response = model.invoke([
                    SystemMessage(content="Anda adalah Asisten AI FastPrint ERP yang presisi, lugas, dan bebas AI slop."),
                    HumanMessage(content=prompt),
                ])
                resp_meta = getattr(response, "response_metadata", {}) or {}
                resolved_model = resp_meta.get("model_name") or resp_meta.get("model")
                if resolved_model:
                    actual_model = str(resolved_model)
                content = getattr(response, "content", None)
                if content and isinstance(content, str) and content.strip():
                    clean_ans, llm_followups = _extract_llm_followups(content.strip())
                    clean_ans = _clean_ai_slop(clean_ans)
                    if clean_ans:
                        final_text = clean_ans
                    if llm_followups:
                        followups = llm_followups
        except Exception as exc:
            logger.warning(f"LLM synthesis failed, falling back to deterministic output: {exc}")
            final_text = deterministic_answer

    # Fallback to deterministic follow-up generation if LLM didn't provide suggestions
    if not followups:
        followups = generate_deterministic_followups(state)
    followups = _finalize_followups(followups, user_query, state)

    if retry_header and not final_text.startswith("> 🔄"):
        final_text = f"{retry_header}{final_text}"

    out_dict: Dict[str, Any] = {
        "final_answer": final_text,
        "deterministic_answer": deterministic_answer,
        "suggested_followups": followups,
        "messages": [AIMessage(content=final_text)],
    }
    if actual_model:
        out_dict["active_model"] = actual_model
    return out_dict
