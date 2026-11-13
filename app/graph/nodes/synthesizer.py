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


def _format_single_tool_presentation(tool: Optional[str], tool_result: Any) -> str:
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
                lines.append("| Tanggal | No. Entri | Referensi | Rekanan | Keterangan | Debit | Kredit | Saldo Berjalan |")
                lines.append("|:---|:---|:---|:---|:---|:---|:---|:---|")
                for it in items:
                    tgl = it.get("date") or "-"
                    num = it.get("entry_number") or "-"
                    ref = it.get("ref") or "-"
                    partner = it.get("partner_name") or "-"
                    desc = it.get("description") or "-"
                    deb = _format_rupiah(it.get("debit", 0.0))
                    kred = _format_rupiah(it.get("credit", 0.0))
                    run_bal = _format_rupiah(it.get("running_balance", 0.0))
                    lines.append(f"| {tgl} | `{num}` | {ref} | {partner} | {desc} | {deb} | {kred} | **{run_bal}** |")
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
            lines.append("- **Status**: ✅ **SEIMBANG (BALANCED)** (Selisih: Rp 0,00)")
        else:
            lines.append(f"- **Status**: ⚠️ **TIDAK SEIMBANG** (Selisih: {_format_rupiah(diff)})")

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
                lines.append(f"- ⚠️ **Tagihan Lewat Jatuh Tempo (Overdue)**: **{overdue_count} faktur**")
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
            lines.append("- **Status**: ✅ **Terekonsiliasi Sempurna (Fully Reconciled)**")
        else:
            lines.append("- **Status**: ⚠️ **Memerlukan Tindak Lanjut Pencocokan Transaksi**")
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

        lines.append("| SKU / Kode | Nama Produk | Perusahaan | Lokasi / Gudang | Qty Fisik |")
        lines.append("|:---|:---|:---|:---|:---|")
        for st_item in items:
            code = st_item.get("code") or st_item.get("sku") or "-"
            name = st_item.get("name") or st_item.get("product_name") or "-"
            comp = st_item.get("company_name") or f"Company #{st_item.get('company_id', 1)}"
            loc = st_item.get("location_name") or (f"Location #{st_item.get('location_id')}" if st_item.get("location_id") else "Semua Lokasi")
            qty = st_item.get("qty_on_hand") or st_item.get("quantity") or 0.0
            lines.append(f"| `{code}` | {name} | {comp} | {loc} | **{qty} unit** |")

        _append_pagination_footer(lines, len(items), total_count)

        # Render breakdown if available (e.g. single product multi-location breakdown)
        if isinstance(tool_result, dict) and tool_result.get("breakdown"):
            lines.append("\n#### Rincian Stok per Lokasi / Rak:")
            lines.append("| Gudang | Lokasi / Rak | Kuantitas |")
            lines.append("|:---|:---|:---|")
            for bd in tool_result["breakdown"]:
                wh = bd.get("warehouse_name") or f"Gudang #{bd.get('warehouse_id', '-')}"
                loc_n = bd.get("location_name") or f"Rak #{bd.get('location_id', '-')}"
                b_qty = bd.get("qty", 0.0)
                lines.append(f"| {wh} | {loc_n} | **{b_qty:g} unit** |")
        return "\n".join(lines)

    # 4. Product Info
    if tool == "get_product_info":
        lines.append("### Informasi Master Produk (Catalog)\n")
        products = tool_result.get("products") if isinstance(tool_result, dict) else (
            tool_result if isinstance(tool_result, list) else [tool_result]
        )
        total_count = _extract_total_count(tool_result, len(products))
        if products:
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
                if "last_unit_price" in p or "price" in p or "sales_price" in p:
                    price_val = p.get("last_unit_price") or p.get("price") or p.get("sales_price", 0.0)
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

    # Default fallback
    return f"Hasil eksekusi data ERP: {str(tool_result)}"


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
            "- **Pengetahuan & SOP Perusahaan** (SOP Retur, Kebijakan, dan Spesifikasi Teknis)"
        )

    # 1. Error Status Handling
    if status == "ERROR":
        msg = error_message or "Terjadi kesalahan internal pada layanan ERP."
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
        return (
            "### Permintaan di Luar Cakupan (Out of Scope)\n\n"
            "Mohon maaf, permintaan Anda tidak dapat diproses karena berada di luar cakupan operasional Asisten AI FastPrint ERP.\n\n"
            "Sistem ini dikhususkan untuk melayani penelusuran data operasional dan pembukuan internal perusahaan:\n"
            "- **Tagihan Vendor / Pemasok** (Hutang usaha & status verifikasi faktur)\n"
            "- **Saldo Akun Buku Kas & Neraca** (Accounting Balance & Kas Kecil)\n"
            "- **Entri Jurnal Umum** (General Journal Entries)\n"
            "- **Stok Barang & Master Produk** (Inventory & Kuantitas Gudang)\n"
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
            "- **Stok Barang & Master Produk** (Inventory & Warehouse Stock)\n\n"
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

                if res:
                    formatted = _format_single_tool_presentation(t_name, res)
                    if formatted:
                        sections.append(formatted)
                else:
                    domain_label = "Modul ERP"
                    if "stock" in t_name or "product" in t_name or "location" in t_name:
                        domain_label = "Inventaris / Stok"
                    elif "balance" in t_name or "invoice" in t_name or "journal" in t_name:
                        domain_label = "Akuntansi / Keuangan"
                    sections.append(
                        f"### Informasi {domain_label} (`{t_name}`)\n\n"
                        f"Tidak ditemukan data atau terjadi kendala saat memuat informasi untuk modul ini."
                    )
        else:
            for key, res in multi_results.items():
                base_tool = key.rsplit("_", 1)[0] if ("_" in key and key.rsplit("_", 1)[1].isdigit()) else key
                if res:
                    formatted = _format_single_tool_presentation(base_tool, res)
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
    return _format_single_tool_presentation(tool, tool_result)


def _extract_llm_followups(llm_output: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Extract structured follow-up recommendations from LLM output block.
    Supports ```json:followups ... ```, ```json ... ```, or [FOLLOWUPS] ... [/FOLLOWUPS].

    Returns:
        Tuple of (clean_markdown_answer, list_of_followup_dicts)
    """
    if not llm_output or not isinstance(llm_output, str):
        return ("", [])

    raw = llm_output.strip()
    followups: List[Dict[str, str]] = []
    clean_text = raw

    # 1. Match codeblock pattern: ```(?:json:followups|json)?\s*(\[\s*\{.*?\}\s*\])\s*```
    codeblock_pat = re.compile(
        r"```(?:json:followups|json)?\s*(\[\s*\{.*?\}\s*\])\s*```",
        re.DOTALL | re.IGNORECASE,
    )
    match = codeblock_pat.search(raw)

    if match:
        json_str = match.group(1).strip()
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and item.get("label") and item.get("query"):
                        followups.append({
                            "label": str(item["label"]).strip()[:30],
                            "query": str(item["query"]).strip(),
                            "icon": str(item.get("icon", "fa-arrow-right")).strip() or "fa-arrow-right",
                        })
                if followups:
                    clean_text = (raw[:match.start()] + raw[match.end():]).strip()
        except Exception as exc:
            logger.debug("Failed parsing LLM followups JSON block: %s", exc)

    # 2. Tagged bracket pattern fallback: [FOLLOWUPS] ... [/FOLLOWUPS]
    if not followups:
        tagged_pat = re.compile(
            r"\[FOLLOWUPS\]\s*(\[\s*\{.*?\}\s*\])\s*\[/FOLLOWUPS\]",
            re.DOTALL | re.IGNORECASE,
        )
        t_match = tagged_pat.search(raw)
        if t_match:
            try:
                parsed = json.loads(t_match.group(1).strip())
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and item.get("label") and item.get("query"):
                            followups.append({
                                "label": str(item["label"]).strip()[:30],
                                "query": str(item["query"]).strip(),
                                "icon": str(item.get("icon", "fa-arrow-right")).strip() or "fa-arrow-right",
                            })
                    if followups:
                        clean_text = (raw[:t_match.start()] + raw[t_match.end():]).strip()
            except Exception as exc:
                logger.debug("Failed parsing LLM tagged followups: %s", exc)

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
        return suggestions[:3]

    # 2. Inventory Domain
    if domain == "inventory" or any(k in tool for k in ("stock", "product", "location", "warehouse", "scrap", "transfer")):
        code = params.get("code") or params.get("product_id")
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

    # 4. General / Knowledge / Guardrail Fallback
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

    return suggestions[:3]


def synthesizer_node(
    state: AgentState,
    llm: Optional[BaseChatModel] = None,
) -> Dict[str, Any]:
    """
    LangGraph pipeline node that synthesizes the final conversational response.

    Args:
        state: Current AgentState.
        llm: Optional BaseChatModel instance for natural language synthesis.

    Returns:
        Partial AgentState update dictionary containing:
            - final_answer: Markdown formatted response string
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

    if model is not None and status == "VALID":
        try:
            prompt = (
                "Anda adalah Asisten AI FastPrint ERP yang profesional dan teliti.\n"
                "Sajikan data laporan berikut dalam Bahasa Indonesia yang formal, ringkas, dan jelas.\n"
                "Pertahankan seluruh angka transaksi, kode, dan rincian tabel Markdown dengan akurat tanpa mengubah nilai.\n"
                "DILARANG KERAS menghitung ulang atau mengestimasi angka nominal keuangan (omzet, HPP, laba kotor, laba bersih, selisih rekonsiliasi, saldo neraca). Seluruh angka wajib dikutip persis dari data laporan terstruktur.\n\n"
                "Selain menyajikan data laporan di atas, susunlah 2 hingga 3 rekomendasi pertanyaan/prompt lanjutan yang paling logis, relevan, dan kontekstual untuk diajukan pengguna berikutnya berdasarkan data riil yang ditemukan di atas (misal: rincian transaksi terkait, mutasi kartu stok produk ini, cek tagihan overdue, atau rekonsiliasi bank).\n\n"
                "ATURAN KETAT REKOMENDASI:\n"
                "1. Wajib hanya menyarankan pertanyaan penelusuran data (READ-ONLY) pada modul FastPrint ERP. DILARANG menyarankan aksi membuat/mengubah transaksi.\n"
                "2. Wajib menggunakan entitas riil (kode SKU, nama vendor, kode akun, nama gudang) yang terdapat pada data di atas.\n"
                "3. Formatkan rekomendasi di akhir respons dalam blok JSON khusus berikut:\n"
                "```json:followups\n"
                "[\n"
                '  {"label": "Label Singkat (1-3 kata)", "query": "Prompt lengkap yang akan dieksekusi", "icon": "fa-icon-name"}\n'
                "]\n"
                "```\n\n"
                f"Pertanyaan Pengguna: {user_query}\n"
                f"Data Laporan Terstruktur:\n{deterministic_answer}"
            )
            response = model.invoke([
                SystemMessage(content="Anda adalah Asisten AI FastPrint ERP."),
                HumanMessage(content=prompt),
            ])
            content = getattr(response, "content", None)
            if content and isinstance(content, str) and content.strip():
                clean_ans, llm_followups = _extract_llm_followups(content.strip())
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

    if retry_header and not final_text.startswith("> 🔄"):
        final_text = f"{retry_header}{final_text}"

    return {
        "final_answer": final_text,
        "suggested_followups": followups,
        "messages": [AIMessage(content=final_text)],
    }
