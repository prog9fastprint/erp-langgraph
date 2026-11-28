"""
Markdown presentation formatters for FastPrint ERP accounting tools.
"""
from typing import Any, Callable, Dict, List, Optional
from app.graph.nodes.formatters.common import (
    _append_pagination_footer,
    _extract_total_count,
    _format_rupiah,
    _resolve_company_display_label,
    _resolve_period_bounds,
)


def format_get_journal_entries(
    tool: str,
    tool_result: Any,
    user_query: Optional[str] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
    default_company_id: Optional[int] = None,
) -> str:
    """Format accounting.get_journal_entries payload into Markdown."""
    lines: List[str] = []
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

        is_bill = any(item.get("type") == "bill_line" or "product_name" in item for item in doc_lines)
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


def format_get_account_balance(
    tool: str,
    tool_result: Any,
    user_query: Optional[str] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
    default_company_id: Optional[int] = None,
) -> str:
    """Format accounting.get_account_balance payload into Markdown."""
    lines: List[str] = []
    accounts = tool_result.get("accounts") or []
    total_count = _extract_total_count(tool_result, len(accounts))
    totals = tool_result.get("totals") or {}
    comp_lbl = _resolve_company_display_label(tool_result, extracted_params, default_company_id)
    d_from, d_to = _resolve_period_bounds(tool_result, extracted_params)
    scope_notes: List[str] = []
    if comp_lbl:
        scope_notes.append(f"Cabang: {comp_lbl}")
    if d_from or d_to:
        scope_notes.append(f"Periode: {d_from or '-'} s/d {d_to or '-'}")
    scope_suffix = f" ({', '.join(scope_notes)})" if scope_notes else ""
    lines.append(f"### Informasi Saldo Chart of Account (Accounting Balance){scope_suffix}\n")

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
        _append_pagination_footer(
            lines,
            len(accounts),
            total_count,
            filter_hint="Klik **Lihat Full Table** atau sebutkan kode/nama akun spesifik untuk menelusuri seluruh data.",
        )
    elif not totals:
        net = _format_rupiah(tool_result.get("net_balance", 0.0))
        deb = _format_rupiah(tool_result.get("total_debit", 0.0))
        kred = _format_rupiah(tool_result.get("total_credit", 0.0))
        lines.append(f"- **Total Debit**: {deb}")
        lines.append(f"- **Total Kredit**: {kred}")
        lines.append(f"- **Saldo Bersih**: **{net}**")
    return "\n".join(lines)


def format_get_general_ledger(
    tool: str,
    tool_result: Any,
    user_query: Optional[str] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
    default_company_id: Optional[int] = None,
) -> str:
    """Format accounting.get_general_ledger payload into Markdown."""
    lines: List[str] = []
    accounts = tool_result.get("accounts") or []
    d_from = tool_result.get("date_from") or "-"
    d_to = tool_result.get("date_to") or "-"
    comp_lbl = _resolve_company_display_label(tool_result, extracted_params, default_company_id)
    comp_tag = f" (Cabang: {comp_lbl})" if comp_lbl else ""
    lines.append(f"### Buku Besar Akun (General Ledger){comp_tag} — Periode: {d_from} s/d {d_to}\n")
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


def _append_hierarchical_account_breakdown(
    lines: List[str],
    breakdown: List[Dict[str, Any]],
    section_heading: str,
    group_col_label: str,
    type_col_label: str,
) -> None:
    """Append P&L or Balance Sheet account breakdown rows, omitting zero-balance accounts."""
    if not breakdown:
        return
    lines.append(f"\n{section_heading}")
    lines.append(f"| {group_col_label} | {type_col_label} | Kode Akun | Nama Akun | Saldo |")
    lines.append("|:---|:---|:---|:---|:---|")
    zero_count = 0
    written_rows = 0
    for grp in breakdown:
        grp_name = grp.get("group_name") or grp.get("name") or "-"
        for tp in grp.get("types", []):
            tp_name = tp.get("type_name") or tp.get("name") or "-"
            for acc in tp.get("accounts", []):
                raw_bal = float(acc.get("balance", 0.0) or 0.0)
                if abs(raw_bal) <= 0.01:
                    zero_count += 1
                    continue
                acc_c = acc.get("account_code") or "-"
                acc_n = acc.get("account_name") or "-"
                lines.append(f"| {grp_name} | {tp_name} | `{acc_c}` | {acc_n} | {_format_rupiah(raw_bal)} |")
                written_rows += 1
    if written_rows == 0:
        lines.append("| - | - | - | *Seluruh akun pada rincian ini bersaldo Rp 0,00* | Rp 0,00 |")
    elif zero_count > 0:
        lines.append(f"\n*Catatan: {zero_count} akun bersaldo Rp 0,00 tidak ditampilkan.*")


def format_get_profit_and_loss(
    tool: str,
    tool_result: Any,
    user_query: Optional[str] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
    default_company_id: Optional[int] = None,
) -> str:
    """Format accounting.get_profit_and_loss payload into Markdown."""
    lines: List[str] = []
    summary = tool_result.get("summary") or {}
    d_from, d_to = _resolve_period_bounds(tool_result, extracted_params)
    comp_lbl = _resolve_company_display_label(tool_result, extracted_params, default_company_id)
    comp_tag = f" ({comp_lbl})" if comp_lbl else ""
    lines.append(f"### Laporan Laba Rugi (Profit & Loss){comp_tag} — Periode: {d_from or '-'} s/d {d_to or '-'}\n")
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
    _append_hierarchical_account_breakdown(
        lines,
        breakdown,
        section_heading="#### Rincian Hierarki Laba Rugi:",
        group_col_label="Grup Akun",
        type_col_label="Tipe / Sub-Tipe",
    )
    return "\n".join(lines)


def format_get_balance_sheet(
    tool: str,
    tool_result: Any,
    user_query: Optional[str] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
    default_company_id: Optional[int] = None,
) -> str:
    """Format accounting.get_balance_sheet payload into Markdown."""
    lines: List[str] = []
    summary = tool_result.get("summary") or {}
    as_of = tool_result.get("as_of_date") or (extracted_params.get("as_of_date") if isinstance(extracted_params, dict) else None) or "-"
    comp_lbl = _resolve_company_display_label(tool_result, extracted_params, default_company_id)
    comp_tag = f" ({comp_lbl})" if comp_lbl else ""
    lines.append(f"### Laporan Neraca Keuangan (Balance Sheet){comp_tag} — Posisi per: {as_of}\n")
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
    _append_hierarchical_account_breakdown(
        lines,
        breakdown,
        section_heading="#### Rincian Akun Neraca:",
        group_col_label="Kategori",
        type_col_label="Tipe Aset/Kewajiban",
    )
    return "\n".join(lines)


def format_get_vendor_bills(
    tool: str,
    tool_result: Any,
    user_query: Optional[str] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
    default_company_id: Optional[int] = None,
) -> str:
    """Format accounting.get_vendor_bills payload into Markdown."""
    lines: List[str] = []
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


def format_get_bank_reconciliation_summary(
    tool: str,
    tool_result: Any,
    user_query: Optional[str] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
    default_company_id: Optional[int] = None,
) -> str:
    """Format accounting.get_bank_reconciliation_summary payload into Markdown."""
    lines: List[str] = []
    summary = tool_result.get("summary") or {}
    j_name = tool_result.get("journal_name") or "Bank Utama"
    acc_c = tool_result.get("bank_account_code") or "-"
    acc_n = tool_result.get("bank_account_name") or "-"
    d_from = tool_result.get("date_from") or "-"
    d_to = tool_result.get("date_to") or "-"

    lines.append(f"### Ringkasan Rekonsiliasi Bank — {j_name}\n")
    lines.append(f"- **Buku Jurnal**: {j_name} (`{acc_c}` — {acc_n})")
    lines.append(f"- **Periode**: {d_from} s/d {d_to}")
    cons_accs = tool_result.get("consolidated_accounts") or []
    if cons_accs:
        cons_labels = ", ".join(
            f"`{ca.get('account_code')}` ({ca.get('company_name') or '-'})"
            for ca in cons_accs
            if isinstance(ca, dict)
        )
        if cons_labels:
            lines.append(f"- **Konsolidasi Akun Cabang**: {cons_labels}")
    lines.append("")

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


def format_reconcile_bank_statement(
    tool: str,
    tool_result: Any,
    user_query: Optional[str] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
    default_company_id: Optional[int] = None,
) -> str:
    """Format accounting.reconcile_bank_statement payload into Markdown."""
    if not isinstance(tool_result, dict):
        return "Hasil rekonsiliasi rekening koran tidak valid."

    meta = tool_result.get("statement_meta") or {}
    checksum = tool_result.get("checksum_gate") or {}
    summary = tool_result.get("summary") or {}
    bank_code = meta.get("bank_code") or "BANK"
    acc_no = meta.get("account_number") or "-"
    acc_holder = meta.get("account_holder") or "-"
    period = meta.get("statement_period") or "-"
    currency = meta.get("currency") or "IDR"
    status = tool_result.get("status") or "DISCREPANCY_FOUND"
    scope = tool_result.get("scope") or "-"

    lines: List[str] = []
    status_label = "KLOP 100% (Selisih Rp 0)" if status == "KLOP_100" else "Terdapat Selisih / Mutasi Belum Terjurnal"
    lines.append(f"### Hasil Rekonsiliasi Rekening Koran {bank_code} `{acc_no}` — {period}\n")
    lines.append(f"- **Pemilik Rekening**: {acc_holder}")
    lines.append(f"- **Mata Uang / Scope**: {currency} &bull; `{scope}`")
    lines.append(f"- **Golden Checksum Gate PDF**: {'**PASS**' if checksum.get('passed') else '**FAIL (Perlu Cek Parser)**'}")
    lines.append(f"- **Status Audit**: **{status_label}**")

    erp_accs = tool_result.get("erp_accounts_used") or []
    if erp_accs:
        acc_labels = ", ".join(
            f"`{a.get('account_code')}` ({a.get('company_name') or a.get('account_name') or '-'})"
            for a in erp_accs
        )
        lines.append(f"- **Akun GL ERP Terkait**: {acc_labels}")

    lines.append("")
    lines.append("| Indikator Mutasi | Rekening Koran Bank | Buku Besar (GL) ERP | Selisih |")
    lines.append("|:---|:---|:---|:---|")
    lines.append(
        f"| **Uang Masuk (Bank CR / GL Debit)** | {_format_rupiah(summary.get('bank_total_cr', 0.0))} | "
        f"{_format_rupiah(summary.get('erp_total_debit', 0.0))} | "
        f"**{_format_rupiah(summary.get('difference_inflow', 0.0))}** |"
    )
    lines.append(
        f"| **Uang Keluar (Bank DB / GL Kredit)** | {_format_rupiah(summary.get('bank_total_db', 0.0))} | "
        f"{_format_rupiah(summary.get('erp_total_credit', 0.0))} | "
        f"**{_format_rupiah(summary.get('difference_outflow', 0.0))}** |"
    )
    lines.append(
        f"\n- **Saldo Awal Bank**: {_format_rupiah(meta.get('saldo_awal', 0.0))} &bull; "
        f"**Saldo Akhir Bank**: **{_format_rupiah(meta.get('saldo_akhir', 0.0))}**"
    )
    lines.append(
        f"- **Transaksi Cocok**: **{summary.get('matched_count', 0)}** &bull; "
        f"Belum di ERP (CR: **{summary.get('unmatched_bank_inflows_count', 0)}**, "
        f"DB: **{summary.get('unmatched_bank_outflows_count', 0)}**) &bull; "
        f"Jurnal GL Tanpa Mutasi Bank: **{summary.get('unmatched_erp_count', 0)}**"
    )

    unm_cr = tool_result.get("unmatched_bank_inflows") or []
    if unm_cr:
        lines.append("\n#### 1. Uang Masuk Bank (CR) Belum Tercatat di ERP")
        lines.append("| No | Tanggal | Nominal | Keterangan Bank | Diagnosis & Rekomendasi (Read-Only) |")
        lines.append("|:---|:---|:---|:---|:---|")
        for idx, item in enumerate(unm_cr[:25], 1):
            desc = str(item.get("description") or "-")[:90].replace("|", "/")
            diag = item.get("diagnosis") or item.get("root_cause") or "-"
            rec = f"{diag} (Debit: `{item.get('recommended_debit')}` / Kredit: `{item.get('recommended_credit')}`)"
            lines.append(f"| {idx} | {item.get('date') or '-'} | {_format_rupiah(item.get('amount', 0.0))} | {desc} | {rec} |")

    unm_db = tool_result.get("unmatched_bank_outflows") or []
    if unm_db:
        lines.append("\n#### 2. Pengeluaran Bank (DB) Belum Tercatat di ERP")
        lines.append("| No | Tanggal | Nominal | Keterangan Bank | Diagnosis & Rekomendasi (Read-Only) |")
        lines.append("|:---|:---|:---|:---|:---|")
        for idx, item in enumerate(unm_db[:25], 1):
            desc = str(item.get("description") or "-")[:90].replace("|", "/")
            diag = item.get("diagnosis") or item.get("root_cause") or "-"
            rec = f"{diag} (Debit: `{item.get('recommended_debit')}` / Kredit: `{item.get('recommended_credit')}`)"
            lines.append(f"| {idx} | {item.get('date') or '-'} | {_format_rupiah(item.get('amount', 0.0))} | {desc} | {rec} |")

    unm_erp = tool_result.get("unmatched_erp_entries") or []
    if unm_erp:
        lines.append("\n#### 3. Jurnal GL ERP Tidak Ditemukan di Rekening Koran")
        lines.append("| No | Tanggal | No. Jurnal | Cabang | Arah | Nominal |")
        lines.append("|:---|:---|:---|:---|:---|:---|")
        for idx, item in enumerate(unm_erp[:25], 1):
            lines.append(
                f"| {idx} | {item.get('date') or '-'} | `{item.get('number') or '-'}` | "
                f"{item.get('company_name') or '-'} | {item.get('direction') or '-'} | {_format_rupiah(item.get('amount', 0.0))} |"
            )

    cross = tool_result.get("cross_account_suspects") or []
    if cross:
        lines.append("\n#### 4. Dugaan Salah Kamar (Tercatat di Rekening Bank Lain)")
        lines.append("| No | Tgl Bank | Nominal | No. Jurnal ERP | Akun ERP | Keterangan |")
        lines.append("|:---|:---|:---|:---|:---|:---|")
        for idx, item in enumerate(cross[:20], 1):
            lines.append(
                f"| {idx} | {item.get('bank_date') or '-'} | {_format_rupiah(item.get('amount', 0.0))} | "
                f"`{item.get('erp_number') or '-'}` | `{item.get('erp_account_code') or '-'}` | {item.get('rule') or '-'} |"
            )

    adjs = tool_result.get("recommended_adjustments") or []
    real_adjs = [a for a in adjs if float(a.get("nominal") or 0.0) > 0]
    if real_adjs:
        lines.append("\n#### Rekomendasi Jurnal Penyesuaian Manual (Read-Only)")
        lines.append("| No | Tanggal | Tipe Penyesuaian | Akun Debit | Akun Kredit | Nominal |")
        lines.append("|:---|:---|:---|:---|:---|:---|")
        for a in real_adjs:
            lines.append(
                f"| {a.get('no')} | {a.get('tanggal')} | {a.get('tipe_penyesuaian')} | "
                f"{a.get('akun_debit')} | {a.get('akun_kredit')} | **{_format_rupiah(a.get('nominal', 0.0))}** |"
            )

    return "\n".join(lines)


ACCOUNTING_FORMATTERS: Dict[str, Callable[..., str]] = {
    "fetch_vendor_invoices": format_get_journal_entries,
    "get_journal_entries": format_get_journal_entries,
    "get_account_balance": format_get_account_balance,
    "get_general_ledger": format_get_general_ledger,
    "get_profit_and_loss": format_get_profit_and_loss,
    "get_balance_sheet": format_get_balance_sheet,
    "get_vendor_bills": format_get_vendor_bills,
    "get_bank_reconciliation_summary": format_get_bank_reconciliation_summary,
    "reconcile_bank_statement": format_reconcile_bank_statement,
}
