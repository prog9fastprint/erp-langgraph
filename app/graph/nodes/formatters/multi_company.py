"""
Multi-company comparison and rollup Markdown formatters for FastPrint ERP.
"""
import re
from typing import Any, Callable, Dict, List

from app.core.branches import COMPANY_NAME_MAP
from app.graph.nodes.formatters.common import (
    _append_pagination_footer,
    _append_zero_branch_footnote,
    _build_evaluative_verdict,
    _extract_total_count,
    _format_contact_table_rows,
    _format_rupiah,
    _is_zero_metric,
    _resolve_mc_period_note,
    _resolve_product_display_price,
)
from app.graph.state import AgentState


def _format_mc_get_account_balance(
    state: AgentState,
    mc_results: Dict[int, Any],
    unauth: List[int],
    lines: List[str],
) -> None:
    """Append multi-company Markdown section for get_account_balance."""
    period_note = _resolve_mc_period_note(mc_results, state.get("extracted_params"))
    zero_branches: List[str] = []
    lines.append(f"### Laporan Saldo Akun Multi-Perusahaan (Combined Accounting Balance){period_note}\n")
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
            if cid in unauth:
                lines.append(f"| {row_idx} | {c_name} | - | - | - | *Akses Dibatasi (HTTP 403)* |")
                row_idx += 1
            else:
                zero_branches.append(c_name)
            continue

        totals = res.get("totals") or {}
        op = float(totals.get("opening_balance", res.get("opening_balance", 0.0)) or 0.0)
        deb = float(totals.get("period_debit", res.get("total_debit", 0.0)) or 0.0)
        kred = float(totals.get("period_credit", res.get("total_credit", 0.0)) or 0.0)
        end = float(totals.get("ending_balance", res.get("net_balance", 0.0)) or 0.0)

        if _is_zero_metric(op, deb, kred, end):
            zero_branches.append(c_name)
            continue

        tot_op += op
        tot_deb += deb
        tot_kred += kred
        tot_end += end

        lines.append(
            f"| {row_idx} | {c_name} | {_format_rupiah(op)} | {_format_rupiah(deb)} | "
            f"{_format_rupiah(kred)} | **{_format_rupiah(end)}** |"
        )
        row_idx += 1

    if row_idx == 1 and zero_branches:
        lines.append("| - | *Seluruh cabang bersaldo Rp 0,00* | Rp 0,00 | Rp 0,00 | Rp 0,00 | **Rp 0,00** |")
    lines.append(
        f"| | **TOTAL GABUNGAN** | **{_format_rupiah(tot_op)}** | **{_format_rupiah(tot_deb)}** | "
        f"**{_format_rupiah(tot_kred)}** | **{_format_rupiah(tot_end)}** |"
    )
    _append_zero_branch_footnote(lines, zero_branches)


def _format_mc_get_profit_and_loss(
    state: AgentState,
    mc_results: Dict[int, Any],
    unauth: List[int],
    lines: List[str],
) -> None:
    """Append multi-company Markdown section for get_profit_and_loss."""
    period_note = _resolve_mc_period_note(mc_results, state.get("extracted_params"))
    zero_branches: List[str] = []
    lines.append(f"### Laporan Laba Rugi Multi-Perusahaan (Combined Profit & Loss){period_note}\n")
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
            if cid in unauth:
                lines.append(f"| {row_idx} | {c_name} | - | - | - | *Akses Dibatasi (HTTP 403)* |")
                row_idx += 1
            else:
                zero_branches.append(c_name)
            continue

        summary = res.get("summary") or res
        rev = float(summary.get("total_revenue", 0.0) or 0.0)
        cogs = float(summary.get("cogs", 0.0) or 0.0)
        gross = float(summary.get("gross_profit", 0.0) or 0.0)
        net = float(summary.get("net_profit", 0.0) or 0.0)

        if _is_zero_metric(rev, cogs, gross, net):
            zero_branches.append(c_name)
            continue

        tot_rev += rev
        tot_cogs += cogs
        tot_gross += gross
        tot_net += net

        lines.append(
            f"| {row_idx} | {c_name} | {_format_rupiah(rev)} | {_format_rupiah(cogs)} | "
            f"{_format_rupiah(gross)} | **{_format_rupiah(net)}** |"
        )
        row_idx += 1

    if row_idx == 1 and zero_branches:
        lines.append("| - | *Seluruh cabang tanpa aktivitas pendapatan/beban* | Rp 0,00 | Rp 0,00 | Rp 0,00 | **Rp 0,00** |")
    lines.append(
        f"| | **TOTAL GABUNGAN** | **{_format_rupiah(tot_rev)}** | **{_format_rupiah(tot_cogs)}** | "
        f"**{_format_rupiah(tot_gross)}** | **{_format_rupiah(tot_net)}** |"
    )
    _append_zero_branch_footnote(lines, zero_branches)

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


def _format_mc_get_balance_sheet(
    state: AgentState,
    mc_results: Dict[int, Any],
    unauth: List[int],
    lines: List[str],
) -> None:
    """Append multi-company Markdown section for get_balance_sheet."""
    zero_branches: List[str] = []
    as_of_note = ""
    for res in mc_results.values():
        if isinstance(res, dict) and res.get("as_of_date"):
            as_of_note = f" (Posisi per: {res['as_of_date']})"
            break
    lines.append(f"### Laporan Neraca Multi-Perusahaan (Combined Balance Sheet){as_of_note}\n")
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
            if cid in unauth:
                lines.append(f"| {row_idx} | {c_name} | - | - | *Akses Dibatasi (HTTP 403)* |")
                row_idx += 1
            else:
                zero_branches.append(c_name)
            continue

        summary = res.get("summary") or res
        ast = float(summary.get("total_assets", 0.0) or 0.0)
        lia = float(summary.get("total_liabilities", 0.0) or 0.0)
        equ = float(summary.get("total_equity", 0.0) or 0.0)

        if _is_zero_metric(ast, lia, equ):
            zero_branches.append(c_name)
            continue

        tot_ast += ast
        tot_lia += lia
        tot_equ += equ

        lines.append(
            f"| {row_idx} | {c_name} | {_format_rupiah(ast)} | {_format_rupiah(lia)} | "
            f"**{_format_rupiah(equ)}** |"
        )
        row_idx += 1

    if row_idx == 1 and zero_branches:
        lines.append("| - | *Seluruh cabang memiliki neraca Rp 0,00* | Rp 0,00 | Rp 0,00 | **Rp 0,00** |")
    lines.append(
        f"| | **TOTAL GABUNGAN** | **{_format_rupiah(tot_ast)}** | **{_format_rupiah(tot_lia)}** | "
        f"**{_format_rupiah(tot_equ)}** |"
    )
    _append_zero_branch_footnote(lines, zero_branches)


def _format_mc_get_stock_level(
    state: AgentState,
    mc_results: Dict[int, Any],
    unauth: List[int],
    lines: List[str],
) -> None:
    """Append multi-company Markdown section for get_stock_level."""
    zero_branches: List[str] = []
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
            if cid in unauth:
                lines.append(f"| {row_idx} | {c_name} | - | - | - | - | - | *Akses Dibatasi (HTTP 403)* |")
                row_idx += 1
            else:
                zero_branches.append(c_name)
            continue

        items = res.get("products") or res.get("stocks") or res.get("items") or [res]
        branch_wrote_row = False
        for it in items:
            sku = it.get("code") or it.get("sku") or "-"
            p_name = it.get("product_name") or it.get("name") or "-"
            qty = float(it.get("quantity", it.get("qty_on_hand", 0.0)) or 0.0)
            avail = float(it.get("available_quantity", qty) or 0.0)
            if _is_zero_metric(qty, avail):
                continue
            breakdown = it.get("breakdown") or []

            tot_qty += qty
            tot_avail += avail
            branch_wrote_row = True

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
        if not branch_wrote_row:
            zero_branches.append(c_name)

    if row_idx == 1 and zero_branches:
        lines.append("| - | *Seluruh cabang memiliki stok 0 unit* | - | - | - | - | 0 unit | 0 unit |")
    lines.append(
        f"| | **TOTAL NASIONAL** | | | | | **{tot_qty:g} unit** | **{tot_avail:g} unit** |"
    )
    _append_zero_branch_footnote(
        lines,
        zero_branches,
        label="Cabang dengan stok 0 / tidak tersedia (tidak ditampilkan di tabel)",
    )


def _format_mc_get_vendor_bills(
    state: AgentState,
    mc_results: Dict[int, Any],
    unauth: List[int],
    lines: List[str],
) -> None:
    """Append multi-company Markdown section for get_vendor_bills."""
    period_note = _resolve_mc_period_note(mc_results, state.get("extracted_params"))
    zero_branches: List[str] = []
    lines.append(f"### Ringkasan Tagihan Vendor Multi-Perusahaan{period_note}\n")
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
            if cid in unauth:
                lines.append(f"| {row_idx} | {c_name} | - | - | - | *Akses Dibatasi (HTTP 403)* |")
                row_idx += 1
            else:
                zero_branches.append(c_name)
            continue

        calc = res.get("calculations") or res.get("summary") or res
        amt = float(calc.get("total_amount", calc.get("total_bill_amount", 0.0)) or 0.0)
        unp = float(calc.get("total_unpaid", 0.0) or 0.0)
        paid = float(calc.get("total_paid", 0.0) or 0.0)

        if _is_zero_metric(amt, unp, paid):
            zero_branches.append(c_name)
            continue

        tot_amt += amt
        tot_unp += unp
        tot_paid += paid

        lines.append(
            f"| {row_idx} | {c_name} | {_format_rupiah(amt)} | {_format_rupiah(unp)} | "
            f"{_format_rupiah(paid)} | Selesai |"
        )
        row_idx += 1

    if row_idx == 1 and zero_branches:
        lines.append("| - | *Seluruh cabang tanpa tagihan vendor* | Rp 0,00 | Rp 0,00 | Rp 0,00 | - |")
    lines.append(
        f"| | **TOTAL GABUNGAN** | **{_format_rupiah(tot_amt)}** | **{_format_rupiah(tot_unp)}** | "
        f"**{_format_rupiah(tot_paid)}** | |"
    )
    _append_zero_branch_footnote(lines, zero_branches)


def _format_mc_get_top_selling_products(
    state: AgentState,
    mc_results: Dict[int, Any],
    unauth: List[int],
    lines: List[str],
) -> None:
    """Append multi-company Markdown section for get_top_selling_products."""
    params = state.get("extracted_params") or {}
    period_note = _resolve_mc_period_note(mc_results, params)
    zero_branches: List[str] = []
    search_term = params.get("search")
    sort_by = params.get("sort_by") or "revenue"
    user_q_low = (state.get("user_query") or "").lower()
    is_branch_ranking_mode = bool(search_term) or bool(
        re.search(r"\b(?:di\s*mana|cabang\s+mana|toko\s+mana|outlet\s+mana)\b", user_q_low)
    )

    if is_branch_ranking_mode:
        branch_rows: List[Dict[str, Any]] = []
        agg_channels: Dict[str, Dict[str, float]] = {}

        for cid in sorted(mc_results.keys()):
            res = mc_results[cid]
            c_name = COMPANY_NAME_MAP.get(cid, f"Cabang {cid}")
            if not res or not isinstance(res, dict):
                if cid not in unauth:
                    zero_branches.append(c_name)
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
            elif cid not in unauth:
                zero_branches.append(c_name)

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
        lines.append(f"### Peringkat Penjualan per Cabang{target_label}{period_note}\n")
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
        _append_zero_branch_footnote(lines, zero_branches)

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

        active_branch_cids = [
            cid for cid in sorted(mc_results.keys())
            if any(p["branch_rev"].get(cid, 0.0) > 0 for p in top_list)
        ]
        if not active_branch_cids:
            active_branch_cids = sorted(mc_results.keys())
        zero_branches = [
            COMPANY_NAME_MAP.get(cid, f"Cabang {cid}")
            for cid in sorted(mc_results.keys())
            if cid not in active_branch_cids and cid not in unauth
        ]

        lines.append(f"### Laporan Produk Pendapatan Tertinggi Multi-Cabang (Top {len(top_list)} Nasional){period_note}\n")
        headers = ["No", "Kode / SKU", "Nama Produk"]
        for cid in active_branch_cids:
            c_name = COMPANY_NAME_MAP.get(cid, f"Cabang {cid}")
            headers.append(f"Omzet {c_name}")
        headers.append("Total Omzet Gabungan")

        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|:" + ":|:".join(["---"] * len(headers)) + ":|")

        branch_totals = {cid: 0.0 for cid in active_branch_cids}
        grand_total = 0.0

        for idx, p in enumerate(top_list, start=1):
            row = [str(idx), f"`{p['code']}`", p["name"]]
            for cid in active_branch_cids:
                b_rev = p["branch_rev"].get(cid, 0.0)
                branch_totals[cid] += b_rev
                row.append(_format_rupiah(b_rev) if b_rev > 0 else "-")
            grand_total += p["total_rev"]
            row.append(f"**{_format_rupiah(p['total_rev'])}**")
            lines.append("| " + " | ".join(row) + " |")

        tot_row = ["", "**TOTAL GABUNGAN**", ""]
        for cid in active_branch_cids:
            tot_row.append(f"**{_format_rupiah(branch_totals[cid])}**")
        tot_row.append(f"**{_format_rupiah(grand_total)}**")
        lines.append("| " + " | ".join(tot_row) + " |")
        _append_zero_branch_footnote(lines, zero_branches)


def _format_mc_get_sales_summary(
    state: AgentState,
    mc_results: Dict[int, Any],
    unauth: List[int],
    lines: List[str],
) -> None:
    """Append multi-company Markdown section for get_sales_summary."""
    ext_params = state.get("extracted_params") or {}
    period_note = _resolve_mc_period_note(mc_results, ext_params)
    zero_branches: List[str] = []
    lines.append(f"### Perbandingan Performa Penjualan Multi-Cabang{period_note}\n")
    lines.append("| Cabang | Omzet Terealisasi | Jumlah Order | Rata-Rata Order (AOV) | Pipeline Quotation |")
    lines.append("|:---|:---|:---:|:---|:---|")
    tot_rev = 0.0
    tot_orders = 0
    tot_pipe = 0.0
    rows_written = 0
    for cid in sorted(mc_results.keys()):
        res = mc_results[cid]
        c_name = COMPANY_NAME_MAP.get(cid, f"Cabang {cid}")
        if not res or not isinstance(res, dict):
            if cid in unauth:
                lines.append(f"| **{c_name}** | *Akses Dibatasi (HTTP 403)* | - | - | - |")
                rows_written += 1
            else:
                zero_branches.append(c_name)
            continue
        conf = res.get("confirmed_metrics") or {}
        rev = float(conf.get("realized_revenue") or 0.0)
        cnt = int(conf.get("orders_count") or 0)
        aov = float(conf.get("average_order_value") or 0.0)
        pipe = res.get("pipeline_metrics") or {}
        pval = float(pipe.get("quotation_potential_value") or 0.0)
        if cnt == 0 and _is_zero_metric(rev, pval):
            zero_branches.append(c_name)
            continue
        tot_rev += rev
        tot_orders += cnt
        tot_pipe += pval
        rows_written += 1
        lines.append(f"| **{c_name}** | **{_format_rupiah(rev)}** | {cnt:,} | {_format_rupiah(aov)} | {_format_rupiah(pval)} |")
    if rows_written == 0 and zero_branches:
        lines.append("| *Seluruh cabang tanpa aktivitas penjualan* | Rp 0,00 | 0 | Rp 0,00 | Rp 0,00 |")
    aov_grand = (tot_rev / tot_orders) if tot_orders > 0 else 0.0
    lines.append(f"| **TOTAL GABUNGAN** | **{_format_rupiah(tot_rev)}** | **{tot_orders:,}** | **{_format_rupiah(aov_grand)}** | **{_format_rupiah(tot_pipe)}** |")
    _append_zero_branch_footnote(lines, zero_branches)
    mc_df = ext_params.get("date_from") or next(
        (r["period"].get("date_from") for r in mc_results.values() if isinstance(r, dict) and isinstance(r.get("period"), dict) and r["period"].get("date_from")),
        None,
    )
    mc_dt = ext_params.get("date_to") or next(
        (r["period"].get("date_to") for r in mc_results.values() if isinstance(r, dict) and isinstance(r.get("period"), dict) and r["period"].get("date_to")),
        None,
    )
    combined_pseudo = {
        "period": {"date_from": mc_df, "date_to": mc_dt},
        "confirmed_metrics": {"realized_revenue": tot_rev, "orders_count": tot_orders, "average_order_value": aov_grand},
    }
    mc_verdict = _build_evaluative_verdict(state.get("user_query") or "", [("get_sales_summary", combined_pseudo, ext_params)])
    if mc_verdict:
        lines.insert(1, f"{mc_verdict}\n")


def _format_mc_get_sales_orders(
    state: AgentState,
    mc_results: Dict[int, Any],
    unauth: List[int],
    lines: List[str],
) -> None:
    """Append multi-company Markdown section for get_sales_orders."""
    ext_params = state.get("extracted_params") or {}
    period_note = _resolve_mc_period_note(mc_results, ext_params)
    zero_branches: List[str] = []
    merged_orders: List[Dict[str, Any]] = []
    branch_rows: List[str] = []
    grand_cnt = 0
    grand_sum = 0.0
    for cid in sorted(mc_results.keys()):
        res = mc_results[cid]
        c_name = COMPANY_NAME_MAP.get(cid, f"Cabang {cid}")
        if not res or not isinstance(res, dict):
            if cid in unauth:
                branch_rows.append(f"| **{c_name}** | - | *Akses Dibatasi (HTTP 403)* | - |")
            else:
                zero_branches.append(c_name)
            continue
        cnt = int(res.get("total_count") or 0)
        tot = float(res.get("total_amount_sum") or 0.0)
        st = res.get("status_filter") or "active"
        if cnt == 0 and _is_zero_metric(tot):
            zero_branches.append(c_name)
            continue
        grand_cnt += cnt
        grand_sum += tot
        branch_rows.append(f"| **{c_name}** | {cnt:,} | **{_format_rupiah(tot)}** | `{st}` |")
        for o in (res.get("orders") or []):
            if isinstance(o, dict):
                row_copy = dict(o)
                if not row_copy.get("company_name"):
                    row_copy["company_name"] = c_name
                merged_orders.append(row_copy)

    if ext_params.get("sort_by") == "amount_desc" or any(
        k in (state.get("user_query") or "").lower()
        for k in ("terbanyak", "terbesar", "tertinggi", "nominal")
    ):
        merged_orders.sort(key=lambda r: float(r.get("total_amount") or 0.0), reverse=True)

    if merged_orders:
        top_o = merged_orders[0]
        lines.append(
            f"Nilai penjualan tertinggi tercatat pada quotation/SO `{top_o.get('so_number') or '-'}` "
            f"(**{top_o.get('company_name') or '-'}**, pelanggan **{top_o.get('customer_name') or '-'}**, status `{top_o.get('status') or '-'}`) "
            f"sebesar **{_format_rupiah(float(top_o.get('total_amount') or 0.0))}**.\n"
        )
        lines.append(f"### Daftar Quotation & Pesanan Penjualan Lintas Cabang{period_note}\n")
        show_mc_cancel = any(bool(o.get("cancel_reason")) for o in merged_orders[:20])
        if show_mc_cancel:
            lines.append("| No | Cabang | No. SO / Quotation | Tanggal | Pelanggan | Kontak HP | Channel | Status | Alasan Batal | Total Nominal |")
            lines.append("|:---:|:---|:---|:---:|:---|:---|:---|:---:|:---|:---|")
        else:
            lines.append("| No | Cabang | No. SO / Quotation | Tanggal | Pelanggan | Kontak HP | Channel | Status | Total Nominal |")
            lines.append("|:---:|:---|:---|:---:|:---|:---|:---|:---:|:---|")
        for idx, o in enumerate(merged_orders[:20], start=1):
            c_phone = o.get("customer_phone") or o.get("sales_phone") or "-"
            if show_mc_cancel:
                lines.append(
                    f"| {idx} | **{o.get('company_name') or '-'}** | `{o.get('so_number') or '-'}` | "
                    f"{o.get('quotation_date') or '-'} | {o.get('customer_name') or '-'} | {c_phone} | "
                    f"{o.get('channel_name') or 'Direct'} | `{o.get('status') or '-'}` | {o.get('cancel_reason') or '-'} | "
                    f"**{_format_rupiah(float(o.get('total_amount') or 0.0))}** |"
                )
            else:
                lines.append(
                    f"| {idx} | **{o.get('company_name') or '-'}** | `{o.get('so_number') or '-'}` | "
                    f"{o.get('quotation_date') or '-'} | {o.get('customer_name') or '-'} | {c_phone} | "
                    f"{o.get('channel_name') or 'Direct'} | `{o.get('status') or '-'}` | **{_format_rupiah(float(o.get('total_amount') or 0.0))}** |"
                )
        lines.append("")

    lines.append(f"### Ringkasan Pesanan Penjualan Multi-Cabang{period_note}\n")
    lines.append("| Cabang | Jumlah Pesanan | Total Nominal | Status Filter |")
    lines.append("|:---|:---:|:---|:---:|")
    if branch_rows:
        lines.extend(branch_rows)
    elif zero_branches:
        lines.append("| *Seluruh cabang tanpa pesanan pada filter ini* | 0 | Rp 0,00 | - |")
    lines.append(f"| **TOTAL GABUNGAN** | **{grand_cnt:,}** | **{_format_rupiah(grand_sum)}** | |")
    _append_zero_branch_footnote(lines, zero_branches)


def _format_mc_get_contacts(
    state: AgentState,
    mc_results: Dict[int, Any],
    unauth: List[int],
    lines: List[str],
) -> None:
    """Append multi-company Markdown section for get_contacts."""
    merged_contacts: List[Dict[str, Any]] = []
    seen_contact_keys = set()
    for cid in sorted(mc_results.keys()):
        res = mc_results[cid]
        if not res or not isinstance(res, dict):
            continue
        for c in (res.get("contacts") or []):
            if not isinstance(c, dict):
                continue
            dedup_key = c.get("contact_id") or c.get("id") or (c.get("name"), c.get("phone"))
            if dedup_key in seen_contact_keys:
                continue
            seen_contact_keys.add(dedup_key)
            merged_contacts.append(c)

    lines.append(f"### Daftar Detail Kontak Lintas Cabang ({len(merged_contacts)} Kontak)\n")
    if merged_contacts:
        lines.append("| No | Nama Kontak | No. HP / WA | Kota | Channel | Sales HP | Order / Total Belanja | Dokumen Terbaru |")
        lines.append("|:---:|:---|:---|:---|:---|:---|:---|:---|")
        lines.extend(_format_contact_table_rows(merged_contacts, max_rows=20))
        _append_pagination_footer(
            lines,
            min(20, len(merged_contacts)),
            len(merged_contacts),
            filter_hint="Sebutkan nama/nomor HP kontak spesifik untuk pencarian lebih fokus.",
        )
    else:
        lines.append("*Tidak ada data kontak yang ditemukan pada cabang yang diperiksa.*")


def _format_mc_get_low_stock_alerts(
    state: AgentState,
    mc_results: Dict[int, Any],
    unauth: List[int],
    lines: List[str],
) -> None:
    """Append multi-company Markdown section for get_low_stock_alerts."""
    exclude_zero = any(
        isinstance(r, dict) and bool(r.get("exclude_zero"))
        for r in mc_results.values()
    )
    total_across_branches = sum(
        _extract_total_count(r, len(r.get("alerts") or []))
        for r in mc_results.values()
        if isinstance(r, dict)
    )
    if exclude_zero:
        lines.append(f"### Produk dengan Stok Tersedikit Multi-Cabang ({total_across_branches} Item)\n")
    else:
        lines.append(f"### Peringatan Stok Menipis / Kritis Multi-Cabang ({total_across_branches} Produk Perlu Perhatian)\n")

    lines.append("| No | Perusahaan / Cabang | SKU | Nama Produk | Gudang | Stok Saat Ini | Batas Minimum | Status |")
    lines.append("|:---|:---|:---|:---|:---|:---|:---|:---|")
    rows_written = 0
    for cid in sorted(mc_results.keys()):
        c_name = COMPANY_NAME_MAP.get(cid, f"Cabang {cid}")
        res = mc_results.get(cid)
        if not res or not isinstance(res, dict):
            continue
        for a in res.get("alerts") or []:
            rows_written += 1
            code = a.get("code") or "-"
            name = a.get("name") or "-"
            comp = a.get("company_name") or c_name
            wh = a.get("warehouse_name") or "-"
            qty = a.get("qty_on_hand", 0.0)
            thresh = a.get("threshold", 0.0)
            status_label = "HABIS" if qty <= 0 else "MENIPIS"
            lines.append(
                f"| {rows_written} | {comp} | `{code}` | {name} | {wh} | **{qty:g} unit** | {thresh:g} | **{status_label}** |"
            )
    if rows_written == 0:
        lines.append("| - | - | - | *Tidak ada produk di bawah ambang batas stok.* | - | - | - | - |")


def _format_mc_get_product_info(
    state: AgentState,
    mc_results: Dict[int, Any],
    unauth: List[int],
    lines: List[str],
) -> None:
    """Append multi-company Markdown section for get_product_info price comparisons."""
    zero_branches: List[str] = []
    lines.append("### Perbandingan Harga Produk Multi-Cabang\n")
    lines.append("| No | Perusahaan / Cabang | SKU | Nama Produk | Harga Jual | Sumber Harga |")
    lines.append("|:---:|:---|:---|:---|:---|:---|")
    row_idx = 1
    for cid in sorted(mc_results.keys()):
        c_name = COMPANY_NAME_MAP.get(cid, f"Cabang {cid}")
        res = mc_results.get(cid)
        if not res or not isinstance(res, dict):
            if cid in unauth:
                lines.append(f"| {row_idx} | {c_name} | - | - | - | *Akses Dibatasi (HTTP 403)* |")
                row_idx += 1
            else:
                zero_branches.append(c_name)
            continue

        if "products" in res:
            prods = [p for p in (res.get("products") or []) if isinstance(p, dict)]
        elif res.get("code") or res.get("name"):
            prods = [res]
        else:
            prods = []

        if not prods:
            zero_branches.append(c_name)
            continue

        for p in prods[:5]:
            sku = p.get("code") or p.get("sku") or p.get("default_code") or "-"
            p_name = p.get("name") or p.get("full_name") or "-"
            uom = p.get("uom")
            uom_name = uom.get("name", "Unit") if isinstance(uom, dict) else (uom or "Unit")
            price_val = _resolve_product_display_price(p)
            p_source = p.get("price_source") or "-"
            lines.append(
                f"| {row_idx} | {c_name} | `{sku}` | {p_name} | **{_format_rupiah(price_val)}** / {uom_name} | `{p_source}` |"
            )
            row_idx += 1

    if row_idx == 1 and zero_branches:
        lines.append("| - | - | - | *Tidak ada produk cocok ditemukan pada cabang terkait.* | - | - |")
    _append_zero_branch_footnote(
        lines,
        zero_branches,
        label="Cabang tanpa data produk cocok (tidak ditampilkan di tabel)",
    )


MULTI_COMPANY_FORMATTERS: Dict[str, Callable[[AgentState, Dict[int, Any], List[int], List[str]], None]] = {
    "get_account_balance": _format_mc_get_account_balance,
    "get_profit_and_loss": _format_mc_get_profit_and_loss,
    "get_balance_sheet": _format_mc_get_balance_sheet,
    "get_stock_level": _format_mc_get_stock_level,
    "fetch_vendor_invoices": _format_mc_get_vendor_bills,
    "get_vendor_bills": _format_mc_get_vendor_bills,
    "get_top_selling_products": _format_mc_get_top_selling_products,
    "get_sales_summary": _format_mc_get_sales_summary,
    "get_sales_orders": _format_mc_get_sales_orders,
    "get_contacts": _format_mc_get_contacts,
    "get_low_stock_alerts": _format_mc_get_low_stock_alerts,
    "get_product_info": _format_mc_get_product_info,
}


def _format_multi_company_presentation(state: AgentState) -> str:
    """
    Format multi-company comparison and network-wide rollups into structured Markdown.
    Includes per-company breakdown, grand total rollup summary, and partial permission warnings.
    """
    tool = state.get("detected_tool") or "get_account_balance"
    mc_results = state.get("multi_company_results") or {}
    unauth = state.get("unauthorized_companies") or []
    lines: List[str] = []

    formatter = MULTI_COMPANY_FORMATTERS.get(tool)
    if formatter is not None:
        formatter(state, mc_results, unauth, lines)
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

    # Append partial authorization warning footer if some branches returned HTTP 403
    if unauth:
        unauth_names = [COMPANY_NAME_MAP.get(cid, f"Company {cid}") for cid in unauth]
        lines.append(
            f"\n> [!WARNING]\n"
            f"> **Catatan Akses Terbatas:** Anda tidak memiliki izin akses untuk cabang berikut (HTTP 403): "
            f"{', '.join(unauth_names)}. Nilai di atas hanya mencakup cabang yang diizinkan."
        )

    return "\n".join(lines)
