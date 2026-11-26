"""
Markdown presentation formatters for FastPrint ERP marketplace tools.
"""
from typing import Any, Callable, Dict, List, Optional
from app.graph.nodes.formatters.common import (
    _append_pagination_footer,
    _extract_total_count,
    _format_currency,
    _format_rupiah,
)


def format_get_account_health(
    tool: str,
    tool_result: Any,
    user_query: Optional[str] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
    default_company_id: Optional[int] = None,
) -> str:
    """Format marketplace.get_account_health payload into Markdown."""
    lines: List[str] = []
    summary = tool_result.get("summary") or {}
    accounts = tool_result.get("accounts") or []
    win_h = tool_result.get("error_window_hours", 24)
    ew = tool_result.get("error_window") or {}
    ew_note = (
        f"Periode: {ew.get('date_from') or '-'} s/d {ew.get('date_to') or '-'}"
        if (isinstance(ew, dict) and (ew.get("date_from") or ew.get("date_to")))
        else f"Window Error: {win_h} Jam"
    )
    lines.append(f"### Kesehatan Akun & Token Marketplace ({ew_note})\n")
    pulled_cnt = summary.get("pulled_orders_count")
    err_cnt_s = int(summary.get("recent_fetch_errors_count") or 0)
    err_ratio = summary.get("error_to_pulled_order_ratio")
    if pulled_cnt is not None:
        pulled_int = int(pulled_cnt)
        ratio_pct = (err_cnt_s / pulled_int * 100.0) if pulled_int > 0 else (float(err_ratio or 0.0) * 100.0)
        pull_metric_str = f" | **Order Berhasil Ditarik**: {pulled_int:,} (**Rasio Error**: {ratio_pct:.2f}%)"
    else:
        pull_metric_str = ""
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
        f"**Error Fetch Order**: {summary.get('recent_fetch_errors_count', 0)}{pull_metric_str}\n"
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
    err_by_type = tool_result.get("error_breakdown_by_type") or []
    if err_by_type:
        lines.append("\n#### Rincian Error Penarikan per Jenis:\n")
        lines.append("| Jenis Error | Jumlah Kejadian |")
        lines.append("|:---|:---:|")
        for et in err_by_type:
            lines.append(f"| `{et.get('error_type') or '-'}` | {int(et.get('count') or 0):,} |")
    err_by_comp = tool_result.get("error_breakdown_by_company") or []
    if err_by_comp:
        lines.append("\n#### Rincian Error Penarikan per Cabang:\n")
        lines.append("| Cabang / Entitas | Jumlah Error |")
        lines.append("|:---|:---:|")
        for ec in err_by_comp:
            lines.append(f"| {ec.get('company_name') or '-'} | {int(ec.get('count') or 0):,} |")
    unassigned = tool_result.get("unassigned_company_errors") or []
    if unassigned:
        lines.append("\n#### Error Fetch Marketplace Tingkat Perusahaan:")
        for ue in unassigned[:5]:
            lines.append(f"- `{ue.get('timestamp', '-')}` [{ue.get('error_type', '-')}] {ue.get('message', '-')}")
    return "\n".join(lines)


def format_get_marketplace_orders(
    tool: str,
    tool_result: Any,
    user_query: Optional[str] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
    default_company_id: Optional[int] = None,
) -> str:
    """Format marketplace.get_marketplace_orders payload into Markdown."""
    lines: List[str] = []
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


def format_get_kilat_order_diagnostics(
    tool: str,
    tool_result: Any,
    user_query: Optional[str] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
    default_company_id: Optional[int] = None,
) -> str:
    """Format marketplace.get_kilat_order_diagnostics payload into Markdown."""
    lines: List[str] = []
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


def format_get_product_mapping_status(
    tool: str,
    tool_result: Any,
    user_query: Optional[str] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
    default_company_id: Optional[int] = None,
) -> str:
    """Format marketplace.get_product_mapping_status payload into Markdown."""
    lines: List[str] = []
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


def format_get_stock_sync_simulation(
    tool: str,
    tool_result: Any,
    user_query: Optional[str] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
    default_company_id: Optional[int] = None,
) -> str:
    """Format marketplace.get_stock_sync_simulation payload into Markdown."""
    lines: List[str] = []
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


def format_get_sync_execution_logs(
    tool: str,
    tool_result: Any,
    user_query: Optional[str] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
    default_company_id: Optional[int] = None,
) -> str:
    """Format marketplace.get_sync_execution_logs payload into Markdown."""
    lines: List[str] = []
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


def format_get_price_comparison(
    tool: str,
    tool_result: Any,
    user_query: Optional[str] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
    default_company_id: Optional[int] = None,
) -> str:
    """Format marketplace.get_price_comparison payload into Markdown."""
    lines: List[str] = []
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


def format_get_fee_and_escrow_anomalies(
    tool: str,
    tool_result: Any,
    user_query: Optional[str] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
    default_company_id: Optional[int] = None,
) -> str:
    """Format marketplace.get_fee_and_escrow_anomalies payload into Markdown."""
    lines: List[str] = []
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


def format_get_deduction_and_settlement_summary(
    tool: str,
    tool_result: Any,
    user_query: Optional[str] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
    default_company_id: Optional[int] = None,
) -> str:
    """Format marketplace.get_deduction_and_settlement_summary payload into Markdown."""
    lines: List[str] = []
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


def format_get_cashflow_and_balance_summary(
    tool: str,
    tool_result: Any,
    user_query: Optional[str] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
    default_company_id: Optional[int] = None,
) -> str:
    """Format marketplace.get_cashflow_and_balance_summary payload into Markdown."""
    lines: List[str] = []
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


def format_get_competitor_insights(
    tool: str,
    tool_result: Any,
    user_query: Optional[str] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
    default_company_id: Optional[int] = None,
) -> str:
    """Format marketplace.get_competitor_insights payload into Markdown."""
    lines: List[str] = []
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


MARKETPLACE_FORMATTERS: Dict[str, Callable[..., str]] = {
    "get_account_health": format_get_account_health,
    "get_marketplace_orders": format_get_marketplace_orders,
    "get_kilat_order_diagnostics": format_get_kilat_order_diagnostics,
    "get_product_mapping_status": format_get_product_mapping_status,
    "get_stock_sync_simulation": format_get_stock_sync_simulation,
    "get_sync_execution_logs": format_get_sync_execution_logs,
    "get_price_comparison": format_get_price_comparison,
    "get_fee_and_escrow_anomalies": format_get_fee_and_escrow_anomalies,
    "get_deduction_and_settlement_summary": format_get_deduction_and_settlement_summary,
    "get_cashflow_and_balance_summary": format_get_cashflow_and_balance_summary,
    "get_competitor_insights": format_get_competitor_insights,
}
