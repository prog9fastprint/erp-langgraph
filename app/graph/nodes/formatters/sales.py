"""
Markdown presentation formatters for FastPrint ERP sales tools.
"""
from typing import Any, Callable, Dict, List, Optional
from app.graph.nodes.formatters.common import (
    _append_pagination_footer,
    _build_evaluative_verdict,
    _format_contact_table_rows,
    _format_rupiah,
    _resolve_company_display_label,
    _resolve_period_bounds,
)


def format_get_top_selling_products(
    tool: str,
    tool_result: Any,
    user_query: Optional[str] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
    default_company_id: Optional[int] = None,
) -> str:
    """Format sales.get_top_selling_products payload into Markdown."""
    lines: List[str] = []
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


def format_get_sales_orders(
    tool: str,
    tool_result: Any,
    user_query: Optional[str] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
    default_company_id: Optional[int] = None,
) -> str:
    """Format sales.get_sales_orders payload into Markdown."""
    lines: List[str] = []
    orders = tool_result.get("orders") if isinstance(tool_result, dict) else (
        tool_result if isinstance(tool_result, list) else []
    )
    total_cnt = tool_result.get("total_count", len(orders)) if isinstance(tool_result, dict) else len(orders)
    tot_sum = float(tool_result.get("total_amount_sum") or sum(float(o.get("total_amount") or 0.0) for o in orders))
    st_filter = tool_result.get("status_filter", "active") if isinstance(tool_result, dict) else "active"
    sort_by = tool_result.get("sort_by") if isinstance(tool_result, dict) else (extracted_params or {}).get("sort_by")
    has_company = any(bool(o.get("company_name")) for o in orders)
    d_from, d_to = _resolve_period_bounds(tool_result, extracted_params)
    period_note = f", Periode: {d_from or '-'} s/d {d_to or '-'}" if (d_from or d_to) else ""

    if orders and sort_by == "amount_desc":
        top_o = orders[0]
        top_so = top_o.get("so_number") or "-"
        top_comp = top_o.get("company_name") or "-"
        top_cust = top_o.get("customer_name") or "-"
        top_st = top_o.get("status") or "-"
        top_amt = float(top_o.get("total_amount") or 0.0)
        lines.append(
            f"Nilai penjualan tertinggi pada filter ini tercatat pada quotation/SO `{top_so}` "
            f"(**{top_comp}**, pelanggan **{top_cust}**, status `{top_st}`) sebesar **{_format_rupiah(top_amt)}**.\n"
        )

    lines.append(f"### Daftar Pesanan Penjualan & Quotation ({len(orders)} dari {total_cnt} Dokumen, Status: `{st_filter}`{period_note})\n")
    show_cancel_col = st_filter in ("cancel", "cancelled") or any(bool(o.get("cancel_reason")) for o in orders)
    if orders:
        if has_company and show_cancel_col:
            lines.append("| No | Cabang | No. SO / Quotation | Tanggal | Pelanggan | Kontak HP | Channel | Status | Alasan Batal | Total Nominal |")
            lines.append("|:---:|:---|:---|:---:|:---|:---|:---|:---:|:---|:---|")
        elif has_company:
            lines.append("| No | Cabang | No. SO / Quotation | Tanggal | Pelanggan | Kontak HP | Channel | Status | Total Nominal |")
            lines.append("|:---:|:---|:---|:---:|:---|:---|:---|:---:|:---|")
        elif show_cancel_col:
            lines.append("| No | No. SO / Quotation | Tanggal | Pelanggan | Kontak HP | Channel | Status | Alasan Batal | Total Nominal |")
            lines.append("|:---:|:---|:---:|:---|:---|:---|:---:|:---|:---|")
        else:
            lines.append("| No | No. SO / Quotation | Tanggal | Pelanggan | Kontak HP | Channel | Status | Total Nominal |")
            lines.append("|:---:|:---|:---:|:---|:---|:---|:---:|:---|")
        for idx, o in enumerate(orders, start=1):
            so_num = o.get("so_number") or "-"
            dt = o.get("quotation_date") or "-"
            cust = o.get("customer_name") or "-"
            c_phone = o.get("customer_phone") or o.get("sales_phone") or "-"
            ch = o.get("channel_name") or "Direct"
            st = o.get("status") or "-"
            cr = o.get("cancel_reason") or "-"
            tot = float(o.get("total_amount") or 0.0)
            if has_company and show_cancel_col:
                comp_n = o.get("company_name") or "-"
                lines.append(f"| {idx} | **{comp_n}** | `{so_num}` | {dt} | {cust} | {c_phone} | {ch} | `{st}` | {cr} | **{_format_rupiah(tot)}** |")
            elif has_company:
                comp_n = o.get("company_name") or "-"
                lines.append(f"| {idx} | **{comp_n}** | `{so_num}` | {dt} | {cust} | {c_phone} | {ch} | `{st}` | **{_format_rupiah(tot)}** |")
            elif show_cancel_col:
                lines.append(f"| {idx} | `{so_num}` | {dt} | {cust} | {c_phone} | {ch} | `{st}` | {cr} | **{_format_rupiah(tot)}** |")
            else:
                lines.append(f"| {idx} | `{so_num}` | {dt} | {cust} | {c_phone} | {ch} | `{st}` | **{_format_rupiah(tot)}** |")
        if has_company and show_cancel_col:
            lines.append(f"| | **TOTAL** | | | | | | | | **{_format_rupiah(tot_sum)}** |")
        elif has_company:
            lines.append(f"| | **TOTAL** | | | | | | | **{_format_rupiah(tot_sum)}** |")
        elif show_cancel_col:
            lines.append(f"| | **TOTAL** | | | | | | | **{_format_rupiah(tot_sum)}** |")
        else:
            lines.append(f"| | **TOTAL** | | | | | | **{_format_rupiah(tot_sum)}** |")
        cancel_sum = tool_result.get("cancel_reason_summary") if isinstance(tool_result, dict) else None
        if cancel_sum and isinstance(cancel_sum, list):
            lines.append("\n#### Rekapitulasi Alasan Pembatalan:")
            lines.append("| Alasan Pembatalan | Jumlah Dokumen |")
            lines.append("|:---|:---:|")
            for cr_row in cancel_sum[:10]:
                lines.append(f"| {cr_row.get('reason') or '-'} | {int(cr_row.get('count') or 0):,} |")
        _append_pagination_footer(
            lines,
            len(orders),
            total_cnt,
            filter_hint="Gunakan filter periode tanggal/status atau klik **Lihat Full Table** untuk melihat seluruh daftar order.",
        )
    else:
        lines.append("*Tidak ada dokumen sales order atau quotation yang sesuai dengan kriteria filter.*")
    return "\n".join(lines)


def format_get_contacts(
    tool: str,
    tool_result: Any,
    user_query: Optional[str] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
    default_company_id: Optional[int] = None,
) -> str:
    """Format sales.get_contacts payload into Markdown."""
    lines: List[str] = []
    contacts = tool_result.get("contacts") if isinstance(tool_result, dict) else (
        tool_result if isinstance(tool_result, list) else []
    )
    total_cnt = tool_result.get("total_count", len(contacts)) if isinstance(tool_result, dict) else len(contacts)
    filters_meta = tool_result.get("filters") if isinstance(tool_result, dict) else {}
    ctx_parts = []
    if isinstance(filters_meta, dict):
        if filters_meta.get("channel_name"):
            ctx_parts.append(f"Channel: **{filters_meta['channel_name']}**")
        if filters_meta.get("product_search"):
            ctx_parts.append(f"Produk: `{filters_meta['product_search']}`")
        if filters_meta.get("date_from") or filters_meta.get("date_to"):
            ctx_parts.append(f"Periode: {filters_meta.get('date_from', '-')} s/d {filters_meta.get('date_to', '-')}")
        if filters_meta.get("status"):
            ctx_parts.append(f"Status Order: `{filters_meta['status']}`")
    ctx_suffix = f" ({', '.join(ctx_parts)})" if ctx_parts else ""

    lines.append(f"### Daftar Detail Kontak ({len(contacts)} dari {total_cnt} Kontak){ctx_suffix}\n")
    if contacts:
        lines.append("| No | Nama Kontak | No. HP / WA | Kota | Channel | Sales HP | Order / Total Belanja | Dokumen Terbaru |")
        lines.append("|:---:|:---|:---|:---|:---|:---|:---|:---|")
        lines.extend(_format_contact_table_rows(contacts))
        _append_pagination_footer(
            lines,
            len(contacts),
            total_cnt,
            filter_hint="Sebutkan nama/nomor HP kontak spesifik atau klik **Lihat Full Table** untuk melihat seluruh data.",
        )

        if len(contacts) <= 3:
            for c in contacts:
                c_name = c.get("name") or "-"
                addr = c.get("address") or "-"
                email = c.get("email") or "-"
                ctype = c.get("contact_type") or "-"
                uname = c.get("username_mp") or "-"
                lines.append(
                    f"\n- **{c_name}** | Tipe: {ctype} | Email: {email} | Username MP: {uname} | Alamat: {addr}"
                )
    else:
        lines.append("*Tidak ada data kontak yang sesuai dengan kriteria filter.*")
    return "\n".join(lines)


def format_get_sales_order_detail(
    tool: str,
    tool_result: Any,
    user_query: Optional[str] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
    default_company_id: Optional[int] = None,
) -> str:
    """Format sales.get_sales_order_detail payload into Markdown."""
    lines: List[str] = []
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
    cust_phone = cust.get("phone") or cust.get("mobile") or cust.get("sales_phone") or "-"
    cust_city = cust.get("city") or ""
    cust_email = cust.get("email") or ""
    cust_addr = cust.get("address") or ""
    cust_extra_parts = []
    if cust_email:
        cust_extra_parts.append(f"Email: {cust_email}")
    if cust_city:
        cust_extra_parts.append(f"Kota: {cust_city}")
    if cust_addr:
        cust_extra_parts.append(f"Alamat: {cust_addr}")
    cust_extra_str = f" | {' | '.join(cust_extra_parts)}" if cust_extra_parts else ""

    fin = tool_result.get("financial_summary") or {}
    tot_amount = float(fin.get("total_amount") or 0.0)
    disc_nom = float(fin.get("discount_nominal") or 0.0)

    lines.append(f"### Detail Sales Order: `{so_num}`{alias_str}\n")
    lines.append(f"- **Cabang / Perusahaan**: {c_name}")
    lines.append(f"- **Status Dokumen**: `{status}` | **Status Packing**: `{packing}`")
    lines.append(f"- **Tanggal Order**: {q_date} | **Termin Pembayaran**: {pay_term}")
    lines.append(f"- **Pelanggan**: **{cust_name}** (Telp/HP: {cust_phone}{cust_extra_str})")
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


def format_get_sales_summary(
    tool: str,
    tool_result: Any,
    user_query: Optional[str] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
    default_company_id: Optional[int] = None,
) -> str:
    """Format sales.get_sales_summary payload into Markdown."""
    lines: List[str] = []
    if not isinstance(tool_result, dict):
        return "Ringkasan data penjualan tidak ditemukan."

    df, dt = _resolve_period_bounds(tool_result, extracted_params)
    comp_lbl = _resolve_company_display_label(tool_result, extracted_params, default_company_id)
    meta_parts: List[str] = []
    if comp_lbl:
        meta_parts.append(f"Cabang: {comp_lbl}")
    if df or dt:
        meta_parts.append(f"Periode {df or '-'} s/d {dt or '-'}")
    p_str = f" ({', '.join(meta_parts)})" if meta_parts else ""

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
    eval_verdict = _build_evaluative_verdict(user_query or "", [("get_sales_summary", tool_result, extracted_params)])
    if eval_verdict:
        lines.append(f"{eval_verdict}\n")
    lines.append(f"- **Total Omzet Terealisasi**: **{_format_rupiah(real_rev)}**")
    lines.append(f"- **Volume Pesanan Dikonfirmasi**: **{orders_cnt:,} pesanan**")
    lines.append(f"- **Average Order Value (AOV)**: **{_format_rupiah(aov)}** / order")
    lines.append(f"- **Pipeline Penawaran (Quotation)**: {pipe_cnt} draft ({_format_rupiah(pipe_val)})")
    lines.append(f"- **Pesanan Dibatalkan (Cancel)**: {canc_cnt} order ({_format_rupiah(canc_val)})\n")

    by_branch = tool_result.get("by_branch") or []
    if by_branch:
        lines.append("#### Kontribusi Omzet per Cabang:\n")
        lines.append("| Cabang | Jumlah Order | Total Omzet | Rata-Rata Order (AOV) | Kontribusi |")
        lines.append("|:---|:---:|:---|:---|:---:|")
        for br in by_branch:
            b_name = br.get("company_name") or f"Company #{br.get('company_id', '-')}"
            b_cnt = int(br.get("order_count") or 0)
            b_rev = float(br.get("total_revenue") or 0.0)
            b_aov = float(br.get("average_order_value") or 0.0)
            b_pct = (b_rev / real_rev * 100.0) if real_rev > 0 else 0.0
            lines.append(f"| **{b_name}** | {b_cnt:,} | **{_format_rupiah(b_rev)}** | {_format_rupiah(b_aov)} | {b_pct:.1f}% |")
        lines.append("")

    monthly_bd = tool_result.get("monthly_breakdown") or []
    if monthly_bd:
        lines.append("#### Tren Omzet Bulanan (Confirmed):\n")
        lines.append("| Bulan | Jumlah Order | Total Omzet |")
        lines.append("|:---:|:---:|:---|")
        for mb in monthly_bd:
            lines.append(f"| `{mb.get('month', '-')}` | {int(mb.get('order_count') or 0):,} | **{_format_rupiah(mb.get('total_revenue') or 0.0)}** |")
        lines.append("")

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


SALES_FORMATTERS: Dict[str, Callable[..., str]] = {
    "get_top_selling_products": format_get_top_selling_products,
    "get_sales_orders": format_get_sales_orders,
    "get_contacts": format_get_contacts,
    "get_sales_order_detail": format_get_sales_order_detail,
    "get_sales_summary": format_get_sales_summary,
}
