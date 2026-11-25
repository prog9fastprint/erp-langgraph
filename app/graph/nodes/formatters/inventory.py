"""
Markdown presentation formatters for FastPrint ERP inventory tools.
"""
from typing import Any, Callable, Dict, List, Optional
from app.core.branches import COMPANY_NAME_MAP
from app.graph.nodes.formatters.common import (
    _append_pagination_footer,
    _extract_total_count,
    _format_rupiah,
)


def format_get_stock_level(
    tool: str,
    tool_result: Any,
    user_query: Optional[str] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
    default_company_id: Optional[int] = None,
) -> str:
    """Format inventory.get_stock_level payload into Markdown."""
    lines: List[str] = []
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

    _append_pagination_footer(
        lines,
        len(items),
        total_count,
        filter_hint="Gunakan kata kunci SKU/nama produk atau nama gudang lebih spesifik, atau klik **Lihat Full Table** untuk menelusuri seluruh data.",
    )

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


def format_get_product_info(
    tool: str,
    tool_result: Any,
    user_query: Optional[str] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
    default_company_id: Optional[int] = None,
) -> str:
    """Format inventory.get_product_info payload into Markdown."""
    lines: List[str] = []
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
                _append_pagination_footer(
                    lines,
                    len(products),
                    total_count,
                    filter_hint="Gunakan kata kunci nama/SKU produk atau kategori lebih spesifik, atau klik **Lihat Full Table** untuk menelusuri seluruh data.",
                )
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
            _append_pagination_footer(
                lines,
                len(products),
                total_count,
                filter_hint="Gunakan kata kunci nama/SKU produk atau kategori lebih spesifik untuk menelusuri data lainnya.",
            )
    return "\n".join(lines)


def format_get_locations(
    tool: str,
    tool_result: Any,
    user_query: Optional[str] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
    default_company_id: Optional[int] = None,
) -> str:
    """Format inventory.get_locations payload into Markdown."""
    lines: List[str] = []
    locs = tool_result.get("locations") if isinstance(tool_result, dict) else (
        tool_result if isinstance(tool_result, list) else [tool_result]
    )
    total_found = _extract_total_count(tool_result, len(locs))
    lines.append(f"### Informasi Daftar Lokasi & Gudang ({total_found} Lokasi Ditemukan)\n")
    if locs:
        lines.append("| ID | Nama Lokasi | Gudang Induk | Tipe | Kategori |")
        lines.append("|:---|:---|:---|:---|:---|")
        for loc in locs:
            lid = loc.get("id", "-")
            lname = loc.get("name", "-")
            wh = loc.get("warehouse_name") or "-"
            ltype = (loc.get("type") or "-").capitalize()
            lcat = (loc.get("category") or "-").capitalize()
            lines.append(f"| `{lid}` | {lname} | {wh} | {ltype} | {lcat} |")
        _append_pagination_footer(
            lines,
            len(locs),
            total_found,
            filter_hint="Gunakan kata kunci nama lokasi atau gudang lebih spesifik, atau klik **Lihat Full Table** untuk menelusuri seluruh data.",
        )
    return "\n".join(lines)


def format_get_stock_history(
    tool: str,
    tool_result: Any,
    user_query: Optional[str] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
    default_company_id: Optional[int] = None,
) -> str:
    """Format inventory.get_stock_history payload into Markdown."""
    lines: List[str] = []
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


def format_get_transfers(
    tool: str,
    tool_result: Any,
    user_query: Optional[str] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
    default_company_id: Optional[int] = None,
) -> str:
    """Format inventory.get_transfers payload into Markdown."""
    lines: List[str] = []
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


def format_get_low_stock_alerts(
    tool: str,
    tool_result: Any,
    user_query: Optional[str] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
    default_company_id: Optional[int] = None,
) -> str:
    """Format inventory.get_low_stock_alerts payload into Markdown."""
    lines: List[str] = []

    alerts = tool_result.get("alerts") if isinstance(tool_result, dict) else (
        tool_result if isinstance(tool_result, list) else [tool_result]
    )
    total_found = _extract_total_count(tool_result, len(alerts))
    exclude_zero = isinstance(tool_result, dict) and bool(tool_result.get("exclude_zero"))
    oos_count = int(tool_result.get("out_of_stock_count") or 0) if isinstance(tool_result, dict) else 0
    threshold_val = float(tool_result.get("threshold") or 0.0) if isinstance(tool_result, dict) else 0.0

    raw_cid = (
        (tool_result.get("company_id") if isinstance(tool_result, dict) else None)
        or (extracted_params.get("company_id") if isinstance(extracted_params, dict) else None)
        or default_company_id
    )
    header_company = (
        (tool_result.get("company_name") if isinstance(tool_result, dict) else None)
        or (alerts[0].get("company_name") if alerts and isinstance(alerts[0], dict) else None)
        or (COMPANY_NAME_MAP.get(int(raw_cid)) if raw_cid is not None else None)
    )
    company_suffix = f" — {header_company}" if header_company else ""

    if exclude_zero:
        lines.append(f"### Produk dengan Stok Tersedikit{company_suffix} (Stok Positif Terendah — {total_found} Item)\n")
        if oos_count > 0:
            lines.append(f"Terdapat **{oos_count} SKU** lain berstatus habis/minus (<= 0 unit) pada cabang ini.\n")
    else:
        lines.append(f"### Peringatan Stok Menipis / Kritis{company_suffix} ({total_found} Produk Perlu Perhatian)\n")
        if threshold_val == 0.0:
            lines.append(
                "*Batas minimum default <= 0 unit (stok habis/minus); sebutkan angka seperti 'di bawah 10' untuk menyertakan stok menipis positif.*\n"
            )

    if alerts:
        lines.append("| No | SKU | Nama Produk | Perusahaan | Gudang | Stok Saat Ini | Batas Minimum | Status |")
        lines.append("|:---|:---|:---|:---|:---|:---|:---|:---|")
        for idx, a in enumerate(alerts, start=1):
            code = a.get("code") or "-"
            name = a.get("name") or "-"
            comp = a.get("company_name") or header_company or "-"
            wh = a.get("warehouse_name") or "-"
            qty = a.get("qty_on_hand", 0.0)
            thresh = a.get("threshold", 0.0)
            status_label = "HABIS" if qty <= 0 else "MENIPIS"
            lines.append(
                f"| {idx} | `{code}` | {name} | {comp} | {wh} | **{qty:g} unit** | {thresh:g} | **{status_label}** |"
            )
        _append_pagination_footer(
            lines,
            len(alerts),
            total_found,
            filter_hint="Gunakan kata kunci nama/SKU produk, kategori, atau gudang lebih spesifik, atau klik **Lihat Full Table** untuk menelusuri seluruh data.",
        )
    elif exclude_zero:
        lines.append("Tidak ditemukan produk dengan stok positif (> 0 unit) yang tersisa pada cabang ini.")
    else:
        lines.append("Semua stok produk terpantau dalam batas aman.")
    return "\n".join(lines)


def format_get_stock_adjustments(
    tool: str,
    tool_result: Any,
    user_query: Optional[str] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
    default_company_id: Optional[int] = None,
) -> str:
    """Format inventory.get_stock_adjustments payload into Markdown."""
    lines: List[str] = []
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


def format_get_scrap_records(
    tool: str,
    tool_result: Any,
    user_query: Optional[str] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
    default_company_id: Optional[int] = None,
) -> str:
    """Format inventory.get_scrap_records payload into Markdown."""
    lines: List[str] = []
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


def format_get_warehouses(
    tool: str,
    tool_result: Any,
    user_query: Optional[str] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
    default_company_id: Optional[int] = None,
) -> str:
    """Format inventory.get_warehouses payload into Markdown."""
    lines: List[str] = []
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
        _append_pagination_footer(
            lines,
            len(whs),
            total_found,
            filter_hint="Gunakan kata kunci nama gudang lebih spesifik, atau klik **Lihat Full Table** untuk menelusuri seluruh data.",
        )
    return "\n".join(lines)


def format_get_product_categories(
    tool: str,
    tool_result: Any,
    user_query: Optional[str] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
    default_company_id: Optional[int] = None,
) -> str:
    """Format inventory.get_product_categories payload into Markdown."""
    lines: List[str] = []
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
        _append_pagination_footer(
            lines,
            len(cats),
            total_found,
            filter_hint="Gunakan kata kunci nama kategori lebih spesifik, atau klik **Lihat Full Table** untuk menelusuri seluruh data.",
        )
    return "\n".join(lines)


INVENTORY_FORMATTERS: Dict[str, Callable[..., str]] = {
    "get_stock_level": format_get_stock_level,
    "get_product_info": format_get_product_info,
    "get_locations": format_get_locations,
    "get_stock_history": format_get_stock_history,
    "get_transfers": format_get_transfers,
    "get_low_stock_alerts": format_get_low_stock_alerts,
    "get_stock_adjustments": format_get_stock_adjustments,
    "get_scrap_records": format_get_scrap_records,
    "get_warehouses": format_get_warehouses,
    "get_product_categories": format_get_product_categories,
}
