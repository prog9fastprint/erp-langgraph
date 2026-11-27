"""
Entity Link Builder for FastPrint ERP AI Assistant.

Extracts structured, validated internal ERP redirect links from raw tool
payloads without injecting URLs into Markdown prose or exported spreadsheets.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

_SAFE_ERP_URL_RE = re.compile(r"^/(?!/)[A-Za-z0-9_\-./?=&%]+$")
_MAX_ENTITY_LINKS = 500
_PLACEHOLDER_LABELS = {"", "-", "—", "none", "null", "n/a", "na"}


def is_safe_erp_path(url: Any) -> bool:
    """Return True if *url* is a relative internal ERP path without unsafe schemes."""
    if not isinstance(url, str):
        return False
    stripped = url.strip()
    if not stripped or ".." in stripped or any(ch.isspace() for ch in stripped):
        return False
    return bool(_SAFE_ERP_URL_RE.match(stripped))


def _positive_int(val: Any) -> Optional[int]:
    """Coerce *val* to a strictly positive integer, returning None otherwise."""
    if val is None or isinstance(val, bool):
        return None
    try:
        parsed = int(val)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _clean_label(val: Any) -> Optional[str]:
    """Normalize a visible entity label, rejecting placeholder strings."""
    if val is None or isinstance(val, bool):
        return None
    text = str(val).strip()
    if text.lower() in _PLACEHOLDER_LABELS:
        return None
    return text


def _make_link(
    label: Any,
    entity_type: str,
    entity_id: Any,
    url: str,
    *,
    field: str = "",
    company_id: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Build a normalized EntityLink payload if label, ID, and URL are all valid."""
    clean_lbl = _clean_label(label)
    pid = _positive_int(entity_id)
    if not clean_lbl or pid is None or not is_safe_erp_path(url):
        return None
    entry: Dict[str, Any] = {
        "label": clean_lbl,
        "type": entity_type,
        "id": pid,
        "url": url,
    }
    if field:
        entry["field"] = field
    cid = _positive_int(company_id)
    if cid is not None:
        entry["company_id"] = cid
    return entry


def _resolve_tvb_name(type_vendor_bill: Any) -> str:
    """Extract lowercase type_vendor_bill name from string or relation dict."""
    if isinstance(type_vendor_bill, Mapping):
        raw = type_vendor_bill.get("name")
    else:
        raw = type_vendor_bill
    return str(raw or "").strip().lower()


def _journal_entry_link(
    entry_id: Any,
    number: Any,
    type_vendor_bill: Any = None,
    *,
    default_to_bill: bool = False,
    field: str = "number",
    company_id: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Map JournalEntry subtypes (Bill, Invoice, Refund, General Journal) to exact edit routes."""
    eid = _positive_int(entry_id)
    if eid is None:
        return None
    tvb = _resolve_tvb_name(type_vendor_bill)
    if "purchase order" in tvb or "landed cost" in tvb or (default_to_bill and not tvb):
        url = f"/accounting/vendors/bill/edit/{eid}"
        etype = "vendor_bill"
    elif tvb == "sales order":
        url = f"/accounting/vendors/invoice/edit/{eid}/"
        etype = "customer_invoice"
    elif tvb == "refund vendor":
        url = f"/accounting/refund-vendor/edit/{eid}"
        etype = "refund_vendor"
    elif tvb == "refund customer":
        url = f"/accounting/refund-customer/edit/{eid}"
        etype = "refund_customer"
    else:
        url = f"/accounting/journal_entries_edit/{eid}/"
        etype = "journal_entry"
    return _make_link(number, etype, eid, url, field=field, company_id=company_id)


def _unwrap_tool_data(raw_payload: Any) -> Optional[Dict[str, Any]]:
    """Unwrap Django ToolEnvelopeResponse ({status, data}) or direct dict payloads."""
    if not isinstance(raw_payload, Mapping):
        return None
    inner = raw_payload.get("data")
    if isinstance(inner, Mapping) and ("status" in raw_payload or "error" in raw_payload):
        return dict(inner)
    return dict(raw_payload)


def _append_valid(out: List[Dict[str, Any]], link: Optional[Dict[str, Any]]) -> None:
    if link is not None:
        out.append(link)


def _extract_product_link(
    row: Mapping[str, Any],
    *,
    allow_plain_id: bool = False,
    company_id: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Extract a ProductVariant link from common flat or nested product dicts."""
    prod_sub = row.get("product")
    if isinstance(prod_sub, Mapping):
        pid = prod_sub.get("id") or prod_sub.get("product_id")
        label = prod_sub.get("code") or prod_sub.get("sku") or prod_sub.get("name")
        link = _make_link(
            label,
            "product",
            pid,
            f"/inventory/product-variants/{pid}/edit/",
            field="code",
            company_id=company_id,
        )
        if link:
            return link

    erp_sub = row.get("erp_product")
    if isinstance(erp_sub, Mapping):
        pid = erp_sub.get("id") or erp_sub.get("product_id")
        label = erp_sub.get("sku") or erp_sub.get("code") or erp_sub.get("name")
        link = _make_link(
            label,
            "product",
            pid,
            f"/inventory/product-variants/{pid}/edit/",
            field="erp_sku",
            company_id=company_id,
        )
        if link:
            return link

    pid = row.get("product_id") or row.get("erp_product_id")
    if pid is None and allow_plain_id:
        pid = row.get("id")
    label = (
        row.get("code")
        or row.get("erp_sku")
        or row.get("sku")
        or row.get("product_code")
        or row.get("product_name")
        or row.get("erp_name")
        or row.get("name")
    )
    field_name = "erp_sku" if row.get("erp_sku") else "code"
    if _positive_int(pid) is None:
        return None
    return _make_link(
        label,
        "product",
        pid,
        f"/inventory/product-variants/{_positive_int(pid)}/edit/",
        field=field_name,
        company_id=company_id,
    )


# ---------------------------------------------------------------------------
# Per-Tool Entity Link Extractors
# ---------------------------------------------------------------------------

def _links_get_journal_entries(data: Dict[str, Any], company_id: Optional[int]) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    entries = data.get("entries")
    rows = entries if isinstance(entries, list) else ([data] if data.get("id") else [])
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        _append_valid(
            links,
            _journal_entry_link(
                row.get("id"),
                row.get("number"),
                row.get("type_vendor_bill"),
                company_id=company_id,
            ),
        )
        _append_valid(
            links,
            _make_link(
                row.get("vendor_name") or row.get("partner_name"),
                "contact",
                row.get("vendor_id") or row.get("partner_id"),
                f"/contact/contact/{_positive_int(row.get('vendor_id') or row.get('partner_id'))}/update/",
                field="partner_name",
                company_id=company_id,
            ),
        )
    return links


def _links_get_vendor_bills(data: Dict[str, Any], company_id: Optional[int]) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    for row in data.get("bills") or []:
        if not isinstance(row, Mapping):
            continue
        _append_valid(
            links,
            _journal_entry_link(
                row.get("id"),
                row.get("number"),
                row.get("type_vendor_bill"),
                default_to_bill=True,
                company_id=company_id,
            ),
        )
        vid = _positive_int(row.get("vendor_id"))
        if vid is not None:
            _append_valid(
                links,
                _make_link(
                    row.get("vendor_name"),
                    "contact",
                    vid,
                    f"/contact/contact/{vid}/update/",
                    field="vendor_name",
                    company_id=company_id,
                ),
            )
    return links


def _links_get_account_balance(data: Dict[str, Any], company_id: Optional[int]) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    accounts = data.get("accounts")
    rows = accounts if isinstance(accounts, list) else ([data] if data.get("account_code") else [])
    for acc in rows:
        if not isinstance(acc, Mapping):
            continue
        aid = _positive_int(acc.get("account_id") or acc.get("id"))
        if aid is not None:
            _append_valid(
                links,
                _make_link(
                    acc.get("account_code"),
                    "account",
                    aid,
                    f"/accounting/chart_of_accounts/edit/{aid}",
                    field="account_code",
                    company_id=company_id,
                ),
            )
    return links


def _links_get_general_ledger(data: Dict[str, Any], company_id: Optional[int]) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    accounts = data.get("accounts")
    acc_list = accounts if isinstance(accounts, list) else [data]
    for acc in acc_list:
        if not isinstance(acc, Mapping):
            continue
        aid = _positive_int(acc.get("account_id") or acc.get("id"))
        if aid is not None:
            _append_valid(
                links,
                _make_link(
                    acc.get("account_code"),
                    "account",
                    aid,
                    f"/accounting/chart_of_accounts/edit/{aid}",
                    field="account_code",
                    company_id=company_id,
                ),
            )
        for item in acc.get("items") or []:
            if not isinstance(item, Mapping):
                continue
            _append_valid(
                links,
                _journal_entry_link(
                    item.get("journal_entry_id"),
                    item.get("entry_number"),
                    item.get("type_vendor_bill"),
                    field="entry_number",
                    company_id=company_id,
                ),
            )
    return links


def _collect_nested_coa_links(node: Any, out: List[Dict[str, Any]], company_id: Optional[int]) -> None:
    """Recursively collect Chart of Accounts links inside P&L / Balance Sheet trees."""
    if isinstance(node, list):
        for child in node:
            _collect_nested_coa_links(child, out, company_id)
        return
    if not isinstance(node, Mapping):
        return
    aid = _positive_int(node.get("account_id") or (node.get("id") if node.get("account_code") else None))
    if aid is not None and node.get("account_code"):
        _append_valid(
            out,
            _make_link(
                node.get("account_code"),
                "account",
                aid,
                f"/accounting/chart_of_accounts/edit/{aid}",
                field="account_code",
                company_id=company_id,
            ),
        )
    for key in ("breakdown", "groups", "types", "accounts", "sections", "lines", "items"):
        child = node.get(key)
        if isinstance(child, (list, Mapping)):
            _collect_nested_coa_links(child, out, company_id)


def _links_financial_statement(data: Dict[str, Any], company_id: Optional[int]) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    _collect_nested_coa_links(data, links, company_id)
    return links


def _links_get_bank_reconciliation_summary(data: Dict[str, Any], company_id: Optional[int]) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    jid = _positive_int(data.get("journal_id"))
    if jid is not None:
        _append_valid(
            links,
            _make_link(
                data.get("journal_name"),
                "accounting_journal",
                jid,
                f"/accounting/journals/update/{jid}",
                field="journal_name",
                company_id=company_id,
            ),
        )
    aid = _positive_int(data.get("bank_account_id"))
    if aid is not None:
        _append_valid(
            links,
            _make_link(
                data.get("bank_account_code"),
                "account",
                aid,
                f"/accounting/chart_of_accounts/edit/{aid}",
                field="bank_account_code",
                company_id=company_id,
            ),
        )
    return links


def _links_get_product_info(data: Dict[str, Any], company_id: Optional[int]) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    rows = data.get("products")
    if not isinstance(rows, list):
        rows = data.get("items") if isinstance(data.get("items"), list) else [data]
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        _append_valid(links, _extract_product_link(row, allow_plain_id=True, company_id=company_id))
        cat = row.get("category")
        if isinstance(cat, Mapping):
            cid = _positive_int(cat.get("id"))
            if cid is not None:
                _append_valid(
                    links,
                    _make_link(
                        cat.get("name"),
                        "category",
                        cid,
                        f"/inventory/product_categories/{cid}/edit/",
                        field="category",
                        company_id=company_id,
                    ),
                )
    return links


def _links_get_stock_level(data: Dict[str, Any], company_id: Optional[int]) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    _append_valid(links, _extract_product_link(data, allow_plain_id=True, company_id=company_id))

    for key in ("products", "items"):
        rows = data.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            _append_valid(
                links,
                _extract_product_link(
                    row,
                    allow_plain_id=(key == "products"),
                    company_id=company_id,
                ),
            )
            wid = _positive_int(row.get("warehouse_id"))
            if wid is not None:
                _append_valid(
                    links,
                    _make_link(
                        row.get("warehouse_name"),
                        "warehouse",
                        wid,
                        f"/inventory/warehouse/{wid}/edit/",
                        field="warehouse_name",
                        company_id=company_id,
                    ),
                )
            lid = _positive_int(row.get("location_id"))
            if lid is not None:
                _append_valid(
                    links,
                    _make_link(
                        row.get("location_name"),
                        "location",
                        lid,
                        f"/inventory/location/{lid}/edit/",
                        field="location_name",
                        company_id=company_id,
                    ),
                )

    for key in ("warehouse_breakdown", "locations", "quants", "stock_by_warehouse", "warehouses"):
        rows = data.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            wid = _positive_int(row.get("warehouse_id") or (row.get("id") if key == "warehouses" else None))
            if wid is not None:
                _append_valid(
                    links,
                    _make_link(
                        row.get("warehouse_name") or row.get("warehouse") or row.get("name"),
                        "warehouse",
                        wid,
                        f"/inventory/warehouse/{wid}/edit/",
                        field="warehouse_name",
                        company_id=company_id,
                    ),
                )
            lid = _positive_int(row.get("location_id") or (row.get("id") if key == "locations" else None))
            if lid is not None:
                _append_valid(
                    links,
                    _make_link(
                        row.get("location_name") or row.get("location") or row.get("full_name") or row.get("name"),
                        "location",
                        lid,
                        f"/inventory/location/{lid}/edit/",
                        field="location_name",
                        company_id=company_id,
                    ),
                )
            _append_valid(links, _extract_product_link(row, allow_plain_id=False, company_id=company_id))
    return links


def _links_get_low_stock_alerts(data: Dict[str, Any], company_id: Optional[int]) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    for row in data.get("alerts") or []:
        if not isinstance(row, Mapping):
            continue
        _append_valid(links, _extract_product_link(row, allow_plain_id=False, company_id=company_id))
        cat = row.get("category")
        if isinstance(cat, Mapping):
            cid = _positive_int(cat.get("id"))
            if cid is not None:
                _append_valid(
                    links,
                    _make_link(
                        cat.get("name"),
                        "category",
                        cid,
                        f"/inventory/product_categories/{cid}/edit/",
                        field="category",
                        company_id=company_id,
                    ),
                )
        wid = _positive_int(row.get("warehouse_id"))
        if wid is not None:
            _append_valid(
                links,
                _make_link(
                    row.get("warehouse_name"),
                    "warehouse",
                    wid,
                    f"/inventory/warehouse/{wid}/edit/",
                    field="warehouse_name",
                    company_id=company_id,
                ),
            )
        lid = _positive_int(row.get("location_id"))
        if lid is not None:
            _append_valid(
                links,
                _make_link(
                    row.get("location_name"),
                    "location",
                    lid,
                    f"/inventory/location/{lid}/edit/",
                    field="location_name",
                    company_id=company_id,
                ),
            )
    return links


def _links_product_list(data: Dict[str, Any], company_id: Optional[int]) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    _append_valid(links, _extract_product_link(data, allow_plain_id=False, company_id=company_id))
    for key in ("products", "items", "forecasts", "velocity", "history", "records"):
        rows = data.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            _append_valid(
                links,
                _extract_product_link(
                    row,
                    allow_plain_id=(key == "products"),
                    company_id=company_id,
                ),
            )
    return links


def _links_get_transfers(data: Dict[str, Any], company_id: Optional[int]) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    rows = data.get("transfers") or data.get("records") or []
    if not rows and data.get("id") and (data.get("ref") or data.get("ref_alias")):
        rows = [data]
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        tid = _positive_int(row.get("id"))
        if tid is not None:
            _append_valid(
                links,
                _make_link(
                    row.get("ref") or row.get("ref_alias") or row.get("name"),
                    "transfer",
                    tid,
                    f"/inventory/stock_transfer/{tid}/edit/",
                    field="ref",
                    company_id=company_id,
                ),
            )
        for line in row.get("lines") or row.get("items") or []:
            if isinstance(line, Mapping):
                _append_valid(links, _extract_product_link(line, allow_plain_id=False, company_id=company_id))
    return links


def _links_get_scrap_records(data: Dict[str, Any], company_id: Optional[int]) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    rows = data.get("scrap_records") or data.get("records") or []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        sid = _positive_int(row.get("id"))
        if sid is not None:
            _append_valid(
                links,
                _make_link(
                    row.get("ref") or row.get("reference") or row.get("name"),
                    "scrap",
                    sid,
                    f"/inventory/scrap/{sid}/edit/",
                    field="ref",
                    company_id=company_id,
                ),
            )
        _append_valid(links, _extract_product_link(row, allow_plain_id=False, company_id=company_id))
    return links


def _links_get_stock_adjustments(data: Dict[str, Any], company_id: Optional[int]) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    rows = data.get("adjustments") or data.get("records") or []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        _append_valid(links, _extract_product_link(row, allow_plain_id=False, company_id=company_id))
        loc = row.get("location")
        if isinstance(loc, Mapping):
            lid = _positive_int(loc.get("id"))
            if lid is not None:
                _append_valid(
                    links,
                    _make_link(
                        loc.get("name") or loc.get("full_name"),
                        "location",
                        lid,
                        f"/inventory/location/{lid}/edit/",
                        field="location",
                        company_id=company_id,
                    ),
                )
    return links


def _links_get_stock_history(data: Dict[str, Any], company_id: Optional[int]) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    rows = data.get("movements") or data.get("history") or data.get("records") or data.get("items") or []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        _append_valid(links, _extract_product_link(row, allow_plain_id=False, company_id=company_id))
        src_id = _positive_int(row.get("source_id"))
        if src_id is not None:
            _append_valid(
                links,
                _make_link(
                    row.get("source_ref"),
                    "transfer",
                    src_id,
                    f"/inventory/stock_transfer/{src_id}/edit/",
                    field="source_ref",
                    company_id=company_id,
                ),
            )
        wh = row.get("warehouse")
        if isinstance(wh, Mapping):
            wid = _positive_int(wh.get("id"))
            if wid is not None:
                _append_valid(
                    links,
                    _make_link(
                        wh.get("name"),
                        "warehouse",
                        wid,
                        f"/inventory/warehouse/{wid}/edit/",
                        field="warehouse",
                        company_id=company_id,
                    ),
                )
        loc = row.get("location")
        if isinstance(loc, Mapping):
            lid = _positive_int(loc.get("id"))
            if lid is not None:
                _append_valid(
                    links,
                    _make_link(
                        loc.get("full_name") or loc.get("name"),
                        "location",
                        lid,
                        f"/inventory/location/{lid}/edit/",
                        field="location",
                        company_id=company_id,
                    ),
                )
    return links


def _links_get_warehouses(data: Dict[str, Any], company_id: Optional[int]) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    for w in data.get("warehouses") or []:
        if not isinstance(w, Mapping):
            continue
        wid = _positive_int(w.get("id"))
        if wid is not None:
            for lbl in (w.get("name"), w.get("full_name")):
                _append_valid(
                    links,
                    _make_link(
                        lbl,
                        "warehouse",
                        wid,
                        f"/inventory/warehouse/{wid}/edit/",
                        field="name",
                        company_id=company_id,
                    ),
                )
        for loc in w.get("locations") or []:
            if not isinstance(loc, Mapping):
                continue
            lid = _positive_int(loc.get("id"))
            if lid is not None:
                _append_valid(
                    links,
                    _make_link(
                        loc.get("full_name") or loc.get("name"),
                        "location",
                        lid,
                        f"/inventory/location/{lid}/edit/",
                        field="location",
                        company_id=company_id,
                    ),
                )
    return links


def _links_get_locations(data: Dict[str, Any], company_id: Optional[int]) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    for loc in data.get("locations") or []:
        if not isinstance(loc, Mapping):
            continue
        lid = _positive_int(loc.get("id"))
        if lid is not None:
            for lbl in (loc.get("full_name"), loc.get("name")):
                _append_valid(
                    links,
                    _make_link(
                        lbl,
                        "location",
                        lid,
                        f"/inventory/location/{lid}/edit/",
                        field="name",
                        company_id=company_id,
                    ),
                )
        wid = _positive_int(loc.get("warehouse_id"))
        if wid is not None:
            _append_valid(
                links,
                _make_link(
                    loc.get("warehouse_name"),
                    "warehouse",
                    wid,
                    f"/inventory/warehouse/{wid}/edit/",
                    field="warehouse_name",
                    company_id=company_id,
                ),
            )
    return links


def _links_get_product_categories(data: Dict[str, Any], company_id: Optional[int]) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    for cat in data.get("categories") or []:
        if not isinstance(cat, Mapping):
            continue
        cid = _positive_int(cat.get("id"))
        if cid is not None:
            _append_valid(
                links,
                _make_link(
                    cat.get("name"),
                    "category",
                    cid,
                    f"/inventory/product_categories/{cid}/edit/",
                    field="name",
                    company_id=company_id,
                ),
            )
    return links


def _links_get_sales_orders(data: Dict[str, Any], company_id: Optional[int]) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    rows = data.get("orders") or data.get("quotations") or []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        oid = _positive_int(row.get("id"))
        if oid is not None:
            _append_valid(
                links,
                _make_link(
                    row.get("so_number") or row.get("mp_invoice_number"),
                    "sales_order",
                    oid,
                    f"/sales/order/quotation/form/{oid}/",
                    field="so_number",
                    company_id=company_id,
                ),
            )
        cust = row.get("customer")
        cid = _positive_int(cust.get("id") if isinstance(cust, Mapping) else (row.get("customer_id") or row.get("contact_id")))
        cname = cust.get("name") if isinstance(cust, Mapping) else row.get("customer_name")
        if cid is not None:
            _append_valid(
                links,
                _make_link(
                    cname,
                    "contact",
                    cid,
                    f"/contact/contact/{cid}/update/",
                    field="customer_name",
                    company_id=company_id,
                ),
            )
    return links


def _links_get_sales_order_detail(data: Dict[str, Any], company_id: Optional[int]) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    order = data.get("order") if isinstance(data.get("order"), Mapping) else data
    oid = _positive_int(order.get("id"))
    if oid is not None:
        _append_valid(
            links,
            _make_link(
                order.get("so_number") or order.get("mp_invoice_number"),
                "sales_order",
                oid,
                f"/sales/order/quotation/form/{oid}/",
                field="so_number",
                company_id=company_id,
            ),
        )
    cust = order.get("customer")
    cid = _positive_int(cust.get("id") if isinstance(cust, Mapping) else order.get("customer_id"))
    cname = cust.get("name") if isinstance(cust, Mapping) else order.get("customer_name")
    if cid is not None:
        _append_valid(
            links,
            _make_link(
                cname,
                "contact",
                cid,
                f"/contact/contact/{cid}/update/",
                field="customer_name",
                company_id=company_id,
            ),
        )
    for line in order.get("lines") or []:
        if isinstance(line, Mapping):
            _append_valid(links, _extract_product_link(line, allow_plain_id=False, company_id=company_id))
    return links


def _links_get_contacts(data: Dict[str, Any], company_id: Optional[int]) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    rows = data.get("contacts")
    if not isinstance(rows, list):
        rows = [data] if data.get("id") and data.get("name") else []
    for c in rows:
        if not isinstance(c, Mapping):
            continue
        cid = _positive_int(c.get("id"))
        if cid is not None:
            _append_valid(
                links,
                _make_link(
                    c.get("name"),
                    "contact",
                    cid,
                    f"/contact/contact/{cid}/update/",
                    field="name",
                    company_id=company_id,
                ),
            )
        for ro in c.get("recent_orders") or []:
            if not isinstance(ro, Mapping):
                continue
            oid = _positive_int(ro.get("id"))
            if oid is not None:
                _append_valid(
                    links,
                    _make_link(
                        ro.get("so_number"),
                        "sales_order",
                        oid,
                        f"/sales/order/quotation/form/{oid}/",
                        field="so_number",
                        company_id=company_id,
                    ),
                )
    return links


def _links_marketplace_orders(data: Dict[str, Any], company_id: Optional[int]) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    rows = data.get("orders") or data.get("anomalies") or data.get("records") or []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        qid = _positive_int(row.get("quotation_id"))
        if qid is not None:
            _append_valid(
                links,
                _make_link(
                    row.get("so_number"),
                    "sales_order",
                    qid,
                    f"/sales/order/quotation/form/{qid}/",
                    field="so_number",
                    company_id=company_id,
                ),
            )
        aid = _positive_int(row.get("account_id"))
        if aid is not None:
            _append_valid(
                links,
                _make_link(
                    row.get("account_name"),
                    "marketplace_account",
                    aid,
                    f"/marketplace/account/{aid}/edit/",
                    field="account_name",
                    company_id=company_id,
                ),
            )
    return links


def _links_get_deduction_and_settlement_summary(data: Dict[str, Any], company_id: Optional[int]) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    for st in data.get("by_store") or []:
        if not isinstance(st, Mapping):
            continue
        aid = _positive_int(st.get("account_id"))
        if aid is not None:
            _append_valid(
                links,
                _make_link(
                    st.get("account_name"),
                    "marketplace_account",
                    aid,
                    f"/marketplace/account/{aid}/edit/",
                    field="account_name",
                    company_id=company_id,
                ),
            )
    for row in data.get("orders") or []:
        if not isinstance(row, Mapping):
            continue
        qid = _positive_int(row.get("quotation_id"))
        if qid is not None:
            _append_valid(
                links,
                _make_link(
                    row.get("so_number"),
                    "sales_order",
                    qid,
                    f"/sales/order/quotation/form/{qid}/",
                    field="so_number",
                    company_id=company_id,
                ),
            )
        jid = _positive_int(row.get("journal_entry_id"))
        if jid is not None:
            _append_valid(
                links,
                _journal_entry_link(
                    jid,
                    row.get("journal_number"),
                    row.get("type_vendor_bill"),
                    field="journal_number",
                    company_id=company_id,
                ),
            )
        aid = _positive_int(row.get("account_id"))
        if aid is not None:
            _append_valid(
                links,
                _make_link(
                    row.get("account_name"),
                    "marketplace_account",
                    aid,
                    f"/marketplace/account/{aid}/edit/",
                    field="account_name",
                    company_id=company_id,
                ),
            )
    return links


def _links_marketplace_accounts(data: Dict[str, Any], company_id: Optional[int]) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    for key in ("accounts", "account_summaries"):
        for acc in data.get(key) or []:
            if not isinstance(acc, Mapping):
                continue
            aid = _positive_int(acc.get("account_id") or acc.get("id"))
            if aid is not None:
                _append_valid(
                    links,
                    _make_link(
                        acc.get("account_name") or acc.get("name"),
                        "marketplace_account",
                        aid,
                        f"/marketplace/account/{aid}/edit/",
                        field="account_name",
                        company_id=company_id,
                    ),
                )
    return links


def _links_marketplace_mapping_items(data: Dict[str, Any], company_id: Optional[int]) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    for key in ("items", "simulations"):
        for row in data.get(key) or []:
            if not isinstance(row, Mapping):
                continue
            _append_valid(links, _extract_product_link(row, allow_plain_id=False, company_id=company_id))
            aid = _positive_int(row.get("account_id"))
            if aid is not None:
                _append_valid(
                    links,
                    _make_link(
                        row.get("account_name"),
                        "marketplace_account",
                        aid,
                        f"/marketplace/account/{aid}/edit/",
                        field="account_name",
                        company_id=company_id,
                    ),
                )
            mid = _positive_int(row.get("mapping_id"))
            if mid is not None:
                mp_sub = row.get("mp_product")
                mp_sku = mp_sub.get("sku_mp") if isinstance(mp_sub, Mapping) else row.get("mp_sku")
                _append_valid(
                    links,
                    _make_link(
                        mp_sku,
                        "marketplace_mapping",
                        mid,
                        f"/marketplace/mapping/{mid}/edit/",
                        field="mp_sku",
                        company_id=company_id,
                    ),
                )
    return links


def _links_get_sync_execution_logs(data: Dict[str, Any], company_id: Optional[int]) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    for key in ("stock_logs", "price_logs", "pcls_logs"):
        for row in data.get(key) or []:
            if not isinstance(row, Mapping):
                continue
            _append_valid(links, _extract_product_link(row, allow_plain_id=False, company_id=company_id))
            aid = _positive_int(row.get("account_id"))
            if aid is not None:
                _append_valid(
                    links,
                    _make_link(
                        row.get("account_name"),
                        "marketplace_account",
                        aid,
                        f"/marketplace/account/{aid}/edit/",
                        field="account_name",
                        company_id=company_id,
                    ),
                )
    return links


def _links_get_competitor_insights(data: Dict[str, Any], company_id: Optional[int]) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    for row in data.get("competitors") or []:
        if not isinstance(row, Mapping):
            continue
        _append_valid(links, _extract_product_link(row, allow_plain_id=False, company_id=company_id))
    return links


_TOOL_LINK_EXTRACTORS: Dict[str, Callable[[Dict[str, Any], Optional[int]], List[Dict[str, Any]]]] = {
    # Accounting
    "get_journal_entries": _links_get_journal_entries,
    "fetch_vendor_invoices": _links_get_journal_entries,
    "get_vendor_bills": _links_get_vendor_bills,
    "get_account_balance": _links_get_account_balance,
    "get_general_ledger": _links_get_general_ledger,
    "get_profit_and_loss": _links_financial_statement,
    "get_balance_sheet": _links_financial_statement,
    "get_bank_reconciliation_summary": _links_get_bank_reconciliation_summary,
    # Inventory
    "get_product_info": _links_get_product_info,
    "get_stock_level": _links_get_stock_level,
    "get_low_stock_alerts": _links_get_low_stock_alerts,
    "get_inventory_velocity_forecast": _links_product_list,
    "get_price_history": _links_product_list,
    "get_transfers": _links_get_transfers,
    "get_scrap_records": _links_get_scrap_records,
    "get_stock_adjustments": _links_get_stock_adjustments,
    "get_stock_history": _links_get_stock_history,
    "get_warehouses": _links_get_warehouses,
    "get_locations": _links_get_locations,
    "get_product_categories": _links_get_product_categories,
    # Sales & Contact
    "get_sales_orders": _links_get_sales_orders,
    "get_sales_order_detail": _links_get_sales_order_detail,
    "get_contacts": _links_get_contacts,
    "get_top_selling_products": _links_product_list,
    # Marketplace
    "get_marketplace_orders": _links_marketplace_orders,
    "get_fee_and_escrow_anomalies": _links_marketplace_orders,
    "get_deduction_and_settlement_summary": _links_get_deduction_and_settlement_summary,
    "get_account_health": _links_marketplace_accounts,
    "get_cashflow_and_balance_summary": _links_marketplace_accounts,
    "get_kilat_order_diagnostics": _links_marketplace_orders,
    "get_product_mapping_status": _links_marketplace_mapping_items,
    "get_price_comparison": _links_marketplace_mapping_items,
    "get_stock_sync_simulation": _links_marketplace_mapping_items,
    "get_sync_execution_logs": _links_get_sync_execution_logs,
    "get_competitor_insights": _links_get_competitor_insights,
}


def build_entity_links_for_tool(
    tool_name: Optional[str],
    raw_payload: Any,
    company_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Build validated entity links for a single tool invocation payload."""
    if not tool_name:
        return []
    extractor = _TOOL_LINK_EXTRACTORS.get(str(tool_name).strip())
    if extractor is None:
        return []
    unwrapped = _unwrap_tool_data(raw_payload)
    if unwrapped is None:
        return []
    effective_cid = _positive_int(company_id) or _positive_int(unwrapped.get("company_id"))
    return extractor(unwrapped, effective_cid)


def _dedupe_entity_links(candidates: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate entity links by (label, url) preserving first-seen order."""
    deduped: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()
    for link in candidates:
        lbl = str(link.get("label") or "").strip()
        url = str(link.get("url") or "").strip()
        if not lbl or not is_safe_erp_path(url):
            continue
        key = (lbl, url)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(link)
        if len(deduped) >= _MAX_ENTITY_LINKS:
            break
    return deduped


def _resolve_tool_name_from_key(raw_key: Any, fallback_tool: Optional[str]) -> Optional[str]:
    """Resolve canonical tool name from a multi_tool_results key (e.g. 'get_stock_level_0')."""
    if isinstance(raw_key, str) and raw_key.strip():
        candidate = raw_key.strip()
        if candidate in _TOOL_LINK_EXTRACTORS:
            return candidate
        stripped = re.sub(r"_\d+$", "", candidate)
        if stripped in _TOOL_LINK_EXTRACTORS:
            return stripped
    return fallback_tool


def build_entity_links(state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """
    Collect and deduplicate clickable ERP entity redirect links across single-tool,
    multi-tool, active_tools, and multi-company execution state.
    """
    if not isinstance(state, Mapping):
        return []

    domain = str(state.get("detected_domain") or "").strip().lower()
    if domain in ("general", "out_of_scope"):
        return []

    query_mode = str(state.get("query_mode") or "").strip().lower()
    if query_mode in ("general", "out_of_scope", "clarification"):
        return []

    v_status = str(state.get("verification_status") or "").strip().upper()
    if "ERROR" in v_status or v_status == "CLARIFICATION":
        return []

    default_company_id = _positive_int(state.get("company_id"))
    detected_tool = state.get("detected_tool")
    collected: List[Dict[str, Any]] = []

    active_tools = state.get("active_tools")
    if isinstance(active_tools, list):
        for plan in active_tools:
            if not isinstance(plan, Mapping):
                continue
            p_tool = plan.get("tool") or plan.get("detected_tool") or detected_tool
            p_cid = _positive_int(plan.get("company_id")) or default_company_id
            p_mc = plan.get("multi_company_results")
            if isinstance(p_mc, Mapping) and p_mc:
                for mc_cid, mc_res in p_mc.items():
                    if mc_res is not None:
                        collected.extend(
                            build_entity_links_for_tool(
                                p_tool,
                                mc_res,
                                company_id=_positive_int(mc_cid) or p_cid,
                            )
                        )
            p_res = plan.get("result") or plan.get("tool_result") or plan.get("tool_raw_result")
            if p_res is not None:
                collected.extend(build_entity_links_for_tool(p_tool, p_res, company_id=p_cid))

    multi_company = state.get("multi_company_results")
    if isinstance(multi_company, Mapping) and multi_company:
        for cid_key, branch in multi_company.items():
            if branch is None:
                continue
            b_cid = _positive_int(cid_key) or default_company_id
            if isinstance(branch, Mapping) and any(
                k in branch for k in ("tool_raw_result", "raw_result", "tool_result", "result")
            ):
                b_tool = branch.get("detected_tool") or branch.get("tool_name") or branch.get("tool") or detected_tool
                b_payload = (
                    branch.get("tool_raw_result")
                    or branch.get("raw_result")
                    or branch.get("tool_result")
                    or branch.get("result")
                )
            else:
                b_tool = detected_tool
                b_payload = branch
            if b_payload is not None:
                collected.extend(build_entity_links_for_tool(b_tool, b_payload, company_id=b_cid))
    elif isinstance(multi_company, list) and multi_company:
        for branch in multi_company:
            if not isinstance(branch, Mapping):
                continue
            b_tool = (
                branch.get("detected_tool")
                or branch.get("tool_name")
                or branch.get("tool")
                or detected_tool
            )
            b_payload = (
                branch.get("tool_raw_result")
                or branch.get("raw_result")
                or branch.get("tool_result")
                or branch.get("result")
                or branch.get("data")
                or branch
            )
            b_cid = _positive_int(branch.get("company_id")) or default_company_id
            collected.extend(build_entity_links_for_tool(b_tool, b_payload, company_id=b_cid))

    multi_tool = state.get("multi_tool_results")
    if isinstance(multi_tool, Mapping) and multi_tool:
        for t_key, t_val in multi_tool.items():
            if t_val is None:
                continue
            resolved_tool = _resolve_tool_name_from_key(t_key, detected_tool)
            collected.extend(
                build_entity_links_for_tool(resolved_tool, t_val, company_id=default_company_id)
            )
    elif isinstance(multi_tool, list) and multi_tool:
        for item in multi_tool:
            if not isinstance(item, Mapping):
                continue
            t_tool = (
                item.get("detected_tool")
                or item.get("tool_name")
                or item.get("tool")
                or detected_tool
            )
            t_payload = (
                item.get("tool_raw_result")
                or item.get("raw_result")
                or item.get("tool_result")
                or item.get("result")
                or item.get("data")
                or item
            )
            t_cid = _positive_int(item.get("company_id")) or default_company_id
            collected.extend(build_entity_links_for_tool(t_tool, t_payload, company_id=t_cid))

    for state_key in ("tool_raw_result", "tool_result"):
        payload = state.get(state_key)
        if payload is not None and detected_tool:
            collected.extend(
                build_entity_links_for_tool(detected_tool, payload, company_id=default_company_id)
            )

    return _dedupe_entity_links(collected)
