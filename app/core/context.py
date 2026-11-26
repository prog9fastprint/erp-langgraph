"""
Context Window & Payload Sanitization Management for FastPrint ERP (TASK-Context).
Provides centralized sliding window message trimming, Django ERP payload pruning,
and contextual summary prompt injection.
"""
from datetime import date, datetime, timedelta, timezone, tzinfo
import re
from typing import Any, Dict, List, Optional, Set
from zoneinfo import ZoneInfo
from langchain_core.messages import BaseMessage

from app.core.config import settings

_FIXED_TIMEZONE_FALLBACKS: Dict[str, tzinfo] = {
    "Asia/Jakarta": timezone(timedelta(hours=7)),
    "Asia/Pontianak": timezone(timedelta(hours=7)),
    "Asia/Makassar": timezone(timedelta(hours=8)),
    "Asia/Jayapura": timezone(timedelta(hours=9)),
    "UTC": timezone.utc,
}


def get_erp_tzinfo() -> tzinfo:
    """Resolve the configured ERP operational timezone with fixed-offset fallback."""
    tz_name = getattr(settings, "APP_TIMEZONE", "Asia/Jakarta") or "Asia/Jakarta"
    try:
        return ZoneInfo(tz_name)
    except Exception:
        return _FIXED_TIMEZONE_FALLBACKS.get(tz_name, timezone(timedelta(hours=7)))


def get_current_erp_datetime() -> datetime:
    """Return current timezone-aware datetime in the configured ERP operational timezone."""
    return datetime.now(get_erp_tzinfo())


def get_current_erp_date() -> date:
    """Return current calendar date in the configured ERP operational timezone."""
    return get_current_erp_datetime().date()

# Internal Django ORM and technical audit fields to strip from ERP payloads
DJANGO_TECHNICAL_FIELDS: Set[str] = {
    "create_uid",
    "write_uid",
    "create_date",
    "write_date",
    "__last_update",
    "access_token",
    "audit_trail",
    "is_active",
    "message_follower_ids",
    "message_ids",
    "activity_ids",
    "website_message_ids",
}


def clean_product_identifier(value: Optional[str]) -> Optional[str]:
    """
    Strip conversational noise or leading prefixes like 'product', 'produk', 'barang', 'item', 'sku', 'kode'.
    Preserves hyphenated SKU codes (e.g. 'SKU-UNKNOWN-999') and clean product names.
    """
    if not value or not isinstance(value, str):
        return value
    cleaned = value.strip().strip("'\"").rstrip("?!.,;: '\"")
    res = re.sub(
        r"^(?:(?:nama\s+(?:produk|barang|product|item)|product|produk|barang|item|sku|kode)(?:\s*:\s*|\s+-\s+|\s+))+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip().rstrip("?!.,;: '\"")
    return res if res else cleaned


def trim_conversation_messages(
    messages: Optional[List[BaseMessage]],
    max_messages: int = 10,
) -> List[BaseMessage]:
    """
    Trim conversation message list to the most recent N messages, preserving order.

    Filters out empty messages and guarantees a bounded context window for downstream nodes.

    Args:
        messages: List of LangChain BaseMessage objects (or None).
        max_messages: Maximum number of recent messages to preserve (default: 10).

    Returns:
        Bounded list of BaseMessage objects.
    """
    if not messages:
        return []

    valid_messages = [
        msg for msg in messages
        if getattr(msg, "content", None) and str(msg.content).strip()
    ]

    if len(valid_messages) <= max_messages:
        return valid_messages

    return valid_messages[-max_messages:]


# Collection keys whose pre-truncation length should be preserved as total_count/total_found
_PAGINATED_COLLECTION_KEYS: Set[str] = {
    "bills",
    "entries",
    "accounts",
    "products",
    "locations",
    "history",
    "transfers",
    "alerts",
    "adjustments",
    "scraps",
    "orders",
    "contacts",
    "records",
    "mappings",
    "exceptions",
    "simulations",
    "stock_logs",
    "price_logs",
    "pcls_logs",
    "anomalies",
    "by_deduction_category",
    "by_store",
    "cashflow_logs",
    "competitors",
    "log_entry_candidates",
    "quotation_note_candidates",
    "fast_moving",
    "slow_moving",
    "restock_recommendations",
    "by_branch",
    "monthly_breakdown",
    "cancel_reason_summary",
    "error_breakdown_by_type",
    "error_breakdown_by_company",
}

_TOTAL_FOUND_COLLECTION_KEYS: Set[str] = {"locations", "alerts"}


def _safe_float(value: Any) -> float:
    """Safely coerce numeric payload values to float, returning 0.0 on None/invalid."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def _compute_bills_calculations(bills: List[Any]) -> Dict[str, Any]:
    """Compute pre-truncation aggregate totals across all vendor bill rows."""
    total_amount = 0.0
    total_paid = 0.0
    total_residual = 0.0
    overdue_count = 0

    for bill in bills:
        if not isinstance(bill, dict):
            continue
        total_amount += _safe_float(bill.get("total_amount"))
        total_paid += _safe_float(bill.get("paid_amount"))
        total_residual += _safe_float(bill.get("residual_balance"))
        if str(bill.get("due_status") or "").strip().lower() == "overdue":
            overdue_count += 1

    return {
        "total_amount": round(total_amount, 2),
        "total_paid": round(total_paid, 2),
        "total_residual": round(total_residual, 2),
        "overdue_count": overdue_count,
    }


def sanitize_erp_payload(
    data: Any,
    max_records: int = 50,
    preserve_fields: Optional[Set[str]] = None,
) -> Any:
    """
    Recursively sanitize and prune ERP data payloads from Django Core.

    Removes technical Django audit fields, null/empty values, and truncates
    excessively large lists to prevent context window overflow while preserving
    all essential financial and inventory business fields.

    Args:
        data: Raw data structure (dict, list, or primitive) from ERP client.
        max_records: Maximum items allowed in large record lists (default: 50).
        preserve_fields: Optional field names exempted from DJANGO_TECHNICAL_FIELDS stripping.

    Returns:
        Cleaned and pruned data structure.
    """
    if data is None:
        return None

    preserved = preserve_fields or set()

    if isinstance(data, list):
        sanitized_list = [
            sanitize_erp_payload(item, max_records=max_records, preserve_fields=preserved)
            for item in data[:max_records]
            if item is not None
        ]
        return sanitized_list

    if isinstance(data, dict):
        sanitized_dict: Dict[str, Any] = {}
        pre_truncation_count: Optional[tuple[str, int]] = None

        for key, value in data.items():
            # 1. Skip technical Django fields unless explicitly preserved for the tool
            if (key in DJANGO_TECHNICAL_FIELDS or key.startswith("_")) and key not in preserved:
                continue

            # 2. Skip null, empty string, or whitespace-only values
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue

            # Track pre-truncation collection length for downstream pagination footers
            if (
                key in _PAGINATED_COLLECTION_KEYS
                and isinstance(value, list)
                and len(value) > max_records
                and pre_truncation_count is None
            ):
                count_key = "total_found" if key in _TOTAL_FOUND_COLLECTION_KEYS else "total_count"
                pre_truncation_count = (count_key, len(value))

            # 3. Recursively sanitize nested structures
            if isinstance(value, (dict, list)):
                cleaned_val = sanitize_erp_payload(value, max_records=max_records, preserve_fields=preserved)
                # Keep calculations, summary, and explicitly track collection keys even if empty
                if cleaned_val or key in (
                    "calculations", "summary", "totals", "accounts", "entries", "lines", "items",
                    "products", "bills", "breakdown", "net_balance_check", "stock_summary", "price_summary",
                    "orders", "contacts", "records", "mappings", "exceptions", "simulations", "stock_logs", "price_logs",
                    "pcls_logs", "anomalies", "by_deduction_category", "by_store", "cashflow_logs",
                    "competitors", "unassigned_company_errors", "log_entry_candidates", "quotation_note_candidates",
                    "fast_moving", "slow_moving", "restock_recommendations", "by_branch", "monthly_breakdown",
                    "cancel_reason_summary", "error_breakdown_by_type", "error_breakdown_by_company",
                ):
                    sanitized_dict[key] = cleaned_val
            else:
                sanitized_dict[key] = value

        # Pre-compute vendor bill aggregates from full list before truncation if absent
        bills_raw = data.get("bills")
        if isinstance(bills_raw, list) and len(bills_raw) > 0 and not sanitized_dict.get("calculations"):
            sanitized_dict["calculations"] = _compute_bills_calculations(bills_raw)

        # Preserve pre-truncation count when total_count/total_found was absent or empty
        if (
            pre_truncation_count is not None
            and not sanitized_dict.get("total_count")
            and not sanitized_dict.get("total_found")
        ):
            sanitized_dict[pre_truncation_count[0]] = pre_truncation_count[1]

        return sanitized_dict

    return data


def build_contextual_prompt(
    system_prompt: str,
    conversation_summary: Optional[str] = None,
) -> str:
    """
    Inject conversation summary context into a base system prompt if available.

    Args:
        system_prompt: Base node instructions.
        conversation_summary: Optional Markdown entity summary from prior turns.

    Returns:
        Augmented system prompt string.
    """
    if not conversation_summary or not conversation_summary.strip():
        return system_prompt

    return (
        f"{system_prompt}\n\n"
        f"### Prior Conversation Context & Entity Summary:\n"
        f"{conversation_summary.strip()}"
    )


def format_deterministic_summary(messages: List[BaseMessage]) -> str:
    """
    Deterministic rule-based summary builder for conversation history.
    Extracts SKU codes, dates, and entity keywords when LLM is unavailable.

    Args:
        messages: Historical BaseMessage list.

    Returns:
        Concise Markdown summary string.
    """
    if not messages:
        return ""

    entities: List[str] = []
    dates_found: List[str] = []

    for msg in messages:
        text = getattr(msg, "content", "") or ""
        # Extract potential SKUs / codes
        skus = re.findall(r"\b([A-Z0-9]+(?:-[A-Z0-9]+)+|\d{3,8})\b", text)
        for sku in skus:
            if sku not in entities and not re.match(r"^\d{4}$", sku):  # exclude plain years
                entities.append(sku)

        # Extract dates
        dates = re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", text)
        for d in dates:
            if d not in dates_found:
                dates_found.append(d)

    summary_lines = ["Rangkuman Konteks Percakapan:"]
    if entities:
        summary_lines.append(f"- Entitas / Kode Terkait: {', '.join(entities[:5])}")
    if dates_found:
        summary_lines.append(f"- Tanggal Teridentifikasi: {', '.join(dates_found[:3])}")
    summary_lines.append(f"- Total Riwayat Giliran: {len(messages)} pesan")

    return "\n".join(summary_lines)


_STRICT_DOC_ID_RE = re.compile(r"\b(?:[A-Za-z0-9_\-]+/)+\d{4}/\d+\b")


def has_strict_document_identifier(params: Optional[Dict[str, Any]]) -> bool:
    """Return True if params target a specific ERP document or Kilat reservation identifier."""
    if not isinstance(params, dict):
        return False
    if params.get("entry_number") or params.get("reference"):
        return True
    search_val = params.get("search")
    return bool(search_val and _STRICT_DOC_ID_RE.search(str(search_val)))


def compute_multi_tool_counts(active_tools: Optional[List[Dict[str, Any]]]) -> Dict[str, int]:
    """Count occurrences of each tool name across active_tools to disambiguate duplicate keys."""
    counts: Dict[str, int] = {}
    for plan in active_tools or []:
        t_name = plan.get("tool", "unnamed_tool")
        counts[t_name] = counts.get(t_name, 0) + 1
    return counts


def get_multi_tool_key(tool_name: str, idx: int, tool_counts: Dict[str, int]) -> str:
    """Return canonical result/status key for an active_tools entry."""
    return f"{tool_name}_{idx}" if tool_counts.get(tool_name, 0) > 1 else tool_name


def should_preserve_plan_on_retry(
    plan: Dict[str, Any],
    key: str,
    multi_verification_status: Optional[Dict[str, str]],
) -> bool:
    """Return True if a multi-tool plan should keep its params/result during retry."""
    if not isinstance(plan, dict):
        return False
    status_val = (multi_verification_status or {}).get(key)
    if plan.get("status") == "success" and status_val == "VALID":
        return True
    return has_strict_document_identifier(plan.get("params"))


SENSITIVE_PARAM_KEYS: Set[str] = {
    "user_token",
    "token",
    "access_token",
    "password",
    "secret",
    "service_key",
    "api_key",
    "authorization",
    "client_secret",
}


def sanitize_tool_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Sanitize ERP tool input parameters for safe client-facing SSE emission.
    Removes authentication tokens, passwords, secrets, internal keys, and empty values.
    """
    if not isinstance(params, dict):
        return {}
    sanitized: Dict[str, Any] = {}
    for k, v in params.items():
        if k in SENSITIVE_PARAM_KEYS or k.startswith("_"):
            continue
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        sanitized[k] = v
    return sanitized


_RECORD_COUNT_KEYS = (
    "total_count",
    "total_found",
    "count",
    "records_count",
    "total_products",
)

_CLEAN_AUDIT_SCANNED_KEYS = (
    "total_accounts",
    "total_evaluated",
    "total_kilat_scanned",
    "total_lines",
    "total_orders",
    "active_competitors",
)

_LOG_TOTAL_KEYS = (
    "stock_logs_total",
    "price_logs_total",
    "pcls_logs_total",
    "cashflow_logs_total",
)

_COLLECTION_LIST_KEYS = (
    "entries",
    "items",
    "accounts",
    "products",
    "stocks",
    "bills",
    "records",
    "lines",
    "locations",
    "transfers",
    "history",
    "adjustments",
    "scraps",
    "warehouses",
    "categories",
    "alerts",
    "orders",
    "contacts",
    "mappings",
    "exceptions",
    "simulations",
    "stock_logs",
    "price_logs",
    "pcls_logs",
    "anomalies",
    "by_store",
    "by_deduction_category",
    "cashflow_logs",
    "competitors",
    "fast_moving",
    "slow_moving",
    "restock_recommendations",
)

_BALANCE_METRIC_KEYS = {
    "net_balance",
    "total_debit",
    "total_credit",
    "grand_net_balance",
    "grand_total_debit",
    "grand_total_credit",
    "total_revenue",
    "total_assets",
    "bank_statement_balance",
}

_KNOWN_RECORD_ATTRS = {
    "id",
    "product_id",
    "product_name",
    "sku",
    "code",
    "default_code",
    "barcode",
    "account_id",
    "account_code",
    "account_name",
    "location_id",
    "location_name",
    "partner_id",
    "partner_name",
    "vendor_id",
    "vendor_name",
    "invoice_id",
    "journal_id",
    "move_id",
    "reference",
    "name",
    "uom",
    "qty_on_hand",
    "quantity",
    "available_quantity",
    "virtual_available",
    "incoming_qty",
    "outgoing_qty",
    "net_balance",
    "total_debit",
    "total_credit",
    "grand_net_balance",
    "amount_total",
    "price_unit",
    "list_price",
    "standard_price",
    "debit",
    "credit",
    "balance",
    "total_revenue",
    "gross_profit",
    "net_profit",
    "cogs",
    "operating_profit",
    "gross_margin_percent",
    "total_assets",
    "total_liabilities",
    "total_equity",
    "total_liabilities_and_equity",
    "bank_statement_balance",
    "erp_book_balance",
    "residual_balance",
    "bill_date",
    "due_date",
    "due_status",
    "closing_balance",
    "running_balance",
}


def resolve_tool_record_count(tool_result: Any) -> int:
    """
    Compute total record count found in ERP tool response envelope.
    Handles error payloads, paginated counts, collections, nested data, and aggregate summaries.
    """
    if tool_result is None:
        return 0
    if isinstance(tool_result, list):
        return len(tool_result)
    if not isinstance(tool_result, dict) or not tool_result:
        return 0

    if (
        tool_result.get("error") is not None
        or tool_result.get("detail") is not None
        or tool_result.get("errors") is not None
        or tool_result.get("success") is False
    ):
        return 0

    if (
        tool_result.get("exclude_zero") is True
        and isinstance(tool_result.get("out_of_stock_count"), (int, float))
        and not isinstance(tool_result.get("out_of_stock_count"), bool)
        and int(tool_result["out_of_stock_count"]) > 0
    ):
        alerts_list = tool_result.get("alerts")
        if isinstance(alerts_list, list) and len(alerts_list) > 0:
            return len(alerts_list)
        return 1

    for count_key in _RECORD_COUNT_KEYS:
        val = tool_result.get(count_key)
        if val is not None and not isinstance(val, bool):
            try:
                c_int = int(val)
                if c_int > 0:
                    return c_int
                summary_blk = tool_result.get("summary")
                if isinstance(summary_blk, dict) and summary_blk:
                    if any(
                        int(summary_blk.get(sk) or 0) > 0
                        for sk in _CLEAN_AUDIT_SCANNED_KEYS
                    ) or ("total_anomalies" in summary_blk and ("date_window" in tool_result or "filters" in tool_result)):
                        return 1
                return 0
            except (ValueError, TypeError):
                pass

    log_totals = [
        tool_result.get(k)
        for k in _LOG_TOTAL_KEYS
        if tool_result.get(k) is not None and not isinstance(tool_result.get(k), bool)
    ]
    has_log_summary = any(
        isinstance(tool_result.get(sk), dict) and tool_result[sk]
        for sk in ("stock_summary", "price_summary")
    )
    if log_totals or has_log_summary:
        try:
            log_sum = sum(int(v) for v in log_totals)
            if log_sum > 0:
                return log_sum
            if any(
                isinstance(tool_result.get(sk), dict) and int(tool_result[sk].get("total") or 0) > 0
                for sk in ("stock_summary", "price_summary")
            ):
                return 1
            return 0
        except (ValueError, TypeError):
            pass

    for list_key in _COLLECTION_LIST_KEYS:
        collection = tool_result.get(list_key)
        if isinstance(collection, list):
            if len(collection) > 0:
                return len(collection)
            summary_blk = tool_result.get("summary")
            if isinstance(summary_blk, dict) and any(
                int(summary_blk.get(sk) or 0) > 0 for sk in _CLEAN_AUDIT_SCANNED_KEYS
            ):
                return 1
            return 0

    if any(
        isinstance(tool_result.get(k), dict) and tool_result[k]
        for k in ("summary", "totals", "confirmed_metrics", "stock_summary", "price_summary")
    ):
        return 1

    if "data" in tool_result and isinstance(tool_result["data"], (dict, list)):
        return resolve_tool_record_count(tool_result["data"])

    if any(k in _BALANCE_METRIC_KEYS for k in tool_result.keys()):
        has_value = any(
            float(v or 0) != 0.0
            for v in tool_result.values()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        )
        return 1 if has_value else 0

    has_business_attr = any(k in _KNOWN_RECORD_ATTRS for k in tool_result.keys())
    has_non_none_value = any(v is not None for v in tool_result.values())
    if has_business_attr and has_non_none_value:
        return 1

    return 0


def chunk_text_for_streaming(text: str, max_chunk_len: int = 35) -> List[str]:
    """
    Split markdown text into clean streaming chunks preserving words and whitespace.
    """
    if not text:
        return []
    words = re.findall(r"\S+\s*", text)
    if not words:
        return [text]
    chunks: List[str] = []
    curr = ""
    for w in words:
        if len(curr) + len(w) > max_chunk_len and curr:
            chunks.append(curr)
            curr = w
        else:
            curr += w
    if curr:
        chunks.append(curr)
    return chunks


ANAPHORA_TOKENS = {"itu", "tersebut", "tadi", "ini", "terkait", "dia", "nya", "sana", "situ"}
ANAPHORIC_CLITICS = (
    "stoknya", "harganya", "barangnya", "produknya", "saldonya",
    "tagihannya", "fakturnya", "invoicenya", "lokasinya", "detailnya",
    "transaksinya", "jurnalnya", "omzetnya", "omsetnya", "ruginya",
    "labanya", "neracanya", "kasnya", "sisanya",
)


def has_anaphoric_reference(text: str) -> bool:
    """Check if text contains Indonesian anaphoric clitics, pronouns, or elliptical clauses."""
    cleaned = (text or "").lower()
    if any(clitic in cleaned for clitic in ANAPHORIC_CLITICS):
        return True
    if any(re.search(rf"\b{re.escape(token)}\b", cleaned) for token in ANAPHORA_TOKENS):
        return True
    if re.search(r"\b(?:berapa|gimana|bagaimana|mana|ada berapa)\s+(?:stok|harga|saldo|tagihan|faktur|detail)\b", cleaned):
        return True
    return False

