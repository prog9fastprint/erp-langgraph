"""
Context Window & Payload Sanitization Management for FastPrint ERP (TASK-Context).
Provides centralized sliding window message trimming, Django ERP payload pruning,
and contextual summary prompt injection.
"""
from datetime import date
import re
from typing import Any, Dict, List, Optional, Set
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

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
    Preserves exact SKU codes and clean product names.
    """
    if not value or not isinstance(value, str):
        return value
    cleaned = value.strip().strip("'\"").rstrip("?!.,;: '\"")
    res = re.sub(
        r"^(?:(?:nama\s+(?:produk|barang|product|item)|product|produk|barang|item|sku|kode)\s*[:\-]?\s*)+",
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


def sanitize_erp_payload(data: Any, max_records: int = 50) -> Any:
    """
    Recursively sanitize and prune ERP data payloads from Django Core.

    Removes technical Django audit fields, null/empty values, and truncates
    excessively large lists to prevent context window overflow while preserving
    all essential financial and inventory business fields.

    Args:
        data: Raw data structure (dict, list, or primitive) from ERP client.
        max_records: Maximum items allowed in large record lists (default: 50).

    Returns:
        Cleaned and pruned data structure.
    """
    if data is None:
        return None

    if isinstance(data, list):
        sanitized_list = [
            sanitize_erp_payload(item, max_records=max_records)
            for item in data[:max_records]
            if item is not None
        ]
        return sanitized_list

    if isinstance(data, dict):
        sanitized_dict: Dict[str, Any] = {}
        pre_truncation_count: Optional[tuple[str, int]] = None

        for key, value in data.items():
            # 1. Skip technical Django fields
            if key in DJANGO_TECHNICAL_FIELDS or key.startswith("_"):
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
                cleaned_val = sanitize_erp_payload(value, max_records=max_records)
                # Keep calculations, summary, and explicitly track collection keys even if empty
                if cleaned_val or key in ("calculations", "summary", "totals", "accounts", "entries", "lines", "items", "products", "bills", "breakdown", "net_balance_check"):
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
