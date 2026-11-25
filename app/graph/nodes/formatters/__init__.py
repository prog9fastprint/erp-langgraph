"""
Domain-organized Markdown presentation formatters for FastPrint ERP tools.
"""
from typing import Any, Callable, Dict, Optional

from app.graph.nodes.formatters.accounting import ACCOUNTING_FORMATTERS
from app.graph.nodes.formatters.common import (
    _append_pagination_footer,
    _extract_total_count,
    _format_currency,
    _format_rag_section,
    _format_rupiah,
)
from app.graph.nodes.formatters.inventory import INVENTORY_FORMATTERS
from app.graph.nodes.formatters.marketplace import MARKETPLACE_FORMATTERS
from app.graph.nodes.formatters.multi_company import (
    MULTI_COMPANY_FORMATTERS,
    _format_multi_company_presentation,
)
from app.graph.nodes.formatters.sales import SALES_FORMATTERS

SINGLE_TOOL_FORMATTERS: Dict[str, Callable[..., str]] = {
    **ACCOUNTING_FORMATTERS,
    **INVENTORY_FORMATTERS,
    **SALES_FORMATTERS,
    **MARKETPLACE_FORMATTERS,
}


def _format_single_tool_presentation(
    tool: Optional[str],
    tool_result: Any,
    user_query: Optional[str] = None,
    extracted_params: Optional[Dict[str, Any]] = None,
    default_company_id: Optional[int] = None,
) -> str:
    """Format an individual tool's response payload into Markdown via domain formatter dispatch."""
    if not tool or not tool_result:
        return ""
    formatter = SINGLE_TOOL_FORMATTERS.get(tool)
    if formatter is not None:
        return formatter(
            tool=tool,
            tool_result=tool_result,
            user_query=user_query,
            extracted_params=extracted_params,
            default_company_id=default_company_id,
        )
    return f"Hasil eksekusi data ERP: {str(tool_result)}"


__all__ = [
    "ACCOUNTING_FORMATTERS",
    "INVENTORY_FORMATTERS",
    "SALES_FORMATTERS",
    "MARKETPLACE_FORMATTERS",
    "SINGLE_TOOL_FORMATTERS",
    "MULTI_COMPANY_FORMATTERS",
    "_format_rupiah",
    "_format_currency",
    "_format_rag_section",
    "_extract_total_count",
    "_append_pagination_footer",
    "_format_single_tool_presentation",
    "_format_multi_company_presentation",
]
