"""
Router Node for FastPrint ERP LangGraph Agent Workflow (TASK-4.2).
Provides high-speed hybrid routing: keyword/regex pattern matching for ERP modules
with negative guardrails (preventing customer-vendor confusion), multi-intent subagent planning,
and LLM fallback.
"""
import logging
import os
import re
import sys
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from pydantic import BaseModel, Field, model_validator

from app.core.config import settings
from app.core.context import trim_conversation_messages
from app.graph.state import AgentState
from app.schemas.tools import TOOL_DOMAIN_MAP
from app.services.llm import get_chat_model

from app.core.branches import (
    COMPANY_NAME_MAP,
    resolve_all_branch_company_ids,
    resolve_branch_company_id,
    resolve_prior_multi_company_scope,
    strip_company_scope_keywords,
)
from app.core.context import (
    extract_period_from_messages,
    has_anaphoric_reference,
    has_sufficient_prior_erp_context,
    is_analytical_thinking_query,
)
from app.graph.nodes.extractors.accounting import resolve_pnl_drilldown_from_history
from app.graph.nodes.extractors.common import resolve_relative_dates
from app.graph.nodes.router_decomposer import (
    _extract_sub_queries,
    _split_conjunction_clauses,
    decompose_user_query,
    find_previous_user_query,
    is_retry_query,
)
from app.graph.nodes.router_patterns import (
    CUSTOMER_TERMS,
    VENDOR_TERMS,
    PURCHASE_INVOICE_TERMS,
    CUSTOMER_BILLING_TERMS,
    ACCOUNTING_PROFIT_LOSS_PATTERNS,
    ACCOUNTING_BALANCE_SHEET_PATTERNS,
    ACCOUNTING_GENERAL_LEDGER_PATTERNS,
    ACCOUNTING_BANK_RECON_PATTERNS,
    ACCOUNTING_VENDOR_BILLS_PATTERNS,
    ACCOUNTING_BALANCE_PATTERNS,
    ACCOUNTING_JOURNAL_PATTERNS,
    INVENTORY_STOCK_PATTERNS,
    INVENTORY_STOCK_HISTORY_PATTERNS,
    INVENTORY_TRANSFERS_PATTERNS,
    INVENTORY_LOW_STOCK_PATTERNS,
    INVENTORY_ADJUSTMENTS_PATTERNS,
    INVENTORY_SCRAP_PATTERNS,
    INVENTORY_WAREHOUSES_PATTERNS,
    INVENTORY_CATEGORIES_PATTERNS,
    INVENTORY_LOCATION_PATTERNS,
    INVENTORY_VELOCITY_FORECAST_PATTERNS,
    INVENTORY_PRICE_HISTORY_PATTERNS,
    SALES_TOP_PRODUCTS_PATTERNS,
    SALES_SUMMARY_PATTERNS,
    SALES_ORDER_DETAIL_PATTERNS,
    SALES_CONTACTS_PATTERNS,
    SALES_ORDERS_PATTERNS,
    MARKETPLACE_WRITE_PATTERNS,
    MARKETPLACE_ACCOUNT_HEALTH_PATTERNS,
    MARKETPLACE_KILAT_PATTERNS,
    MARKETPLACE_ORDERS_PATTERNS,
    MARKETPLACE_MAPPING_PATTERNS,
    MARKETPLACE_STOCK_SYNC_SIM_PATTERNS,
    MARKETPLACE_SYNC_LOGS_PATTERNS,
    MARKETPLACE_PRICE_COMPARISON_PATTERNS,
    MARKETPLACE_FEE_ANOMALIES_PATTERNS,
    MARKETPLACE_DEDUCTION_SETTLEMENT_PATTERNS,
    MARKETPLACE_CASHFLOW_BALANCE_PATTERNS,
    MARKETPLACE_COMPETITOR_PATTERNS,
    ALL_MARKETPLACE_PATTERN_GROUPS,
    INVENTORY_PRODUCT_PATTERNS,
    KNOWLEDGE_RAG_PATTERNS,
    GREETING_PATTERNS,
    GENERAL_INQUIRY_PATTERNS,
    OUT_OF_SCOPE_PATTERNS,
    RETRY_PATTERNS,
    _CASUAL_NON_ERP_INTERROGATIVE_RE,
    _FOLLOWUP_ELLIPTICAL_PREFIX_RE,
    _STANDALONE_NON_FOLLOWUP_INTERROGATIVE_RE,
    _match_any,
    _find_first_match,
    is_allowed_general_query,
    is_general_help_inquiry,
    is_meta_question,
    is_pure_greeting,
)

logger = logging.getLogger("router_node")


class PlannedToolCall(BaseModel):
    """Structured representation of an individual ERP tool call planned by router."""
    tool: str = Field(..., description="Target ERP tool name.")
    domain: Literal["accounting", "inventory", "sales", "marketplace", "hr", "purchasing", "general"] = Field(
        ...,
        description="Target ERP domain: 'accounting', 'inventory', 'sales', or 'marketplace'."
    )
    sub_query: Optional[str] = Field(
        None,
        description="Specific user query clause or segment relevant to this tool call."
    )

    @model_validator(mode="after")
    def validate_tool_domain(self) -> "PlannedToolCall":
        """Ensure domain aligns with the tool's registered domain."""
        from app.services.tool_registry_service import get_tool_registry_service
        registry = get_tool_registry_service()
        registered = registry.get_tool_by_name(self.tool)
        if self.tool not in TOOL_DOMAIN_MAP and registered is None:
            raise ValueError(f"Unregistered tool: '{self.tool}'. Tool must be registered in TOOL_DOMAIN_MAP or ToolRegistryService.")
        expected = TOOL_DOMAIN_MAP.get(self.tool) or (registered.get("domain") if registered else None)
        if expected and self.domain != expected:
            self.domain = expected
        return self


class RouterDecision(BaseModel):
    """Structured output schema for LLM router fallback."""
    domain: Literal["accounting", "inventory", "sales", "marketplace", "knowledge", "general", "out_of_scope"] = Field(
        ...,
        description=(
            "Target ERP module domain: 'accounting', 'inventory', 'sales', 'marketplace', "
            "'knowledge' (SOPs, specs, policies), 'general' (STRICTLY ONLY pure greetings, thanks, bot capability/help, or session meta-questions), "
            "or 'out_of_scope' (ANY non-ERP topic such as trivia, weather, recipes, coding, chitchat, unsupported modules like HR/payroll, or write requests)."
        ),
    )
    query_mode: Literal["live_erp", "vector_rag", "hybrid", "general", "out_of_scope", "contextual_analysis"] = Field(
        "live_erp",
        description="Execution mode: 'vector_rag' for SOP/policies/specs, 'live_erp' for real-time transactions/stock, 'hybrid' for combined questions, 'contextual_analysis' for follow-up analytical questions on prior report data, 'general' (greetings/help only), or 'out_of_scope' (anything outside ERP capabilities).",
    )
    tool: Optional[str] = Field(
        None,
        description="Target tool name to invoke in ERP backend, or null if knowledge / general / out_of_scope."
    )
    rag_query: Optional[str] = Field(
        None,
        description="Extracted search query for RAG knowledge retrieval if query_mode is vector_rag or hybrid."
    )
    reasoning: Optional[str] = Field(
        None,
        description="Brief explanation of why this domain and tool were selected."
    )

    @model_validator(mode="after")
    def clean_tool_for_general_and_out_of_scope(self) -> "RouterDecision":
        """Enforce null tool whenever domain is general, out_of_scope, or knowledge."""
        if self.domain in ("general", "out_of_scope", "knowledge") or self.query_mode == "vector_rag":
            self.tool = None
        return self


class MultiRouterDecision(BaseModel):
    """Structured output schema for LLM router supporting multi-intent planning."""
    domain: Literal["accounting", "inventory", "sales", "marketplace", "knowledge", "general", "out_of_scope"] = Field(
        ...,
        description=(
            "Primary target ERP module domain: 'accounting', 'inventory', 'sales', 'marketplace', "
            "'knowledge' (SOPs/specs/policies), 'general' (STRICTLY ONLY pure greetings, thanks, bot capability/help, or session meta-questions), "
            "or 'out_of_scope' (ANY non-ERP topic such as general knowledge, trivia, recipes, weather, coding, chitchat, unsupported modules, or write operations)."
        ),
    )
    query_mode: Literal["live_erp", "vector_rag", "hybrid", "general", "out_of_scope", "contextual_analysis"] = Field(
        "live_erp",
        description="Execution mode: 'vector_rag', 'live_erp', 'hybrid', 'contextual_analysis', 'general' (greetings/help only), or 'out_of_scope' (anything outside ERP capabilities).",
    )
    tool: Optional[str] = Field(
        None,
        description="Primary target tool name (for backward compatibility), or null if general/out_of_scope/knowledge."
    )
    planned_tools: List[PlannedToolCall] = Field(
        default_factory=list,
        description="List of planned tool calls when one or more live ERP tools are needed."
    )
    rag_query: Optional[str] = Field(
        None,
        description="Extracted search query for RAG knowledge retrieval if query_mode is vector_rag or hybrid."
    )
    reasoning: Optional[str] = Field(
        None,
        description="Brief explanation of why these domains and tools were selected."
    )

    @model_validator(mode="after")
    def reconcile_and_clean_tools(self) -> "MultiRouterDecision":
        """Clean tools on non-tool domains and synchronize primary tool with planned_tools."""
        if self.domain in ("general", "out_of_scope", "knowledge") or self.query_mode in ("vector_rag", "contextual_analysis"):
            self.tool = None
            self.planned_tools = []
            return self

        # If planned_tools is populated but primary tool is omitted, set from first planned tool
        if self.planned_tools and not self.tool:
            self.tool = self.planned_tools[0].tool
            self.domain = self.planned_tools[0].domain

        # If primary tool is specified without planned_tools, synthesize planned_tools
        if self.tool and not self.planned_tools:
            from app.services.tool_registry_service import get_tool_registry_service
            registry = get_tool_registry_service()
            registered = registry.get_tool_by_name(self.tool)
            d = TOOL_DOMAIN_MAP.get(self.tool) or (registered.get("domain") if registered else self.domain)
            if d in ("accounting", "inventory", "sales", "marketplace"):
                self.planned_tools = [PlannedToolCall(tool=self.tool, domain=d)]

        return self



def _find_previous_erp_query(messages: Optional[List[BaseMessage]]) -> Optional[str]:
    """
    Find the latest substantive HumanMessage that routes to a concrete ERP domain,
    allowing intermediate short follow-ups ('semua cabang', 'breakdown datanya') to chain.
    """
    if not messages:
        return None
    for msg in reversed(messages[:-1]):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") in ("human", "user"):
            content_str = str(getattr(msg, "content", "") or "").strip()
            if not content_str or is_retry_query(content_str):
                continue
            try:
                cand = classify_query_rule_based(content_str)
                if isinstance(cand, dict) and cand.get("detected_domain") in (
                    "accounting", "inventory", "sales", "marketplace"
                ):
                    return content_str
            except Exception as exc:
                logger.debug("Prior ERP query probe failed: %s", exc)
    return find_previous_user_query(messages)


_NON_PRODUCT_SALES_TOKENS = {
    "perusahaan", "cabang", "toko", "outlet", "kantor", "pusat", "semua", "seluruh",
    "hari", "minggu", "bulan", "tahun", "kemarin", "ini", "itu", "lalu", "terakhir",
    "shopee", "tiktok", "tokopedia", "lazada", "marketplace", "offline", "online",
    "confirmed", "quotation", "cancel", "cancelled", "paid", "invoiced", "draft",
    "total", "ringkasan", "laporan", "summary", "performa", "rata", "aov", "di", "ke",
    "dari", "pada", "untuk", "dan", "serta", "berapa", "cek", "tampilkan", "mana",
    "dimana", "terbanyak", "tertinggi", "terbesar", "terlaris", "keseluruhan",
    "surabaya", "jakarta", "bandung", "medan", "makassar", "makasar", "bekasi",
    "tangerang", "malang", "mainstore", "main", "store", "headoffice", "head", "office",
    "ho", "ms", "sby", "jkt", "bdg", "mdn", "mks", "bks", "tgr", "mlg", "fastmart",
    "ytd", "mtd", "q1", "q2", "q3", "q4", "kuartal", "triwulan", "semester",
    "nya", "list", "daftar", "target", "bisa", "engga", "enggak", "gak", "tidak",
    "ya", "yg", "yang", "menurutmu", "owner", "minta", "m", "miliar", "milyar",
    "jt", "juta", "terhadap", "sekarang", "saat", "apakah", "impact", "dampak",
}

_EXPLICIT_PL_RE = re.compile(
    r"\b(laba\s*(?:dan\s*)?rugi(?:nya)?|rugi\s*laba(?:nya)?|p&l|profit\s*(?:and|&)\s*loss|income\s*statement|gross\s*margin|laba\s*kotor|laba\s*bersih|hpp|cogs)\b",
    re.IGNORECASE,
)


def _has_product_scoped_sales_intent(cleaned: str) -> bool:
    """Return True when a sales/omzet query targets a specific product or SKU rather than company-wide totals."""
    if not cleaned or _EXPLICIT_PL_RE.search(cleaned):
        return False
    if any(b in cleaned for b in ("faktur", "tagihan", "invoice", "bill", "piutang", "kuitansi", "bon", "sales order", "quotation", "so/")):
        return False
    if not re.search(r"\b(?:penjualan|omzet|omset|terjual)\b", cleaned):
        return False
    if re.search(r"\bsku\s*[:#]?\s*[a-z0-9._\-]+\b", cleaned):
        return True
    # Pattern: "<product> penjualan terbanyak ..."
    m_pre = re.search(r"^([a-z0-9\-_]+(?:\s+[a-z0-9\-_]+)?)\s+penjualan\s+(?:terbanyak|tertinggi|terbesar)\b", cleaned)
    if m_pre:
        words = [w for w in m_pre.group(1).split() if w not in _NON_PRODUCT_SALES_TOKENS]
        if words:
            return True
    # Pattern: "(omzet )?penjualan <product_phrase>"
    m_post = re.search(r"\b(?:(?:omzet|omset)\s+)?(?:penjualan|omzet|omset|terjual)\s+(?:produk\s+|barang\s+)?([a-z0-9\-\_\.\+]+(?:\s+[a-z0-9\-\_\.\+]+){0,4})", cleaned)
    if m_post:
        phrase_tokens = m_post.group(1).split()
        first_tok = phrase_tokens[0]
        if (
            first_tok not in _NON_PRODUCT_SALES_TOKENS
            and not re.match(r"^(?:20\d\d|q[1-4]|\d+(?:[.,]\d+)?(?:m|jt|rb|k)?)$", first_tok)
        ):
            return True
    return False


def classify_query_rule_based(query: str) -> Optional[Dict[str, Any]]:
    """
    Deterministic rule-based classifier for ERP queries.
    Detects single or multi-intent requests across inventory and accounting.
    Returns dictionary with dual representation (active_tools and primary scalar keys).
    """
    cleaned = query.lower().strip()
    has_customer = any(term in cleaned for term in CUSTOMER_TERMS)
    has_vendor = any(re.search(rf"\b{re.escape(term)}(nya)?\b", cleaned) for term in VENDOR_TERMS)
    has_purchase_invoice = any(kw in cleaned for kw in PURCHASE_INVOICE_TERMS)
    is_rag = _match_any(KNOWLEDGE_RAG_PATTERNS, cleaned)
    is_greeting_or_general = is_pure_greeting(cleaned) or is_general_help_inquiry(cleaned)

    # 0. Session Meta-Questions (period / company scope / latest ERP date)
    if is_meta_question(cleaned):
        return {
            "detected_domain": "general",
            "detected_tool": None,
            "query_mode": "general",
            "rag_query": None,
            "is_multi_tool": False,
            "detected_domains": ["general"],
            "detected_tools": [],
            "active_tools": [],
        }

    # 0.1 Fast-Path Out-of-Scope Guardrail (including direct marketplace write attempts)
    is_mp_diagnostic_q = bool(
        re.search(r"\b(kenapa|mengapa|apa\s+penyebab|log|riwayat|history|simulasi|kapan|gagal|error|status)\b", cleaned)
    )
    if _match_any(OUT_OF_SCOPE_PATTERNS, cleaned) or (
        _match_any(MARKETPLACE_WRITE_PATTERNS, cleaned) and not is_mp_diagnostic_q
    ):
        return {
            "detected_domain": "out_of_scope",
            "detected_tool": None,
            "query_mode": "out_of_scope",
            "rag_query": None,
            "is_multi_tool": False,
            "detected_domains": ["out_of_scope"],
            "detected_tools": [],
            "active_tools": [],
        }

    # 0.2 Multi-Intent Accumulation across Accounting & Inventory
    candidates: List[Tuple[str, str, int]] = []

    # Accounting: Profit & Loss (Laba Rugi)
    m_pl = _find_first_match(ACCOUNTING_PROFIT_LOSS_PATTERNS, cleaned)
    if m_pl:
        candidates.append(("get_profit_and_loss", "accounting", m_pl.start()))

    # Accounting: Balance Sheet (Neraca Keuangan)
    m_bs = _find_first_match(ACCOUNTING_BALANCE_SHEET_PATTERNS, cleaned)
    if m_bs:
        candidates.append(("get_balance_sheet", "accounting", m_bs.start()))

    # Accounting: Bank Reconciliation Summary
    m_recon = _find_first_match(ACCOUNTING_BANK_RECON_PATTERNS, cleaned)
    if m_recon:
        candidates.append(("get_bank_reconciliation_summary", "accounting", m_recon.start()))

    # Accounting: General Ledger (Buku Besar)
    m_gl = _find_first_match(ACCOUNTING_GENERAL_LEDGER_PATTERNS, cleaned)
    if m_gl:
        candidates.append(("get_general_ledger", "accounting", m_gl.start()))

    # Accounting: Vendor bills & AP Aging
    m_bills = _find_first_match(ACCOUNTING_VENDOR_BILLS_PATTERNS, cleaned)
    if m_bills:
        candidates.append(("get_vendor_bills", "accounting", m_bills.start()))

    # Accounting: Vendor invoices (legacy alias or general vendor invoice keywords)
    if (has_vendor and any(kw in cleaned for kw in ("tagihan", "faktur", "invoice", "bill", "hutang"))) or has_purchase_invoice:
        vendor_invoice_kws = (
            "tagihan", "faktur", "invoice", "bill", "hutang",
            "vendor", "supplier", "pemasok",
            "faktur pembelian", "tagihan pembelian", "invoice pembelian",
        )
        matches = [cleaned.find(kw) for kw in vendor_invoice_kws if kw in cleaned]
        if matches and not m_bills:
            candidates.append(("fetch_vendor_invoices", "accounting", min(matches)))
    elif has_vendor and not is_rag and not m_bills:
        matches = [cleaned.find(kw) for kw in ("vendor", "supplier", "pemasok", "hutang") if kw in cleaned]
        if matches:
            candidates.append(("fetch_vendor_invoices", "accounting", min(matches)))

    # Accounting: Specific document number reference (e.g. PC/2026/00002, BDG/JE/2026/00009, JE/2026/001)
    doc_match = re.search(r"\b((?:[A-Za-z0-9_\-]+/)+\d{4}/\d+)\b", cleaned)
    if doc_match and not any(c[0] in ("fetch_vendor_invoices", "get_vendor_bills", "get_journal_entries") for c in candidates):
        doc_code = doc_match.group(1).upper()
        if "JE" in doc_code:
            candidates.append(("get_journal_entries", "accounting", doc_match.start()))
        else:
            candidates.append(("fetch_vendor_invoices", "accounting", doc_match.start()))

    # Accounting: Journal entries
    journal_match = None
    if ("jurnal" in cleaned or "journal" in cleaned) and any(kw in cleaned for kw in ("entri", "transaksi", "daftar", "list", "riwayat", "terakhir", "hari", "bulan", "minggu")):
        m = _find_first_match([r"\bjurnal(nya)?\b", r"\bjournal\b"], cleaned)
        if m:
            journal_match = m.start()
            candidates.append(("get_journal_entries", "accounting", journal_match))
    else:
        m = _find_first_match(ACCOUNTING_JOURNAL_PATTERNS, cleaned)
        if m:
            journal_match = m.start()
            candidates.append(("get_journal_entries", "accounting", journal_match))

    # Accounting: Account balance
    m_bal = _find_first_match(ACCOUNTING_BALANCE_PATTERNS, cleaned)
    if m_bal:
        # Avoid duplicate balance tool when user specifically asked about journal entries with saldo
        is_journal_saldo_only = (
            journal_match is not None
            and re.search(r"\bsaldo\s+(di|dalam)?\s*jurnal\b", cleaned, re.IGNORECASE)
            and not re.search(r"\b(kas|bank|neraca|buku kas|saldo akun)\b", cleaned, re.IGNORECASE)
        )
        if not is_journal_saldo_only:
            candidates.append(("get_account_balance", "accounting", m_bal.start()))

    # Inventory: Stock history (Kartu Stok / Riwayat Mutasi)
    m_history = _find_first_match(INVENTORY_STOCK_HISTORY_PATTERNS, cleaned)
    if m_history:
        candidates.append(("get_stock_history", "inventory", m_history.start()))

    # Inventory: Transfers (Surat Jalan / Transfer Gudang / DO)
    m_transfers = _find_first_match(INVENTORY_TRANSFERS_PATTERNS, cleaned)
    if m_transfers:
        candidates.append(("get_transfers", "inventory", m_transfers.start()))

    # Inventory: Low stock alerts (Stok Menipis / Kritis / Habis)
    m_low_stock = _find_first_match(INVENTORY_LOW_STOCK_PATTERNS, cleaned)
    if m_low_stock:
        candidates.append(("get_low_stock_alerts", "inventory", m_low_stock.start()))

    # Inventory: Stock adjustments (Stock Opname / Selisih Stok)
    m_adjustments = _find_first_match(INVENTORY_ADJUSTMENTS_PATTERNS, cleaned)
    if m_adjustments:
        candidates.append(("get_stock_adjustments", "inventory", m_adjustments.start()))

    # Inventory: Scrap records (Barang Rusak / Cacat / Pemusnahan)
    m_scrap = _find_first_match(INVENTORY_SCRAP_PATTERNS, cleaned)
    if m_scrap:
        candidates.append(("get_scrap_records", "inventory", m_scrap.start()))

    # Inventory: Warehouses (Master Gudang)
    m_warehouses = _find_first_match(INVENTORY_WAREHOUSES_PATTERNS, cleaned)
    if m_warehouses:
        candidates.append(("get_warehouses", "inventory", m_warehouses.start()))

    # Inventory: Product categories (Master Kategori Produk)
    m_categories = _find_first_match(INVENTORY_CATEGORIES_PATTERNS, cleaned)
    if m_categories:
        candidates.append(("get_product_categories", "inventory", m_categories.start()))

    # Inventory: Velocity & Restock Forecast (Fast-Moving / Slow-Moving / Restock)
    m_velocity = _find_first_match(INVENTORY_VELOCITY_FORECAST_PATTERNS, cleaned)
    if m_velocity:
        candidates.append(("get_inventory_velocity_forecast", "inventory", m_velocity.start()))

    # Inventory: Stock levels (Physical on-hand inventory)
    m_stock = _find_first_match(INVENTORY_STOCK_PATTERNS, cleaned)
    if m_stock and not m_history and not m_low_stock and not m_adjustments and not m_scrap and not m_velocity:
        candidates.append(("get_stock_level", "inventory", m_stock.start()))

    # Inventory: Locations (Rak / Lokasi Penyimpanan)
    # When stock inquiry is matched, stock breakdown already reports locations/racks;
    # only add get_locations if explicit listing terms like 'daftar gudang' / 'gudang aktif' appear
    m_loc = _find_first_match(INVENTORY_LOCATION_PATTERNS, cleaned)
    if m_loc and not m_warehouses:
        if m_stock:
            explicit_location_listing = [
                r"\b(daftar|list)\s+(gudang|lokasi|rak)\b",
                r"\b(gudang|lokasi|rak)\s+(apa\s+saja|aktif|tersedia)\b",
                r"\bada\s+(gudang|lokasi|rak)\b",
            ]
            if _find_first_match(explicit_location_listing, cleaned):
                candidates.append(("get_locations", "inventory", m_loc.start()))
        else:
            candidates.append(("get_locations", "inventory", m_loc.start()))

    # Sales: Contacts (Contact details & channel/product buyer lookup, Top N customers)
    m_sales_contacts = _find_first_match(SALES_CONTACTS_PATTERNS, cleaned)
    if m_sales_contacts and not any(kw in cleaned for kw in CUSTOMER_BILLING_TERMS):
        candidates.append(("get_contacts", "sales", m_sales_contacts.start()))

    # Sales: Top selling products by revenue / qty (and product-scoped sales/omzet queries)
    m_top_sales = _find_first_match(SALES_TOP_PRODUCTS_PATTERNS, cleaned)
    if m_top_sales and m_sales_contacts and not re.search(r"\b(?:produk|barang|item|sku)\b", cleaned):
        m_top_sales = None
    is_product_sales = _has_product_scoped_sales_intent(cleaned) and not m_sales_contacts and not m_velocity
    if m_top_sales or is_product_sales:
        pos = m_top_sales.start() if m_top_sales else max(0, cleaned.find("penjualan"))
        candidates.append(("get_top_selling_products", "sales", pos))

    # Sales: Summary / Omzet (only when NOT scoped to a specific product or SKU)
    m_sales_summary = _find_first_match(SALES_SUMMARY_PATTERNS, cleaned)
    if m_sales_summary and not m_top_sales and not is_product_sales and not m_sales_contacts and not m_velocity:
        candidates.append(("get_sales_summary", "sales", m_sales_summary.start()))

    # Suppress 'omzet'-only Profit & Loss trigger when sales domain or marketplace+omzet owns the query
    if (m_sales_summary or m_top_sales or is_product_sales) and not _EXPLICIT_PL_RE.search(cleaned):
        candidates = [c for c in candidates if c[0] != "get_profit_and_loss"]

    # Sales: Order Detail
    m_sales_detail = _find_first_match(SALES_ORDER_DETAIL_PATTERNS, cleaned)
    if m_sales_detail and not m_top_sales and not is_product_sales and not m_sales_summary and not m_sales_contacts and not m_velocity:
        candidates.append(("get_sales_order_detail", "sales", m_sales_detail.start()))

    # Sales: Orders List & Quotations
    m_sales_orders = _find_first_match(SALES_ORDERS_PATTERNS, cleaned)
    if m_sales_orders and not m_top_sales and not is_product_sales and not m_sales_summary and not m_sales_detail and not m_sales_contacts and not m_velocity:
        candidates.append(("get_sales_orders", "sales", m_sales_orders.start()))

    # Marketplace Domain: 11 Specialized Operational & Financial Diagnostic Tools
    mp_tool_matches = [
        ("get_account_health", _find_first_match(MARKETPLACE_ACCOUNT_HEALTH_PATTERNS, cleaned)),
        ("get_kilat_order_diagnostics", _find_first_match(MARKETPLACE_KILAT_PATTERNS, cleaned)),
        ("get_marketplace_orders", _find_first_match(MARKETPLACE_ORDERS_PATTERNS, cleaned)),
        ("get_product_mapping_status", _find_first_match(MARKETPLACE_MAPPING_PATTERNS, cleaned)),
        ("get_stock_sync_simulation", _find_first_match(MARKETPLACE_STOCK_SYNC_SIM_PATTERNS, cleaned)),
        ("get_sync_execution_logs", _find_first_match(MARKETPLACE_SYNC_LOGS_PATTERNS, cleaned)),
        ("get_price_comparison", _find_first_match(MARKETPLACE_PRICE_COMPARISON_PATTERNS, cleaned)),
        ("get_fee_and_escrow_anomalies", _find_first_match(MARKETPLACE_FEE_ANOMALIES_PATTERNS, cleaned)),
        ("get_deduction_and_settlement_summary", _find_first_match(MARKETPLACE_DEDUCTION_SETTLEMENT_PATTERNS, cleaned)),
        ("get_cashflow_and_balance_summary", _find_first_match(MARKETPLACE_CASHFLOW_BALANCE_PATTERNS, cleaned)),
        ("get_competitor_insights", _find_first_match(MARKETPLACE_COMPETITOR_PATTERNS, cleaned)),
    ]
    matched_mp_tools = set()
    for mp_tname, mp_m in mp_tool_matches:
        if mp_m:
            matched_mp_tools.add(mp_tname)
            candidates.append((mp_tname, "marketplace", mp_m.start()))

    # Collision suppression when marketplace-specific signals are present
    if matched_mp_tools:
        has_explicit_coa_cue = bool(
            re.search(r"\b(coa|chart\s+of\s+accounts|kode\s+akun|buku\s+besar|rekening\s+bank|kas\s+kecil|1[0-9]{3,7})\b", cleaned)
        )
        suppressed_tools = set()
        if not has_explicit_coa_cue and matched_mp_tools & {
            "get_cashflow_and_balance_summary",
            "get_account_health",
            "get_deduction_and_settlement_summary",
            "get_fee_and_escrow_anomalies",
        }:
            suppressed_tools.add("get_account_balance")
        if "get_deduction_and_settlement_summary" in matched_mp_tools and not re.search(
            r"\b(entri\s+jurnal|jurnal\s+umum|debit\s+kredit)\b", cleaned
        ):
            suppressed_tools.add("get_journal_entries")
        if matched_mp_tools & {"get_stock_sync_simulation", "get_sync_execution_logs", "get_product_mapping_status"}:
            if not re.search(r"\b(cek\s+stok\s+(?:gudang|fisik|rak)|stok\s+(?:gudang|rak))\b", cleaned):
                suppressed_tools.add("get_stock_level")
        if matched_mp_tools & {"get_marketplace_orders", "get_kilat_order_diagnostics"}:
            suppressed_tools.update({"get_sales_orders", "get_sales_order_detail"})
        if "get_price_comparison" in matched_mp_tools:
            suppressed_tools.add("get_product_info")
        if "get_competitor_insights" in matched_mp_tools and not re.search(
            r"\b(?:bandingkan|perbandingan|pricelist|harga\s+erp|vs\s+erp)\b", cleaned
        ):
            suppressed_tools.add("get_price_comparison")
        if suppressed_tools:
            candidates = [c for c in candidates if c[0] not in suppressed_tools]

    # Inventory: Price Change History (ERP Pricelist + Marketplace + PCLS)
    m_price_hist = _find_first_match(INVENTORY_PRICE_HISTORY_PATTERNS, cleaned)
    if (
        m_price_hist
        and "get_competitor_insights" not in matched_mp_tools
        and "get_sync_execution_logs" not in matched_mp_tools
    ):
        candidates.append(("get_price_history", "inventory", m_price_hist.start()))
        if "get_price_comparison" in matched_mp_tools and not re.search(
            r"\b(?:bandingkan|perbandingan|selisih|compare|vs)\b", cleaned
        ):
            candidates = [c for c in candidates if c[0] != "get_price_comparison"]

    # Inventory: Product info
    if (
        not is_rag
        and not m_velocity
        and not m_price_hist
        and not m_top_sales
        and not is_product_sales
        and not m_sales_summary
        and not m_sales_detail
        and not m_sales_orders
        and not m_sales_contacts
        and "get_price_comparison" not in matched_mp_tools
        and (not m_low_stock or _find_first_match([r"\bharga\b", r"\bpricelist\b"], cleaned))
    ):
        # If stock level pattern matched, only add product_info if explicit product catalog detail terms are present
        # (avoid treating "stok produk ...", "sku ...", or "barcode ..." as a separate product_info tool call)
        if m_stock:
            explicit_catalog_patterns = [
                r"\bharga(nya)?\b",
                r"\bdetail barang\b",
                r"\bvarian\b",
                r"\bkategori\b",
                r"\bsatuan\b",
                r"\buom\b",
                r"\bpricelist\b",
            ]
            m_prod = _find_first_match(explicit_catalog_patterns, cleaned)
        else:
            m_prod = _find_first_match(INVENTORY_PRODUCT_PATTERNS, cleaned)

        if m_prod:
            candidates.append(("get_product_info", "inventory", m_prod.start()))

    # Sort candidates by chronological appearance in query
    candidates.sort(key=lambda x: x[2])

    # De-duplicate candidate tools preserving chronological order
    seen_tools = set()
    unique_candidates: List[Tuple[str, str, int]] = []
    for c in candidates:
        if c[0] not in seen_tools:
            seen_tools.add(c[0])
            unique_candidates.append(c)
    candidates = unique_candidates

    # 1. Negative Guardrail: Customer Invoice / Billing Inquiries
    has_customer_billing = has_customer and (
        any(term in cleaned for term in CUSTOMER_BILLING_TERMS)
        or (bool(candidates) and all(c[1] not in ("inventory", "sales", "marketplace") for c in candidates))
        or not candidates
    )
    if has_customer_billing and not has_vendor and not has_purchase_invoice and not is_rag:
        return {
            "detected_domain": "general",
            "detected_tool": None,
            "query_mode": "general",
            "rag_query": None,
            "is_multi_tool": False,
            "detected_domains": ["general"],
            "detected_tools": [],
            "active_tools": [],
        }

    # 1.1 Fast-Path Non-Vendor Billing Guardrail (Customer / Retail / Ambiguous Billing)
    billing_keywords = (
        "tagihan",
        "faktur",
        "invoice",
        "bill",
        "piutang",
        "kuitansi",
        "bon",
    )
    has_billing_kw = any(term in cleaned for term in billing_keywords)
    if has_billing_kw and not has_vendor and not has_purchase_invoice and not candidates and not is_rag:
        return {
            "detected_domain": "general",
            "detected_tool": None,
            "query_mode": "general",
            "rag_query": None,
            "is_multi_tool": False,
            "detected_domains": ["general"],
            "detected_tools": [],
            "active_tools": [],
        }

    # 2. Check for Hybrid (both RAG knowledge and Live tool intents)
    if is_rag and candidates:
        primary = candidates[0]
        sub_queries = _extract_sub_queries(query, candidates)
        active_tools = [
            {
                "tool": c[0],
                "domain": c[1],
                "sub_query": sub_queries[i],
                "params": {},
                "status": "pending",
            }
            for i, c in enumerate(candidates)
        ]
        return {
            "detected_domain": primary[1],
            "detected_tool": primary[0],
            "query_mode": "hybrid",
            "rag_query": query,
            "is_multi_tool": len(candidates) > 1,
            "detected_domains": list(dict.fromkeys(c[1] for c in candidates)),
            "detected_tools": [c[0] for c in candidates],
            "active_tools": active_tools,
        }

    # 3. Check for Pure Vector RAG (SOP, specs, policies)
    if is_rag and not candidates:
        return {
            "detected_domain": "knowledge",
            "detected_tool": None,
            "query_mode": "vector_rag",
            "rag_query": query,
            "is_multi_tool": False,
            "detected_domains": ["knowledge"],
            "detected_tools": [],
            "active_tools": [],
        }

    # 4. Pure Live ERP Tool Match (Single or Multi-Tool)
    if candidates:
        primary = candidates[0]
        sub_queries = _extract_sub_queries(query, candidates)
        active_tools = [
            {
                "tool": c[0],
                "domain": c[1],
                "sub_query": sub_queries[i],
                "params": {},
                "status": "pending",
            }
            for i, c in enumerate(candidates)
        ]
        return {
            "detected_domain": primary[1],
            "detected_tool": primary[0],
            "query_mode": "live_erp",
            "rag_query": None,
            "is_multi_tool": len(candidates) > 1,
            "detected_domains": list(dict.fromkeys(c[1] for c in candidates)),
            "detected_tools": [c[0] for c in candidates],
            "active_tools": active_tools,
        }

    # 5. Greeting / General Conversational Inquiry (only when no ERP/RAG intent matched)
    if is_greeting_or_general:
        return {
            "detected_domain": "general",
            "detected_tool": None,
            "query_mode": "general",
            "rag_query": None,
            "is_multi_tool": False,
            "detected_domains": ["general"],
            "detected_tools": [],
            "active_tools": [],
        }

    return None


_HQ_CROSS_BRANCH_TOOLS = {
    "get_sales_orders",
    "get_contacts",
    "get_top_selling_products",
    "get_sales_summary",
    "get_inventory_velocity_forecast",
    "get_price_history",
    "get_account_health",
    "get_marketplace_orders",
    "get_kilat_order_diagnostics",
    "get_product_mapping_status",
    "get_stock_sync_simulation",
    "get_sync_execution_logs",
    "get_price_comparison",
    "get_fee_and_escrow_anomalies",
    "get_deduction_and_settlement_summary",
    "get_cashflow_and_balance_summary",
    "get_competitor_insights",
}

_HQ_PIN_RE = re.compile(r"\b(?:head\s*office|headoffice|kantor\s*pusat|ho|hq)\b", re.IGNORECASE)


def _should_hq_auto_expand(
    active_cid: Optional[int],
    explicit_single: Optional[int],
    multi_cids: List[int],
    detected_tool: Optional[str],
    query_low: str,
) -> bool:
    """
    Determine whether an active HQ/Pusat session should automatically expand across branches:
    - HEAD OFFICE (id=3): auto-expand all operational cross-branch sales, customer, marketplace,
      and velocity tools when no single branch is named (unless user explicitly pins HQ/HO).
    - Kantor Pusat (id=1): expand only explicit cross-branch sales probes or anaphoric continuations
      to keep default unit-test single-company behavior intact.
    - Accounting and warehouse stock-by-warehouse tools stay strict single-company unless requested.
    """
    if explicit_single is not None or multi_cids or _HQ_PIN_RE.search(query_low):
        return False
    if active_cid == 3:
        return detected_tool in _HQ_CROSS_BRANCH_TOOLS
    if active_cid == 1 and detected_tool in (
        "get_sales_orders", "get_contacts", "get_top_selling_products", "get_sales_summary"
    ):
        return bool(
            re.search(
                r"\b(?:quotation\s+mana|penjualan\s+.*?\s+mana|apakah\s+ada\s+penjualan|ter\s*banyak\s+di\s+quotation)\b",
                query_low,
            )
            or has_anaphoric_reference(query_low)
        )
    return False


def _compute_company_scope(
    text: str,
    state: AgentState,
    detected_tool: Optional[str],
    fallback_to_active: bool = False,
) -> Optional[Tuple[int, List[int], bool]]:
    """Resolve (company_id, company_ids, is_multi_company) from text and prior state."""

    multi_cids, is_all = resolve_all_branch_company_ids(text)
    explicit_single = resolve_branch_company_id(text)
    text_low = (text or "").lower()
    active_cid = int(state.get("company_id") or settings.DEFAULT_COMPANY_ID)

    is_branch_ranking = bool(
        detected_tool in ("get_top_selling_products", "get_stock_level")
        and re.search(r"\b(?:di\s*mana|cabang\s+mana|toko\s+mana|outlet\s+mana|gudang\s+mana)\b", text_low)
    )
    is_hq_expand = _should_hq_auto_expand(
        active_cid=active_cid,
        explicit_single=explicit_single,
        multi_cids=multi_cids,
        detected_tool=detected_tool,
        query_low=text_low,
    )

    if is_all or (is_branch_ranking and explicit_single is None and not multi_cids) or is_hq_expand:
        auth_cids = [int(c) for c in (state.get("authorized_company_ids") or []) if c]
        eff_cids = auth_cids if (is_hq_expand and len(auth_cids) > 1) else list(COMPANY_NAME_MAP.keys())
        return eff_cids[0], eff_cids, True
    if len(multi_cids) > 1:
        return multi_cids[0], multi_cids, True
    if explicit_single is not None or len(multi_cids) == 1:
        cid = explicit_single if explicit_single is not None else multi_cids[0]
        return cid, [cid], False
    if state.get("is_multi_company") and state.get("company_ids"):
        eff_cids = [int(c) for c in state["company_ids"] if c]
        if len(eff_cids) > 1 or not fallback_to_active:
            return (eff_cids or [active_cid])[0], eff_cids, True
    recovered_mc = resolve_prior_multi_company_scope(state.get("messages"), text)
    if recovered_mc and (len(recovered_mc) > 1 or not fallback_to_active):
        return recovered_mc[0], recovered_mc, True
    if fallback_to_active:
        return active_cid, [active_cid], False
    return None


def _annotate_multi_company(res: Dict[str, Any], state: AgentState, user_query: str) -> Dict[str, Any]:
    if not isinstance(res, dict):
        return res
    if res.pop("_per_clause_company_scoped", False):
        return res
    scope = _compute_company_scope(
        user_query,
        state=state,
        detected_tool=res.get("detected_tool"),
        fallback_to_active=False,
    )
    if scope is not None:
        cid, cids, is_mc = scope
        res["is_multi_company"] = is_mc
        res["company_id"] = cid
        res["company_ids"] = cids
    return res


def _extract_stock_location_hint(text: str) -> Optional[str]:
    """Extract normalized location/warehouse hint from a stock clause for merge eligibility."""
    m = re.search(
        r"(?:\bdi\s+(?:gudang\s+|lokasi\s+)?|\b(?:gudang|lokasi)\s+)([A-Za-z0-9\_\-]+)",
        text or "",
        re.IGNORECASE,
    )
    return m.group(1).strip().lower() if m else None


def _resolve_clause_company_scope(
    clause: str,
    state: AgentState,
    detected_tool: Optional[str],
) -> Tuple[int, List[int], bool]:
    """Resolve company_id, company_ids, and is_multi_company strictly for one sub-question clause."""
    scope = _compute_company_scope(
        clause,
        state=state,
        detected_tool=detected_tool,
        fallback_to_active=True,
    )
    if scope is not None:
        return scope
    active_cid = int(state.get("company_id") or settings.DEFAULT_COMPANY_ID)
    return active_cid, [active_cid], False


def _enforce_general_vs_out_of_scope(
    query: str,
    detected_domain: Optional[str],
    query_mode: Optional[str],
    has_active_tools: bool = False,
) -> Tuple[str, str]:
    """
    Post-LLM deterministic guardrail: coerce LLM 'general' decisions to 'out_of_scope'
    unless the query is strictly a pure greeting, bot help/capability inquiry, session meta-question,
    or customer billing guardrail.
    """
    domain = detected_domain or "out_of_scope"
    mode = query_mode or ("out_of_scope" if domain == "out_of_scope" else "live_erp")
    if has_active_tools:
        return domain, mode
    if domain == "out_of_scope" or mode == "out_of_scope":
        return "out_of_scope", "out_of_scope"
    if domain == "general" or mode == "general":
        if is_allowed_general_query(query):
            return "general", "general"
        return "out_of_scope", "out_of_scope"
    return domain, mode


def _classify_non_tool_category(clause: str, domain: Optional[str]) -> str:
    """Categorize a non-tool clause for per-clause synthesis and guardrail notices."""
    cleaned = (clause or "").lower()
    if domain == "out_of_scope":
        if _match_any(MARKETPLACE_WRITE_PATTERNS, cleaned):
            return "marketplace_write"
        return "out_of_scope"
    has_cust = any(t in cleaned for t in CUSTOMER_TERMS)
    has_vend = any(re.search(rf"\b{re.escape(t)}(nya)?\b", cleaned) for t in VENDOR_TERMS)
    has_purch = any(kw in cleaned for kw in PURCHASE_INVOICE_TERMS)
    if (has_cust or any(t in cleaned for t in ("tagihan", "faktur", "invoice", "bill", "piutang", "kuitansi", "bon"))) and not has_vend and not has_purch:
        return "customer_billing"
    if is_pure_greeting(cleaned) or is_general_help_inquiry(cleaned) or is_meta_question(cleaned):
        return "general"
    return "out_of_scope"


def _invoke_llm_clause_router(
    clause: str,
    state: AgentState,
    model: BaseChatModel,
) -> Optional[Dict[str, Any]]:
    """Invoke structured MultiRouterDecision LLM fallback for a single unmatched clause."""
    try:
        system_prompt = (
            "You are a Supervisor Agent and Multi-Domain Tool Router for FastPrint ERP.\n"
            "Classify this single user question into target module domain ('accounting', 'inventory', 'sales', 'marketplace', 'knowledge', 'general', 'out_of_scope') "
            "and query_mode ('live_erp', 'vector_rag', 'hybrid', 'general', 'out_of_scope').\n"
            "Populate planned_tools when live ERP data is needed, or rag_query when SOP/specs/policy knowledge is needed.\n"
            "STRICT BOUNDARY:\n"
            "- Use 'general' ONLY for pure greetings (halo, selamat pagi), thanks (terima kasih), bot capability/help (/scope, bisa bantu apa), or session meta-questions.\n"
            "- Use 'out_of_scope' for ANY topic unrelated to FastPrint ERP (general knowledge, trivia, recipes, weather, coding, chitchat, unsupported modules like HR/payroll) or write requests."
        )
        raw_history = (state.get("messages") or [])[:-1]
        history_msgs = trim_conversation_messages(raw_history, max_messages=settings.MAX_WINDOW_MESSAGES)
        structured_model = model.with_structured_output(MultiRouterDecision)
        decision = structured_model.invoke([
            SystemMessage(content=system_prompt),
            *history_msgs,
            HumanMessage(content=clause),
        ])
        detected_domain = getattr(decision, "domain", "out_of_scope")
        query_mode = getattr(decision, "query_mode", "out_of_scope")
        detected_tool = getattr(decision, "tool", None)
        rag_query = getattr(decision, "rag_query", None)
        raw_planned = getattr(decision, "planned_tools", None) or []

        if detected_domain in ("general", "out_of_scope", "knowledge") or query_mode == "vector_rag":
            detected_tool = None
            raw_planned = []

        active_tools: List[Dict[str, Any]] = []
        for pt in raw_planned:
            pt_tool = pt.tool if hasattr(pt, "tool") else pt.get("tool")
            pt_domain = pt.domain if hasattr(pt, "domain") else pt.get("domain", TOOL_DOMAIN_MAP.get(pt_tool, detected_domain))
            pt_sub = (pt.sub_query if hasattr(pt, "sub_query") else pt.get("sub_query")) or clause
            if pt_tool and pt_domain in ("accounting", "inventory", "sales", "marketplace"):
                active_tools.append({
                    "tool": pt_tool,
                    "domain": pt_domain,
                    "sub_query": pt_sub,
                    "params": {},
                    "status": "pending",
                })
        if not active_tools and detected_tool and detected_domain in ("accounting", "inventory", "sales", "marketplace"):
            active_tools.append({
                "tool": detected_tool,
                "domain": detected_domain,
                "sub_query": clause,
                "params": {},
                "status": "pending",
            })

        detected_domain, query_mode = _enforce_general_vs_out_of_scope(
            clause,
            detected_domain,
            query_mode,
            has_active_tools=bool(active_tools),
        )

        return {
            "detected_domain": active_tools[0]["domain"] if active_tools else detected_domain,
            "detected_tool": active_tools[0]["tool"] if active_tools else None,
            "query_mode": query_mode,
            "rag_query": rag_query or (clause if query_mode in ("vector_rag", "hybrid") else None),
            "is_multi_tool": len(active_tools) > 1,
            "detected_domains": list(dict.fromkeys(t["domain"] for t in active_tools)) if active_tools else [detected_domain],
            "detected_tools": [t["tool"] for t in active_tools],
            "active_tools": active_tools,
        }
    except Exception:
        return None


def _build_dynamic_clause_result(dyn_cands: List[Dict[str, Any]], clause: str) -> Optional[Dict[str, Any]]:
    """Build normalized single-clause route result from semantic tool retrieval candidates."""
    if not dyn_cands:
        return None
    best = dyn_cands[0]
    return {
        "detected_domain": best["domain"],
        "detected_tool": best["tool_name"],
        "query_mode": "live_erp",
        "rag_query": None,
        "is_multi_tool": False,
        "detected_domains": [best["domain"]],
        "detected_tools": [best["tool_name"]],
        "active_tools": [{
            "tool": best["tool_name"],
            "domain": best["domain"],
            "sub_query": clause,
            "params": {},
            "status": "pending",
        }],
    }


def _classify_decomposed_clauses_core(
    clauses: List[str],
    state: AgentState,
    model: Optional[BaseChatModel],
    model_factory: Optional[Callable[[], Optional[BaseChatModel]]] = None,
) -> Tuple[List[str], str, List[Tuple[int, str, Optional[Dict[str, Any]]]]]:
    """Run rule and lazy LLM classification on each decomposed clause before dynamic fallback."""
    shared_prefix_clause = ""
    working_clauses = list(clauses)
    if working_clauses:
        m_prefix = re.match(
            r"^\s*((?:untuk|di)\s+(?:cabang\s+|toko\s+|kantor\s+)?[A-Za-z0-9\s]+?)\s*:\s*(.+)$",
            working_clauses[0],
            re.IGNORECASE,
        )
        if m_prefix:
            if resolve_branch_company_id(m_prefix.group(1)) is not None:
                shared_prefix_clause = m_prefix.group(1).strip()
                working_clauses = [m_prefix.group(2).strip()] + list(working_clauses[1:])

    classified_items: List[Tuple[int, str, Optional[Dict[str, Any]]]] = []
    for q_idx, raw_clause in enumerate(working_clauses):
        clause = raw_clause.strip()
        if not clause:
            continue
        c_res = classify_query_rule_based(clause)
        if c_res is None:
            active_model = model if model is not None else (model_factory() if model_factory else None)
            if active_model is not None:
                c_res = _invoke_llm_clause_router(clause, state, active_model)
        classified_items.append((q_idx, clause, c_res))

    return working_clauses, shared_prefix_clause, classified_items


def _assemble_decomposed_route(
    clauses: List[str],
    shared_prefix_clause: str,
    classified_items: List[Tuple[int, str, Optional[Dict[str, Any]]]],
    user_query: str,
    state: AgentState,
) -> Dict[str, Any]:
    """Assemble final multi-question state delta from classified clauses."""
    from app.services.tool_registry_service import get_tool_registry_service

    collected_plans: List[Dict[str, Any]] = []
    rag_queries: List[str] = []
    non_tool_questions: List[Dict[str, Any]] = []
    first_explicit_scope: Optional[Tuple[int, List[int], bool]] = None

    for q_idx, clause, c_res in classified_items:

        if c_res is None:
            c_low = clause.lower().strip()
            if (
                is_pure_greeting(c_low)
                or is_general_help_inquiry(c_low)
                or is_meta_question(c_low)
            ):
                non_tool_questions.append({
                    "question_index": q_idx,
                    "question_text": clause,
                    "category": "general",
                    "domain": "general",
                })
            else:
                non_tool_questions.append({
                    "question_index": q_idx,
                    "question_text": clause,
                    "category": "out_of_scope",
                    "domain": "out_of_scope",
                })
            continue

        c_mode = c_res.get("query_mode")
        c_domain = c_res.get("detected_domain")
        c_tools = c_res.get("active_tools") or []

        if c_mode in ("vector_rag", "hybrid") or c_domain == "knowledge":
            rq = (c_res.get("rag_query") or clause).strip()
            if rq and rq not in rag_queries:
                rag_queries.append(rq)

        if c_tools:
            # Merge stock-breakdown elaboration clauses (e.g. "Berikan jumlah unit per lokasi (MS) dan semua lokasi")
            # back into an already-planned get_stock_level instead of spawning a spurious get_locations call.
            prior_stock_plan = next(
                (p for p in collected_plans if p.get("tool") == "get_stock_level"),
                None,
            )
            if (
                prior_stock_plan is not None
                and len(c_tools) == 1
                and c_tools[0].get("tool") == "get_locations"
                and re.search(
                    r"\b(?:jumlah\s+unit|unit\s+per\s+lokasi|per\s+lokasi|semua\s+lokasi|setiap\s+(?:lokasi|gudang))\b",
                    clause,
                    re.IGNORECASE,
                )
                and not re.search(
                    r"\b(?:daftar|list)\s+(?:gudang|lokasi|rak)\b|\b(?:gudang|lokasi|rak)\s+apa\s+saja\b",
                    clause,
                    re.IGNORECASE,
                )
            ):
                prior_stock_plan["sub_query"] = f"{prior_stock_plan.get('sub_query') or ''} {clause}".strip()
                prior_stock_plan["question_text"] = f"{prior_stock_plan.get('question_text') or ''} {clause}".strip()
                elab_cids, elab_is_all = resolve_all_branch_company_ids(clause)
                if elab_is_all:
                    all_cids = list(COMPANY_NAME_MAP.keys())
                    prior_stock_plan["is_multi_company"] = True
                    prior_stock_plan["company_ids"] = all_cids
                    prior_stock_plan["company_id"] = all_cids[0]
                elif elab_cids and prior_stock_plan.get("is_multi_company"):
                    merged_cids = list(dict.fromkeys([*(prior_stock_plan.get("company_ids") or []), *elab_cids]))
                    prior_stock_plan["company_ids"] = merged_cids
                continue

            scope_text = f"{shared_prefix_clause} {clause}".strip() if shared_prefix_clause else clause
            clause_has_explicit_branch = bool(
                resolve_branch_company_id(scope_text) is not None
                or resolve_all_branch_company_ids(scope_text)[0]
                or resolve_all_branch_company_ids(scope_text)[1]
            )
            if (
                q_idx > 0
                and not clause_has_explicit_branch
                and first_explicit_scope is not None
                and has_anaphoric_reference(clause)
            ):
                c_cid, c_cids, c_is_mc = first_explicit_scope
            else:
                c_cid, c_cids, c_is_mc = _resolve_clause_company_scope(
                    scope_text,
                    state=state,
                    detected_tool=c_tools[0].get("tool"),
                )
            if first_explicit_scope is None and (clause_has_explicit_branch or c_is_mc):
                first_explicit_scope = (c_cid, c_cids, c_is_mc)
            for t_plan in c_tools:
                p_copy = dict(t_plan)
                p_copy["question_index"] = q_idx
                p_copy["question_text"] = clause
                p_copy["company_id"] = c_cid
                p_copy["company_ids"] = c_cids
                p_copy["is_multi_company"] = c_is_mc
                collected_plans.append(p_copy)
        elif c_mode not in ("vector_rag",):
            nt_cat = _classify_non_tool_category(clause, c_domain)
            nt_domain = "out_of_scope" if nt_cat in ("out_of_scope", "marketplace_write") else (c_domain or "general")
            non_tool_questions.append({
                "question_index": q_idx,
                "question_text": clause,
                "category": nt_cat,
                "domain": nt_domain,
            })

    # Smart merge: merge adjacent/same-scope non-anaphoric get_stock_level clauses into 1 batch call
    merged_plans: List[Dict[str, Any]] = []
    stock_bucket_idx: Dict[Tuple[Any, ...], int] = {}
    for plan in collected_plans:
        t_name = plan.get("tool")
        sub_q = plan.get("sub_query") or ""
        loc_hint = _extract_stock_location_hint(sub_q)
        scope_key = (
            plan.get("company_id"),
            tuple(plan.get("company_ids") or []),
            bool(plan.get("is_multi_company")),
            loc_hint,
        )
        if t_name == "get_stock_level" and not has_anaphoric_reference(sub_q):
            if scope_key in stock_bucket_idx:
                existing = merged_plans[stock_bucket_idx[scope_key]]
                existing["sub_query"] = f"{existing.get('sub_query', '')} dan {sub_q}".strip()
                existing["question_text"] = f"{existing.get('question_text', '')} dan {plan.get('question_text', '')}".strip()
                continue
            stock_bucket_idx[scope_key] = len(merged_plans)
        merged_plans.append(plan)

    registry = get_tool_registry_service()
    detected_tools = [p["tool"] for p in merged_plans]
    detected_domains = list(dict.fromkeys(p["domain"] for p in merged_plans))
    avail = [m for tn in dict.fromkeys(detected_tools) if (m := registry.get_tool_by_name(tn))]

    if merged_plans and rag_queries:
        query_mode = "hybrid"
    elif merged_plans:
        query_mode = "live_erp"
    elif rag_queries:
        query_mode = "vector_rag"
    elif non_tool_questions and all(n.get("domain") == "out_of_scope" for n in non_tool_questions):
        query_mode = "out_of_scope"
    else:
        query_mode = "general"

    primary_plan = merged_plans[0] if merged_plans else None
    if primary_plan:
        primary_domain = primary_plan["domain"]
        primary_tool = primary_plan["tool"]
        top_cid = primary_plan.get("company_id") or state.get("company_id") or settings.DEFAULT_COMPANY_ID
        top_cids = primary_plan.get("company_ids") or [top_cid]
        top_is_mc = bool(primary_plan.get("is_multi_company"))
    else:
        primary_domain = "knowledge" if rag_queries else (non_tool_questions[0]["domain"] if non_tool_questions else "general")
        primary_tool = None
        top_cid = state.get("company_id") or settings.DEFAULT_COMPANY_ID
        top_cids = state.get("company_ids") or [top_cid]
        top_is_mc = bool(state.get("is_multi_company"))

    effective_sub_questions = (
        [user_query]
        if (len(merged_plans) == 1 and not rag_queries and not non_tool_questions)
        else clauses
    )

    return {
        "detected_domain": primary_domain,
        "detected_tool": primary_tool,
        "query_mode": query_mode,
        "rag_query": rag_queries[0] if len(rag_queries) == 1 else (" | ".join(rag_queries) if rag_queries else None),
        "rag_queries": rag_queries,
        "is_multi_tool": len(merged_plans) > 1,
        "detected_domains": detected_domains or [primary_domain],
        "detected_tools": detected_tools,
        "active_tools": merged_plans,
        "available_tools": avail,
        "sub_questions": effective_sub_questions,
        "non_tool_questions": non_tool_questions,
        "company_id": top_cid,
        "company_ids": top_cids,
        "is_multi_company": top_is_mc,
        "_per_clause_company_scoped": True,
    }


def _route_decomposed_clauses(
    clauses: List[str],
    user_query: str,
    state: AgentState,
    model: Optional[BaseChatModel],
    model_factory: Optional[Callable[[], Optional[BaseChatModel]]] = None,
) -> Dict[str, Any]:
    """
    Route multiple decomposed sub-questions independently (sync), applying smart merge
    for identical same-branch/location stock queries and isolating per-clause company scopes.
    """
    from app.services.tool_registry_service import get_tool_registry_service

    working_clauses, shared_prefix_clause, classified_items = _classify_decomposed_clauses_core(
        clauses=clauses,
        state=state,
        model=model,
        model_factory=model_factory,
    )
    resolved_items: List[Tuple[int, str, Optional[Dict[str, Any]]]] = []
    for q_idx, clause, c_res in classified_items:
        if c_res is None and settings.DYNAMIC_TOOL_RETRIEVAL_ENABLED:
            try:
                registry = get_tool_registry_service()
                dyn_cands = registry.search_tools_sync(
                    query=clause,
                    user_roles=state.get("user_roles"),
                    top_k=settings.TOOL_RETRIEVAL_TOP_K,
                    distance_threshold=settings.TOOL_RETRIEVAL_DISTANCE_THRESHOLD,
                )
                c_res = _build_dynamic_clause_result(dyn_cands, clause)
            except Exception as exc:
                logger.warning("Dynamic retrieval error for clause '%s': %s", clause, exc)
        resolved_items.append((q_idx, clause, c_res))

    return _assemble_decomposed_route(
        clauses=working_clauses,
        shared_prefix_clause=shared_prefix_clause,
        classified_items=resolved_items,
        user_query=user_query,
        state=state,
    )


async def _aroute_decomposed_clauses(
    clauses: List[str],
    user_query: str,
    state: AgentState,
    model: Optional[BaseChatModel],
    model_factory: Optional[Callable[[], Optional[BaseChatModel]]] = None,
) -> Dict[str, Any]:
    """Async counterpart of _route_decomposed_clauses avoiding sync nested event loops."""
    from app.services.tool_registry_service import get_tool_registry_service

    working_clauses, shared_prefix_clause, classified_items = _classify_decomposed_clauses_core(
        clauses=clauses,
        state=state,
        model=model,
        model_factory=model_factory,
    )
    resolved_items: List[Tuple[int, str, Optional[Dict[str, Any]]]] = []
    for q_idx, clause, c_res in classified_items:
        if c_res is None and settings.DYNAMIC_TOOL_RETRIEVAL_ENABLED:
            try:
                registry = get_tool_registry_service()
                dyn_cands = await registry.search_tools(
                    query=clause,
                    user_roles=state.get("user_roles"),
                    top_k=settings.TOOL_RETRIEVAL_TOP_K,
                    distance_threshold=settings.TOOL_RETRIEVAL_DISTANCE_THRESHOLD,
                )
                c_res = _build_dynamic_clause_result(dyn_cands, clause)
            except Exception as exc:
                logger.warning("Async dynamic retrieval error for clause '%s': %s", clause, exc)
        resolved_items.append((q_idx, clause, c_res))

    return _assemble_decomposed_route(
        clauses=working_clauses,
        shared_prefix_clause=shared_prefix_clause,
        classified_items=resolved_items,
        user_query=user_query,
        state=state,
    )


def _collect_trailing_human_query(messages: Optional[List[BaseMessage]]) -> str:
    """Collect the trailing unanswered HumanMessage(s) when state['user_query'] is omitted."""
    if not messages:
        return ""
    trailing: List[str] = []
    for msg in reversed(messages):
        msg_type = getattr(msg, "type", "")
        if isinstance(msg, HumanMessage) or msg_type in ("human", "user"):
            txt = str(getattr(msg, "content", "") or "").strip()
            if txt:
                trailing.append(txt)
        else:
            break
    if trailing:
        trailing.reverse()
        return "\n".join(trailing)
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") in ("human", "user"):
            return str(getattr(msg, "content", "") or "").strip()
    return ""


def router_node(
    state: AgentState,
    llm: Optional[BaseChatModel] = None,
) -> Dict[str, Any]:
    """
    LangGraph pipeline node that determines the domain and target tools from user query.
    Returns AgentState update dictionary with active_tools and synchronized scalar fields.
    """
    user_query = state.get("user_query")
    if not user_query and state.get("messages"):
        user_query = _collect_trailing_human_query(state["messages"])

    user_query = (user_query or "").strip()
    raw_res = _router_node_impl(state=state, llm=llm, user_query=user_query)
    return _annotate_multi_company(raw_res, state=state, user_query=user_query)


async def arouter_node(
    state: AgentState,
    llm: Optional[BaseChatModel] = None,
) -> Dict[str, Any]:
    """
    Asynchronous counterpart of router_node that awaits dynamic semantic tool retrieval
    directly on the running event loop without spawning background threads.
    """
    user_query = state.get("user_query")
    if not user_query and state.get("messages"):
        user_query = _collect_trailing_human_query(state["messages"])

    user_query = (user_query or "").strip()
    raw_res = await _arouter_node_impl(state=state, llm=llm, user_query=user_query)
    return _annotate_multi_company(raw_res, state=state, user_query=user_query)


def _prepare_router_context(
    state: AgentState,
    llm: Optional[BaseChatModel],
    user_query: str,
) -> Tuple[Optional[Dict[str, Any]], bool, Optional[str], str, Callable[[], Optional[BaseChatModel]]]:
    """Evaluate empty/retry early exits and construct lazy LLM getter."""
    cached_model: Optional[BaseChatModel] = llm
    model_resolved = llm is not None
    is_test_env = "unittest" in sys.modules or os.environ.get("FASTPRINT_TEST_ENV") == "true"

    def _get_model() -> Optional[BaseChatModel]:
        nonlocal cached_model, model_resolved
        if model_resolved:
            return cached_model
        model_resolved = True
        if not is_test_env:
            try:
                cached_model = get_chat_model()
            except Exception as exc:
                logger.debug("Lazy router LLM initialization failed: %s", exc)
                cached_model = None
        return cached_model

    if not user_query:
        return (
            {
                "detected_domain": "general",
                "detected_tool": None,
                "query_mode": "general",
                "rag_query": None,
                "rag_queries": [],
                "sub_questions": [],
                "non_tool_questions": [],
                "is_multi_tool": False,
                "detected_domains": ["general"],
                "detected_tools": [],
                "active_tools": [],
            },
            False,
            None,
            user_query,
            _get_model,
        )

    is_retry = is_retry_query(user_query)
    prior_query = find_previous_user_query(state.get("messages")) if is_retry else None
    if is_retry:
        if prior_query:
            return None, True, prior_query, prior_query, _get_model
        return (
            {
                "detected_domain": "general",
                "detected_tool": None,
                "query_mode": "general",
                "rag_query": None,
                "rag_queries": [],
                "sub_questions": [],
                "non_tool_questions": [],
                "error_message": "NO_PREVIOUS_QUERY_TO_RETRY",
                "is_multi_tool": False,
                "detected_domains": ["general"],
                "detected_tools": [],
                "active_tools": [],
            },
            False,
            None,
            user_query,
            _get_model,
        )

    return None, False, None, user_query, _get_model


def _router_single_and_llm_steps(
    state: AgentState,
    user_query: str,
    is_retry: bool,
    prior_query: Optional[str],
    get_model_fn: Callable[[], Optional[BaseChatModel]],
) -> Optional[Dict[str, Any]]:
    """Execute single-clause rules, conversational follow-up resolution, and LLM fallback."""

    # Step 1: Fast Rule / Regex Pattern Matching (Single-Clause)
    try:
        rule_result = classify_query_rule_based(user_query)
        if rule_result is not None:
            from app.services.tool_registry_service import get_tool_registry_service
            registry = get_tool_registry_service()
            avail = []
            for t_name in rule_result.get("detected_tools", []):
                meta = registry.get_tool_by_name(t_name)
                if meta:
                    avail.append(meta)
            rule_result["available_tools"] = avail
            rule_result["sub_questions"] = [user_query]
            rule_result["rag_queries"] = [rule_result["rag_query"]] if rule_result.get("rag_query") else []
            rule_result["non_tool_questions"] = []
            if is_retry and prior_query:
                rule_result["user_query"] = prior_query
                rule_result["retry_instruction"] = f"Mengulang permintaan: {prior_query}"
            return rule_result
    except Exception as exc:
        logger.warning(f"Error in rule-based classifier for query '{user_query}': {exc}")

    # Step 1.2: P&L / Balance Sheet Account Drill-Down Follow-up (e.g. 'coba berikan detail dari Penjualan Surabaya')
    history_messages = state.get("messages") or []
    if len(history_messages) > 1:
        drill_match = resolve_pnl_drilldown_from_history(user_query, history_messages)
        if drill_match:
            from app.services.tool_registry_service import get_tool_registry_service
            registry = get_tool_registry_service()
            gl_meta = registry.get_tool_by_name("get_general_ledger")
            drill_res: Dict[str, Any] = {
                "detected_domain": "accounting",
                "detected_tool": "get_general_ledger",
                "query_mode": "live_erp",
                "rag_query": None,
                "rag_queries": [],
                "is_multi_tool": False,
                "detected_domains": ["accounting"],
                "detected_tools": ["get_general_ledger"],
                "active_tools": [{
                    "tool": "get_general_ledger",
                    "domain": "accounting",
                    "sub_query": user_query,
                    "params": {},
                    "status": "pending",
                }],
                "available_tools": [gl_meta] if gl_meta else [],
                "sub_questions": [user_query],
                "non_tool_questions": [],
            }
            if resolve_branch_company_id(user_query) is None and not resolve_all_branch_company_ids(user_query)[1]:
                prior_q_acc = _find_previous_erp_query(history_messages)
                if prior_q_acc:
                    prior_cid = resolve_branch_company_id(prior_q_acc)
                    if prior_cid is not None:
                        drill_res["company_id"] = prior_cid
                        drill_res["company_ids"] = [prior_cid]
                        drill_res["is_multi_company"] = False
            return drill_res

    # Step 1.5: Fast Conversational Follow-up / Slot Refinement Check
    # Matches branch/location mentions ('di mainstore', 'di cabang bandung'),
    # retry/check follow-ups ('coba cek lagi dimainstore'), or entity refinements
    # when an active ERP inquiry exists in prior messages.
    if len(history_messages) > 1:
        prior_q = _find_previous_erp_query(history_messages)
        if prior_q:

            clean_uq = user_query.strip().rstrip("?!.,;:'\"> ").lower()
            is_casual_oos = bool(_CASUAL_NON_ERP_INTERROGATIVE_RE.search(clean_uq))
            is_location_interrogative = clean_uq in {
                "dimana", "di mana", "cabang mana", "toko mana", "gudang mana", "outlet mana",
            }
            _NON_BRANCH_FOLLOWUP_TOKENS = {
                "wa", "whatsapp", "shopee", "tiktok", "tokopedia", "lazada", "goshop",
                "website", "offline", "online", "quotation", "penawaran", "order", "so",
                "contact", "kontak", "produk", "barang", "faktur", "invoice", "jurnal",
                "akunting", "accounting", "keuangan", "inventori", "inventory", "penjualan", "sales", "marketplace",
                "mana", "sini", "situ", "atas", "bawah", "dalam", "luar",
                "tabel", "table", "semua", "cabang", "perusahaan", "sana",
            }
            m_loc_hint = re.search(r"\b(?:di\s+|cabang\s+|gudang\s+|lokasi\s+|rak\s+)([A-Za-z0-9\_\-]+)", clean_uq)
            is_branch_mention = bool(resolve_branch_company_id(clean_uq)) or (
                bool(m_loc_hint) and m_loc_hint.group(1).lower() not in _NON_BRANCH_FOLLOWUP_TOKENS
            )
            _, is_all_scope_followup = resolve_all_branch_company_ids(clean_uq)
            is_followup_retry = is_retry_query(clean_uq) or bool(
                re.search(r"\b(coba\s+(?:(?:di)?cek\s+|(?:di)?periksa\s+|(?:di)?cari\s+)?lagi|(?:di)?cek\s+lagi|coba\s+lagi|ulang(i)?|tes\s+lagi|(?:di)?cek\s+ulang)\b", clean_uq)
            )
            has_sales_entity_shift = (
                _match_any(SALES_CONTACTS_PATTERNS, clean_uq)
                or _match_any(SALES_ORDERS_PATTERNS, clean_uq)
                or any(term in clean_uq for term in ("contact", "kontak", "quotation", "penawaran"))
            )
            is_non_erp_or_general_turn = (
                is_casual_oos
                or is_pure_greeting(clean_uq)
                or is_general_help_inquiry(clean_uq)
                or is_meta_question(clean_uq)
                or _match_any(OUT_OF_SCOPE_PATTERNS, clean_uq)
                or _match_any(MARKETPLACE_WRITE_PATTERNS, clean_uq)
            )
            is_short_followup = (
                len(clean_uq.split()) <= 7
                and not has_sales_entity_shift
                and not is_non_erp_or_general_turn
                and (is_branch_mention or is_all_scope_followup or is_followup_retry or is_location_interrogative)
            )

            if is_short_followup:
                try:
                    prior_result = classify_query_rule_based(prior_q)
                    if prior_result is not None and prior_result.get("detected_domain") in ("accounting", "inventory", "sales", "marketplace"):
                        if is_location_interrogative:
                            effective_sub = f"{prior_q} {clean_uq}".strip()
                            effective_user = effective_sub
                        elif is_branch_mention or is_all_scope_followup:
                            effective_sub = user_query
                            sanitized_prior = strip_company_scope_keywords(prior_q)
                            effective_user = f"{sanitized_prior} {user_query}".strip() if sanitized_prior else user_query
                        else:
                            effective_sub = prior_q
                            effective_user = prior_q
                        followup_tools = [
                            {
                                "tool": t["tool"],
                                "domain": t["domain"],
                                "sub_query": effective_sub,
                                "params": {},
                                "status": "pending",
                            }
                            for t in prior_result.get("active_tools", [])
                        ]
                        return {
                            "user_query": effective_user,
                            "detected_domain": prior_result.get("detected_domain"),
                            "detected_tool": prior_result.get("detected_tool"),
                            "query_mode": prior_result.get("query_mode", "live_erp"),
                            "rag_query": None,
                            "is_multi_tool": prior_result.get("is_multi_tool", False),
                            "detected_domains": prior_result.get("detected_domains", []),
                            "detected_tools": prior_result.get("detected_tools", []),
                            "active_tools": followup_tools,
                        }
                except Exception as exc:
                    logger.warning(f"Error resolving conversational follow-up: {exc}")

            # Heuristic Short Follow-Up: Require an explicit follow-up cue (branch/scope, temporal filter,
            # anaphoric/elliptical continuation, or short bare non-interrogative entity refinement) so
            # short Turn-2 off-topic questions are never hijacked by the prior ERP tool.
            has_new_domain_intent = (
                any(term in clean_uq for term in CUSTOMER_TERMS)
                or any(term in clean_uq for term in VENDOR_TERMS)
                or has_sales_entity_shift
                or is_non_erp_or_general_turn
                or _match_any(ACCOUNTING_BALANCE_PATTERNS, clean_uq)
                or _match_any(ACCOUNTING_JOURNAL_PATTERNS, clean_uq)
                or _match_any(KNOWLEDGE_RAG_PATTERNS, clean_uq)
            )
            uq_words = clean_uq.split()
            has_temporal_followup = resolve_relative_dates(clean_uq) != (None, None)
            has_anaphoric_or_elliptical_cue = (
                has_anaphoric_reference(clean_uq)
                or bool(_FOLLOWUP_ELLIPTICAL_PREFIX_RE.search(clean_uq))
            )
            is_bare_entity_refinement = (
                1 <= len(uq_words) <= 4
                and not user_query.strip().endswith("?")
                and not _STANDALONE_NON_FOLLOWUP_INTERROGATIVE_RE.search(clean_uq)
            )
            has_followup_cue = (
                is_branch_mention
                or is_all_scope_followup
                or has_temporal_followup
                or has_anaphoric_or_elliptical_cue
                or is_bare_entity_refinement
            )
            if len(uq_words) <= 6 and has_followup_cue and not has_new_domain_intent:
                try:
                    prior_result = classify_query_rule_based(prior_q)
                    if prior_result is not None and prior_result.get("detected_domain") in ("accounting", "inventory", "sales", "marketplace"):
                        if is_branch_mention or is_all_scope_followup:
                            sanitized_prior = strip_company_scope_keywords(prior_q)
                            effective_sub = f"{sanitized_prior} {user_query}".strip() if sanitized_prior else user_query
                        else:
                            effective_sub = f"{prior_q} {user_query}"
                        followup_tools = [
                            {
                                "tool": t["tool"],
                                "domain": t["domain"],
                                "sub_query": effective_sub,
                                "params": {},
                                "status": "pending",
                            }
                            for t in prior_result.get("active_tools", [])
                        ]
                        return {
                            "user_query": effective_sub,
                            "detected_domain": prior_result.get("detected_domain"),
                            "detected_tool": prior_result.get("detected_tool"),
                            "query_mode": prior_result.get("query_mode", "live_erp"),
                            "rag_query": None,
                            "is_multi_tool": prior_result.get("is_multi_tool", False),
                            "detected_domains": prior_result.get("detected_domains", []),
                            "detected_tools": prior_result.get("detected_tools", []),
                            "active_tools": followup_tools,
                        }
                except Exception as exc:
                    logger.warning(f"Error in heuristic short follow-up routing: {exc}")

    # Step 2: Fallback to LLM Structured Output (Context-Aware)
    model = get_model_fn()

    if model is not None:
        try:
            system_prompt = (
                "You are a Supervisor Agent and Multi-Domain Tool Router for FastPrint ERP.\n"
                "Classify user inquiries into target module domains ('accounting', 'inventory', 'sales', 'marketplace', 'knowledge', 'general', 'out_of_scope') "
                "and determine the execution query_mode ('live_erp', 'vector_rag', 'hybrid', 'general', 'out_of_scope').\n\n"
                "MULTI-INTENT & MULTI-TOOL PLANNING:\n"
                "- If the user inquiry asks for multiple pieces of ERP data spanning one or more domains "
                "(e.g., checking stock levels in inventory AND checking account balances or vendor invoices in accounting), "
                "plan each required tool call in 'planned_tools' with its specific tool name, domain, and focused sub_query.\n"
                "- Populate 'tool' and 'domain' with the primary (first) tool call for backward compatibility.\n"
                "- If only a single tool is required, include it in both 'tool' and 'planned_tools'.\n\n"
                "ROUTING RULES:\n"
                "1. 'vector_rag' / domain='knowledge': Qualitative questions about SOPs, return policies, product technical specifications (GSM, size), warranty, or contract terms. tool=null, planned_tools=[].\n"
                "2. 'live_erp': Numerical / real-time operational inquiries about stock on hand, cash balances, journal ledger, vendor bills/invoices, sales orders/revenue, or marketplace operations/diagnostics.\n"
                "3. 'hybrid': Combined queries asking about both SOP/specs rules AND live ERP numbers/stock. Plan live ERP tool(s) in planned_tools and set rag_query.\n"
                "4. 'contextual_analysis': Analytical or 'why' follow-up questions about detailed ERP report data already shown in conversation history when no new branch/period/tool is requested. Set domain to the prior ERP domain ('accounting', 'inventory', 'sales', 'marketplace'), tool=null, planned_tools=[].\n"
                "5. 'general': STRICTLY ONLY pure greetings (halo, selamat pagi), thanks (terima kasih), asking about this bot's ERP capabilities/help (/scope, bisa bantu apa), or session meta-questions. NEVER use 'general' for general knowledge, trivia, recipes, weather, coding, casual chitchat, or unsupported modules. tool=null, planned_tools=[].\n"
                "6. 'out_of_scope': ANY inquiry unrelated to FastPrint ERP data (general knowledge, trivia, recipes, weather, coding, entertainment, politics, personal advice), unsupported ERP modules (e.g., HR/payroll/leave), or write/mutation requests (e.g., pushing stock/price to Shopee/TikTok or merging Kilat orders). tool=null, planned_tools=[].\n\n"
                "CONTEXT AWARENESS:\n"
                "Evaluate user query in context of conversational history.\n"
                "If user asks to retry or follow up, infer intended domain and target ERP tools from history.\n\n"
                "ACCOUNTING TOOLS GUIDE:\n"
                "- 'get_profit_and_loss': Profit & loss, income statement, laba rugi, omzet vs HPP, gross margin, net profit.\n"
                "- 'get_balance_sheet': Balance sheet, neraca keuangan, total assets, liabilities, equity, balance equation.\n"
                "- 'get_general_ledger': General ledger, buku besar akun, mutasi kronologis, running balance per akun.\n"
                "- 'get_bank_reconciliation_summary': Bank reconciliation, selisih rekening koran vs buku bank ERP.\n"
                "- 'get_vendor_bills': Vendor bills, faktur pembelian, hutang vendor, tagihan jatuh tempo, AP aging.\n"
                "- 'get_account_balance': Chart of accounts balances, saldo kas/bank/akun periode tertentu.\n"
                "- 'get_journal_entries': General journal vouchers/entries, memorial entries.\n\n"
                "SALES TOOLS GUIDE:\n"
                "- 'get_top_selling_products': Top selling products by volume or revenue.\n"
                "- 'get_sales_orders': Sales order & quotation list (filterable by channel, product_search, status, sort_by).\n"
                "- 'get_sales_order_detail': Single SO / quotation detail, customer contact, lines, and AWB/courier.\n"
                "- 'get_sales_summary': Executive revenue/omzet summary and channel breakdown.\n"
                "- 'get_contacts': Customer/buyer contact details (phone, email, address, city, channel, product buyers).\n\n"
                "INVENTORY PRICE HISTORY:\n"
                "- 'get_price_history': Historical product price changes across ERP Pricelist, Marketplace PriceUpdateLog, and PCLS plus current pricelist snapshot.\n\n"
                "MARKETPLACE TOOLS GUIDE (All READ-ONLY):\n"
                "- 'get_account_health': Store readiness, OAuth token expiry, shop moderation status, missing configs, fetch errors.\n"
                "- 'get_marketplace_orders': Shipping SLA (overdue/due_soon), missing AWB, cancellations, ERP vs marketplace status mismatch.\n"
                "- 'get_kilat_order_diagnostics': Shopee Pesanan Kilat stuck reservations (invoice_updated=false) & duplicate collisions.\n"
                "- 'get_product_mapping_status': Unmapped SKUs, bundle qty_multiplier, duplicate mappings, ExceptionProduct.\n"
                "- 'get_stock_sync_simulation': Dry-run determine_stock() simulation, shared_stock_companies, vs last StockUpdateLog.\n"
                "- 'get_sync_execution_logs': StockUpdateLog, PriceUpdateLog, and PCLS price sync failures/skips.\n"
                "- 'get_price_comparison': ERP Pricelist/Wholesale vs marketplace listing prices.\n"
                "- 'get_fee_and_escrow_anomalies': Negative fee, high fee %, zero expected income, ERP vs MP total mismatch.\n"
                "- 'get_deduction_and_settlement_summary': Fee deduction breakdown by category, gross vs nett, unposted settlement journals.\n"
                "- 'get_cashflow_and_balance_summary': Marketplace seller wallet (available vs pending escrow), snapshots, settlements/withdrawals.\n"
                "- 'get_competitor_insights': Competitor price drops, sold velocity, and stale audits.\n\n"
                "CRITICAL GUARDRAILS:\n"
                "- References to specific ERP transaction document numbers (e.g., 'PC/...', 'JE/...', 'INV/...') must be routed to domain='accounting' (fetch_vendor_invoices or get_journal_entries).\n"
                "- Customer / pelanggan inquiries targeting invoices, billing, or accounts receivable must NOT be routed to 'fetch_vendor_invoices'. Set domain='general', tool=null, planned_tools=[].\n"
                "- Customer / pelanggan inquiries asking about inventory stock, warehouse locations, or product details must be routed to domain='inventory'.\n"
                "- Requests to push/update marketplace stock/price or merge/process Kilat orders are write operations: set domain='out_of_scope', query_mode='out_of_scope', tool=null, planned_tools=[].\n"
                "- Whenever domain is 'general' or 'out_of_scope', tool MUST be null and planned_tools MUST be empty."
            )
            raw_history = (state.get("messages") or [])[:-1]
            history_msgs = trim_conversation_messages(
                raw_history,
                max_messages=settings.MAX_WINDOW_MESSAGES,
            )
            structured_model = model.with_structured_output(MultiRouterDecision)
            messages = [
                SystemMessage(content=system_prompt),
                *history_msgs,
                HumanMessage(content=user_query),
            ]
            decision = structured_model.invoke(messages)

            detected_domain = getattr(decision, "domain", "out_of_scope")
            query_mode = getattr(decision, "query_mode", "out_of_scope")
            detected_tool = getattr(decision, "tool", None)
            rag_query = getattr(decision, "rag_query", None)
            raw_planned_tools = getattr(decision, "planned_tools", None) or []

            active_tools: List[Dict[str, Any]] = []

            if detected_domain in ("general", "out_of_scope", "knowledge") or query_mode in ("vector_rag", "contextual_analysis"):
                detected_tool = None
                raw_planned_tools = []

            if raw_planned_tools:
                for pt in raw_planned_tools:
                    pt_tool = pt.tool if hasattr(pt, "tool") else pt.get("tool")
                    pt_domain = pt.domain if hasattr(pt, "domain") else pt.get("domain", TOOL_DOMAIN_MAP.get(pt_tool, detected_domain))
                    pt_sub_query = (pt.sub_query if hasattr(pt, "sub_query") else pt.get("sub_query")) or user_query
                    if pt_tool and pt_domain in ("accounting", "inventory", "sales", "marketplace"):
                        active_tools.append({
                            "tool": pt_tool,
                            "domain": pt_domain,
                            "sub_query": pt_sub_query,
                            "params": {},
                            "status": "pending",
                        })

            if not active_tools and detected_tool and detected_domain in ("accounting", "inventory", "sales", "marketplace"):
                active_tools.append({
                    "tool": detected_tool,
                    "domain": detected_domain,
                    "sub_query": user_query,
                    "params": {},
                    "status": "pending",
                })

            detected_domain, query_mode = _enforce_general_vs_out_of_scope(
                user_query,
                detected_domain,
                query_mode,
                has_active_tools=bool(active_tools),
            )

            if active_tools:
                primary = active_tools[0]
                detected_tool = primary["tool"]
                detected_domain = primary["domain"]
                detected_tools = [t["tool"] for t in active_tools]
                detected_domains = list(dict.fromkeys(t["domain"] for t in active_tools))
                is_multi_tool = len(active_tools) > 1
            else:
                detected_tools = []
                detected_domains = [detected_domain] if detected_domain else []
                is_multi_tool = False

            result: Dict[str, Any] = {
                "detected_domain": detected_domain,
                "detected_tool": detected_tool,
                "query_mode": query_mode,
                "rag_query": rag_query or (user_query if query_mode in ("vector_rag", "hybrid") else None),
                "is_multi_tool": is_multi_tool,
                "detected_domains": detected_domains,
                "detected_tools": detected_tools,
                "active_tools": active_tools,
            }
            from app.services.tool_registry_service import get_tool_registry_service
            registry = get_tool_registry_service()
            avail = []
            for t_name in detected_tools:
                meta = registry.get_tool_by_name(t_name)
                if meta:
                    avail.append(meta)
            result["available_tools"] = avail

            if is_retry and prior_query:
                result["user_query"] = prior_query
                result["retry_instruction"] = f"Mengulang permintaan: {prior_query}"
            return result
        except Exception as exc:
            logger.debug("LLM router fallback failed for '%s': %s", user_query, exc)

    return None


def _build_dynamic_single_result(
    candidates: List[Dict[str, Any]],
    user_query: str,
    is_retry: bool,
    prior_query: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Build single-tool dynamic retrieval route dictionary when candidates are present."""
    if not candidates:
        return None
    best_match = candidates[0]
    best_tool = best_match["tool_name"]
    best_domain = best_match["domain"]
    active_tools = [
        {
            "tool": best_tool,
            "domain": best_domain,
            "sub_query": user_query,
            "params": {},
            "status": "pending",
        }
    ]
    result_dynamic: Dict[str, Any] = {
        "detected_domain": best_domain,
        "detected_tool": best_tool,
        "query_mode": "live_erp",
        "rag_query": None,
        "is_multi_tool": False,
        "detected_domains": [best_domain],
        "detected_tools": [best_tool],
        "active_tools": active_tools,
        "available_tools": candidates,
    }
    if is_retry and prior_query:
        result_dynamic["user_query"] = prior_query
        result_dynamic["retry_instruction"] = f"Mengulang permintaan: {prior_query}"
    return result_dynamic


def _general_fallback_result() -> Dict[str, Any]:
    return {
        "detected_domain": "out_of_scope",
        "detected_tool": None,
        "query_mode": "out_of_scope",
        "rag_query": None,
        "is_multi_tool": False,
        "detected_domains": ["out_of_scope"],
        "detected_tools": [],
        "active_tools": [],
        "available_tools": [],
    }


_LIVE_DIAGNOSTIC_OR_DRILLDOWN_TOOLS = {
    "get_sync_execution_logs",
    "get_kilat_order_diagnostics",
    "get_fee_and_escrow_anomalies",
    "get_inventory_velocity_forecast",
    "get_general_ledger",
    "get_stock_sync_simulation",
    "get_price_comparison",
    "get_price_history",
}


def _infer_prior_erp_domain_and_tool(
    history_messages: List[BaseMessage],
    state: AgentState,
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve prior ERP domain and tool from checkpoint snapshot, prior HumanMessage, or prior AIMessage heading."""
    prior_domain = state.get("prior_detected_domain")
    prior_tool = state.get("prior_detected_tool")
    if prior_domain in ("accounting", "inventory", "sales", "marketplace") and prior_tool:
        return prior_domain, prior_tool

    prior_q = _find_previous_erp_query(history_messages)
    if prior_q:
        try:
            prior_probe = classify_query_rule_based(prior_q)
            if isinstance(prior_probe, dict) and prior_probe.get("detected_domain") in (
                "accounting", "inventory", "sales", "marketplace"
            ):
                return prior_probe.get("detected_domain"), prior_probe.get("detected_tool")
        except Exception:
            pass

    for msg in reversed(history_messages[:-1]):
        if getattr(msg, "type", "") != "ai" and msg.__class__.__name__ != "AIMessage":
            continue
        txt = str(getattr(msg, "content", "") or "").lower()
        if "laba rugi" in txt or "profit & loss" in txt:
            return "accounting", "get_profit_and_loss"
        if "neraca keuangan" in txt or "balance sheet" in txt:
            return "accounting", "get_balance_sheet"
        if "buku besar" in txt or "general ledger" in txt:
            return "accounting", "get_general_ledger"
        if "rekonsiliasi bank" in txt:
            return "accounting", "get_bank_reconciliation_summary"
        if "tagihan faktur vendor" in txt or "vendor bills" in txt:
            return "accounting", "get_vendor_bills"
        if "ringkasan eksekutif penjualan" in txt or "sales summary" in txt:
            return "sales", "get_sales_summary"
        if "perputaran stok" in txt or "velocity" in txt:
            return "inventory", "get_inventory_velocity_forecast"
        if "marketplace" in txt or "potongan fee" in txt or "saldo dompet" in txt:
            return "marketplace", prior_tool
        if "stok" in txt:
            return "inventory", prior_tool
        break
    return prior_domain, prior_tool


def _try_route_contextual_analysis(
    user_query: str,
    state: AgentState,
) -> Optional[Dict[str, Any]]:
    """
    Route pure analytical/thinking follow-ups directly to synthesizer when detailed
    ERP data is already present in conversation history and no scope/period/tool changed.
    """
    history_messages = state.get("messages") or []
    if len(history_messages) <= 1:
        return None
    if not is_analytical_thinking_query(user_query):
        return None

    clean_uq = user_query.strip().rstrip("?!.,;:'\"> ").lower()
    if is_retry_query(clean_uq) or is_meta_question(clean_uq):
        return None
    if _CASUAL_NON_ERP_INTERROGATIVE_RE.search(clean_uq):
        return None
    if (
        _match_any(OUT_OF_SCOPE_PATTERNS, clean_uq)
        or is_pure_greeting(clean_uq)
        or is_general_help_inquiry(clean_uq)
        or _match_any(MARKETPLACE_WRITE_PATTERNS, clean_uq)
    ):
        return None
    if resolve_pnl_drilldown_from_history(user_query, history_messages):
        return None

    # Any explicit branch or company scope change requires fresh tool execution
    if resolve_branch_company_id(user_query) is not None:
        return None
    multi_cids, is_all = resolve_all_branch_company_ids(user_query)
    if multi_cids or is_all:
        return None

    # Any new date/period different from prior displayed period requires fresh tool execution
    df, dt = resolve_relative_dates(user_query)
    if df is not None or dt is not None:
        prior_df, prior_dt, prior_asof = extract_period_from_messages(history_messages)
        if not ((df, dt) == (prior_df, prior_dt) and (prior_df or prior_dt)) and not (
            dt and dt == prior_asof
        ):
            return None

    prior_domain, prior_tool = _infer_prior_erp_domain_and_tool(history_messages, state)
    if prior_domain not in ("accounting", "inventory", "sales", "marketplace"):
        return None

    # Defer if the query targets guardrails, RAG knowledge, or a different/live diagnostic tool
    probe = classify_query_rule_based(user_query)
    if probe is not None:
        p_domain = probe.get("detected_domain")
        p_mode = probe.get("query_mode")
        p_tools = probe.get("detected_tools") or []
        if p_domain in ("general", "out_of_scope", "knowledge") or p_mode in ("vector_rag", "hybrid"):
            return None
        if any(t in _LIVE_DIAGNOSTIC_OR_DRILLDOWN_TOOLS for t in p_tools) and prior_tool not in p_tools:
            return None
        if prior_tool and any(t != prior_tool for t in p_tools):
            return None
        if not prior_tool and p_tools:
            return None

    if not has_sufficient_prior_erp_context(
        history_messages,
        prior_tool_result=state.get("prior_tool_result"),
        prior_tool_name=prior_tool,
    ):
        return None

    return {
        "detected_domain": prior_domain,
        "detected_tool": None,
        "query_mode": "contextual_analysis",
        "verification_status": "VALID",
        "rag_query": None,
        "rag_queries": [],
        "is_multi_tool": False,
        "detected_domains": [prior_domain],
        "detected_tools": [],
        "active_tools": [],
        "available_tools": [],
        "sub_questions": [user_query],
        "non_tool_questions": [],
    }


def _router_node_impl(
    state: AgentState,
    llm: Optional[BaseChatModel] = None,
    user_query: str = "",
) -> Dict[str, Any]:
    early_res, is_retry, prior_query, eff_query, get_model_fn = _prepare_router_context(
        state=state,
        llm=llm,
        user_query=user_query,
    )
    if early_res is not None:
        return early_res

    if not is_retry:
        ctx_res = _try_route_contextual_analysis(eff_query, state)
        if ctx_res is not None:
            return ctx_res

    clauses = decompose_user_query(eff_query)
    if len(clauses) > 1:
        try:
            decomposed_res = _route_decomposed_clauses(
                clauses,
                user_query=eff_query,
                state=state,
                model=None,
                model_factory=get_model_fn,
            )
            if is_retry and prior_query:
                decomposed_res["user_query"] = prior_query
                decomposed_res["retry_instruction"] = f"Mengulang permintaan: {prior_query}"
            return decomposed_res
        except Exception as exc:
            logger.warning("Multi-question decomposition failed for '%s': %s", eff_query, exc)

    step_res = _router_single_and_llm_steps(
        state=state,
        user_query=eff_query,
        is_retry=is_retry,
        prior_query=prior_query,
        get_model_fn=get_model_fn,
    )
    if step_res is not None:
        return step_res

    # Step 2.5: Dynamic Semantic Tool Retrieval via PGVector / In-Memory Catalog
    if settings.DYNAMIC_TOOL_RETRIEVAL_ENABLED:
        try:
            from app.services.tool_registry_service import get_tool_registry_service
            registry = get_tool_registry_service()
            candidates = registry.search_tools_sync(
                query=eff_query,
                user_roles=state.get("user_roles"),
                top_k=settings.TOOL_RETRIEVAL_TOP_K,
                distance_threshold=settings.TOOL_RETRIEVAL_DISTANCE_THRESHOLD,
            )
            dyn_res = _build_dynamic_single_result(candidates, eff_query, is_retry, prior_query)
            if dyn_res is not None:
                return dyn_res
        except Exception as exc:
            logger.warning(f"Error in dynamic semantic tool retrieval during router execution: {exc}")

    return _general_fallback_result()


async def _arouter_node_impl(
    state: AgentState,
    llm: Optional[BaseChatModel] = None,
    user_query: str = "",
) -> Dict[str, Any]:
    early_res, is_retry, prior_query, eff_query, get_model_fn = _prepare_router_context(
        state=state,
        llm=llm,
        user_query=user_query,
    )
    if early_res is not None:
        return early_res

    if not is_retry:
        ctx_res = _try_route_contextual_analysis(eff_query, state)
        if ctx_res is not None:
            return ctx_res

    clauses = decompose_user_query(eff_query)
    if len(clauses) > 1:
        try:
            decomposed_res = await _aroute_decomposed_clauses(
                clauses,
                user_query=eff_query,
                state=state,
                model=None,
                model_factory=get_model_fn,
            )
            if is_retry and prior_query:
                decomposed_res["user_query"] = prior_query
                decomposed_res["retry_instruction"] = f"Mengulang permintaan: {prior_query}"
            return decomposed_res
        except Exception as exc:
            logger.warning("Async multi-question decomposition failed for '%s': %s", eff_query, exc)

    step_res = _router_single_and_llm_steps(
        state=state,
        user_query=eff_query,
        is_retry=is_retry,
        prior_query=prior_query,
        get_model_fn=get_model_fn,
    )
    if step_res is not None:
        return step_res

    if settings.DYNAMIC_TOOL_RETRIEVAL_ENABLED:
        try:
            from app.services.tool_registry_service import get_tool_registry_service
            registry = get_tool_registry_service()
            candidates = await registry.search_tools(
                query=eff_query,
                user_roles=state.get("user_roles"),
                top_k=settings.TOOL_RETRIEVAL_TOP_K,
                distance_threshold=settings.TOOL_RETRIEVAL_DISTANCE_THRESHOLD,
            )
            dyn_res = _build_dynamic_single_result(candidates, eff_query, is_retry, prior_query)
            if dyn_res is not None:
                return dyn_res
        except Exception as exc:
            logger.warning(f"Error in async dynamic semantic tool retrieval during router execution: {exc}")

    return _general_fallback_result()

__all__ = [
    "PlannedToolCall",
    "RouterDecision",
    "MultiRouterDecision",
    "TOOL_DOMAIN_MAP",
    "decompose_user_query",
    "is_retry_query",
    "find_previous_user_query",
    "_extract_sub_queries",
    "_split_conjunction_clauses",
    "classify_query_rule_based",
    "router_node",
    "arouter_node",
    "CUSTOMER_TERMS",
    "VENDOR_TERMS",
    "PURCHASE_INVOICE_TERMS",
    "CUSTOMER_BILLING_TERMS",
    "ACCOUNTING_PROFIT_LOSS_PATTERNS",
    "ACCOUNTING_BALANCE_SHEET_PATTERNS",
    "ACCOUNTING_GENERAL_LEDGER_PATTERNS",
    "ACCOUNTING_BANK_RECON_PATTERNS",
    "ACCOUNTING_VENDOR_BILLS_PATTERNS",
    "ACCOUNTING_BALANCE_PATTERNS",
    "ACCOUNTING_JOURNAL_PATTERNS",
    "INVENTORY_STOCK_PATTERNS",
    "INVENTORY_STOCK_HISTORY_PATTERNS",
    "INVENTORY_TRANSFERS_PATTERNS",
    "INVENTORY_LOW_STOCK_PATTERNS",
    "INVENTORY_ADJUSTMENTS_PATTERNS",
    "INVENTORY_SCRAP_PATTERNS",
    "INVENTORY_WAREHOUSES_PATTERNS",
    "INVENTORY_CATEGORIES_PATTERNS",
    "INVENTORY_LOCATION_PATTERNS",
    "INVENTORY_VELOCITY_FORECAST_PATTERNS",
    "INVENTORY_PRICE_HISTORY_PATTERNS",
    "SALES_TOP_PRODUCTS_PATTERNS",
    "SALES_SUMMARY_PATTERNS",
    "SALES_ORDER_DETAIL_PATTERNS",
    "SALES_CONTACTS_PATTERNS",
    "SALES_ORDERS_PATTERNS",
    "MARKETPLACE_WRITE_PATTERNS",
    "MARKETPLACE_ACCOUNT_HEALTH_PATTERNS",
    "MARKETPLACE_KILAT_PATTERNS",
    "MARKETPLACE_ORDERS_PATTERNS",
    "MARKETPLACE_MAPPING_PATTERNS",
    "MARKETPLACE_STOCK_SYNC_SIM_PATTERNS",
    "MARKETPLACE_SYNC_LOGS_PATTERNS",
    "MARKETPLACE_PRICE_COMPARISON_PATTERNS",
    "MARKETPLACE_FEE_ANOMALIES_PATTERNS",
    "MARKETPLACE_DEDUCTION_SETTLEMENT_PATTERNS",
    "MARKETPLACE_CASHFLOW_BALANCE_PATTERNS",
    "MARKETPLACE_COMPETITOR_PATTERNS",
    "ALL_MARKETPLACE_PATTERN_GROUPS",
    "INVENTORY_PRODUCT_PATTERNS",
    "KNOWLEDGE_RAG_PATTERNS",
    "GREETING_PATTERNS",
    "GENERAL_INQUIRY_PATTERNS",
    "OUT_OF_SCOPE_PATTERNS",
    "RETRY_PATTERNS",
    "_match_any",
    "_find_first_match",
    "is_meta_question",
]
