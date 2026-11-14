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
from typing import Any, Dict, List, Literal, Optional, Tuple
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from pydantic import BaseModel, Field, model_validator

from app.core.config import settings
from app.core.context import trim_conversation_messages
from app.graph.state import AgentState, ToolCallPlan
from app.schemas.tools import TOOL_DOMAIN_MAP
from app.services.llm import get_chat_model

logger = logging.getLogger("router_node")


class PlannedToolCall(BaseModel):
    """Structured representation of an individual ERP tool call planned by router."""
    tool: Literal[
        "fetch_vendor_invoices",
        "get_account_balance",
        "get_journal_entries",
        "get_general_ledger",
        "get_profit_and_loss",
        "get_balance_sheet",
        "get_vendor_bills",
        "get_bank_reconciliation_summary",
        "get_stock_level",
        "get_product_info",
        "get_locations",
        "get_stock_history",
        "get_transfers",
        "get_low_stock_alerts",
        "get_stock_adjustments",
        "get_scrap_records",
        "get_warehouses",
        "get_product_categories",
    ] = Field(..., description="Target ERP tool name.")
    domain: Literal["accounting", "inventory"] = Field(
        ...,
        description="Target ERP domain: 'accounting' or 'inventory'."
    )
    sub_query: Optional[str] = Field(
        None,
        description="Specific user query clause or segment relevant to this tool call."
    )

    @model_validator(mode="after")
    def validate_tool_domain(self) -> "PlannedToolCall":
        """Ensure domain aligns with the tool's registered domain."""
        expected = TOOL_DOMAIN_MAP.get(self.tool)
        if expected and self.domain != expected:
            self.domain = expected
        return self


class RouterDecision(BaseModel):
    """Structured output schema for LLM router fallback."""
    domain: Literal["accounting", "inventory", "knowledge", "general", "out_of_scope"] = Field(
        ...,
        description="Target ERP module domain: 'accounting', 'inventory', 'knowledge' (SOPs, specs, policies), 'general' (greetings/help), or 'out_of_scope' (unrelated queries outside ERP)."
    )
    query_mode: Literal["live_erp", "vector_rag", "hybrid", "general", "out_of_scope"] = Field(
        "live_erp",
        description="Execution mode: 'vector_rag' for SOP/policies/specs, 'live_erp' for real-time transactions/stock, 'hybrid' for combined questions, 'general', or 'out_of_scope'."
    )
    tool: Optional[Literal[
        "fetch_vendor_invoices",
        "get_account_balance",
        "get_journal_entries",
        "get_general_ledger",
        "get_profit_and_loss",
        "get_balance_sheet",
        "get_vendor_bills",
        "get_bank_reconciliation_summary",
        "get_stock_level",
        "get_product_info",
        "get_locations",
        "get_stock_history",
        "get_transfers",
        "get_low_stock_alerts",
        "get_stock_adjustments",
        "get_scrap_records",
        "get_warehouses",
        "get_product_categories",
    ]] = Field(
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
    domain: Literal["accounting", "inventory", "knowledge", "general", "out_of_scope"] = Field(
        ...,
        description="Primary target ERP module domain: 'accounting', 'inventory', 'knowledge', 'general', or 'out_of_scope'."
    )
    query_mode: Literal["live_erp", "vector_rag", "hybrid", "general", "out_of_scope"] = Field(
        "live_erp",
        description="Execution mode: 'vector_rag', 'live_erp', 'hybrid', 'general', or 'out_of_scope'."
    )
    tool: Optional[Literal[
        "fetch_vendor_invoices",
        "get_account_balance",
        "get_journal_entries",
        "get_general_ledger",
        "get_profit_and_loss",
        "get_balance_sheet",
        "get_vendor_bills",
        "get_bank_reconciliation_summary",
        "get_stock_level",
        "get_product_info",
        "get_locations",
        "get_stock_history",
        "get_transfers",
        "get_low_stock_alerts",
        "get_stock_adjustments",
        "get_scrap_records",
        "get_warehouses",
        "get_product_categories",
    ]] = Field(
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
        if self.domain in ("general", "out_of_scope", "knowledge") or self.query_mode == "vector_rag":
            self.tool = None
            self.planned_tools = []
            return self

        # If planned_tools is populated but primary tool is omitted, set from first planned tool
        if self.planned_tools and not self.tool:
            self.tool = self.planned_tools[0].tool
            self.domain = self.planned_tools[0].domain

        # If primary tool is specified without planned_tools, synthesize planned_tools
        if self.tool and not self.planned_tools:
            d = TOOL_DOMAIN_MAP.get(self.tool, self.domain)
            if d in ("accounting", "inventory"):
                self.planned_tools = [PlannedToolCall(tool=self.tool, domain=d)]

        return self


# Disambiguation term sets
CUSTOMER_TERMS = {"pelanggan", "customer", "piutang", "penjualan", "sales", "buyer", "klien"}
VENDOR_TERMS = {"vendor", "supplier", "pemasok", "hutang"}
PURCHASE_INVOICE_TERMS = ("faktur pembelian", "tagihan pembelian", "invoice pembelian")
CUSTOMER_BILLING_TERMS = (
    "tagihan",
    "faktur",
    "invoice",
    "bill",
    "piutang",
    "hutang",
    "pembayaran",
    "bayar",
)

ACCOUNTING_PROFIT_LOSS_PATTERNS = [
    r"\b(laba\s*(?:dan\s*)?rugi|rugi\s*laba|p&l|profit\s*(?:and|&)\s*loss|income\s*statement|omzet|omset|gross\s*margin|laba\s*kotor|laba\s*bersih)\b",
]

ACCOUNTING_BALANCE_SHEET_PATTERNS = [
    r"\b(neraca(?:\s*keuangan)?|balance\s*sheet|posisi\s*keuangan|total\s*aset|aktiva\s*(?:dan\s*)?pasiva|keseimbangan\s*neraca)\b",
]

ACCOUNTING_GENERAL_LEDGER_PATTERNS = [
    r"\b(buku\s*besar|general\s*ledger|ledger\s*akun|kartu\s*buku\s*besar|running\s*balance|mutasi\s*rekening)\b",
]

ACCOUNTING_BANK_RECON_PATTERNS = [
    r"\b(rekonsiliasi\s*bank|bank\s*reconciliation|rekon\s*bank|selisih\s*(?:rekening\s*koran|bank)|statement\s*balance)\b",
]

ACCOUNTING_VENDOR_BILLS_PATTERNS = [
    r"\b(vendor\s*bills?|ap\s*aging|umur\s*hutang|jatuh\s*tempo|overdue|due\s*soon)\b",
]

ACCOUNTING_BALANCE_PATTERNS = [
    r"\bsaldo(nya)?\b",
    r"\bbalance\b",
    r"\bkas\b",
    r"\bkas kecil\b",
    r"\bbuku kas\b",
    r"\bsaldo akun\b",
]

ACCOUNTING_JOURNAL_PATTERNS = [
    r"\bjurnal(nya)?\b",
    r"\bjournal\b",
    r"\bentri jurnal\b",
    r"\bdebit kredit\b",
]

INVENTORY_STOCK_PATTERNS = [
    r"\bstok(nya)?\b",
    r"\bstock(nya)?\b",
    r"\bsisa barang(nya)?\b",
    r"\bsisa stok(nya)?\b",
    r"\bkuantitas(nya)?\b",
    r"\bqty\b",
    r"\bon hand\b",
]

INVENTORY_STOCK_HISTORY_PATTERNS = [
    r"\bkartu\s+stok\b",
    r"\briwayat\s+(mutasi|stok|pergerakan)\b",
    r"\bmutasi\s+stok\b",
    r"\bpergerakan\s+(stok|barang)\b",
    r"\bbuku\s+besar\s+stok\b",
    r"\bstock\s+history\b",
    r"\bhistory\s+stok\b",
    r"\bkartu\s+persediaan\b",
]

INVENTORY_TRANSFERS_PATTERNS = [
    r"\bsurat\s+jalan\b",
    r"\btransfer\s+(barang|gudang|stok|internal)\b",
    r"\bmutasi\s+(antar\s+gudang|barang)\b",
    r"\bpengiriman\s+barang\b",
    r"\bpenerimaan\s+barang\b",
    r"\bdelivery\s+order\b",
    r"\bwh/(?:in|out)/\d+\b",
    r"\bdokumen\s+transfer\b",
]

INVENTORY_LOW_STOCK_PATTERNS = [
    r"\bstok(nya)?\s+(menipis|habis|kritis|minimum|kurang|minus|sedikit)\b",
    r"\bstok(nya)?\s+.*?(menipis|habis|kritis|minimum)\b",
    r"\blow\s+stock\b",
    r"\balert\s+stok\b",
    r"\bstok\s+alert\b",
    r"\bdi\s*bawah\s+minimum\b",
    r"\bperingatan\s+stok\b",
]

INVENTORY_ADJUSTMENTS_PATTERNS = [
    r"\bstock\s+opname\b",
    r"\bstok\s+opname\b",
    r"\bpenyesuaian\s+stok\b",
    r"\badjustment\s+stok\b",
    r"\bselisih\s+stok\b",
    r"\bkoreksi\s+stok\b",
    r"\bhasil\s+opname\b",
]

INVENTORY_SCRAP_PATTERNS = [
    r"\bbarang\s+(rusak|cacat|scrap)\b",
    r"\bscrap\s*(order|record)?\b",
    r"\bpemusnahan\s+barang\b",
    r"\bproduk\s+rusak\b",
    r"\bcatatan\s+scrap\b",
]

INVENTORY_WAREHOUSES_PATTERNS = [
    r"\bmaster\s+gudang\b",
    r"\b(daftar|list)\s+master\s+gudang\b",
    r"\bhierarki\s+gudang\b",
    r"\bmaster\s+warehouses?\b",
    r"\binfo\s+master\s+gudang\b",
]

INVENTORY_CATEGORIES_PATTERNS = [
    r"\bkategori\s+(produk|barang)\b",
    r"\bmaster\s+kategori\b",
    r"\bdaftar\s+kategori\b",
    r"\bproduct\s+categor(?:y|ies)\b",
]

INVENTORY_LOCATION_PATTERNS = [
    r"\b(daftar|list|semua|info|data)\s+(lokasi|rak)\b",
    r"\b(daftar|list)\s+(lokasi|rak)\b",
    r"\bada\s+(lokasi|rak)\b",
    r"\b(lokasi|rak)\s+(apa\s+saja|tersedia|aktif)\b",
    r"^(lokasi|rak)\b",
    r"\b(daftar|list|semua|info|data)\s+(gudang|lokasi|warehouse)\b",
    r"\b(daftar|list)\s+(gudang|lokasi|warehouse)\b",
    r"\bada\s+(gudang|lokasi|warehouse)\b",
    r"\b(gudang|lokasi|warehouse)\s+(apa\s+saja|tersedia|aktif)\b",
    r"^(gudang|lokasi|warehouse)\b",
]

INVENTORY_PRODUCT_PATTERNS = [
    r"\bproduk(nya)?\b",
    r"\bproduct\b",
    r"\bharga(nya)?\b",
    r"\bharga\s+(barang|jual|beli)?(nya)?\b",
    r"\bharga barang\b",
    r"\bharga jual\b",
    r"\bharga beli\b",
    r"\bdetail barang\b",
    r"\bsku\b",
    r"\bbarcode\b",
    r"\bvarian\b",
]

KNOWLEDGE_RAG_PATTERNS = [
    r"\bsop\b",
    r"\bkebijakan\b",
    r"\bprosedur\b",
    r"\baturan\b",
    r"\b(per)?syarat(an)?\b",
    r"\b(ke)?tentuan\b",
    r"\b(di|me)?retur(kan)?(nya)?\b",
    r"\b(ber)?garansi\b",
    r"\bspesifikasi\b",
    r"\b(gsm|gramasi|ketebalan)\b",
    r"\bprofil\s+(vendor|supplier|customer|pelanggan)\b",
    r"\btermin\s+(pembayaran|kredit)\b",
    r"\bcara\s+(klaim|retur|pengajuan)\b",
    r"\b(di)?klaim\b",
]

GREETING_PATTERNS = [
    r"^(halo|hai|hi|hey|hei|helo|hello)\b",
    r"^(selamat\s+(pagi|siang|sore|malam|petang))\b",
    r"^(assalamu\s*['’]?alaikum|sampurasun|kulonuwun)\b",
    r"^(ping|p\b|tes|test|testing)\b",
    r"^(terima\s*kasih|makasih|thanks|thank\s*you|matur\s*nuwun)\b",
]

GENERAL_INQUIRY_PATTERNS = [
    r"^(bisa\s+bantu\s+apa|kamu\s+siapa|fitur\s+apa|menu|help|bantuan|siapa\s+kamu|apa\s+yang\s+bisa\s+kamu\s+lakukan)\b",
]

OUT_OF_SCOPE_PATTERNS = [
    r"\b(system\s*design|arsitektur\s*(sistem)?|source\s*code|prompt\s*(kamu|anda)|system\s*prompt)\b",
    r"\b(buatkan\s+(puisi|cerita|kode|script)|ceritakan\s+(lelucon|dongeng)|siapa\s+presiden|ramalan\s+cuaca)\b",
]

RETRY_PATTERNS = [
    r"^(?:(?:bisa\s+)?tolong\s+|mohon\s+|silakan\s+|bisa\s+)?(?:coba\s+(?:(?:di)?cek\s+|(?:di)?periksa\s+|(?:di)?cari\s+|(?:di)?lihat\s+)?(?:lagi|ulang)|(?:coba\s+)?(?:di)?cek\s+(?:sekali\s+)?(?:lagi|ulang)|(?:coba\s+)?(?:di)?periksa\s+(?:sekali\s+)?(?:lagi|ulang)|(?:coba\s+)?(?:di)?cari\s+(?:sekali\s+)?(?:lagi|ulang)|ulang(i)?|coba\s+ulang|ulangi\s+lagi|sekali\s+lagi|coba\s+sekali\s+lagi|retry|tes\s+lagi)(?:\s+dong|\s+ya|\s+deh|\s+bro|\s+kak|\s+mas|\s+mbak|\s+min|\s+gan|\s+pak|\s+bu)?$",
]


def _match_any(patterns: list, text: str) -> bool:
    """Check if any regex pattern matches the text."""
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def _find_first_match(patterns: list, text: str) -> Optional[re.Match]:
    """Find the earliest regex pattern match in text."""
    best: Optional[re.Match] = None
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            if best is None or m.start() < best.start():
                best = m
    return best


def _extract_sub_queries(query: str, candidates: List[Tuple[str, str, int]]) -> List[str]:
    """
    Split multi-intent query into focused sub-queries along Indonesian conjunctions.
    Falls back to the full query if clause count does not align with candidates count.
    """
    if len(candidates) <= 1:
        return [query] * len(candidates)

    parts = re.split(r"\b(?:dan|serta|sekaligus|juga|lalu)\b|;|(?<!\d),(?!\d)", query, flags=re.IGNORECASE)
    cleaned_parts = [p.strip() for p in parts if p.strip()]

    if len(cleaned_parts) == len(candidates):
        return cleaned_parts

    return [query] * len(candidates)


def is_retry_query(text: str) -> bool:
    """Check if the text represents a conversational retry command."""
    cleaned = (text or "").strip().rstrip("?!.,;: '\"").lower()
    return _match_any(RETRY_PATTERNS, cleaned)


def find_previous_user_query(messages: Optional[List[BaseMessage]]) -> Optional[str]:
    """
    Find the most recent HumanMessage content prior to the current turn that is not a retry command.
    """
    if not messages:
        return None
    for msg in reversed(messages[:-1]):
        if isinstance(msg, HumanMessage):
            content = getattr(msg, "content", "") or ""
            content_str = str(content).strip()
            if content_str and not is_retry_query(content_str):
                return content_str
    return None


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

    # 0. Sapaan / Percakapan Kasual / Bantuan Cepat
    if _match_any(GREETING_PATTERNS, cleaned) or _match_any(GENERAL_INQUIRY_PATTERNS, cleaned):
        has_erp_intent = (
            _match_any(ACCOUNTING_BALANCE_PATTERNS, cleaned)
            or _match_any(ACCOUNTING_JOURNAL_PATTERNS, cleaned)
            or _match_any(ACCOUNTING_PROFIT_LOSS_PATTERNS, cleaned)
            or _match_any(ACCOUNTING_BALANCE_SHEET_PATTERNS, cleaned)
            or _match_any(ACCOUNTING_GENERAL_LEDGER_PATTERNS, cleaned)
            or _match_any(ACCOUNTING_BANK_RECON_PATTERNS, cleaned)
            or _match_any(ACCOUNTING_VENDOR_BILLS_PATTERNS, cleaned)
            or _match_any(INVENTORY_STOCK_PATTERNS, cleaned)
            or _match_any(INVENTORY_LOCATION_PATTERNS, cleaned)
            or _match_any(INVENTORY_PRODUCT_PATTERNS, cleaned)
            or has_vendor
            or has_purchase_invoice
            or is_rag
        )
        if not has_erp_intent:
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

    # 0.1 Fast-Path Out-of-Scope Guardrail
    if _match_any(OUT_OF_SCOPE_PATTERNS, cleaned):
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

    # Inventory: Stock levels (Physical on-hand inventory)
    m_stock = _find_first_match(INVENTORY_STOCK_PATTERNS, cleaned)
    if m_stock and not m_history and not m_low_stock and not m_adjustments and not m_scrap:
        candidates.append(("get_stock_level", "inventory", m_stock.start()))

    # Inventory: Locations (Rak / Lokasi Penyimpanan)
    m_loc = _find_first_match(INVENTORY_LOCATION_PATTERNS, cleaned)
    if m_loc and not m_warehouses:
        candidates.append(("get_locations", "inventory", m_loc.start()))

    # Inventory: Product info
    if not is_rag:
        # If stock level pattern matched, only add product_info if explicit product detail terms are present
        # (avoid treating "stok produk ..." or "produk ... berapa stoknya" as a separate product_info tool call)
        explicit_product_patterns = [
            r"\bharga(nya)?\b",
            r"\bharga\s+(barang|jual|beli)?(nya)?\b",
            r"\bharga barang\b",
            r"\bharga jual\b",
            r"\bharga beli\b",
            r"\bdetail barang\b",
            r"\bsku\b",
            r"\bbarcode\b",
            r"\bvarian\b",
        ]
        if m_stock:
            m_prod = _find_first_match(explicit_product_patterns, cleaned)
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
        or (bool(candidates) and all(c[1] != "inventory" for c in candidates))
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

    return None


def _annotate_multi_company(res: Dict[str, Any], state: AgentState, user_query: str) -> Dict[str, Any]:
    if not isinstance(res, dict):
        return res
    from app.graph.nodes.extractor import resolve_all_branch_company_ids, COMPANY_NAME_MAP
    multi_cids, is_all = resolve_all_branch_company_ids(user_query)
    is_multi_co = bool(state.get("is_multi_company") or is_all or (len(multi_cids) > 1))
    if is_multi_co:
        res["is_multi_company"] = True
        effective_cids = (
            state.get("company_ids")
            if (state.get("company_ids") and len(state.get("company_ids")) > 1 and not is_all and not multi_cids)
            else (multi_cids if multi_cids else (state.get("company_ids") or list(COMPANY_NAME_MAP.keys())))
        )
        if effective_cids:
            res["company_ids"] = effective_cids
            res["company_id"] = effective_cids[0]
    return res


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
        last_msg = state["messages"][-1]
        user_query = getattr(last_msg, "content", "") or ""

    user_query = (user_query or "").strip()
    raw_res = _router_node_impl(state=state, llm=llm, user_query=user_query)
    return _annotate_multi_company(raw_res, state=state, user_query=user_query)


def _router_node_impl(
    state: AgentState,
    llm: Optional[BaseChatModel] = None,
    user_query: str = "",
) -> Dict[str, Any]:
    if not user_query:
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

    # Step 0: Fast Conversational Retry Check ('coba lagi', 'ulang', 'retry')
    is_retry = is_retry_query(user_query)
    prior_query = find_previous_user_query(state.get("messages")) if is_retry else None
    if is_retry:
        if prior_query:
            try:
                rule_result = classify_query_rule_based(prior_query)
                if rule_result is not None:
                    rule_result["user_query"] = prior_query
                    rule_result["retry_instruction"] = f"Mengulang permintaan: {prior_query}"
                    return rule_result
            except Exception as exc:
                logger.warning(f"Error in rule-based classifier during retry: {exc}")
            user_query = prior_query
        else:
            return {
                "detected_domain": "general",
                "detected_tool": None,
                "query_mode": "general",
                "rag_query": None,
                "error_message": "NO_PREVIOUS_QUERY_TO_RETRY",
                "is_multi_tool": False,
                "detected_domains": ["general"],
                "detected_tools": [],
                "active_tools": [],
            }

    # Step 1: Fast Rule / Regex Pattern Matching
    try:
        rule_result = classify_query_rule_based(user_query)
        if rule_result is not None:
            return rule_result
    except Exception as exc:
        logger.warning(f"Error in rule-based classifier for query '{user_query}': {exc}")

    # Step 1.5: Fast Conversational Follow-up / Slot Refinement Check
    # Matches branch/location mentions ('di mainstore', 'di cabang bandung'),
    # retry/check follow-ups ('coba cek lagi dimainstore'), or entity refinements
    # when an active ERP inquiry exists in prior messages.
    history_messages = state.get("messages") or []
    if len(history_messages) > 1:
        prior_q = find_previous_user_query(history_messages)
        if prior_q:
            from app.graph.nodes.extractor import resolve_branch_company_id

            clean_uq = user_query.strip().rstrip("?!.,;: '\"").lower()
            is_branch_mention = bool(resolve_branch_company_id(clean_uq)) or bool(
                re.search(r"\b(?:di\s+|cabang\s+|gudang\s+|lokasi\s+|rak\s+)([A-Za-z0-9\_\-]+)", clean_uq)
            )
            is_followup_retry = is_retry_query(clean_uq) or bool(
                re.search(r"\b(coba\s+(?:(?:di)?cek\s+|(?:di)?periksa\s+|(?:di)?cari\s+)?lagi|(?:di)?cek\s+lagi|coba\s+lagi|ulang(i)?|tes\s+lagi|(?:di)?cek\s+ulang)\b", clean_uq)
            )
            is_short_followup = len(clean_uq.split()) <= 7 and (is_branch_mention or is_followup_retry)

            if is_short_followup:
                try:
                    prior_result = classify_query_rule_based(prior_q)
                    if prior_result is not None and prior_result.get("detected_domain") in ("accounting", "inventory"):
                        effective_sub = user_query if is_branch_mention else prior_q
                        effective_user = f"{prior_q} {user_query}" if is_branch_mention else prior_q
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

            # Heuristic Short Follow-Up: If query <= 6 words with no new domain intent,
            # and prior ERP query exists, treat as conversational follow-up to prior domain
            has_new_domain_intent = (
                any(term in clean_uq for term in CUSTOMER_TERMS)
                or any(term in clean_uq for term in VENDOR_TERMS)
                or _match_any(GREETING_PATTERNS, clean_uq)
                or _match_any(OUT_OF_SCOPE_PATTERNS, clean_uq)
                or _match_any(ACCOUNTING_BALANCE_PATTERNS, clean_uq)
                or _match_any(ACCOUNTING_JOURNAL_PATTERNS, clean_uq)
                or _match_any(KNOWLEDGE_RAG_PATTERNS, clean_uq)
            )
            if len(clean_uq.split()) <= 6 and not has_new_domain_intent:
                try:
                    prior_result = classify_query_rule_based(prior_q)
                    if prior_result is not None and prior_result.get("detected_domain") in ("accounting", "inventory"):
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
    model = llm
    is_test_env = "unittest" in sys.modules or os.environ.get("FASTPRINT_TEST_ENV") == "true"
    if model is None and not is_test_env:
        try:
            model = get_chat_model()
        except Exception:
            # If no API key configured in test/offline environment, safely fallback
            model = None

    if model is not None:
        try:
            system_prompt = (
                "You are a Supervisor Agent and Multi-Domain Tool Router for FastPrint ERP.\n"
                "Classify user inquiries into target module domains ('accounting', 'inventory', 'knowledge', 'general', 'out_of_scope') "
                "and determine the execution query_mode ('live_erp', 'vector_rag', 'hybrid', 'general', 'out_of_scope').\n\n"
                "MULTI-INTENT & MULTI-TOOL PLANNING:\n"
                "- If the user inquiry asks for multiple pieces of ERP data spanning one or more domains "
                "(e.g., checking stock levels in inventory AND checking account balances or vendor invoices in accounting), "
                "plan each required tool call in 'planned_tools' with its specific tool name, domain, and focused sub_query.\n"
                "- Populate 'tool' and 'domain' with the primary (first) tool call for backward compatibility.\n"
                "- If only a single tool is required, include it in both 'tool' and 'planned_tools'.\n\n"
                "ROUTING RULES:\n"
                "1. 'vector_rag' / domain='knowledge': Qualitative questions about SOPs, return policies, product technical specifications (GSM, size), warranty, or contract terms. tool=null, planned_tools=[].\n"
                "2. 'live_erp': Numerical / real-time operational inquiries about stock on hand, cash balances, journal ledger, or vendor bills/invoices.\n"
                "3. 'hybrid': Combined queries asking about both SOP/specs rules AND live ERP numbers/stock. Plan live ERP tool(s) in planned_tools and set rag_query.\n"
                "4. 'general': Pure greetings, polite chitchat, or asking about bot capabilities. tool=null, planned_tools=[].\n"
                "5. 'out_of_scope': Inquiries unrelated to FastPrint ERP data or operations. tool=null, planned_tools=[].\n\n"
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
                "CRITICAL GUARDRAILS:\n"
                "- References to specific ERP transaction document numbers (e.g., 'PC/...', 'JE/...', 'INV/...') must be routed to domain='accounting' (fetch_vendor_invoices or get_journal_entries).\n"
                "- Customer / pelanggan inquiries targeting invoices, billing, or accounts receivable must NOT be routed to 'fetch_vendor_invoices'. Set domain='general', tool=null, planned_tools=[].\n"
                "- Customer / pelanggan inquiries asking about inventory stock, warehouse locations, or product details must be routed to domain='inventory'.\n"
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

            detected_domain = getattr(decision, "domain", "general")
            query_mode = getattr(decision, "query_mode", "live_erp")
            detected_tool = getattr(decision, "tool", None)
            rag_query = getattr(decision, "rag_query", None)
            raw_planned_tools = getattr(decision, "planned_tools", None) or []

            active_tools: List[Dict[str, Any]] = []

            if detected_domain in ("general", "out_of_scope", "knowledge") or query_mode == "vector_rag":
                detected_tool = None
                raw_planned_tools = []

            if raw_planned_tools:
                for pt in raw_planned_tools:
                    pt_tool = pt.tool if hasattr(pt, "tool") else pt.get("tool")
                    pt_domain = pt.domain if hasattr(pt, "domain") else pt.get("domain", TOOL_DOMAIN_MAP.get(pt_tool, detected_domain))
                    pt_sub_query = (pt.sub_query if hasattr(pt, "sub_query") else pt.get("sub_query")) or user_query
                    if pt_tool and pt_domain in ("accounting", "inventory"):
                        active_tools.append({
                            "tool": pt_tool,
                            "domain": pt_domain,
                            "sub_query": pt_sub_query,
                            "params": {},
                            "status": "pending",
                        })

            if not active_tools and detected_tool and detected_domain in ("accounting", "inventory"):
                active_tools.append({
                    "tool": detected_tool,
                    "domain": detected_domain,
                    "sub_query": user_query,
                    "params": {},
                    "status": "pending",
                })

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
            if is_retry and prior_query:
                result["user_query"] = prior_query
                result["retry_instruction"] = f"Mengulang permintaan: {prior_query}"
            return result
        except Exception:
            pass

    # Default fallback when unclassified
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
