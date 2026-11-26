"""
Hybrid Semantic Retrieval + Scoped LLM NLU Router with Conservative Disambiguation.

Implements the target NLU pipeline defined in nlu_architecture_blueprint.md:
  1. Minimal Hard Guardrail Regex (greeting/help/meta, write-block/out-of-scope/explicit
     customer billing, strict document ID fast-path, contextual analysis & P&L drill-down).
  2. Per-Clause Hybrid Retrieval via ToolRegistryService.search_tools_hybrid (PGVector Cosine
     + PostgreSQL FTS RRF + Conditional RBAC) with context enrichment for short follow-ups.
  3. Dynamic Scoped LLM Router (MultiRouterDecision) seeing only Top-K retrieved tool cards
     and conservative disambiguation rules.
"""
import asyncio
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.branches import resolve_all_branch_company_ids, resolve_branch_company_id
from app.core.config import settings
from app.core.context import (
    has_anaphoric_reference,
    trim_conversation_messages,
)
from app.graph.nodes.extractors.accounting import resolve_pnl_drilldown_from_history
from app.graph.nodes.router_decomposer import (
    decompose_user_query,
    find_previous_user_query,
)
from app.graph.nodes.router_patterns import (
    CUSTOMER_BILLING_TERMS,
    CUSTOMER_TERMS,
    MARKETPLACE_WRITE_PATTERNS,
    OUT_OF_SCOPE_PATTERNS,
    PURCHASE_INVOICE_TERMS,
    VENDOR_TERMS,
    _match_any,
    is_general_help_inquiry,
    is_meta_question,
    is_pure_greeting,
)
from app.graph.state import AgentState
from app.schemas.tools import TOOL_DOMAIN_MAP
from app.services.tool_registry_service import (
    get_tool_registry_service,
    merge_rrf_candidates,
)

logger = logging.getLogger("router_semantic")

# Deterministic ERP mutation verbs blocked unless framed as a read-only diagnostic question
_ERP_WRITE_PATTERNS = [
    r"\b(?:buat(?:kan)?|tambah(?:kan)?|bikin(?:kan)?|input|posting|approve|setujui|hapus|delete|batalkan|cancel|ubah|edit|koreksi)\s+"
    r"(?:jurnal|voucher|faktur|invoice|tagihan|sales\s+order|so\b|transfer|surat\s+jalan|stok\s+opname|stock\s+adjustment|scrap)\b",
    r"\b(?:lakukan|eksekusi|jalankan)\s+(?:stock\s+opname|penyesuaian\s+stok|transfer\s+stok|posting\s+jurnal|push\s+stok|update\s+harga)\b",
    r"\b(?:update|push|sinkron(?:kan)?|ubah|edit|naik(?:kan)?|turun(?:kan)?)\s+"
    r"(?:data\s+)?(?:stok|stock|harga|price)\s+(?:[\w\-]+\s+){0,3}?(?:ke|di)\s+"
    r"(?:shopee|tiktok(?:\s*shop)?|tokopedia|lazada|marketplace)\b",
]

_DIAGNOSTIC_READ_INTENT_RE = re.compile(
    r"\b(?:kenapa|mengapa|apa\s+penyebab|bagaimana|gimana|cara|sop|prosedur|log|riwayat|history|simulasi|kapan|gagal|error|status|daftar|list|cek|lihat|tampilkan|berapa)\b",
    re.IGNORECASE,
)

# Explicit customer-side billing indicators (ambiguous "cek tagihan <nama>" is intentionally excluded
# so it flows to Scoped LLM Router for conservative clarification chips).
_EXPLICIT_CUSTOMER_BILLING_PHRASES = (
    "faktur penjualan",
    "invoice penjualan",
    "tagihan penjualan",
    "bon kasir",
    "bon toko",
    "kuitansi",
    "kwitansi",
    "piutang",
)

# Strict document identifier patterns with 100% deterministic tool mappings
_STRICT_TRANSFER_DOC_RE = re.compile(r"\bWH/(?:OUT|IN|INT)/\d+\b", re.IGNORECASE)
_STRICT_SALES_ORDER_DOC_RE = re.compile(
    r"\b(?:[A-Z0-9_\-]+/SO/\d+|SO/\d{4}/\d+)\b",
    re.IGNORECASE,
)
_STRICT_JOURNAL_DOC_RE = re.compile(
    r"\b(?:[A-Z0-9_\-]+/)*JE/\d{4}/\d+\b",
    re.IGNORECASE,
)
_STRICT_PURCHASE_VOUCHER_DOC_RE = re.compile(
    r"\b(?:[A-Z0-9_\-]+/)*PC/\d{4}/\d+\b",
    re.IGNORECASE,
)

_MAX_SEMANTIC_CANDIDATES = 12


def is_explicit_customer_billing_query(text: str) -> bool:
    """
    Return True strictly when the query explicitly targets unsupported customer billing/AR
    (e.g. 'tagihan customer Budi', 'piutang', 'faktur penjualan', 'kuitansi').
    Ambiguous entity billing like 'cek tagihan Frendy' returns False so Scoped LLM Router
    can offer clickable clarification chips (Vendor Bills vs Sales Orders).
    """
    cleaned = (text or "").lower().strip()
    if not cleaned:
        return False
    has_vendor = any(re.search(rf"\b{re.escape(term)}(nya)?\b", cleaned) for term in VENDOR_TERMS)
    has_purchase = any(kw in cleaned for kw in PURCHASE_INVOICE_TERMS)
    if has_vendor or has_purchase:
        return False

    if any(phrase in cleaned for phrase in _EXPLICIT_CUSTOMER_BILLING_PHRASES):
        return True

    has_customer = any(re.search(rf"\b{re.escape(term)}(nya)?\b", cleaned) for term in CUSTOMER_TERMS)
    has_billing = any(kw in cleaned for kw in CUSTOMER_BILLING_TERMS)
    return bool(has_customer and has_billing)


def is_hard_write_block_query(text: str) -> bool:
    """Return True for explicit marketplace/ERP write or mutation commands (non-diagnostic)."""
    cleaned = (text or "").lower().strip()
    if not cleaned:
        return False
    if _DIAGNOSTIC_READ_INTENT_RE.search(cleaned):
        return False
    return _match_any(MARKETPLACE_WRITE_PATTERNS, cleaned) or _match_any(_ERP_WRITE_PATTERNS, cleaned)


def match_strict_document_identifier(query: str) -> Optional[Tuple[str, str]]:
    """
    Map unambiguous ERP document reference codes directly to (tool_name, domain).
    Returns None if no strict code is detected.
    """
    if not query:
        return None
    if _STRICT_TRANSFER_DOC_RE.search(query):
        return ("get_transfers", "inventory")
    if _STRICT_SALES_ORDER_DOC_RE.search(query):
        return ("get_sales_order_detail", "sales")
    if _STRICT_JOURNAL_DOC_RE.search(query):
        return ("get_journal_entries", "accounting")
    if _STRICT_PURCHASE_VOUCHER_DOC_RE.search(query):
        return ("fetch_vendor_invoices", "accounting")
    return None


def _build_single_tool_route(
    tool_name: str,
    domain: str,
    user_query: str,
    is_retry: bool = False,
    prior_query: Optional[str] = None,
    available_tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build normalized single-tool AgentState delta."""
    registry = get_tool_registry_service()
    if available_tools is None:
        meta = registry.get_tool_by_name(tool_name)
        available_tools = [meta] if meta else []
    res: Dict[str, Any] = {
        "detected_domain": domain,
        "detected_tool": tool_name,
        "query_mode": "live_erp",
        "rag_query": None,
        "rag_queries": [],
        "is_multi_tool": False,
        "detected_domains": [domain],
        "detected_tools": [tool_name],
        "active_tools": [
            {
                "tool": tool_name,
                "domain": domain,
                "sub_query": user_query,
                "params": {},
                "status": "pending",
            }
        ],
        "available_tools": available_tools,
        "sub_questions": [user_query],
        "non_tool_questions": [],
        "needs_clarification": False,
        "clarification_question": None,
        "clarification_options": [],
    }
    if is_retry and prior_query:
        res["user_query"] = prior_query
        res["retry_instruction"] = f"Mengulang permintaan: {prior_query}"
    return res


def _build_non_tool_route(domain: str, query_mode: str, user_query: str) -> Dict[str, Any]:
    """Build normalized non-tool AgentState delta (general / out_of_scope)."""
    return {
        "detected_domain": domain,
        "detected_tool": None,
        "query_mode": query_mode,
        "rag_query": None,
        "rag_queries": [],
        "is_multi_tool": False,
        "detected_domains": [domain],
        "detected_tools": [],
        "active_tools": [],
        "available_tools": [],
        "sub_questions": [user_query],
        "non_tool_questions": [],
        "needs_clarification": False,
        "clarification_question": None,
        "clarification_options": [],
    }


def check_hard_guardrails(
    user_query: str,
    state: AgentState,
    clauses: List[str],
    is_retry: bool = False,
    prior_query: Optional[str] = None,
    contextual_analysis_fn: Optional[Callable[[str, AgentState], Optional[Dict[str, Any]]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Evaluate the 4 zero-risk deterministic hard guardrails (+ P&L drill-down fast-path):
      1. Pure Greeting / Help / Session Meta
      2. Write-Operation Block, Explicit Out-of-Scope, & Explicit Customer Billing
      3. Strict Document Identifier Fast-Path (single-clause only)
      4. Contextual Analysis & P&L Account Drill-Down Short-Circuit
    """
    cleaned = (user_query or "").lower().strip()
    if not cleaned:
        return _build_non_tool_route("general", "general", user_query)

    # 1. Pure greeting, bot help/capabilities, or session meta-question (<5ms, zero LLM cost)
    if is_pure_greeting(cleaned) or is_general_help_inquiry(cleaned) or is_meta_question(cleaned):
        return _build_non_tool_route("general", "general", user_query)

    # 2. Hard security / scope blocks (only short-circuit whole query when single-clause)
    if len(clauses) <= 1:
        if _match_any(OUT_OF_SCOPE_PATTERNS, cleaned) or is_hard_write_block_query(cleaned):
            return _build_non_tool_route("out_of_scope", "out_of_scope", user_query)
        if is_explicit_customer_billing_query(cleaned):
            return _build_non_tool_route("general", "general", user_query)

    # 3. Strict document identifier fast-path (single-clause only to avoid dropping 2nd intent)
    if len(clauses) <= 1:
        doc_match = match_strict_document_identifier(user_query)
        if doc_match is not None:
            tool_name, domain = doc_match
            return _build_single_tool_route(
                tool_name=tool_name,
                domain=domain,
                user_query=user_query,
                is_retry=is_retry,
                prior_query=prior_query,
            )

    # 4. Contextual analytical follow-up & P&L/Balance-Sheet account drill-down fast-paths
    if not is_retry:
        if contextual_analysis_fn is not None:
            ctx_res = contextual_analysis_fn(user_query, state)
            if ctx_res is not None:
                ctx_res.setdefault("needs_clarification", False)
                ctx_res.setdefault("clarification_question", None)
                ctx_res.setdefault("clarification_options", [])
                return ctx_res

        history_messages = state.get("messages") or []
        if len(history_messages) > 1:
            drill_match = resolve_pnl_drilldown_from_history(user_query, history_messages)
            if drill_match:
                drill_res = _build_single_tool_route(
                    tool_name="get_general_ledger",
                    domain="accounting",
                    user_query=user_query,
                )
                if (
                    resolve_branch_company_id(user_query) is None
                    and not resolve_all_branch_company_ids(user_query)[1]
                ):
                    prior_q_hist = find_previous_user_query(history_messages)
                    if prior_q_hist:
                        prior_cid = resolve_branch_company_id(prior_q_hist)
                        if prior_cid is not None:
                            drill_res["company_id"] = prior_cid
                            drill_res["company_ids"] = [prior_cid]
                            drill_res["is_multi_company"] = False
                return drill_res

    return None


def _is_short_followup_or_clarification_reply(
    user_query: str,
    state: AgentState,
) -> bool:
    """Detect short follow-ups ('di mainstore', 'bulan lalu') or replies to prior clarification."""
    history_messages = state.get("messages") or []
    if len(history_messages) <= 1:
        return False

    words = [w for w in re.findall(r"[a-z0-9]+", (user_query or "").lower()) if w]
    if len(words) <= 5:
        return True
    if has_anaphoric_reference(user_query):
        return True

    # Check if the preceding assistant message ended with a question mark (likely a clarification prompt)
    for msg in reversed(history_messages[:-1]):
        if isinstance(msg, AIMessage) or getattr(msg, "type", "") in ("ai", "assistant"):
            content = str(getattr(msg, "content", "") or "").strip()
            if content.endswith("?"):
                return True
            break
    return False


def _build_retrieval_queries(
    user_query: str,
    clauses: List[str],
    state: AgentState,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Build per-clause retrieval queries plus optional prior-tool context injection so
    short follow-ups and clarification replies never suffer Top-K recall misses.
    """
    queries = [c.strip() for c in clauses if c and c.strip()] or [user_query]
    injected_tools: List[Dict[str, Any]] = []
    registry = get_tool_registry_service()

    if _is_short_followup_or_clarification_reply(user_query, state):
        history_messages = state.get("messages") or []
        prior_q = find_previous_user_query(history_messages)
        if prior_q and prior_q.strip().lower() != user_query.strip().lower():
            enriched = f"{user_query} {prior_q}".strip()
            if enriched not in queries:
                queries.append(enriched)

        prior_tool = state.get("prior_detected_tool")
        if prior_tool:
            meta = registry.get_tool_by_name(prior_tool)
            if meta:
                injected_item = dict(meta)
                injected_item.setdefault("rrf_score", 0.015)
                injected_item.setdefault("distance", None)
                injected_tools.append(injected_item)

    return queries, injected_tools


async def aretrieve_candidate_tools(
    user_query: str,
    clauses: List[str],
    state: AgentState,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Run parallel per-clause hybrid search and fuse candidates via RRF."""
    registry = get_tool_registry_service()
    effective_top_k = top_k if top_k is not None else settings.TOOL_RETRIEVAL_TOP_K
    user_roles = state.get("user_roles")
    retrieval_queries, injected_tools = _build_retrieval_queries(user_query, clauses, state)

    tasks = [
        registry.search_tools_hybrid(
            query=q,
            user_roles=user_roles,
            top_k=effective_top_k,
            distance_threshold=settings.TOOL_RETRIEVAL_DISTANCE_THRESHOLD,
        )
        for q in retrieval_queries
    ]
    clause_results = await asyncio.gather(*tasks, return_exceptions=True)
    valid_lists: List[List[Dict[str, Any]]] = []
    for res in clause_results:
        if isinstance(res, list):
            valid_lists.append(res)
        elif isinstance(res, Exception):
            logger.warning("Per-clause hybrid retrieval failed: %s", res)

    if injected_tools:
        valid_lists.append(injected_tools)

    max_cands = _MAX_SEMANTIC_CANDIDATES if len(retrieval_queries) > 1 else effective_top_k
    return merge_rrf_candidates(valid_lists, max_results=max_cands)


def retrieve_candidate_tools_sync(
    user_query: str,
    clauses: List[str],
    state: AgentState,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Synchronous counterpart of aretrieve_candidate_tools for sync router_node."""
    registry = get_tool_registry_service()
    effective_top_k = top_k if top_k is not None else settings.TOOL_RETRIEVAL_TOP_K
    user_roles = state.get("user_roles")
    retrieval_queries, injected_tools = _build_retrieval_queries(user_query, clauses, state)

    valid_lists: List[List[Dict[str, Any]]] = []
    for q in retrieval_queries:
        try:
            res = registry.search_tools_hybrid_sync(
                query=q,
                user_roles=user_roles,
                top_k=effective_top_k,
                distance_threshold=settings.TOOL_RETRIEVAL_DISTANCE_THRESHOLD,
            )
            valid_lists.append(res)
        except Exception as exc:
            logger.warning("Sync per-clause hybrid retrieval failed for '%s': %s", q, exc)

    if injected_tools:
        valid_lists.append(injected_tools)

    max_cands = _MAX_SEMANTIC_CANDIDATES if len(retrieval_queries) > 1 else effective_top_k
    return merge_rrf_candidates(valid_lists, max_results=max_cands)


def _format_candidate_tool_cards(candidates: List[Dict[str, Any]]) -> str:
    """Render compact Top-K candidate tool cards (description, key params, negative_hints)."""
    if not candidates:
        return "(Tidak ada kandidat tool ERP yang cocok secara langsung dengan query ini.)"

    lines: List[str] = []
    for idx, cand in enumerate(candidates, start=1):
        t_name = cand.get("tool_name", "")
        domain = cand.get("domain", "")
        desc = cand.get("description", "")
        neg = (cand.get("negative_hints") or "").strip()

        params_schema = cand.get("params_schema") or {}
        props = params_schema.get("properties", {}) if isinstance(params_schema, dict) else {}
        param_names = [p for p in props.keys() if p != "company_id"]
        required_params = [
            r for r in (params_schema.get("required") or [])
            if isinstance(params_schema, dict) and r != "company_id"
        ]

        card = f"{idx}. {t_name} [{domain}]: {desc}"
        if param_names:
            req_tag = f" (wajib: {', '.join(required_params)})" if required_params else ""
            card += f"\n   Params: {', '.join(param_names)}{req_tag}"
        if neg:
            card += f"\n   Bukan untuk: {neg}"
        lines.append(card)
    return "\n".join(lines)


def build_scoped_router_system_prompt(
    candidates: List[Dict[str, Any]],
    state: AgentState,
    clauses: List[str],
) -> str:
    """Construct dynamic Scoped LLM Router system prompt from Top-K retrieved candidates."""
    tool_cards_block = _format_candidate_tool_cards(candidates)

    prior_context_lines: List[str] = []
    if state.get("prior_detected_tool"):
        prior_context_lines.append(
            f"- Tool sebelumnya: {state['prior_detected_tool']} (domain: {state.get('prior_detected_domain')})"
        )
    if state.get("company_id"):
        prior_context_lines.append(f"- Active company_id sesi: {state['company_id']}")
    if len(clauses) > 1:
        prior_context_lines.append(f"- Terdeteksi {len(clauses)} klausa/sub-pertanyaan: {clauses}")
    context_block = "\n".join(prior_context_lines) if prior_context_lines else "- Tidak ada konteks tool sebelumnya."

    return (
        "Anda adalah NLU Intent & Tool Router untuk FastPrint ERP AI Orchestrator (READ-ONLY).\n"
        "Tugas Anda adalah memetakan permintaan pengguna ke mode eksekusi dan tool ERP yang PALING TEPAT "
        "berdasarkan daftar kandidat tool hasil retrieval di bawah ini serta riwayat percakapan.\n\n"
        f"KANDIDAT TOOL (Top-K Retrieval):\n{tool_cards_block}\n\n"
        f"KONTEKS SESI:\n{context_block}\n\n"
        "MODE EKSEKUSI (query_mode & domain):\n"
        "1. 'live_erp': Pertanyaan data operasional/angka real-time yang cocok dengan >=1 tool pada KANDIDAT TOOL. "
        "Isi `planned_tools` (bisa lebih dari 1 untuk multi-intent; isi `sub_query` sesuai klausa relevan).\n"
        "2. 'vector_rag' (domain='knowledge'): Pertanyaan kualitatif tentang SOP, kebijakan, aturan retur, garansi, "
        "spesifikasi teknis produk (GSM/ketebalan), atau termin pembayaran. Set `tool=null`, `planned_tools=[]`, isi `rag_query`.\n"
        "3. 'hybrid': Gabungan pertanyaan SOP/kebijakan DAN data ERP live. Isi `planned_tools` DAN `rag_query`.\n"
        "4. 'contextual_analysis': Pertanyaan lanjutan analitis ('kenapa naik?', 'jelaskan angkanya') atas data laporan "
        "yang sudah ditampilkan di riwayat percakapan tanpa meminta cabang/periode/tool baru.\n"
        "5. 'clarification' (domain='clarification', needs_clarification=true): HANYA sesuai ATURAN KLARIFIKASI KONSERVATIF di bawah.\n"
        "6. 'general': HANYA sapaan murni, terima kasih, pertanyaan kemampuan bot, atau meta-pertanyaan sesi. `tool=null`, `planned_tools=[]`.\n"
        "7. 'out_of_scope': Topik non-ERP (trivia, cuaca, resep, coding, berita), modul tidak didukung (HR/payroll/cuti), "
        "atau permintaan tulis/mutasi data (push/ubah stok/harga, buat/hapus jurnal/order). `tool=null`, `planned_tools=[]`.\n\n"
        "ATURAN KLARIFIKASI KONSERVATIF (WAJIB DIPATUHI):\n"
        "- Aktifkan `query_mode='clarification'` dan `needs_clarification=true` HANYA JIKA:\n"
        "  (a) Query bentrok kuat antara >= 2 kandidat tool berbeda makna "
        "(contoh: 'cek order minggu ini' -> get_sales_orders vs get_marketplace_orders; "
        "'cek mutasi bulan ini' -> get_stock_history vs get_general_ledger; "
        "'cek tagihan <nama>' tanpa konteks vendor vs pelanggan -> get_vendor_bills vs get_sales_orders).\n"
        "  (b) Tool target membutuhkan objek/identifier wajib yang tidak ada di query maupun riwayat "
        "(contoh: 'cek detail SO' tanpa nomor SO/resi, atau 'cek kartu stok' / 'riwayat perubahan harga' tanpa nama produk/SKU).\n"
        "- DILARANG KERAS meminta klarifikasi hanya karena cabang/company atau rentang tanggal tidak disebutkan "
        "(sistem selalu memakai default company_id sesi dan periode default tool).\n"
        "- Jika riwayat percakapan sudah memuat jawaban klarifikasi pengguna (misal user membalas 'yang marketplace' atau mengklik chip), "
        "langsung eksekusi tool target tanpa bertanya ulang.\n"
        "- Saat meminta klarifikasi: tulis `clarification_question` 1-2 kalimat singkat Bahasa Indonesia, "
        "dan berikan 2-4 `clarification_options` (`label` 2-5 kata TANPA emoji, `query` prompt lengkap siap eksekusi).\n\n"
        "GUARDRAIL WAJIB:\n"
        "- Pilih nama tool HANYA dari daftar KANDIDAT TOOL di atas (jangan mengarang nama tool baru).\n"
        "- Perhatikan catatan 'Bukan untuk' pada setiap kandidat tool sebelum memilih.\n"
        "- Permintaan tagihan/piutang pelanggan eksplisit BUKAN `get_vendor_bills`/`fetch_vendor_invoices` -> set `domain='general'`, `query_mode='general'`."
    )


def _build_state_from_semantic_decision(
    decision: Any,
    user_query: str,
    clauses: List[str],
    candidates: List[Dict[str, Any]],
    state: AgentState,
    is_retry: bool,
    prior_query: Optional[str],
    enforce_general_fn: Callable[..., Tuple[str, str]],
) -> Dict[str, Any]:
    """Post-validate MultiRouterDecision and convert it into an AgentState update delta."""
    registry = get_tool_registry_service()
    detected_domain = getattr(decision, "domain", "out_of_scope")
    query_mode = getattr(decision, "query_mode", "out_of_scope")
    detected_tool = getattr(decision, "tool", None)
    rag_query = getattr(decision, "rag_query", None)
    needs_clarification = bool(getattr(decision, "needs_clarification", False))
    raw_planned_tools = getattr(decision, "planned_tools", None) or []

    # 1. Conservative Disambiguation / Clarification branch
    if needs_clarification or query_mode == "clarification" or detected_domain == "clarification":
        question = str(getattr(decision, "clarification_question", None) or "").strip()
        if not question:
            question = "Mohon perjelas permintaan Anda dengan memilih salah satu opsi di bawah ini."
        raw_opts = getattr(decision, "clarification_options", None) or []
        serialized_opts: List[Dict[str, str]] = []
        for opt in raw_opts[:4]:
            if hasattr(opt, "model_dump"):
                serialized_opts.append(opt.model_dump())
            elif isinstance(opt, dict):
                serialized_opts.append({
                    "label": str(opt.get("label") or "").strip(),
                    "query": str(opt.get("query") or "").strip(),
                    "icon": str(opt.get("icon") or "fa-arrow-right").strip(),
                })
        return {
            "detected_domain": "clarification",
            "detected_tool": None,
            "query_mode": "clarification",
            "verification_status": "CLARIFICATION",
            "rag_query": None,
            "rag_queries": [],
            "is_multi_tool": False,
            "detected_domains": ["clarification"],
            "detected_tools": [],
            "active_tools": [],
            "available_tools": candidates,
            "sub_questions": clauses,
            "non_tool_questions": [],
            "needs_clarification": True,
            "clarification_question": question,
            "clarification_options": serialized_opts,
        }

    # 2. Clean tools for non-tool domains/modes
    if detected_domain in ("general", "out_of_scope", "knowledge") or query_mode in (
        "vector_rag",
        "contextual_analysis",
    ):
        detected_tool = None
        raw_planned_tools = []

    active_tools: List[Dict[str, Any]] = []
    for idx, pt in enumerate(raw_planned_tools):
        pt_tool = pt.tool if hasattr(pt, "tool") else pt.get("tool")
        if not pt_tool:
            continue
        registered = registry.get_tool_by_name(pt_tool)
        if pt_tool not in TOOL_DOMAIN_MAP and registered is None:
            logger.warning("Semantic router ignored unregistered LLM tool '%s'", pt_tool)
            continue
        pt_domain = TOOL_DOMAIN_MAP.get(pt_tool) or (registered.get("domain") if registered else None)
        if pt_domain not in ("accounting", "inventory", "sales", "marketplace"):
            continue
        pt_sub_query = (pt.sub_query if hasattr(pt, "sub_query") else pt.get("sub_query")) or user_query
        active_tools.append({
            "tool": pt_tool,
            "domain": pt_domain,
            "sub_query": pt_sub_query,
            "question_index": idx,
            "question_text": pt_sub_query,
            "params": {},
            "status": "pending",
        })

    if not active_tools and detected_tool:
        registered = registry.get_tool_by_name(detected_tool)
        resolved_domain = TOOL_DOMAIN_MAP.get(detected_tool) or (
            registered.get("domain") if registered else None
        )
        if resolved_domain in ("accounting", "inventory", "sales", "marketplace"):
            active_tools.append({
                "tool": detected_tool,
                "domain": resolved_domain,
                "sub_query": user_query,
                "question_index": 0,
                "question_text": user_query,
                "params": {},
                "status": "pending",
            })

    # Post-LLM deterministic general vs out_of_scope guardrail
    detected_domain, query_mode = enforce_general_fn(
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
        if query_mode not in ("live_erp", "hybrid"):
            query_mode = "hybrid" if rag_query else "live_erp"
    else:
        detected_tool = None
        detected_tools = []
        detected_domains = [detected_domain] if detected_domain else []
        is_multi_tool = False
        if query_mode == "hybrid":
            query_mode = "vector_rag" if rag_query else "out_of_scope"

    eff_rag_query = rag_query or (user_query if query_mode in ("vector_rag", "hybrid") else None)
    avail_tools: List[Dict[str, Any]] = []
    seen_avail: set[str] = set()
    for t_name in detected_tools:
        meta = registry.get_tool_by_name(t_name)
        if meta and t_name not in seen_avail:
            seen_avail.add(t_name)
            avail_tools.append(meta)
    for cand in candidates:
        c_name = cand.get("tool_name")
        if c_name and c_name not in seen_avail:
            seen_avail.add(c_name)
            avail_tools.append(cand)

    result: Dict[str, Any] = {
        "detected_domain": detected_domain,
        "detected_tool": detected_tool,
        "query_mode": query_mode,
        "rag_query": eff_rag_query,
        "rag_queries": [eff_rag_query] if eff_rag_query else [],
        "is_multi_tool": is_multi_tool,
        "detected_domains": detected_domains,
        "detected_tools": detected_tools,
        "active_tools": active_tools,
        "available_tools": avail_tools,
        "sub_questions": clauses,
        "non_tool_questions": [],
        "needs_clarification": False,
        "clarification_question": None,
        "clarification_options": [],
    }
    if query_mode == "contextual_analysis":
        result["verification_status"] = "VALID"
    if is_retry and prior_query:
        result["user_query"] = prior_query
        result["retry_instruction"] = f"Mengulang permintaan: {prior_query}"
    return result


async def aroute_semantic_nlu(
    state: AgentState,
    user_query: str,
    get_model_fn: Callable[[], Optional[BaseChatModel]],
    is_retry: bool = False,
    prior_query: Optional[str] = None,
    contextual_analysis_fn: Optional[Callable[[str, AgentState], Optional[Dict[str, Any]]]] = None,
    enforce_general_fn: Optional[Callable[..., Tuple[str, str]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Asynchronous Hybrid Semantic NLU Router pipeline:
      Hard Guardrails -> Per-Clause Hybrid Retrieval -> Dynamic Scoped LLM Router.
    Returns None if LLM is unavailable or fails so caller can fall back cleanly to legacy_regex.
    """
    from app.graph.nodes.router import MultiRouterDecision, _enforce_general_vs_out_of_scope

    clauses = decompose_user_query(user_query) or [user_query]
    guardrail_res = check_hard_guardrails(
        user_query=user_query,
        state=state,
        clauses=clauses,
        is_retry=is_retry,
        prior_query=prior_query,
        contextual_analysis_fn=contextual_analysis_fn,
    )
    if guardrail_res is not None:
        return guardrail_res

    model = get_model_fn()
    if model is None:
        logger.debug("Semantic NLU router skipped: no chat model available.")
        return None

    candidates = await aretrieve_candidate_tools(
        user_query=user_query,
        clauses=clauses,
        state=state,
    )
    system_prompt = build_scoped_router_system_prompt(candidates=candidates, state=state, clauses=clauses)
    raw_history = (state.get("messages") or [])[:-1]
    history_msgs = trim_conversation_messages(
        raw_history,
        max_messages=min(settings.MAX_WINDOW_MESSAGES, 6),
    )
    messages = [
        SystemMessage(content=system_prompt),
        *history_msgs,
        HumanMessage(content=user_query),
    ]

    try:
        structured_model = model.with_structured_output(MultiRouterDecision)
        decision = await structured_model.ainvoke(
            messages,
            config={"tags": ["internal_no_stream"]},
        )
    except Exception as exc:
        logger.warning("Async Scoped LLM Router failed for '%s': %s", user_query, exc)
        return None

    return _build_state_from_semantic_decision(
        decision=decision,
        user_query=user_query,
        clauses=clauses,
        candidates=candidates,
        state=state,
        is_retry=is_retry,
        prior_query=prior_query,
        enforce_general_fn=enforce_general_fn or _enforce_general_vs_out_of_scope,
    )


def route_semantic_nlu_sync(
    state: AgentState,
    user_query: str,
    get_model_fn: Callable[[], Optional[BaseChatModel]],
    is_retry: bool = False,
    prior_query: Optional[str] = None,
    contextual_analysis_fn: Optional[Callable[[str, AgentState], Optional[Dict[str, Any]]]] = None,
    enforce_general_fn: Optional[Callable[..., Tuple[str, str]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Synchronous counterpart of aroute_semantic_nlu for sync router_node execution.
    Returns None if LLM is unavailable or fails so caller can fall back cleanly to legacy_regex.
    """
    from app.graph.nodes.router import MultiRouterDecision, _enforce_general_vs_out_of_scope

    clauses = decompose_user_query(user_query) or [user_query]
    guardrail_res = check_hard_guardrails(
        user_query=user_query,
        state=state,
        clauses=clauses,
        is_retry=is_retry,
        prior_query=prior_query,
        contextual_analysis_fn=contextual_analysis_fn,
    )
    if guardrail_res is not None:
        return guardrail_res

    model = get_model_fn()
    if model is None:
        logger.debug("Sync semantic NLU router skipped: no chat model available.")
        return None

    candidates = retrieve_candidate_tools_sync(
        user_query=user_query,
        clauses=clauses,
        state=state,
    )
    system_prompt = build_scoped_router_system_prompt(candidates=candidates, state=state, clauses=clauses)
    raw_history = (state.get("messages") or [])[:-1]
    history_msgs = trim_conversation_messages(
        raw_history,
        max_messages=min(settings.MAX_WINDOW_MESSAGES, 6),
    )
    messages = [
        SystemMessage(content=system_prompt),
        *history_msgs,
        HumanMessage(content=user_query),
    ]

    try:
        structured_model = model.with_structured_output(MultiRouterDecision)
        decision = structured_model.invoke(messages)
    except Exception as exc:
        logger.warning("Sync Scoped LLM Router failed for '%s': %s", user_query, exc)
        return None

    return _build_state_from_semantic_decision(
        decision=decision,
        user_query=user_query,
        clauses=clauses,
        candidates=candidates,
        state=state,
        is_retry=is_retry,
        prior_query=prior_query,
        enforce_general_fn=enforce_general_fn or _enforce_general_vs_out_of_scope,
    )
