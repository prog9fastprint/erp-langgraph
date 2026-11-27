# Codebase Audit Report — `erp-langgraph`

**Baseline Audit:** Read-only full audit of `app/`, `cli.py`, `scripts/`, `requirements.txt`, `Dockerfile`, `docker-compose.yml`, `.github/workflows/deploy.yml`, `.env.example` (678 tests passing in 45.8s).

**Post-Change Verification Audit:** Verified full working-tree diff (`59` tracked files modified/deleted, plus modular subpackages `app/core/branches.py`, `app/graph/nodes/formatters/`, `app/graph/nodes/extractors/`, `app/graph/nodes/router_patterns.py`, `app/graph/nodes/router_decomposer.py`, [pyproject.toml](file:///D:/erp-fastprint/erp-langgraph/pyproject.toml), and [test_audit_remediations.py](file:///D:/erp-fastprint/erp-langgraph/tests/test_audit_remediations.py)), full test suite (`700/700` passing in `19.2s`, exit `0`, `-58%` runtime, `0` outbound network/DB leaks), repo-wide `ruff check app cli.py scripts tests` (`All checks passed!`, exit `0`), and `codegraph sync` (`Done`). Changes are uncommitted on `HEAD` (`40b13b4`).

**Assumed Load:** Multi-user FastAPI production service (port 8002) with concurrent HTTP + SSE chat streams across multiple ERP companies, plus single-user CLI/ingestion scripts.

---

## Remediation Summary

### Overall Status (30 Findings)

| Status | Count | Findings |
|---|---|---|
| **Fixed** | **25** | `#1, #2, #5, #6, #7, #8, #9, #10, #11, #12, #14, #15, #16, #17, #18, #19, #20, #21, #22, #23, #24, #26, #28, #29, #30` |
| **Partially Fixed (Accepted Tradeoff)** | **5** | `#3, #4, #13, #25, #27` |
| **Residual Gaps / Deferred** | **0** | None |

### Effect by Audit Category

| Category | Completed Effect | Accepted Tradeoffs |
|---|---|---|
| **Dead Code Removal** | Deleted all 6 deprecated/unused files (`agent.py`, `erp_client.py`, `app/tools/*`, `prompts/erp_system_prompt.md`, `-880 LOC`) + `COPY prompts` in `Dockerfile`; deleted 2 unused Pydantic schemas, 3 dead config settings, `python-dotenv` entry, all `109+` unused imports across `app/`, `cli.py`, `scripts/`, and `tests/`, plus dead ternaries, unused locals, and subsumed regexes. | `*ResponseSchema` / `*DataSchema` (`~497L` in [tools.py](file:///D:/erp-fastprint/erp-langgraph/app/schemas/tools.py)) intentionally kept as Django response contract tests (`#27`). |
| **Duplicate / Redundant Logic** | Unified 3 record counters into `context.resolve_tool_record_count()`; consolidated duplicate PGVector DDL into `VectorService.initialize_table()`; deduplicated all 6 verbatim duplication pairs across pipeline nodes; unified 4 sales date extractors into `_apply_sales_date_filters()`; replaced 33 copy-pasted `DjangoCoreClient.fetch_*` methods (`-497L`) and per-request `tool_node.py` `dispatch_map` with declarative binding from `TOOL_DOMAIN_MAP`. | None. |
| **Bugs, Security & Bad Practices** | Fixed runtime `NameError` (`#1`), `OPENAI_API_KEY` OpenRouter leak (`#2`), unverified `user_roles` spoofing (`#3`), hardcoded login secret (`#4`), SSE stream token/JSON/error bugs (`#5`), sales date/limit corruption (`#6`), unreachable multi-company retry (`#7`), greeting hijacking (`#8`), non-transactional RAG delete (`#9`), cross-loop `asyncpg` crash (`#10`), lock leak (`#11`), eager LLM init (`#12`), branch false positives including `"ms word/excel/office"` (`#13`), multi-company stock `products` key (`#14`), parallel peer backfill (`#19`), token cache scoping (`#20`), streaming circuit-breaker gaps (`#21`), SQL table identifier validation (`#22`), `None` metadata & state typing (`#23`), container/deploy/docs hardening (`#25`), hermetic test `.env` isolation (`#26`), repo-wide `ruff check` in CI (`#28`), and CLI/queue/logging bugs (`#30`). | Dev/test auth defaults & `ERP_API_TOKEN` fallback (`#3`); MD5 login wire format awaiting Django Core HMAC migration (`#4`); 3-letter branch shorthand codes (`#13`); `docker-compose.yml` `network_mode: "host"` (`#25`). |
| **Maintainability & Verification** | Extracted [branches.py](file:///D:/erp-fastprint/erp-langgraph/app/core/branches.py), split `synthesizer.py` (`3,864L -> 1,657L`) into `formatters/` dispatch dicts, split `extractor.py` (`2,855L -> 1,144L`) into `extractors/`, split `router.py` (`2,183L -> 1,806L`) into `router_patterns.py` + `router_decomposer.py`, shrunk `django_client.py` (`1,099L -> 602L`), and removed cross-node imports from `extractor.py` / `verifier.py`. Tests: `678 -> 700` passing, runtime `45.8s -> 19.2s` (`-58%`), `0` network/DB leaks, `0` ruff errors. | All public & test-imported symbols re-exported cleanly for zero test breakage. |

---

## High Priority

### 1. `[FIXED]` Runtime `NameError` Crash in Multi-Company `get_sales_orders` Formatter
- **Category:** Anti-pattern / Bug
- **Original Issue:** `_format_multi_company_presentation(state)` referenced undefined `user_query.lower()` when `ext_params.get("sort_by") != "amount_desc"`, crashing with `NameError`.
- **Change Made:** Replaced `user_query.lower()` with `(state.get("user_query") or "").lower()` (now in [multi_company.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/formatters/multi_company.py#L508-L511)).
- **Effect & Verification:** Multi-company `get_sales_orders` without `sort_by="amount_desc"` sorts and renders cleanly. Verified by `test_1_multi_company_sales_orders_no_name_error` in [test_audit_remediations.py](file:///D:/erp-fastprint/erp-langgraph/tests/test_audit_remediations.py#L54-L95).

### 2. `[FIXED]` `OPENAI_API_KEY` Exfiltrated to OpenRouter on Startup and Fallback
- **Category:** Security
- **Original Issue:** `_sync_fetch_free_models_catalog` and `get_chat_model("openrouter")` fell back to `settings.OPENAI_API_KEY`, and `app/main.py` lifespan called `afetch_free_models_catalog()` unconditionally.
- **Change Made:** Removed `or settings.OPENAI_API_KEY` fallback in [llm.py](file:///D:/erp-fastprint/erp-langgraph/app/services/llm.py#L131) and [llm.py](file:///D:/erp-fastprint/erp-langgraph/app/services/llm.py#L370). Gated lifespan catalog pre-warm in [main.py](file:///D:/erp-fastprint/erp-langgraph/app/main.py#L53-L59) on `OPENROUTER_API_KEY` or `MODEL_PROVIDER == "openrouter"`.
- **Effect & Verification:** OpenAI-only deployments never send `OPENAI_API_KEY` to `openrouter.ai`. Verified by `test_2_openai_key_never_leaks_to_openrouter` in [test_audit_remediations.py](file:///D:/erp-fastprint/erp-langgraph/tests/test_audit_remediations.py#L285-L298).

### 3. `[PARTIALLY FIXED]` Untrusted Client RBAC Roles (`user_roles`) and Permissive Auth Defaults
- **Category:** Security
- **Original Issue:** Client JSON `ChatRequest.user_roles` was trusted unconditionally (allowing `{"user_roles": ["admin"]}` RBAC bypass), and `REQUIRE_USER_TOKEN` / `REQUIRE_SERVICE_KEY` defaulted to `False`.
- **Change Made:** Updated `_resolve_and_validate_auth()` ([main.py](file:///D:/erp-fastprint/erp-langgraph/app/main.py#L182-L224)) to verify `X-Internal-Service-Key` and enforce JWT tokens in both `production` and `staging` (`L215-L218`). Updated `_build_workflow_kwargs()` ([main.py](file:///D:/erp-fastprint/erp-langgraph/app/main.py#L227-L255)) to strip client-supplied `user_roles` to `[]` whenever `service_key_verified` is `False`.
- **Effect & Verification:** Closes body-level RBAC role spoofing in all environments and blocks unauthenticated staging requests (`HTTP 401`). Verified by `test_3_unverified_user_roles_stripped_and_staging_enforced`.
- **Accepted Tradeoff:** `REQUIRE_USER_TOKEN` / `REQUIRE_SERVICE_KEY` stay `False` in `development`/`test` and `ERP_API_TOKEN` service-key fallback stays intact for backward compatibility.

### 4. `[PARTIALLY FIXED]` Hardcoded ERP Login Secret with Predictable MD5 Key
- **Category:** Security / Redundant Logic
- **Original Issue:** Static `"erp_master_nexus-{now}"` MD5 secret was hardcoded in both `django_client.py` and `erp_client.py`.
- **Change Made:** Added `ERP_LOGIN_SECRET` setting in [config.py](file:///D:/erp-fastprint/erp-langgraph/app/core/config.py#L43) and [.env.example](file:///D:/erp-fastprint/erp-langgraph/.env.example#L32). Updated `_generate_secret_key()` in [django_client.py](file:///D:/erp-fastprint/erp-langgraph/app/services/django_client.py#L240-L250) to require `settings.ERP_LOGIN_SECRET` and raise `DjangoAuthError` when unset. Deleted duplicate `erp_client.py` (`#16`).
- **Effect & Verification:** Zero hardcoded `"erp_master_nexus"` references remain across the repository. Verified by `test_4_erp_login_secret_required`.
- **Accepted Tradeoff:** Wire format still uses time-based MD5 pending coordinated Django Core `/external_api/get_token/` migration to HMAC-SHA256. **Deployment action:** configure `ERP_LOGIN_SECRET` in `.env` wherever auto-login is active.

### 5. `[FIXED]` Broken SSE Streaming: Duplicate Tokens, Raw `json:followups` Leak, and Double `event: error`
- **Category:** Anti-pattern / Bug
- **Original Issue:** `/api/v1/chat/stream` streamed deterministic tokens followed by raw LLM chunks (including ````json:followups``` JSON), duplicated `general_nt_items`, and emitted two `event: error` frames on failure.
- **Change Made:**
  - Tagged internal/general/followup/polish `model.ainvoke()` calls in `asynthesizer_node` with `config={"tags": ["internal_no_stream"]}` ([synthesizer.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/synthesizer.py#L1468-L1505)).
  - Filtered `internal_no_stream` events in `astream_workflow_events` ([workflow.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/workflow.py#L868-L870)).
  - Dispatched immediate `<10ms` tokens only for non-polished paths and emitted clean post-processed `out_dict["final_answer"]` once for polished responses.
  - Removed duplicate pre-raise `yield ("error", ...)` in [workflow.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/workflow.py) so [main.py](file:///D:/erp-fastprint/erp-langgraph/app/main.py#L403-L410) emits a single sanitized error frame.
- **Effect & Verification:** Concatenated `token` events match `complete.final_answer` with zero `json:followups` leakage and single `event: error` frames. Verified by `test_5_sse_streaming_suppresses_internal_tokens_and_duplicate_errors` and verified `RunnableWithFallbacks` tag propagation.

### 6. `[FIXED]` Sales Fallback Extractors Corrupt Explicit Dates and Truncate `limit`
- **Category:** Anti-pattern / Bug
- **Original Issue:** Four sales extractors ran `\b(20\d\d)\b` before `resolve_relative_dates()`, turning explicit dates like `2026-03-15` into full-year ranges; `_fallback_extract_sales_orders` treated date day numbers (`tanggal 5 maret 2026`) as `limit=5`.
- **Change Made:** Added shared `_apply_sales_date_filters()` ([sales.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/extractors/sales.py#L113-L135)) running `resolve_relative_dates()` first and isolating bare 4-digit years via `(?<![-/\d])\b(20\d\d)\b(?![-/\d])`; wired all four sales extractors. Required explicit prefix or trailing noun in `_fallback_extract_sales_orders` `limit` regex.
- **Effect & Verification:** Explicit single/range ISO dates stay intact and day-of-month integers no longer overwrite `limit=20`. Verified by `test_6_explicit_iso_dates_and_limit_extraction`.

### 7. `[FIXED]` Multi-Company Verifier Marks All-Empty Responses as `VALID` (Unreachable Retry Branch)
- **Category:** Bug / Dead Branch
- **Original Issue:** `any(r is not None for r in mc_results.values())` treated empty dicts `{1: {"entries": [], "total_count": 0}}` as `VALID`, making retry and `EMPTY` unreachable.
- **Change Made:** Removed the non-`None` short-circuit in Path MC ([verifier.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/verifier.py#L89)) and multi-tool Path A ([verifier.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/verifier.py#L156)).
- **Effect & Verification:** All-empty multi-company payloads now transition through `RETRY` -> `EMPTY`, while populated branch payloads (including `{"stocks": [...]}`) evaluate to `VALID`. Verified by `test_7_all_empty_multi_company_triggers_retry_and_empty`.

### 8. `[FIXED]` Greeting Fast-Path Hijacks Valid Sales and Inventory Queries
- **Category:** Bug
- **Original Issue:** Upfront `has_erp_intent` allowlist omitted all `SALES_*` patterns and 7 Inventory pattern groups, misrouting `"Halo, cek daftar pesanan hari ini"` as `general`.
- **Change Made:** Deleted the manual allowlist in [router.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/router.py#L634-L645) and deferred the greeting/general return to Step 5 after all ERP/RAG candidate matching runs.
- **Effect & Verification:** Greetings prefixed to Sales or Inventory queries route to their ERP tools while plain greetings stay `general`. Verified by `test_8_greetings_with_sales_or_inventory_intent_route_to_erp`.

### 9. `[FIXED]` Non-Transactional Delete-Then-Insert in Knowledge Ingestion
- **Category:** Anti-pattern / Data Loss
- **Original Issue:** `ingest_document()` deleted existing chunks before calling Gemini embeddings and inserted without a transaction, leaving partial wipes on mid-file API errors.
- **Change Made:** Computed all chunk embeddings upfront before connecting to PostgreSQL, then wrapped `DELETE` + `INSERT` inside `async with conn.transaction():` in [ingest_knowledge.py](file:///D:/erp-fastprint/erp-langgraph/scripts/ingest_knowledge.py#L91-L137).
- **Effect & Verification:** Embedding failures (429/timeout) exit before touching PostgreSQL; DB writes are atomic. Verified by `test_9_ingest_document_transactional_no_wipe_on_embed_failure`.

### 10. `[FIXED]` Cross-Event-Loop `asyncpg` Pool Failure in `search_tools_sync`
- **Category:** Bug / Performance
- **Original Issue:** Sync `router_node` spawned a second event loop for `search_tools_sync()`, reusing `VectorService._pool` bound to FastAPI's main loop (`RuntimeError`).
- **Change Made:** Scoped `VectorService` pools and pool locks per event loop via `WeakKeyDictionary` ([vector_service.py](file:///D:/erp-fastprint/erp-langgraph/app/services/vector_service.py#L37-L108)). Added async `arouter_node` ([router.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/router.py#L1192-L1206)) awaiting `registry.search_tools(...)` directly and registered it in `create_workflow_graph()` ([workflow.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/workflow.py#L143)).
- **Effect & Verification:** Production graph runs semantic tool lookup directly on the active loop with zero thread-spawn or cross-loop pool collisions; sync callers get isolated loop pools. Verified by `test_10_vector_service_per_loop_pool_and_async_router`.

---

## Medium Priority

### 11. `[FIXED]` Unbounded Memory Leak in `_THREAD_LOCKS_BY_LOOP`
- **Category:** Performance / Scale
- **Original Issue:** `_THREAD_LOCKS_BY_LOOP[loop]` retained every `thread_id` lock permanently.
- **Change Made:** Switched inner map to `weakref.WeakValueDictionary[str, asyncio.Lock]` at [workflow.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/workflow.py#L511-L527).
- **Effect & Verification:** Idle locks are automatically garbage-collected. Verified by `test_11_thread_lock_weakref_reclaimed`.

### 12. `[FIXED]` Eager & Duplicate `get_chat_model()` Instantiation on Fast-Path Queries
- **Category:** Performance / Redundant Logic
- **Original Issue:** `router.py` called `get_chat_model()` eagerly before regex matching (and a second time on fallback); `extractor.py` initialized the LLM before Tier-1 deterministic fast paths.
- **Change Made:** Added lazy cached `_get_model()` closures in `_prepare_router_context()` ([router.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/router.py#L1209-L1231)) and thread-safe `_get_model()` in `extractor_node()` ([extractor.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/extractor.py#L915-L927)).
- **Effect & Verification:** Zero `get_chat_model()` calls on single-clause regex routing and Tier-1 extraction. Verified by `test_12_single_clause_rule_hit_does_not_call_get_chat_model`.

### 13. `[PARTIALLY FIXED]` Unprefixed Short Branch Tokens Hijack `company_id`
- **Category:** Bug / Fragile Logic
- **Original Issue:** Optional branch prefixes caused `"SOP HR tentang cuti"`, `"detail contact Sandy"`, `"latency 200 ms"`, and `"MS Word/Excel/Office"` to resolve wrong company IDs.
- **Change Made:** Moved branch resolution into [branches.py](file:///D:/erp-fastprint/erp-langgraph/app/core/branches.py#L93-L141); required explicit branch prefixes for `_PREFIX_ONLY_BRANCH_KEYS = {"hr", "sandy"}`, skipped digit-preceded `"ms"` units (`200 ms`), and guarded non-branch `"ms"` software phrases (`_NON_BRANCH_MS_FOLLOW_RE`: `ms word`, `ms excel`, `ms office`, `ms powerpoint`, `ms access`, `ms sql`, `ms teams`, `ms windows`, etc.).
- **Effect & Verification:** All reproduced false positives (`SOP HR`, `Sandy`, `200 ms`, `ms excel`, `ms word`, `ms office`) return `None`, while `(MS)`, `MS/SO/...`, `cabang hr`, and `stok sby` keep working. Verified by `test_13_ambiguous_branch_aliases_and_valid_codes`.
- **Accepted Tradeoff:** 3-letter branch codes (`sby`/`jkt`/`bdg`/etc.), `(MS)`, and `HO` remain unprefixed by design to preserve shorthand UX and existing branch tests.

### 14. `[FIXED]` Multi-Company `get_stock_level` Formatter Misses `products` Key
- **Category:** Bug
- **Original Issue:** Multi-company stock formatter only checked `stocks`/`items`, rendering bogus `- / 0 unit` rows for `{"products": [...]}` responses.
- **Change Made:** Updated `_format_mc_get_stock_level` in [multi_company.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/formatters/multi_company.py#L207) to `res.get("products") or res.get("stocks") or res.get("items") or [res]`.
- **Effect & Verification:** Renders real SKU and product names from `products` payloads. Verified by `test_14_multi_company_stock_level_supports_products_key`.

### 15. `[FIXED]` Conflicting PGVector DDL Creates Duplicate HNSW Indexes
- **Category:** Redundant Logic / Performance
- **Original Issue:** `ai_knowledge_chunks` DDL was duplicated between `vector_service.py` and `ingest_knowledge.py` with different schemas and two different HNSW index names.
- **Change Made:** Removed duplicate `CREATE_TABLE_SQL` from [ingest_knowledge.py](file:///D:/erp-fastprint/erp-langgraph/scripts/ingest_knowledge.py#L51-L60) and delegated `init_vector_database()` to `VectorService.initialize_table()`. Unified DDL in [vector_service.py](file:///D:/erp-fastprint/erp-langgraph/app/services/vector_service.py#L155-L176) (`CREATE EXTENSION IF NOT EXISTS vector;`, `embedding vector(768) NOT NULL`, `updated_at`, single HNSW index `idx_ai_knowledge_chunks_embedding`).
- **Effect & Verification:** Single canonical schema and single HNSW index. Verified by `test_10_vector_service_per_loop_pool_and_async_router`.

### 16. `[FIXED]` Deprecated Legacy ReAct Stack & Unreachable Tool Definitions (~880 Lines Dead Code)
- **Category:** Dead Code
- **Original Issue:** Deprecated ReAct graph, static/dynamic tools (with 8 unreachable `@tool` defs), sync `ERPClient`, and unused `prompts/erp_system_prompt.md` sat unused.
- **Change Made:** Deleted `app/graph/agent.py` (`-109L`), `app/tools/static.py` (`-370L`), `app/tools/dynamic.py` (`-104L`), `app/tools/__init__.py` (`-1L`), `app/services/erp_client.py` (`-210L`), and `prompts/erp_system_prompt.md` (`-86L`) along with `COPY prompts ./prompts` in [Dockerfile](file:///D:/erp-fastprint/erp-langgraph/Dockerfile). Cleaned re-exports in [app/graph/__init__.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/__init__.py) and [app/services/__init__.py](file:///D:/erp-fastprint/erp-langgraph/app/services/__init__.py); updated legacy test blocks.
- **Effect & Verification:** `-880 LOC` dead code removed with `0` remaining references across the repository. Verified by `test_16_legacy_react_stack_deleted`.

### 17. `[FIXED]` Diverging Record-Counting Implementations (`context.py` vs `verifier.py` vs `cli.py`)
- **Category:** Redundant Logic / Bug
- **Original Issue:** Three diverging counters disagreed on error envelopes (`context.py` returned `1` on `{"success": False, "error": "..."}`) and missed collection keys.
- **Change Made:** Upgraded `resolve_tool_record_count()` in [context.py](file:///D:/erp-fastprint/erp-langgraph/app/core/context.py#L530-L636) as the single canonical counter; delegated `verifier._calculate_records_count()` ([verifier.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/verifier.py#L21-L23)), [tool_node.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/tool_node.py), and [cli.py](file:///D:/erp-fastprint/erp-langgraph/cli.py#L110), [L273](file:///D:/erp-fastprint/erp-langgraph/cli.py#L273) to it.
- **Effect & Verification:** Consistent `0` count on error payloads across SSE `tool_end`, verifier, and CLI; deleted ~200 lines of duplicate logic from `verifier.py` and eliminated the `tool_node -> verifier` import. Verified by `test_17_unified_record_counter_handles_errors`.

### 18. `[FIXED]` Large Verbatim Code Duplication Across Core Pipeline Nodes (~560+ Lines)
- **Category:** Redundant Logic / Maintainability
- **Original Issue:** Six large duplicated blocks existed across `synthesizer.py`, `workflow.py`, `router.py`, `verifier.py`, and `tool_node.py`.
- **Change Made:** Consolidated all six pairs into shared helpers (`_prepare_synthesis_context`, `_build_followup_only_prompt`, `_build_polish_prompt`, `_finalize_synthesis_output`, `_render_single_plan_section` in [synthesizer.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/synthesizer.py); `_build_node_completion_events` in [workflow.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/workflow.py#L658-L747); `_compute_company_scope` in [router.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/router.py#L649-L699); `_is_explicit_daily_existence` in [verifier.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/verifier.py#L26-L45); `_is_permission_error` in [tool_node.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/tool_node.py#L594)).
- **Effect & Verification:** Eliminated ~450+ lines of copy-pasted logic and unified permission/rendering/event semantics across sync and async paths.

### 19. `[FIXED]` Parallel Extraction Defeats Tier-1 Peer Harvesting
- **Category:** Bug / Maintainability
- **Original Issue:** Concurrent `ThreadPoolExecutor` extraction read empty peer `params` before futures finished.
- **Change Made:** Added sequential post-join `_backfill_peer_entities(working_active_tools)` in [extractor.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/extractor.py#L786-L846).
- **Effect & Verification:** Deterministically propagates resolved SKU/product and vendor entities across peer plans after parallel extraction completes. Verified by `test_19_backfill_peer_entities_is_deterministic`.

### 20. `[FIXED]` Global `_CACHED_SYSTEM_TOKEN` Scoping & `service_key` Conflation
- **Category:** Anti-pattern / Concurrency
- **Original Issue:** Single global `_CACHED_SYSTEM_TOKEN` mixed tokens across hosts/usernames, `_get_token_lock()` was unsynchronized, and `self.service_key` ignored `INTERNAL_SERVICE_KEY`.
- **Change Made:** Replaced single token with `_CACHED_SYSTEM_TOKENS: Dict[Tuple[str, str], str]` keyed by `(base_url, username)` ([django_client.py](file:///D:/erp-fastprint/erp-langgraph/app/services/django_client.py#L89)); guarded `_get_token_lock()` with `_SHARED_CLIENT_LOCK`; prioritized `INTERNAL_SERVICE_KEY` for `self.service_key`.
- **Effect & Verification:** Host/user tokens are isolated and `X-Internal-Service-Key` uses `INTERNAL_SERVICE_KEY` first. Verified by `test_20_token_cache_isolated_by_host_and_username`.

### 21. `[FIXED]` Circuit-Breaker & LLM Fallback Gaps
- **Category:** Anti-pattern / Reliability
- **Original Issue:** `CircuitBreakerChatOpenAI` did not wrap `stream`/`astream`, OpenRouter returned plain `ChatOpenAI`, OpenAI -> Gemini fallback ignored `is_gemini_unhealthy()` and disabled multi-key rotation, and catalog refresh tasks were fire-and-forget.
- **Change Made:** Overrode `stream` and `astream` on `CircuitBreakerChatOpenAI` ([llm.py](file:///D:/erp-fastprint/erp-langgraph/app/services/llm.py#L98-L111)); wrapped OpenRouter in `CircuitBreakerChatOpenAI(..., provider_name="openrouter")` ([L390-L402](file:///D:/erp-fastprint/erp-langgraph/app/services/llm.py#L390-L402)); gated OpenAI -> Gemini fallback on `not is_gemini_unhealthy()` with `_build_gemini_model(..., enable_key_fallback=True)` ([L470-L482](file:///D:/erp-fastprint/erp-langgraph/app/services/llm.py#L470-L482)); deduplicated/tracked background refresh via `_REFRESH_TASK` ([L23](file:///D:/erp-fastprint/erp-langgraph/app/services/llm.py#L23), [L249-L270](file:///D:/erp-fastprint/erp-langgraph/app/services/llm.py#L249-L270)).
- **Effect & Verification:** Streaming auth/quota failures trip circuit breakers across OpenAI/Gemini/OpenRouter, multi-key Gemini rotation works during OpenAI failover, and background refresh tasks are tracked. Verified by `test_21_openrouter_uses_circuit_breaker_and_astream_trips`.

### 22. `[FIXED]` Unvalidated SQL Table Interpolation & Stale Incremental Sync in `ToolRegistryService`
- **Category:** Security / Bug
- **Original Issue:** `self.table_name` was interpolated directly into SQL without validation, and `initialize_catalog(force_update=False)` ignored changed tool descriptions.
- **Change Made:** Added `re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", resolved_table)` check in `__init__` ([tool_registry_service.py](file:///D:/erp-fastprint/erp-langgraph/app/services/tool_registry_service.py#L389-L392)) and compared `existing_tools.get(t_name) == t_desc` in `initialize_catalog()` ([L483](file:///D:/erp-fastprint/erp-langgraph/app/services/tool_registry_service.py#L483)).
- **Effect & Verification:** Rejects unsafe SQL identifiers with `ValueError` and updates modified tool descriptions incrementally. Verified by `test_22_tool_registry_sql_identifier_and_desc_update`.

### 23. `[FIXED]` State Type Mismatches & Unsafe `None` Metadata Access
- **Category:** Weak Typing / Bug
- **Original Issue:** `rag_results_by_query` was typed/initialized as a list instead of a dict, `metadata: None` raised `AttributeError`, and `sub_questions` string entries broke RAG ordering.
- **Change Made:** Fixed `AgentState.rag_results_by_query` type (`Optional[Dict[str, Dict[str, Any]]]`) and initial default (`{}`) in [state.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/state.py#L114), [L188](file:///D:/erp-fastprint/erp-langgraph/app/graph/state.py#L188); guarded `(c.get("metadata") or {}).get("source")` in [rag_node.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/rag_node.py#L87), [L173](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/rag_node.py#L173) and [common.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/formatters/common.py#L37); handled `isinstance(sq, str)` indices in [synthesizer.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/synthesizer.py).
- **Effect & Verification:** Eliminates `None` metadata crashes and preserves sub-question order. Verified by `test_23_none_metadata_chunks_and_state_init`.

### 24. `[FIXED]` Monolithic God Functions, 5-Way Tool Registry Fragmentation, and Cross-Node Coupling
- **Category:** Maintainability
- **Original Issue:** Monolithic files (`synthesizer.py` `3,890L`, `extractor.py` `2,777L`, `router.py` `2,021L`), 515L of boilerplate `fetch_*` wrappers in `django_client.py`, per-request 35-line `dispatch_map` in `tool_node.py`, and circular lazy imports across pipeline nodes.
- **Change Made:**
  - Extracted shared branch registry & helpers into [branches.py](file:///D:/erp-fastprint/erp-langgraph/app/core/branches.py) (`199L`) and `has_anaphoric_reference` into [context.py](file:///D:/erp-fastprint/erp-langgraph/app/core/context.py#L660-L678), eliminating all cross-node imports from `extractor.py` in `router.py`, `tool_node.py`, `synthesizer.py`, and `workflow.py`.
  - Replaced 33 boilerplate `fetch_*` wrappers in [django_client.py](file:///D:/erp-fastprint/erp-langgraph/app/services/django_client.py#L565-L598) (`1,099L -> 602L`) and the per-request `dispatch_map` in [tool_node.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/tool_node.py#L33-L46) with declarative binding derived from `TOOL_DOMAIN_MAP` (preserving `AsyncMock(spec=DjangoCoreClient)` compatibility and avoiding redundant `_clean_params` passes).
  - Split `synthesizer.py` (`3,864L -> 1,657L`) into `app/graph/nodes/formatters/` ([__init__.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/formatters/__init__.py), [common.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/formatters/common.py), [accounting.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/formatters/accounting.py), [inventory.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/formatters/inventory.py), [sales.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/formatters/sales.py), [marketplace.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/formatters/marketplace.py), [multi_company.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/formatters/multi_company.py)) driven by `SINGLE_TOOL_FORMATTERS` and `MULTI_COMPANY_FORMATTERS` dispatch dicts.
  - Split `extractor.py` (`2,855L -> 1,144L`) into `app/graph/nodes/extractors/` ([__init__.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/extractors/__init__.py), [common.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/extractors/common.py), [accounting.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/extractors/accounting.py), [inventory.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/extractors/inventory.py), [sales.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/extractors/sales.py), [marketplace.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/extractors/marketplace.py)).
  - Split `router.py` (`2,183L -> 1,806L`) into [router_patterns.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/router_patterns.py) (`330L`) and [router_decomposer.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/router_decomposer.py) (`176L`), eliminating `extractor.py -> router.py` circular imports.
- **Effect & Verification:** Clean acyclic domain module architecture with backward-compatible re-exports and zero test breakage. Verified by `test_24_modular_refactor_and_decoupled_imports` and all 700 unit tests.

### 25. `[PARTIALLY FIXED]` Container, CI/CD, and Production Hardening Gaps
- **Category:** Best-Practice Violations / Security
- **Original Issue:** `Dockerfile` ran as `root` with `gcc` left installed, `deploy.sh` swallowed `/health` failures with `|| true` and repeated git reset, `docker-compose.yml` used `network_mode: "host"`, and `app/main.py` exposed `/docs`/`/redoc` in production with `reload=True`.
- **Change Made:**
  - [Dockerfile](file:///D:/erp-fastprint/erp-langgraph/Dockerfile#L9-L24): purges `gcc` after `pip install`, removes unused `COPY prompts`, and runs as non-root `USER appuser`.
  - [deploy.sh](file:///D:/erp-fastprint/erp-langgraph/scripts/deploy.sh#L11-L14), [L37-L51](file:///D:/erp-fastprint/erp-langgraph/scripts/deploy.sh#L37-L51) & [deploy.yml](file:///D:/erp-fastprint/erp-langgraph/.github/workflows/deploy.yml#L70): skips duplicate git sync via `SKIP_GIT_SYNC=1` and runs a 10x2s `/health` retry loop exiting `1` on failure.
  - [main.py](file:///D:/erp-fastprint/erp-langgraph/app/main.py#L102-L109), [L451-L456](file:///D:/erp-fastprint/erp-langgraph/app/main.py#L451-L456): disables `/docs` & `/redoc` when `APP_ENV=production` and restricts `reload` to `development`.
- **Effect & Verification:** Hardens container and deploy failure detection. Verified by `test_25_production_hides_swagger_docs`.
- **Accepted Tradeoff:** [docker-compose.yml](file:///D:/erp-fastprint/erp-langgraph/docker-compose.yml#L8) retains `network_mode: "host"` to avoid breaking VPS `.env` `127.0.0.1` URLs for Django/Postgres.

### 26. `[FIXED]` Unit Tests Make Live Outbound Gemini API & PostgreSQL Calls
- **Category:** Testing / Reliability
- **Original Issue:** Running `unittest discover tests` triggered live Gemini embedding calls, OpenRouter GETs, `erp-dev.fastprint.id` POSTs, and PostgreSQL pool connection attempts because `.env` leaked into `settings`.
- **Change Made:** Gated Path 1 (`vector_service.search_knowledge`) in [rag_node.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/rag_node.py#L34) on `client is None`, and added `_is_test_runtime()` in [config.py](file:///D:/erp-fastprint/erp-langgraph/app/core/config.py#L114-L123) to instantiate `settings = Settings(_env_file=None)` under `unittest` / `FASTPRINT_TEST_ENV=true` / `APP_ENV=test`.
- **Effect & Verification:** Full 700-test suite log confirms `0` calls to `generativelanguage.googleapis.com`, `0` calls to `openrouter.ai`, `0` calls to `erp-dev.fastprint.id`, and `0` live PostgreSQL connection failures (`45.8s -> 19.2s`, `-58%` runtime). Verified by `test_26_rag_node_with_injected_client_skips_vector_service` and `test_26_and_28_hermetic_test_settings_and_ci_lint`.

---

## Low Priority

### 27. `[PARTIALLY FIXED]` Unused Schemas, Settings, and Transitive Dependency
- **Category:** Dead Code
- **Original Issue:** Unused schemas (`DecomposedSubQuestion`, `QueryDecompositionDecision`, `*ResponseSchema`), dead settings (`GEMINI_MODEL_OCR_AIR`, `USE_DYNAMIC_TOOLS`, `ERP_MULTI_COMPANY_MAX_BRANCHES`), unwired `LOG_LEVEL`, inverted `GEMINI_API_KEY` aliases, and redundant `python-dotenv`.
- **Change Made:** Deleted `DecomposedSubQuestion` and `QueryDecompositionDecision` from [router.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/router.py); deleted `GEMINI_MODEL_OCR_AIR`, `USE_DYNAMIC_TOOLS`, and `ERP_MULTI_COMPANY_MAX_BRANCHES` from [config.py](file:///D:/erp-fastprint/erp-langgraph/app/core/config.py) and [.env.example](file:///D:/erp-fastprint/erp-langgraph/.env.example); removed `python-dotenv>=1.0.0` from [requirements.txt](file:///D:/erp-fastprint/erp-langgraph/requirements.txt); wired `settings.LOG_LEVEL` to `logging.basicConfig()` in [main.py](file:///D:/erp-fastprint/erp-langgraph/app/main.py#L31-L32); aligned `GEMINI_API_KEY` `AliasChoices` priority in [config.py](file:///D:/erp-fastprint/erp-langgraph/app/core/config.py#L101-L104).
- **Effect & Verification:** Zero-reference dead config/schema items removed; `LOG_LEVEL` controls runtime log verbosity.
- **Accepted Tradeoff:** `*DataSchema` / `*ResponseSchema` (`~497L` in [tools.py](file:///D:/erp-fastprint/erp-langgraph/app/schemas/tools.py)) and `RouterDecision` intentionally retained for Django response contract tests.

### 28. `[FIXED]` 17 Unused Imports Across 11 Source Files (No Linter in CI)
- **Category:** Dead Code
- **Original Issue:** 17 unused imports across 11 production/script files, pre-existing unused imports in `tests/`, and no linter step in CI.
- **Change Made:** Removed all unused imports and `E`/`F` lint issues across `app/`, [cli.py](file:///D:/erp-fastprint/erp-langgraph/cli.py), `scripts/`, and all `tests/*.py` (`109` unused imports + `19` `E`/`F` issues cleaned). Added [pyproject.toml](file:///D:/erp-fastprint/erp-langgraph/pyproject.toml) (`select = ["E", "F"]`, `ignore = ["E501"]`) and a `Lint with Ruff` step (`ruff check app cli.py scripts tests`) to [.github/workflows/deploy.yml](file:///D:/erp-fastprint/erp-langgraph/.github/workflows/deploy.yml#L24-L32).
- **Effect & Verification:** `ruff check app cli.py scripts tests` passes with `0` errors (`All checks passed!`, exit `0`). Verified by `test_26_and_28_hermetic_test_settings_and_ci_lint`.

### 29. `[FIXED]` Minor Dead Expressions, Redundant Regexes, In-Place Mutation, and Test Leak
- **Category:** Dead Code / Anti-pattern
- **Original Issue:** No-op ternary in `extractor.py:L793`, unreachable ternary branch and in-place `raw_result` mutation + `_mock_return_value` check in `tool_node.py`, subsumed regexes in `router.py`, unused `active_tools` locals in `workflow.py`.
- **Change Made:** Fixed `params["search" if " " in prev_code else "code"] = prev_code` ([inventory.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/extractors/inventory.py#L253-L254)); shallow-copied `raw_result = dict(raw_result)` ([tool_node.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/tool_node.py#L492)), simplified dead ternary, and replaced `_mock_return_value` with `type(client) is DjangoCoreClient` ([L967](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/tool_node.py#L967)); deduplicated subsumed regexes in [router_patterns.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/router_patterns.py) and [router.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/router.py); removed unused `active_tools` locals in [workflow.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/workflow.py).
- **Effect & Verification:** All 5 listed micro-defects and dead expressions resolved.

### 30. `[FIXED]` CLI Model/Company Bugs, Swallowed Exceptions, and Unbounded SSE Queue
- **Category:** Anti-pattern / Error Handling
- **Original Issue:** `cli.py` misreported Gemini model names and couldn't revert sticky `co 2 -> co 1`; bare `except Exception: pass` swallowed LLM/validation errors; `rag_node.py` returned `error=None` on unexpected exceptions; SSE queue was unbounded.
- **Change Made:** Updated [cli.py](file:///D:/erp-fastprint/erp-langgraph/cli.py#L155) to `get_default_model_name()` and tracked `company_explicit_pending` ([L172](file:///D:/erp-fastprint/erp-langgraph/cli.py#L172), [L209](file:///D:/erp-fastprint/erp-langgraph/cli.py#L209), [L238-L240](file:///D:/erp-fastprint/erp-langgraph/cli.py#L238-L240)); logged caught exceptions at `logger.debug` in `router.py`, `extractor.py`, and `synthesizer.py`; updated [rag_node.py](file:///D:/erp-fastprint/erp-langgraph/app/graph/nodes/rag_node.py#L69) to return `error=str(exc), is_client_error=True`; bounded SSE queue to `asyncio.Queue(maxsize=256)` at [main.py](file:///D:/erp-fastprint/erp-langgraph/app/main.py#L373).
- **Effect & Verification:** All originally flagged locations resolved.

---

## Quick Wins Status

| # | Fix | Status | Verified Result |
|---|---|---|---|
| **#1** | Fix `user_query` -> `(state.get("user_query") or "")` | **Fixed** | `NameError` eliminated; covered by regression test |
| **#2** | Stop sending `OPENAI_API_KEY` to OpenRouter + gate startup pre-warm | **Fixed** | Zero OpenAI key leakage to OpenRouter |
| **#6** | Reorder `resolve_relative_dates()` before year regex & fix `limit` regex | **Fixed** | Explicit ISO dates & default `limit=20` preserved |
| **#7** | Remove `any(r is not None ...)` short-circuit in `verifier.py` | **Fixed** | Empty multi-company responses trigger `RETRY` -> `EMPTY` |
| **#8** | Defer `GREETING_PATTERNS` until after ERP/RAG candidate matching | **Fixed** | Greeting-prefixed Sales/Inventory queries route to ERP |
| **#5** | Filter `internal_no_stream` tags in SSE & drop duplicate `event: error` | **Fixed** | Clean SSE token stream + single error frame |
| **#11 & #12** | `WeakValueDictionary` thread locks + lazy `get_chat_model()` | **Fixed** | Idle locks GC'd; `0` LLM init on regex fast paths |
| **#16 & #28** | Delete deprecated ReAct stack/prompt & enforce repo-wide `ruff check` | **Fixed** | `-880 LOC` dead code deleted; `0` ruff errors in CI & locally |

---

## ERP & PGVector Data Safety Verification

- **Django ERP Transactional Data:** Completely safe and untouched. All 34 ERP tools in `erp-langgraph` are read-only (`POST /ai/tools/<domain>/<tool_name>/` queries only), and this service holds no direct database connection to Django Core.
- **Test-Suite Isolation (`#26`):** Final 700-test verification run logged `0` outbound requests to Django ERP (`erp-dev.fastprint.id`), `0` to PostgreSQL, `0` to Gemini, and `0` to OpenRouter.
- **PGVector Tables (`ai_knowledge_chunks`, `ai_tool_catalog`):**
  - `VectorService.initialize_table()` ([vector_service.py](file:///D:/erp-fastprint/erp-langgraph/app/services/vector_service.py#L155-L176)) uses `CREATE TABLE IF NOT EXISTS` and `CREATE INDEX IF NOT EXISTS` — existing chunks are never dropped or overwritten on startup.
  - [ingest_knowledge.py](file:///D:/erp-fastprint/erp-langgraph/scripts/ingest_knowledge.py#L91-L137) computes all embeddings before opening DB transactions and wraps `DELETE` + `INSERT` inside `conn.transaction()`, preventing partial document wipes on API errors.
- **Fail-Closed Auth (`#3`, `#4`):** Missing `ERP_LOGIN_SECRET` raises `DjangoAuthError` and unverified `user_roles` are stripped to `[]`; neither can mutate ERP records.

---

## Operational Notes Before Deployment

- **`ERP_LOGIN_SECRET` (`#4`):** Configure `ERP_LOGIN_SECRET` in local/staging/VPS `.env` files wherever programmatic `ERP_USERNAME` / `ERP_PASSWORD` auto-login is active.
- **Auth & RBAC (`#3`):** Ensure the Django BFF forwards verified `X-Internal-Service-Key` when passing `user_roles`, and sends valid user JWTs on both `APP_ENV=staging` and `APP_ENV=production`.
- **Cross-Repo Follow-Up (`#4`):** Coordinate with Django Core to migrate `/external_api/get_token/` from time-based MD5 to HMAC-SHA256.
