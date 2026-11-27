# FastPrint ERP AI Orchestrator — Agent Operating Guidelines (`GEMINI.md`)

This document serves as the canonical operational specification and codebase guide for AI coding assistants (Antigravity `agy`, Cursor, Claude Code, Copilot) and software engineers working on the `erp-langgraph` repository.

---

## 1. Project Overview & Tech Stack

`erp-langgraph` is an autonomous AI Orchestrator service designed to interface with the FastPrint ERP Django backend (`ai_tools` layer). It executes intelligent business queries, accounting ledger analysis, and inventory monitoring via a compiled LangGraph `StateGraph` pipeline.

### Core Architecture & Components
- **Runtime & Language**: Python 3.11+
- **API Service**: FastAPI (served via `uvicorn` on port `8002`)
- **Agent Framework**: LangGraph (`langgraph.graph.StateGraph`), LangChain Core (`langchain_core`)
- **Settings & Validation**: Pydantic v2 (`pydantic-settings`, `BaseModel`)
- **HTTP Clients**:
  - `app/services/django_client.py`: `DjangoCoreClient` — Asynchronous HTTP client (`httpx.AsyncClient`) communicating with ERP Django (`/ai/tools/`).
- **LLM Engine**:
  - `app/services/llm.py`: Multi-provider factory supporting OpenAI, Google Gemini (multi-key), and OpenRouter.
  - Implements circuit-breaker failover and dynamic free-tier OpenRouter fallback when primary models are unavailable or rate-limited.
- **Canonical Architecture Reference**:
  - See `docs/ARCHITECTURE_MAP.md` for the complete 4-zone codebase map, 7-node StateGraph trace, `AgentState` map, and document index.

---

## 2. Developer Workflows & Essential Commands

Always activate the project virtual environment before executing commands:

### Environment Setup
```powershell
# PowerShell (Windows)
.\.venv\Scripts\Activate.ps1
```

### Running Tests
The test suite utilizes the standard Python `unittest` framework. **Do not execute `pytest` unless installed.**
```powershell
# Run the complete test suite (~850 tests)
.\.venv\Scripts\python.exe -m unittest discover tests

# Run a specific test module
.\.venv\Scripts\python.exe -m unittest tests/test_workflow.py

# Run an individual test case
.\.venv\Scripts\python.exe -m unittest tests.test_router_extractor_nodes.TestRouterNode.test_greeting_routes_to_general
```

### Running Applications
```powershell
# Interactive Terminal CLI Runner
python cli.py

# FastAPI Orchestrator Service (Port 8002)
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

---

## 3. LangGraph StateGraph Architecture

The system executes a 7-node StateGraph pipeline with conditional routing, RAG integration, and adaptive retry loops. For exact function line numbers and end-to-end request traces, read `docs/ARCHITECTURE_MAP.md`.

```mermaid
graph TD
    START([START]) --> Summarizer[summarizer_node]
    Summarizer --> Router[arouter_node]
    Router -->|live_erp / hybrid| Extractor[extractor_node]
    Router -->|vector_rag| RagNode[rag_node]
    Router -->|general / out_of_scope / clarification / contextual_analysis| Synthesizer[asynthesizer_node]
    Extractor --> ToolNode[tool_node]
    ToolNode -->|hybrid| RagNode
    ToolNode -->|live_erp| Verifier[verifier_node]
    RagNode -->|hybrid| Verifier
    RagNode -->|vector_rag| Synthesizer
    Verifier -->|RETRY < 2| Extractor
    Verifier -->|VALID / EMPTY / ERROR| Synthesizer
    Synthesizer --> END([END])
```

### State Model (`AgentState` in `app/graph/state.py`)
`AgentState` is a `TypedDict` tracking:
- `messages`, `user_query`, `user_token`, `user_roles`, `conversation_summary`: Conversational memory and auth context.
- `company_id`, `company_ids`, `authorized_company_ids`, `is_multi_company`: Multi-company scoping across 17 FastPrint branches.
- `query_mode`, `detected_domain`, `detected_tool`, `active_tools`, `rag_queries`, `sub_questions`: Routing mode (`live_erp`, `vector_rag`, `hybrid`, `general`, `out_of_scope`, `contextual_analysis`, `clarification`) and tool execution plans.
- `extracted_params`, `tool_result`, `multi_tool_results`, `multi_company_results`, `prior_tool_result`: Validated parameters, raw ERP results, and prior-turn snapshots.
- `verification_status`, `retry_count`, `retry_instruction`: Status (`VALID`, `EMPTY`, `RETRY`, `ERROR`, `CLARIFICATION`) and retry tracking (max `2`).
- `final_answer`, `suggested_followups`, `table_query_meta`, `entity_links`, `needs_clarification`: UI presentation and SSE output fields.

### Pipeline Nodes (`app/graph/nodes/`)
1. **`summarizer_node` (`summarizer.py`)**: Summarizes and trims long conversation histories beyond `SUMMARIZATION_THRESHOLD_MESSAGES`.
2. **`arouter_node` (`router.py` / `router_semantic.py`)**: Classifies intent across `accounting`, `inventory`, `sales`, `marketplace`, `knowledge`, `general`, `out_of_scope`, and `clarification` (supports `NLU_ROUTER_MODE`: `legacy_regex`, `semantic_llm`, `shadow`).
3. **`extractor_node` (`extractor.py` + `extractors/`)**: 3-tier entity/date parameter extraction validated against `app/schemas/tools.py`, with retry filter relaxation.
4. **`tool_node` (`tool_node.py`)**: Dispatches concurrent single/multi-tool and multi-company requests via `DjangoCoreClient` to `POST /ai/tools/<domain>/<tool_name>/`.
5. **`rag_node` (`rag_node.py`)**: Semantic SOP/knowledge retrieval via PostgreSQL PGVector (`VectorService`) with Django fallback.
6. **`verifier_node` (`verifier.py`)**: Evaluates record count via `resolve_tool_record_count()`. Retries back to `extractor_node` up to 2 times when 0 records are found, otherwise routes to `synthesizer`.
7. **`asynthesizer_node` (`synthesizer.py` + `formatters/`)**: Builds deterministic Markdown tables, IDR formatting, `entity_links`, `table_query_meta`, analytical follow-ups, and clarification fast-path responses.

---

## 4. ERP Domain Rules & Safety Guardrails

### 4.1 Strict Multi-Company Scoping
- FastPrint ERP is strictly multi-company. Every single ERP query and tool call **MUST** include a valid `company_id`.
- Default company fallback is `1` (`settings.DEFAULT_COMPANY_ID`). Never dispatch tool requests with an empty or `None` company ID.

### 4.2 Read vs. Write Safety (Human-in-the-Loop)
- **Read Operations (`is_write: false`)**: Execute autonomously (e.g., `get_account_balance`, `get_stock_level`, `get_journal_entries`, `get_product_info`).
- **Write Operations (`is_write: true`)**: Modifications (creating journal vouchers, moving stock, adjusting balances) **STRICTLY REQUIRE** human confirmation.
- The agent must summarize proposed parameters to the user and receive affirmative confirmation before sending `"confirm": true` to the backend.

### 4.3 Accounting & Inventory Knowledge
- **Dates**: All dates must strictly format as `YYYY-MM-DD` (ISO 8601).
- **Chart of Accounts (COA)**:
  - `1xxx`: Assets | `2xxx`: Liabilities | `3xxx`: Equity
  - `4xxx`: Revenue | `5xxx`: Cost of Goods Sold (COGS) | `6xxx`: Expenses
- **Inventory Stock Quants**:
  - `quantity`: Physical warehouse inventory count.
  - `available_quantity`: Physical count minus reserved order stock.
  - Always prefer exact SKU (`default_code`) or Barcode lookup before broad keyword matching.

### 4.4 Data Grounding & Anti-Hallucination
- Only state financial figures and stock numbers present in `tool_raw_result` or `calculations`.
- If an entity or transaction does not exist, explicitly inform the user rather than estimating or hallucinating.
- Always cite reference numbers (`JE/2026/...`, SKU, Account Code) for cross-system verification.

---

## 5. Agent Coding Guidelines & Contribution Rules

When modifying or adding features to this repository:

1. **State Immutability in StateGraph**:
   - Node functions must return a dictionary delta (e.g. `{"verification_status": VerificationStatus.VALID, "retry_count": count + 1}`).
   - Never mutate the input `state` dictionary in-place.
2. **Type Annotations & Validation**:
   - Use explicit Python type hints (`typing.Optional`, `typing.Dict`, `typing.List`, `typing.Any`).
   - Validate external inputs through Pydantic models in `app/schemas/tools.py`.
3. **Async ERP Client Usage**:
   - StateGraph pipeline nodes are asynchronous and use `DjangoCoreClient` (`app/services/django_client.py`).
4. **Preserve Documentation & Comments**:
   - Do not remove existing docstrings, module headers, or `TASK-X.X` reference tags.
5. **Code Intelligence with CodeGraph**:
   - For codebase exploration, architectural queries, diagnosing bugs, or surveying symbols before an edit, prioritize using **CodeGraph** (`codegraph_explore` MCP tool or `codegraph explore <query>` CLI).
   - **Mandatory for `/grill-me`**: When handling `/grill-me`, the agent **must** use CodeGraph first to research and answer design/architecture questions from the codebase before asking the user. Only query the user on high-level preferences, business logic choices, or requirements that cannot be discovered from the code.
   - Use CodeGraph to inspect symbol call paths, blast radius, and affected test files in a single pass instead of broad grep/read loops.
   - Treat code blocks returned by `codegraph_explore` as already read; avoid redundant file reads.
   - Run `codegraph sync` after adding or renaming major symbols to keep the local graph index synchronized.
6. **Continuous Verification**:
   - Before finishing any turn or declaring a task complete, run the full test suite:
     `.\.venv\Scripts\python.exe -m unittest discover tests`
   - All tests must pass with zero failures.

---

## 6. Communication & Anti-AI-Slop Standards

All generated text, commit messages, PR descriptions, and agent responses must be direct, sharp, and human. Eliminate generic AI filler and corporate buzzwords:

### 6.1 Direct, Concrete Prose
- **Lead with the Answer**: Answer questions or state findings immediately. Cut throat-clearing openers (*"Certainly!"*, *"I'd be happy to help"*, *"Here's the thing"*, *"Let me be clear"*).
- **Show, Don't Tell**: Let facts, code line numbers, and concrete metrics carry the weight. Never label a point *"crucial"*, *"pivotal"*, or *"vital"*—demonstrate its importance with evidence.
- **No Dramatic Reveals & Binary Contrasts**: Avoid binary setups (*"It's not X, it's Y"* / *"Not just X, but Y"*) and colon reveals (*"The best part: it works"*). State points directly in complete sentences.
- **No Summary Recaps**: Never end with redundant recap paragraphs (*"In conclusion"*, *"Overall"*, *"In summary"*). Stop on the last concrete action, result, or verifiable takeaway.

### 6.2 Banned AI Buzzwords & Corporate Slop
- **Strictly Banned**: *delve, foster, leverage, utilize, facilitate, empower, streamline, robust, cutting-edge, paradigm shift, game changer, tapestry, realm, beacon, multifaceted, intricate, paramount, transformative, elevate, embark, supercharge, harness, ever-evolving*.
- **Empty Adverbs to Strip**: *fundamentally, crucially, inherently, inevitably, literally, truly*.
- **Empty Filler Phrases to Cut**: *it's worth noting, it's important to note, at the end of the day, when it comes to, at its core, going forward*.

### 6.3 Functional, Clean Formatting
- No decorative emojis in technical headings or bullet lists.
- Avoid sprinkling bold text haphazardly across sentences for fake enthusiasm.
- Format must follow technical clarity: prefer short code blocks, exact file links, and clean markdown tables over prose lists.
