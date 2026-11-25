# CodeGraph Intelligence Rules (`codegraph.md`)

This rule governs codebase exploration, impact analysis, and symbol surveying using CodeGraph for the `erp-langgraph` repository.

---

## 1. Operating Principles

- **Primary Exploration Tool**: Prioritize `codegraph` (`codegraph_explore` MCP tool or `codegraph explore <query>` CLI) for:
  - Answering architectural and flow questions.
  - Surveying symbols, functions, classes, and call hierarchies before making edits.
  - Diagnosing bugs, unexpected behavior, or broken links across the pipeline.
  - Answering codebase questions during `/grill-me` sessions before asking the user.
- **Mandatory for `/grill-me`**:
  - Whenever the agent handles `/grill-me`, the slash command specifies: *"If a question can be answered by exploring the codebase, explore the codebase instead."*
  - The agent **MUST** execute CodeGraph (`codegraph_explore` MCP tool or `codegraph explore` CLI) to thoroughly survey the codebase, architecture, symbols, schemas, and call flows before querying the user.
  - Never ask the user questions that can be answered by inspecting the graph via CodeGraph.
- **Anti-Redundancy**: Source code blocks returned by `codegraph_explore` are current on-disk, line-numbered, and verbatim. Treat them as already read; **never perform redundant file reads** on code blocks already displayed by CodeGraph.
- **Blast Radius Analysis**: Before modifying or refactoring shared functions (e.g. pipeline nodes, schemas, clients), check the blast radius to determine all callers, callees, and impacted test suites.

---

## 2. Essential Commands & Workflows

### MCP Tool Usage (`call_mcp_tool`)
- **Explore Flow & Symbols**:
  - `ServerName`: `codegraph`
  - `ToolName`: `codegraph_explore`
  - `Arguments`:
    ```json
    {
      "projectPath": "C:/laragon/www/erp-langgraph",
      "query": "router_node extractor_node tool_node"
    }
    ```

### Terminal CLI Workflows
```powershell
# 1. Multi-symbol or flow exploration
codegraph explore "create_workflow_graph decide_after_router"

# 2. Find all upstream callers of a function or class
codegraph callers router_node

# 3. Find all downstream functions called by a symbol
codegraph callees extractor_node

# 4. Analyze blast radius before modifying a symbol
codegraph impact AgentState

# 5. Identify test files affected by changed source files
codegraph affected app/graph/workflow.py

# 6. Synchronize index after adding or renaming symbols
codegraph sync

# 7. Check index health and node counts
codegraph status
```

---

## 3. Index Maintenance
- The index database resides in `.codegraph/` (configured in `.gitignore`).
- Run `codegraph sync` after generating new modules or refactoring existing node signatures.
