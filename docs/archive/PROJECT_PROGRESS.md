# Project Progress: FastPOS, Helpdesk, & Price History Integration

Tracking implementasi bertahap di dua repository (`erp-langgraph` & `ERP/ai_tools`) untuk menangani laporan komplain/retur, analitik FastPOS, riwayat perubahan harga produk, serta query produk multi-aspek (`"coba cek barang A penjualannya, returnya, perubahan harganya"`).

---

## Ringkasan Status Fase

| Fase | Ruang Lingkup | Status | Verifikasi |
|---|---|---|---|
| **Phase 1a** | `erp-langgraph`: Kontrak Handoff + `inventory.get_price_history` (schema, registry, router, extractor, `tool_node`, formatter, unit test) | **Done** | 752 tests OK, `ruff check` 0 errors |
| **Phase 1b** | `ERP/ai_tools`: Implementasi `inventory.get_price_history` (`LogEntry`, `PriceUpdateLog`, `InventoryUpdatePriceLog`, `PricelistLineModel`) + registry + unit test | **Done** | 119 tests OK + read-only live ORM smoke test |
| **Phase 2** | Domain `helpdesk`: `get_complaints` + `get_product_returns` (Django + LangGraph) + disambiguasi RAG `retur` | **Pending** | — |
| **Phase 3** | Domain `fastpos`: `get_product_performance`, `get_stock_alerts`, `get_list_orders`, `get_stock_opname` (Django + LangGraph) | **Pending** | — |
| **Phase 4** | Multi-aspect product fan-out, cross-tool backfill, ringkasan eksekutif deterministik, E2E test, `codegraph sync` | **Pending** | — |

---

## Detail Implementasi Selesai (Phase 1)

### Phase 1a — `C:\laragon\www\erp-langgraph`
- **Handoff & Kontrak API**: [docs/AI_TOOLS_HANDOFF.md](docs/AI_TOOLS_HANDOFF.md) (Sections `6.4 FastPOS`, `6.5 Helpdesk`, `6.6 inventory.get_price_history`).
- **Schema & Registry**: [app/schemas/tools.py](app/schemas/tools.py) (`PriceHistoryFilterSchema`, `TOOL_DOMAIN_MAP`), [app/services/tool_registry_service.py](app/services/tool_registry_service.py) (`CORE_TOOL_DEFINITIONS`).
- **Router & Guardrails**: [app/graph/nodes/router_patterns.py](app/graph/nodes/router_patterns.py) (`INVENTORY_PRICE_HISTORY_PATTERNS`), [app/graph/nodes/router.py](app/graph/nodes/router.py) (`_HQ_CROSS_BRANCH_TOOLS`, collision suppression dengan `get_competitor_insights`, `get_sync_execution_logs`, `get_price_comparison`, `get_product_info`).
- **Extractor**: [app/graph/nodes/extractors/inventory.py](app/graph/nodes/extractors/inventory.py) (`_extract_product_target_for_metric_query`, `_fallback_extract_price_history`, default 90 hari), [app/graph/nodes/extractors/__init__.py](app/graph/nodes/extractors/__init__.py), [app/graph/nodes/extractor.py](app/graph/nodes/extractor.py).
- **Tool Node, Context, & Formatter**: [app/graph/nodes/tool_node.py](app/graph/nodes/tool_node.py) (`supports_native_mc`, sanitasi `search`), [app/core/context.py](app/core/context.py) (`changes`), [app/graph/nodes/formatters/inventory.py](app/graph/nodes/formatters/inventory.py) (`format_get_price_history`).
- **Tests**: [tests/test_fastpos_helpdesk_price_tools.py](tests/test_fastpos_helpdesk_price_tools.py) (9 unit/integration tests).

### Phase 1b — `C:\laragon\www\ERP`
- **Tool Implementation**: [ai_tools/tools/inventory/get_price_history.py](file:///C:/laragon/www/ERP/ai_tools/tools/inventory/get_price_history.py) (`GetPriceHistoryTool`, `is_write=False`, multi-company ACL via `resolve_authorized_companies`).
  - Menggabungkan `ess.LogEntry` (`model='PricelistLineModel'`), `marketplace.PriceUpdateLog` (wholesale tier + filter no-op retail `old_price == new_price`), `inventory.InventoryUpdatePriceLog` (`service_type='sales_pricelist'`), dan snapshot aktif `sales.PricelistLineModel`.
- **Registry**: [ai_tools/tools/inventory/registry.py](file:///C:/laragon/www/ERP/ai_tools/tools/inventory/registry.py) (`INVENTORY_TOOLS['get_price_history']`).
- **Tests**: [ai_tools/tests_price_history.py](file:///C:/laragon/www/ERP/ai_tools/tests_price_history.py) (12 unit tests `SimpleTestCase`).

---

## Rencana Fase Berikutnya

1. **Phase 2 — Helpdesk (`get_complaints`, `get_product_returns`)**
   - Django `ai_tools/tools/helpdesk/`: agregasi `Tiket` (exclude `Q_SERVICE_TICKETS` default + 4 breakdown `reportingView.py`), `ReturnTransfer`, dan `RefundsLine` (`refund customer`).
   - LangGraph: domain `helpdesk`, schema, extractor YTD, formatter, dan disambiguasi `KNOWLEDGE_RAG_PATTERNS` agar pertanyaan SOP (`"SOP retur"`, `"bisa diretur"`) tetap ke RAG sementara pertanyaan transaksional (`"returnya"`, `"cek retur"`) masuk ke `helpdesk.get_product_returns`.
2. **Phase 3 — FastPOS (`get_product_performance`, `get_stock_alerts`, `get_list_orders`, `get_stock_opname`)**
   - Django `ai_tools/tools/fastpos/`: ekspos metrik `ProductStockRecap`, `ProductRecommendationRecap`, `CogsDailySnapshot`, `BcgHistory`, `FastposIndex`, `ListOrder`, dan `StockOpname` (dengan BOM explosion MTH).
   - LangGraph: domain `fastpos`, 4 schema/extractor/formatter, dan supresi tabrakan dengan `get_stock_adjustments` serta `get_top_selling_products`.
3. **Phase 4 — Multi-Aspect Product Orchestration**
   - Parallel fan-out untuk query seperti `"coba cek barang A penjualannya, returnya, perubahan harganya"` -> `sales.get_top_selling_products` (atau `fastpos.get_product_performance` jika eksplisit FastPOS) + `helpdesk.get_product_returns` + `inventory.get_price_history`, cross-tool product backfill, dan blok **Ringkasan Eksekutif Produk** deterministik.
