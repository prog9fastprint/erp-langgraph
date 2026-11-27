# Roadmap Progress: FastPrint ERP AI Orchestrator (`erp-langgraph`)

Pelacakan eksekusi perbaikan berdasarkan [PROJECT_ANALYSIS.md](PROJECT_ANALYSIS.md#L475-L505).

---

## Ringkasan Status Roadmap

| Prioritas | ID | Item | Status | Verifikasi |
|---|---|---|---|---|
| **P0** | **P0-1** | Tutup Celah Keamanan `authorized_company_ids`, Batas Input, Rate Limiter, & Auth Strictness | **Done** (2026-11-27) | `850` unit tests OK (`32.93s`), `ruff check` 0 errors |
| **P0** | **P0-2** | Migrasi Checkpointer ke PostgreSQL (`AsyncPostgresSaver`) | **Pending** | — |
| **P0** | **P0-3** | Validasi Live & Cutover `NLU_ROUTER_MODE=semantic_llm` | **Pending** | — |
| **P1** | **P1-1** | Konversi `extractor_node` & `summarizer_node` ke Async Murni | **Pending** | — |
| **P1** | **P1-2** | Observability Produksi (JSON Logs, Token Cost Tracking, `/metrics`) | **Pending** | — |
| **P1** | **P1-3** | Ekspansi Korpus Pengetahuan RAG + Hybrid Search di `VectorService` | **Pending** | — |
| **P1** | **P1-4** | Migrasi Login ERP `MD5` -> `HMAC-SHA256` & Pemisahan Service Key | **Partial** (Inbound done; outbound & MD5 pending koordinasi Django Core) | — |
| **P2** | **P2-1** | Decommissioning Regex Legacy & Penyatuan Manifest Tool | **Pending** | — |
| **P2** | **P2-2** | Implementasi Domain `helpdesk` (2 Tools) & `fastpos` (4 Tools) | **Pending** (lihat [PROJECT_PROGRESS.md](PROJECT_PROGRESS.md)) | — |
| **P2** | **P2-3** | Pre-Seed Catalog di CI/Deploy, Lock Dependencies, & Zero-Downtime Deploy | **Pending** | — |
| **P2** | **P2-4** | Multi-Key Gemini Rotation pada `VectorService` & TTL Cache Lookup Master | **Pending** | — |
| **P3** | **P3-1** | Penyelarasan Dokumentasi & Automated Doc-Check | **Pending** | — |
| **P3** | **P3-2** | Dukungan Penuh Write Tools + Endpoint Resume HITL | **Pending** | — |
| **P3** | **P3-3** | Shared Redis Cache untuk Outage Circuit-Breaker & Embedding | **Pending** | — |

---

## Detail Implementasi Selesai

### P0-1 — API Security, Input Bounds, Rate Limiter & Auth Alignment (2026-11-27)

1. **Pemisahan Inbound `INTERNAL_SERVICE_KEY` & Gating `authorized_company_ids`**
   - [app/main.py](app/main.py): Menghapus fallback `ERP_API_TOKEN` pada `_resolve_and_validate_auth()` (fail-closed `HTTP 403` saat `REQUIRE_SERVICE_KEY=True` dan `INTERNAL_SERVICE_KEY` kosong).
   - [app/main.py](app/main.py): `_build_workflow_kwargs()` membuang `authorized_company_ids` menjadi `[]` (sama seperti `user_roles`) jika `X-Internal-Service-Key` tidak terverifikasi.
   - **Efek:** Mencegah spoofing daftar cabang dan eskalasi hak akses menggunakan token ERP biasa.

2. **Penyelarasan Strict `user_token` & Clamping Fan-Out Multi-Company**
   - [app/core/config.py](app/core/config.py): Menambahkan `Settings.is_strict_user_token_required()` (`REQUIRE_USER_TOKEN=True` atau `APP_ENV` `production`/`staging`).
   - [app/graph/nodes/tool_node.py](app/graph/nodes/tool_node.py), [app/graph/nodes/rag_node.py](app/graph/nodes/rag_node.py): Menggunakan `settings.is_strict_user_token_required()` pada `DjangoCoreClient` agar tidak pernah fallback ke login sistem/admin di staging/produksi.
   - [app/graph/workflow.py](app/graph/workflow.py), [app/graph/nodes/router.py](app/graph/nodes/router.py): Membatasi fan-out `"semua cabang"`, ranking cabang, dan HQ auto-expand ke `authorized_company_ids` terverifikasi (penyebutan 1 cabang eksplisit tetap diteruskan ke Django untuk pesan `HTTP 403`).
   - **Efek:** Menutup celah fallback kredensial admin pada node eksekusi dan membatasi fan-out multi-cabang sesuai hak akses user.

3. **Batas Payload `ChatRequest` & Truncation History Aman**
   - [app/schemas/chat.py](app/schemas/chat.py): Batas keras `HTTP 422` untuk `query` (`<= 4.000` char), `thread_id` (`<= 128` char), `user_id` (`<= 128` char), `company_ids` / `authorized_company_ids` (`<= 50` item), `user_roles` (`<= 100` item), `user_token` / `auth_token` (`<= 8.192` char), dan `history` ekstrem (`<= 100` item, `content <= 50.000` char).
   - [app/graph/workflow.py](app/graph/workflow.py): `_hydrate_messages_from_history()` memotong `history` aman ke `MAX_HYDRATED_HISTORY_MESSAGES = 30` pesan terbaru.
   - **Efek:** Memblokir *prompt-flooding DoS* dan pembengkakan token tanpa memutus sesi percakapan panjang.

4. **In-Process Sliding-Window Rate Limiter**
   - [app/core/rate_limit.py](app/core/rate_limit.py), [app/main.py](app/main.py), [.env.example](.env.example): `SlidingWindowRateLimiter` (`RATE_LIMIT_MAX_REQUESTS=30`, `RATE_LIMIT_WINDOW_SECONDS=60`, `TRUST_PROXY_HEADERS=false`, `max_buckets=10_000` LRU) pada `/api/v1/chat` dan `/api/v1/chat/stream` (`HTTP 429` + `Retry-After`), dengan key priority verified `user_id` -> SHA-256 `token` -> client IP. `/health` dan `/health/ready` dikecualikan.
   - **Efek:** Mencegah satu pengguna atau pengirim tanpa token membanjiri API dan membakar kuota LLM, dengan isolasi kuota per user di balik Django BFF.

5. **Pengujian**
   - [tests/test_security_p0_hardening.py](tests/test_security_p0_hardening.py): 21 test cases baru untuk seluruh guard P0-1.
   - [tests/test_production_readiness.py](tests/test_production_readiness.py), [tests/test_adversarial_stress.py](tests/test_adversarial_stress.py): Diperbarui sesuai kebijakan *fail-closed* `INTERNAL_SERVICE_KEY` dan batas `query` 4.000 karakter.
