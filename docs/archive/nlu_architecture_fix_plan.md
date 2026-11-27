# Plan Modernisasi & Cutover NLU Routing (`erp-langgraph`)

**Scope:** P0-3 (Hardening, Live Evaluation & Cutover `NLU_ROUTER_MODE=semantic_llm`), diikuti fase terpisah P2-1 (Decommission Legacy Regex).  
**Status:** Proposed — disesuaikan dengan state aktual repo pada commit `0c46131` (2026-11-27).  
**Referensi Kode & Blueprint:** [nlu_architecture_blueprint.md](nlu_architecture_blueprint.md), [PROJECT_ANALYSIS.md](PROJECT_ANALYSIS.md), [ROADMAP_PROGRESS.md](ROADMAP_PROGRESS.md).

---

## 1. Baseline Aktual Repo (Commit `0c46131`)

Pondasi Hybrid Semantic NLU sebagian besar **sudah terimplementasi** di repo:
- Enriched 36 tool catalog + PGVector Cosine + PostgreSQL FTS (`'simple'`) RRF + Conditional RBAC + in-memory fallback: [app/services/tool_registry_service.py](app/services/tool_registry_service.py) (`1.513` LOC), [app/scripts/seed_tools.py](app/scripts/seed_tools.py).
- Semantic router (`check_hard_guardrails`, per-clause retrieval, dynamic Top-K prompt, conservative disambiguation, `SemanticRouterOutput`): [app/graph/nodes/router_semantic.py](app/graph/nodes/router_semantic.py) (`1.013` LOC).
- Feature flag `NLU_ROUTER_MODE` (`legacy_regex` | `semantic_llm` | `shadow`) & background shadow task: [app/core/config.py](app/core/config.py#L145-L160), [app/graph/nodes/router.py](app/graph/nodes/router.py#L2146-L2261) (`2.533` LOC), [app/graph/nodes/router_patterns.py](app/graph/nodes/router_patterns.py) (`488` LOC).
- Golden dataset 100 kasus & evaluator CLI (`--mode offline|live`, `--output`, `--markdown`, `--fail-on-gate`): [tests/fixtures/nlu_golden_cases.json](tests/fixtures/nlu_golden_cases.json), [scripts/eval_nlu_router.py](scripts/eval_nlu_router.py) (`445` LOC).
- Unit test NLU & evaluator: [tests/test_semantic_nlu_router.py](tests/test_semantic_nlu_router.py), [tests/test_eval_nlu_router.py](tests/test_eval_nlu_router.py).

### Hasil Reproduksi Baseline Offline (`0c46131`)

Dieksekusi via `.\.venv\Scripts\python.exe scripts/eval_nlu_router.py --mode offline`:

| Metrik | Hasil Aktual (`0c46131`) | Catatan |
|---|---|---|
| Total Golden Cases | `100` | 36/36 core tools tercakup; 12 guardrail, 6 clarification, 2 multi-turn, 2 hybrid, 5 multi-tool |
| Offline Retrieval Top-K Recall | `100.0%` (`78/78`) | In-memory fallback saja, bukan bukti recall PGVector live |
| Legacy Regex Accuracy | `84.04%` (`79/94`) | 6 kasus klarifikasi tidak dihitung di penyebut legacy |
| Guardrail Regressions | `0 / 12` | `PASS` |
| Legacy P95 Latency | `44.57 ms` | Baseline pembanding offline |
| Full Unit Test Suite & Lint | `850` tests OK, Ruff `0` error | Terverifikasi pada `0c46131` |
| Default `NLU_ROUTER_MODE` (kode) | `legacy_regex` | Catatan: `.env` VPS staging sudah override ke `semantic_llm` |

---

## 2. Empat Celah Nyata yang Harus Ditutup Sebelum Cutover

Plan asli (21 task A-01–A-21) disederhanakan menjadi **4 Batch P0-3** karena sebagian besar pondasi sudah ada, sementara audit kode menemukan 4 celah konkret berikut:

| # | Celah di Kode Saat Ini | Lokasi | Risiko |
|---|---|---|---|
| **G1** | `_build_state_from_semantic_decision()` hanya memvalidasi tool terhadap `TOOL_DOMAIN_MAP` / `registry`, **bukan** terhadap `candidates` (shortlist Top-K) maupun `user_roles` RBAC. Injeksi `prior_detected_tool` dan fast-path di `check_hard_guardrails()` juga belum mengecek `user_roles`. | [router_semantic.py:405-451](app/graph/nodes/router_semantic.py#L405-L451), [512-520](app/graph/nodes/router_semantic.py#L512-L520), [773-811](app/graph/nodes/router_semantic.py#L773-L811) | Halusinasi / prompt injection LLM bisa memilih tool di luar shortlist atau di luar izin RBAC user. |
| **G2** | `_try_route_branch_switch_followup()` untuk `"semua cabang"` langsung menetapkan `company_ids = list(text_cids)` (17 cabang) tanpa clamping ke `authorized_company_ids`. | [router_semantic.py:314-332](app/graph/nodes/router_semantic.py#L314-L332) | Bypass proteksi multi-company P0-1 ([router.py:904-907](app/graph/nodes/router.py#L904-L907)) saat user mengirim follow-up `"semua cabang"` di mode `semantic_llm`. |
| **G3** | Hard write-block & explicit customer billing dibatasi `if len(clauses) <= 1`; pada kueri multi-klausa campuran (*"cek omzet bulan ini dan update stok ke Shopee"*), klausa tulis lolos ke LLM dan `non_tool_questions` selalu kosong (`[]`). | [router_semantic.py:398-403](app/graph/nodes/router_semantic.py#L398-L403), [863](app/graph/nodes/router_semantic.py#L863) | Mutasi terlarang pada kueri majemuk hanya bergantung pada prompt LLM, tidak ditolak deterministik. |
| **G4** | `eval_nlu_router.py` tidak menolak zero-denominator, tidak membedakan `pgvector_fts` vs `in_memory_fallback`, tidak memisahkan error/timeout LLM (`None`) dari salah rute, dan tanpa metadata run (`commit_sha`, `provider`, `model`). Shadow log di `router.py` juga mencatat `query[:200]` mentah tanpa correlation ID dan sulit dihitung denominatornya saat `NLU_SHADOW_LOG_MISMATCH_ONLY=True`. | [eval_nlu_router.py:151-192](scripts/eval_nlu_router.py#L151-L192), [255-275](scripts/eval_nlu_router.py#L255-L275), [router.py:2158-2261](app/graph/nodes/router.py#L2158-L2261) | False PASS pada benchmark live (mis. diam-diam pakai in-memory fallback) & kebocoran query bisnis di log `INFO`. |

---

## 3. Batasan Scope & Invariant

### In Scope (P0-3)
- Penutupan celah G1–G4 pada `router_semantic.py`, `tool_registry_service.py`, `router.py` (shadow log), dan `scripts/eval_nlu_router.py`.
- Unit test regresi dan fault-injection offline (tanpa DB/network/API key).
- Seeding `ai_tool_catalog` & live evaluation di staging (PGVector + Gemini).
- Perbaikan cluster failure berbasis bukti live eval (metadata tool card / prompt / disambiguation), tanpa menambah intent regex baru.
- Shadow check, rollback rehearsal, dan cutover default `NLU_ROUTER_MODE=semantic_llm` dengan fallback `legacy_regex` tetap aktif.

### Out of Scope (Dipisah ke Workstream Lain)
- Rewrite menjadi ReAct / ubah struktur 7-node StateGraph.
- Migrasi `AsyncPostgresSaver` (P0-2), async `extractor_node`/`summarizer_node` (P1-1), observability `/metrics` penuh (P1-2), protokol login Django (P1-4), dan penyatuan manifest extractor/formatter lintas sistem.
- Penghapusan kode `legacy_regex` / `router_patterns.py` (ditunda ke **Phase 4 / P2-1** setelah cutover stabil).

### Invariant Wajib
1. Tidak ada write tool yang dieksekusi; klausa write pada kueri tunggal maupun majemuk ditolak deterministik.
2. Pada jalur scoped LLM, tidak ada tool di luar kandidat shortlist atau di luar izin `user_roles` (saat `user_roles` terverifikasi non-empty).
3. Router tidak boleh memperluas `authorized_company_ids` (termasuk pada follow-up `"semua cabang"`).
4. Mode `clarification` tidak memanggil tool ERP (`active_tools == []`) dan tetap kompatibel dengan `ChatResponse` + SSE.
5. Prompt LLM bukan otoritas keamanan; semua boundary ditegakkan di kode Python.

---

## 4. Exit Gates Cutover (P0-3)

Kelima gate dari [nlu_architecture_blueprint.md](nlu_architecture_blueprint.md#L229-L234) dan [PROJECT_ANALYSIS.md](PROJECT_ANALYSIS.md) berlaku bersamaan pada live eval:

| Gate | Target | Aturan Validasi |
|---|---:|---|
| Retrieval Top-K Recall | `>= 98.0%` | Wajib berjalan di backend `pgvector_fts` (bukan 100% `in_memory_fallback`); penyebut `> 0`. |
| Semantic Routing Accuracy | `>= 95.0%` & `>= Legacy` | Exact-match mode + himpunan tool pada seluruh kasus; error/timeout dihitung gagal (tidak dibuang). |
| False-Clarification Rate | `<= 3.0%` | Kasus non-ambigu yang diklarifikasi / total kasus non-ambigu (`clear_cases > 0`). |
| P95 Latency Delta | `<= 800 ms` | `semantic_p95_ms - legacy_p95_ms` pada boundary router yang sama; jalankan minimal 3 run comparable dan simpan semua hasil. |
| Guardrail Regressions | `== 0` | Seluruh hard guard (write-block, out-of-scope, customer vs vendor, doc-ID, RBAC) lulus tanpa bergantung pada fallback legacy. |
| Shadow Mismatch (Staging) | `< 5%` | Diukur dari shadow completed decisions; timeout/error dan klarifikasi sah dilaporkan terpisah agar tidak menyesatkan. |

---

## 5. Rencana Eksekusi Bertahap

### Batch 1 — Semantic Router Security & Multi-Clause Hardening *(Lokal, Offline)*

Menutup celah **G1, G2, G3** sebelum menjalankan live benchmark.

- [ ] **NLU-1.1 — Enforce Shortlist & Conditional RBAC Boundary**
  - **File:** [app/graph/nodes/router_semantic.py](app/graph/nodes/router_semantic.py), [app/services/tool_registry_service.py](app/services/tool_registry_service.py)
  - **Perubahan:**
    - Ekspos helper tunggal `_is_tool_authorized_for_roles(tool_meta, user_roles)` berbasis `_resolve_rbac_roles()` agar aturan Conditional RBAC identik di retrieval, injeksi `prior_detected_tool`, hard fast-path, dan post-validation LLM.
    - Di `_build_state_from_semantic_decision()` ([router_semantic.py:773-811](app/graph/nodes/router_semantic.py#L773-L811)): terima tool dari output LLM **hanya** jika `pt_tool` ada dalam `candidates` (shortlist hasil retrieval) **dan** lolos `_is_tool_authorized_for_roles()`. Tool yang tidak valid/tidak diizinkan ditolak dan dicatat `warning`; jika semua tool ditolak, kembalikan state aman tanpa eksekusi (bukan jatuh ke eksekusi liar).
    - Terapkan pemeriksaan `_is_tool_authorized_for_roles()` pada injeksi `prior_detected_tool` ([router_semantic.py:512-520](app/graph/nodes/router_semantic.py#L512-L520)) serta fast-path doc-ID / P&L drill-down / branch-switch di `check_hard_guardrails()`.
- [ ] **NLU-1.2 — Clamp `authorized_company_ids` pada Branch-Switch Follow-Up**
  - **File:** [app/graph/nodes/router_semantic.py](app/graph/nodes/router_semantic.py#L314-L332)
  - **Perubahan:** Saat `is_all=True` di `_try_route_branch_switch_followup()`, batasi `company_ids` ke `state.get("authorized_company_ids")` (bila non-empty) persis seperti [router.py:904-907](app/graph/nodes/router.py#L904-L907).
- [ ] **NLU-1.3 — Per-Clause Write & Explicit Customer-Billing Filter pada Multi-Intent**
  - **File:** [app/graph/nodes/router_semantic.py](app/graph/nodes/router_semantic.py#L398-L403)
  - **Perubahan:** Untuk kueri dengan `len(clauses) > 1`, pisahkan klausa yang memicu `is_hard_write_block_query()`, `OUT_OF_SCOPE_PATTERNS`, atau `is_explicit_customer_billing_query()` sebelum retrieval & LLM call:
    - Jika seluruh klausa ditolak → langsung return `out_of_scope` / `general` tanpa panggilan LLM.
    - Jika sebagian klausa valid dan sebagian ditolak → jalankan retrieval + Scoped LLM hanya untuk klausa valid, dan simpan klausa yang ditolak ke `non_tool_questions` agar synthesizer tetap menjelaskan penolakan tanpa mengeksekusi mutasi.
- [ ] **NLU-1.4 — Unit & Fault-Injection Regression Tests**
  - **File:** [tests/test_semantic_nlu_router.py](tests/test_semantic_nlu_router.py)
  - **Cakupan test baru:**
    - Mock LLM memilih tool palsu, tool valid tapi di luar `candidates`, atau tool terlarang oleh `user_roles` (single & partial multi-tool) → dipastikan tidak masuk `active_tools`.
    - Follow-up `"semua cabang"` dengan `authorized_company_ids=[3, 4]` → `company_ids` ter-clamp ke `[3, 4]`.
    - Kueri campuran `"Cek omzet bulan ini, apa SOP retur barang cetak, dan update stok tinta ke Shopee sekarang"` → klausa write ditolak ke `non_tool_questions`, klausa sales + RAG tetap diproses.
    - Fault-injection: LLM timeout / malformed output / candidate list kosong → fallback terukur tanpa crash atau unauthorized tool.
  - **Verifikasi:** `.\.venv\Scripts\python.exe -m unittest tests/test_semantic_nlu_router.py` lalu full suite + `ruff check`.

---

### Batch 2 — Evaluator Reliability & Auditable Shadow Logging *(Lokal, Offline)*

Menutup celah **G4** agar hasil benchmark live dan log shadow di staging bisa dipercaya.

- [ ] **NLU-2.1 — Tandai Retrieval Backend pada Candidate Metadata**
  - **File:** [app/services/tool_registry_service.py](app/services/tool_registry_service.py#L985-L1010) & [1490-1497](app/services/tool_registry_service.py#L1490-L1497)
  - **Perubahan:** Tambahkan field `retrieval_backend` (`"pgvector_fts"` pada `_hydrate_db_record()` untuk jalur hybrid, `"pgvector"` untuk `search_tools()`, dan `"in_memory_fallback"` pada `search_tools_fallback()`). Tidak mengubah kontrak downstream.
- [ ] **NLU-2.2 — Perkuat `scripts/eval_nlu_router.py` & Unit Test Evaluator**
  - **File:** [scripts/eval_nlu_router.py](scripts/eval_nlu_router.py), [tests/test_eval_nlu_router.py](tests/test_eval_nlu_router.py)
  - **Perubahan:**
    - Guard zero-denominator pada `evaluate_exit_gates()`: jika penyebut gate (`retrieval_cases`, `semantic_cases`, `clear_cases`, `guardrail_cases`) bernilai `0` pada mode `live`, status gate adalah `FAIL` (bukan diam-diam `PASS`).
    - Tambah gate/peringatan eksplisit jika di mode `live` seluruh kasus retrieval jatuh ke `in_memory_fallback` (`pgvector_retrieval_cases == 0`).
    - Pisahkan metrik `semantic_error_cases` (`semantic_res is None` / exception) dari wrong-route misses, serta klasifikasi gagal per kasus (`retrieval_miss`, `routing_miss`, `false_clarification`, `missed_clarification`, `guardrail_regression`, `llm_error`).
    - Tambahkan metadata run pada output JSON & Markdown: `commit_sha`, `timestamp_utc`, `router_provider`, `router_model`, `dataset_path`, `pgvector_cases`, `fallback_cases`.
    - Tambahkan unit test batas threshold (`== threshold` PASS, melampaui threshold FAIL, zero-denominator FAIL, simulasi `semantic_res is None`) di [tests/test_eval_nlu_router.py](tests/test_eval_nlu_router.py).
- [ ] **NLU-2.3 — Sanitasi & Correlation pada Shadow Logging**
  - **File:** [app/graph/nodes/router.py](app/graph/nodes/router.py#L2158-L2261), [tests/test_semantic_nlu_router.py](tests/test_semantic_nlu_router.py)
  - **Perubahan:**
    - Di `_build_shadow_diff_record()`: ganti field `query` mentah dengan `query_hash` (SHA-256 12 char) + `query_len` agar teks sensitif bisnis/PII tidak tertulis di log `INFO`.
    - Sertakan `thread_id` (jika ada di state), `completed` (`status == "ok"`), `is_clarification_diff` (dipisah dari `is_tool_mismatch`), dan `legacy_ms`/`semantic_ms`.
    - Pastikan dalam mode `shadow`, hasil legacy tetap authoritative dan task shadow tidak pernah memutasi state atau mengeksekusi tool ERP.
  - **Verifikasi:** `.\.venv\Scripts\python.exe -m unittest tests/test_eval_nlu_router.py tests/test_semantic_nlu_router.py`, full suite + `ruff check` + `eval_nlu_router.py --mode offline`.

---

### Batch 3 — Staging Seed, Live Evaluation & Cluster Fixes *(VPS Staging: PGVector + Gemini)*

**Prasyarat:** Batch 1 & Batch 2 selesai dan ter-deploy (atau dijalankan di environment yang terhubung ke PostgreSQL PGVector + Gemini API key aktif).

- [ ] **NLU-3.1 — Seed Katalog 36 Tool di Staging**
  - Jalankan: `python -m app.scripts.seed_tools --update-existing`
  - Verifikasi 36 tool aktif memiliki `search_text`, `search_tsv`, dan `embedding` 768-dim.
- [ ] **NLU-3.2 — Jalankan Live Benchmark (3 Run Comparable)**
  - Jalankan 3 kali dengan model router aktif (`gemini-2.5-flash-lite` atau `gemini-2.5-flash`):
    ```bash
    python scripts/eval_nlu_router.py --mode live \
      --output docs/nlu-eval/live_run_1.json \
      --markdown docs/nlu-eval/live_run_1.md \
      --fail-on-gate
    ```
  - Pastikan laporan menunjukkan `pgvector_cases > 0` (bukan fallback in-memory) dan catat seluruh 3 run (jangan hanya ambil run terbaik).
- [ ] **NLU-3.3 — Perbaikan Failure Cluster *(Conditional)***
  - Jika ada retrieval miss → perbaiki `sample_queries` / `negative_hints` pada `CORE_TOOL_DEFINITIONS` di [tool_registry_service.py](app/services/tool_registry_service.py), reseed, ulang live eval.
  - Jika ada wrong-tool / false-clarification / latency miss → perbaiki prompt/disambiguation di [router_semantic.py](app/graph/nodes/router_semantic.py) atau sesuaikan `NLU_ROUTER_MODEL` / concurrency. **Jangan** menambah intent regex baru atau mengubah expected label golden case demi meloloskan model.
  - Ulang NLU-3.2 sampai **seluruh 5 exit gate PASS** pada seluruh run.

---

### Batch 4 — Shadow Sanity Check, Rollback Rehearsal & Default Cutover

**Prasyarat:** Batch 3 seluruh 5 exit gate PASS.

- [ ] **NLU-4.1 — Shadow Check & Rollback Rehearsal di Staging**
  - Uji `NLU_ROUTER_MODE=shadow` dengan `NLU_SHADOW_LOG_MISMATCH_ONLY=false` selama jendela observasi pragmatis (1–2 hari kerja / >= 50 request staging + smoke REST/SSE single-tool, multi-tool, RAG, hybrid, clarification turn-2, write-block).
  - Review setiap mismatch (bedakan semantic yang lebih benar vs bug nyata) — target mismatch `< 5%`, `0` guardrail regression.
  - Latih rollback cepat: ubah env ke `NLU_ROUTER_MODE=legacy_regex`, restart container, verifikasi mode kembali ke legacy, lalu kembalikan ke `semantic_llm`.
- [ ] **NLU-4.2 — Ubah Default Kode ke `semantic_llm`**
  - **File:** [app/core/config.py](app/core/config.py#L145-L148), [.env.example](.env.example), [ROADMAP_PROGRESS.md](ROADMAP_PROGRESS.md), unit test config.
  - Ubah default `NLU_ROUTER_MODE` menjadi `"semantic_llm"` (override eksplisit `legacy_regex` & `shadow` tetap dipertahankan; fallback ke `legacy_regex` saat LLM mati tetap aktif; pastikan unit test hermetic tidak bocor ke network).
  - **Checkpoint P0-3 Selesai.**

---

### Phase 4 (Ditunda — P2-1 Decommissioning Legacy Regex, Pasca-Stabilisasi)

Hanya dikerjakan setelah Batch 4 berjalan stabil minimal 1 minggu tanpa insiden fallback/guardrail:
1. **Audit dependensi legacy**: inventarisir helper di `router.py` dan `router_patterns.py` yang masih dipakai semantic/extractor (`retain` vs `remove`).
2. **Isolasi hard guards**: pisahkan guard deterministik yang dipertahankan dari legacy intent classifier.
3. **Pengganti fallback aman**: rancang degraded path tanpa regex legacy saat LLM gagal (hard guards + klarifikasi/unavailable aman, bukan tebakan tool).
4. **Hapus intent regex bertahap & satukan sync/async wrapper**: pangkas dead code di `router.py`/`router_patterns.py` dalam PR kecil terpisah dan sinkronkan dokumentasi.

---

## 6. Ringkasan Efek Perubahan terhadap Sistem

| Batch | Efek Langsung terhadap Sistem |
|---|---|
| **Batch 1 (Hardening)** | Menutup risiko prompt-injection / halusinasi tool di luar shortlist atau RBAC, menutup celah bypass `authorized_company_ids` pada follow-up `"semua cabang"` di mode semantic, dan memastikan klausa write pada kueri majemuk ditolak deterministik sebelum menyentuh LLM. Langsung mengamankan VPS staging yang sudah memakai `NLU_ROUTER_MODE=semantic_llm`. |
| **Batch 2 (Evaluator & Shadow)** | Menghilangkan risiko *false PASS* pada live benchmark (mendeteksi diam-diam jatuh ke in-memory fallback, zero-denominator, dan error/timeout LLM), menghasilkan artifact JSON/Markdown yang bisa diaudit antar-commit, dan menghapus kebocoran teks query mentah dari log shadow `INFO`. |
| **Batch 3 (Live Eval & Fix)** | Memberikan bukti numerik terukur atas 5 exit gate (recall >=98%, accuracy >=95%, false clarification <=3%, P95 delta <=800ms, 0 guardrail regression) dengan Gemini + PGVector sungguhan, sekaligus mengkalibrasi model/tool card sebelum cutover. |
| **Batch 4 (Cutover)** | Menjadikan `semantic_llm` jalur routing bawaan (meningkatkan akurasi dari baseline regex `84.04%`, menangani kasus klarifikasi `0/6` → aktif, dan memudahkan penambahan tool baru cukup via `CORE_TOOL_DEFINITIONS` tanpa menambah regex) dengan rollback instan 1 env var (`NLU_ROUTER_MODE=legacy_regex`). |

---

## 7. Perintah Verifikasi & Rollback

```powershell
# 1. Offline eval baseline / regresi (tanpa API key & tanpa DB)
.\.venv\Scripts\python.exe scripts/eval_nlu_router.py --mode offline

# 2. Targeted NLU & evaluator tests
.\.venv\Scripts\python.exe -m unittest tests/test_semantic_nlu_router.py tests/test_eval_nlu_router.py

# 3. Full regression suite + Ruff lint
.\.venv\Scripts\python.exe -m unittest discover tests
.\.venv\Scripts\ruff.exe check app cli.py scripts tests
```

```bash
# 4. Seeding & live eval (di VPS / staging dengan PGVector + Gemini aktif)
python -m app.scripts.seed_tools --update-existing
python scripts/eval_nlu_router.py --mode live --output live_eval.json --markdown live_eval.md --fail-on-gate
```

**Rollback Instan (jika terjadi insiden pasca-cutover):**
Set `NLU_ROUTER_MODE=legacy_regex` di `.env` VPS → jalankan `docker compose up -d --force-recreate` → verifikasi `/health/ready` dan smoke test 1 kueri chat. Tidak memerlukan migrasi skema database maupun revert kode selama Phase 4 (P2-1) belum dijalankan.
