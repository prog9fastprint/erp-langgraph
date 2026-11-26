# FastPrint ERP — AI Tools Handoff Context & Integration Specification

Dokumen spesifikasi resmi untuk menghubungkan agen AI eksternal (LangGraph, LangChain, CrewAI, AutoGen, atau custom agent) dengan layer `ai_tools` di FastPrint ERP.

---

## 1. Arsitektur & Gambaran Umum

FastPrint ERP menyediakan gateway API khusus AI (`/ai/tools/`) yang memisahkan eksekusi model AI dari core Django ORM.

```
┌──────────────────────────────────────────────┐
│            External AI Agent                 │
│         (LangGraph / LangChain)              │
└──────────────────────┬───────────────────────┘
                       │ HTTP REST (JSON)
                       │ Header: Authorization: Bearer <JWT>
                       │ Header: X-AI-Source: AI_TOOL
                       ▼
┌──────────────────────────────────────────────┐
│               FastPrint ERP                  │
│                                              │
│  ├── CheckSessionMiddleware (JWT & ACL)      │
│  ├── SystemErrorLogMiddleware (Audit Trail)  │
│  └── /ai/tools/ (CSRF-Exempt Dispatcher)     │
│       ├── Domain Resolver                    │
│       ├── Company Access Validator           │
│       ├── Write Confirmation Guard           │
│       └── Domain Registries                  │
│            ├── accounting                    │
│            │    ├── get_account_balance      │
│            │    ├── get_journal_entries      │
│            │    ├── get_general_ledger       │
│            │    ├── get_profit_and_loss      │
│            │    ├── get_balance_sheet        │
│            │    ├── get_vendor_bills         │
│            │    └── get_bank_reconciliation_summary │
│            ├── inventory                     │
│            │    ├── get_stock_level          │
│            │    ├── get_product_info         │
│            │    └── get_price_history        │
│            ├── sales                         │
│            │    ├── get_top_selling_products │
│            │    ├── get_sales_orders         │
│            │    ├── get_sales_order_detail   │
│            │    ├── get_sales_summary        │
│            │    └── get_contacts             │
│            ├── fastpos                       │
│            │    ├── get_product_performance  │
│            │    ├── get_stock_alerts         │
│            │    ├── get_list_orders          │
│            │    └── get_stock_opname         │
│            ├── helpdesk                      │
│            │    ├── get_complaints           │
│            │    └── get_product_returns      │
│            └── marketplace                   │
│                 ├── get_account_health       │
│                 ├── get_marketplace_orders   │
│                 ├── get_kilat_order_diagnostics │
│                 ├── get_product_mapping_status │
│                 ├── get_stock_sync_simulation │
│                 ├── get_sync_execution_logs  │
│                 ├── get_price_comparison     │
│                 ├── get_fee_and_escrow_anomalies │
│                 ├── get_deduction_and_settlement_summary │
│                 ├── get_cashflow_and_balance_summary │
│                 └── get_competitor_insights  │
└──────────────────────────────────────────────┘
```

---

## 2. Prasyarat di Sisi ERP

Untuk mengaktifkan endpoint AI di ERP Django:
1. Pastikan variabel lingkungan ini aktif di `.env` utama ERP:
   ```env
   AI_ENABLED=True
   ```
2. Ketika `AI_ENABLED=True`:
   - Django mendaftarkan app `ai_tools` ke `INSTALLED_APPS`.
   - Django me-route endpoint `/ai/` ke `ai_tools.urls`.

---

## 3. Protokol Autentikasi & Keamanan

### 3.1 HTTP Headers Wajib
Setiap pemanggilan API tool wajib menyertakan:
```http
Authorization: Bearer <jwt_access_token>
Content-Type: application/json
X-AI-Source: AI_TOOL
```

### 3.2 Programmatic Auto-Login (MD5 Secret Key)
Jika token JWT belum tersedia atau expired, agent dapat memperoleh access token baru secara otomatis melalui endpoint login ERP:

- **Endpoint**: `POST /external_api/get_token/`
- **Headers**:
  ```http
  Content-Type: application/json
  key: <md5_hash_string>
  ```
- **Algoritma `key`**:
  ```python
  import hashlib
  from datetime import datetime

  now = datetime.now().strftime("%Y-%m-%d-%H:%M")
  raw_string = f"erp_master_nexus-{now}"
  secret_key = hashlib.md5(raw_string.encode()).hexdigest()
  ```
- **Payload Request**:
  ```json
  {
    "username": "your_erp_username",
    "password": "your_erp_password"
  }
  ```
- **Response Sukses (HTTP 200)**:
  ```json
  {
    "access": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "refresh": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
  }
  ```

---

## 4. Multi-Company Scoping

- **Isolasi Data**: Data ERP terisolasi ketat berdasarkan `company_id`.
- **Validasi Akses**: Server otomatis memeriksa apakah user yang terautentikasi memiliki akses ke `company_id` yang diminta.
- **Error Unauthorized**: Jika user tidak memiliki akses ke perusahaan tersebut, API mengembalikan HTTP 403 `PERMISSION_DENIED`.

---

## 5. Kontrak API Endpoints

### 5.1 Tool Discovery
Mengambil katalog semua domain dan tool yang aktif beserta JSON schema parameternya.

- **Method & URL**: `GET /ai/tools/`
- **Response**:
  ```json
  {
    "success": true,
    "data": {
      "total_tools": 4,
      "tools": [
        {
          "domain": "accounting",
          "name": "get_account_balance",
          "description": "Calculate total debit, total credit, and net balance...",
          "is_write": false,
          "params_schema": {
            "type": "object",
            "properties": {
              "account_id": {"type": "integer", "description": "..."},
              "account_code": {"type": "string", "description": "..."},
              "date_from": {"type": "string", "format": "date", "description": "..."},
              "date_to": {"type": "string", "format": "date", "description": "..."}
            },
            "required": []
          }
        }
      ]
    }
  }
  ```

### 5.2 Tool Execution
Mengeksekusi tool tertentu dalam domain tertentu.

- **Method & URL**: `POST /ai/tools/<str:domain>/<str:tool_name>/`
- **Payload Request Structure**:
  ```json
  {
    "company_id": 1,
    "params": {
      "param_key": "param_value"
    },
    "confirm": false
  }
  ```
- **Format Response Sukses**:
  ```json
  {
    "success": true,
    "data": { ... }
  }
  ```
- **Format Response Gagal**:
  ```json
  {
    "success": false,
    "error": {
      "message": "Deskripsi error.",
      "code": "ERROR_CODE_STRING",
      "details": { ... }
    }
  }
  ```

---

## 6. Katalog 4 Pilot Tools & Spesifikasi Lengkap

### 6.1 Domain: Accounting

#### Tool 1: `accounting.get_account_balance`
- **Tujuan**: Menghitung saldo total debit, kredit, dan saldo bersih akun buku besar dalam rentang tanggal tertentu.
- **Tipe**: Read-Only (`is_write: false`)
- **Parameter**:
  | Field | Tipe | Wajib | Keterangan |
  |---|---|---|---|
  | `account_id` | integer | Opsional | ID spesifik akun |
  | `account_code` | string | Opsional | Kode akun (misal `"111100"`) |
  | `date_from` | string (YYYY-MM-DD) | Opsional | Batas awal tanggal mutasi |
  | `date_to` | string (YYYY-MM-DD) | Opsional | Batas akhir tanggal mutasi |

- **Contoh Request Payload**:
  ```json
  {
    "company_id": 1,
    "params": {
      "account_code": "111100",
      "date_from": "2026-01-01",
      "date_to": "2026-12-31"
    }
  }
  ```

- **Contoh Response Data**:
  ```json
  {
    "success": true,
    "data": {
      "accounts": [
        {
          "account_id": 12,
          "account_code": "111100",
          "account_name": "Kas Kecil Operasional",
          "total_debit": 150000000.0,
          "total_credit": 75000000.0,
          "net_balance": 75000000.0,
          "balance_position": "debit"
        }
      ],
      "summary": {
        "total_debit": 150000000.0,
        "total_credit": 75000000.0,
        "net_balance": 75000000.0
      }
    }
  }
  ```

---

#### Tool 2: `accounting.get_journal_entries`
- **Tujuan**: Mengambil entri jurnal umum beserta baris item (debit/kredit, akun lawan).
- **Tipe**: Read-Only (`is_write: false`)
- **Parameter**:
  | Field | Tipe | Wajib | Keterangan |
  |---|---|---|---|
  | `journal_id` | integer | Opsional | ID buku jurnal |
  | `entry_number` | string | Opsional | Filter nomor jurnal (misal `"JE/2026/001"`) |
  | `state` | string | Opsional | Status jurnal: `"draft"`, `"posted"`, `"cancelled"` |
  | `date_from` | string (YYYY-MM-DD) | Opsional | Filter tanggal >= YYYY-MM-DD |
  | `date_to` | string (YYYY-MM-DD) | Opsional | Filter tanggal <= YYYY-MM-DD |
  | `account_id` | integer | Opsional | Filter jurnal yang menyentuh ID akun ini |
  | `limit` | integer | Opsional | Maksimal baris diambil (default 20, maks 100) |
  | `offset` | integer | Opsional | Offset pagination (default 0) |

---

### 6.2 Domain: Inventory

#### Tool 3: `inventory.get_stock_level`
- **Tujuan**: Memeriksa kuantitas fisik (`quantity`) dan stok siap jual (`available_quantity`) per gudang.
- **Tipe**: Read-Only (`is_write: false`)
- **Parameter**:
  | Field | Tipe | Wajib | Keterangan |
  |---|---|---|---|
  | `product_id` | integer | Opsional | ID produk atau varian |
  | `warehouse_id` | integer | Opsional | ID gudang spesifik |
  | `sku` | string | Opsional | SKU produk |
  | `barcode` | string | Opsional | Barcode produk |
  | `limit` | integer | Opsional | Maksimal baris (default 20, maks 100) |
  | `offset` | integer | Opsional | Pagination offset (default 0) |

- **Contoh Response Data**:
  ```json
  {
    "success": true,
    "data": {
      "total_records": 1,
      "quants": [
        {
          "product_id": 450,
          "product_name": "Tinta Art Paper 100ml Black",
          "sku": "INK-AP-BLK-100",
          "barcode": "8991234567890",
          "warehouse_id": 3,
          "warehouse_name": "Gudang Utama Surabaya",
          "quantity": 250.0,
          "available_quantity": 240.0
        }
      ]
    }
  }
  ```

---

#### Tool 4: `inventory.get_product_info`
- **Tujuan**: Mengambil master produk, harga jual (pricelist), varian, dan satuan ukuran (UoM).
- **Tipe**: Read-Only (`is_write: false`)
- **Parameter**:
  | Field | Tipe | Wajib | Keterangan |
  |---|---|---|---|
  | `product_id` | integer | Opsional | ID spesifik produk |
  | `query` | string | Opsional | Pencarian teks nama produk |
  | `barcode` | string | Opsional | Barcode produk |
  | `sku` | string | Opsional | SKU produk |
  | `include_variants` | boolean | Opsional | Menampilkan detail varian (default `true`) |
  | `limit` | integer | Opsional | Maksimal hasil (default 20, maks 50) |

---

## 7. Keamanan Operasi Tulis (Write Safety & HITL)

- Pada setiap tool yang melakukan mutasi data (`is_write: true`):
  1. Payload request **wajib** menyertakan `"confirm": true`.
  2. Jika dikirim dengan `"confirm": false` atau tanpa field `confirm`, backend menolak eksekusi dan mengembalikan:
     ```json
     {
       "success": false,
       "error": {
         "code": "CONFIRMATION_REQUIRED",
         "message": "Tool '...' performs data modifications and requires explicit confirmation. Send 'confirm': true."
       }
     }
     ```
  3. Pada agen LangGraph, gunakan mekanisme **Human-in-the-Loop (HITL)** via `interrupt_before=["tools"]` sehingga agent meminta persetujuan manusia sebelum mengeksekusi aksi tulis ke database.

---

## 8. Panduan Prompting & Konteks Domain ERP untuk LLM

Saat mengintegrasikan LLM dengan ERP ini, sertakan panduan domain berikut di System Prompt agent Anda:
1. **Multi-Company Default**: Selalu pastikan `company_id` telah ditentukan. Jika user tidak menyebutkan perusahaan, tanyakan kepada user atau gunakan ID perusahaan default yang diset di environment.
2. **Aturan Bagan Akun Akuntansi (COA)**:
   - `1xxx` = Aset (Kas, Bank, Piutang, Persediaan) -> Bertambah di Debit.
   - `2xxx` = Kewajiban (Hutang Dagang, Pajak) -> Bertambah di Kredit.
   - `3xxx` = Modal / Ekuitas -> Bertambah di Kredit.
   - `4xxx` = Pendapatan / Sales -> Bertambah di Kredit.
   - `5xxx` = Beban Pokok Penjualan (HPP / COGS) -> Bertambah di Debit.
   - `6xxx` = Beban Operasional & Administrasi -> Bertambah di Debit.
3. **Format Tanggal**: Selalu gunakan format tanggal ISO `YYYY-MM-DD`.
- **Perbedaan Stok**:
  - `quantity`: Jumlah fisik nyata yang tersimpan di gudang.
  - `available_quantity`: Jumlah fisik dikurangi kuantitas yang sedang di-booking/direservasi pesanan belum selesai.

---

### 6.3 Domain: Marketplace (`is_write: false`)

Semua 11 tool pada domain `marketplace` bersifat read-only dan mendukung parameter opsional `company_ids: list[int]` untuk konsolidasi multi-cabang (ERP mengembalikan HTTP 403 `PERMISSION_DENIED` jika ada ID cabang di `company_ids` yang tidak diizinkan untuk user tersebut):

| Tool Name | Endpoint | Fungsi Utama |
|---|---|---|
| `get_account_health` | `POST /ai/tools/marketplace/get_account_health/` | Cek kesiapan akun toko, token OAuth (tanpa secret), konfigurasi `AccountOrder`/`AccountProperties`, status toko, dan error fetch Celery |
| `get_marketplace_orders` | `POST /ai/tools/marketplace/get_marketplace_orders/` | Monitoring order marketplace, SLA kirim (`overdue_shipping`, `due_soon`), resi AWB kosong, pembatalan, dan selisih status ERP vs platform |
| `get_kilat_order_diagnostics` | `POST /ai/tools/marketplace/get_kilat_order_diagnostics/` | Diagnosa read-only Pesanan Kilat Shopee (`invoice_updated=false`), duplikat `QuotationMarketplaceModel`, dan kandidat `LogEntry` |
| `get_product_mapping_status` | `POST /ai/tools/marketplace/get_product_mapping_status/` | Audit SKU unmapped, aturan `qty_multiplier`, duplikat mapping aktif, dan `ExceptionProduct` |
| `get_stock_sync_simulation` | `POST /ai/tools/marketplace/get_stock_sync_simulation/` | Simulasi dry-run `determine_stock()` dengan pooling `shared_stock_companies` (masked jika cabang tidak berhak) vs `StockUpdateLog` |
| `get_sync_execution_logs` | `POST /ai/tools/marketplace/get_sync_execution_logs/` | Log eksekusi `StockUpdateLog`, `PriceUpdateLog`, dan `InventoryUpdatePriceLog` (error ternormalisasi, header/secret ter-redaksi) |
| `get_price_comparison` | `POST /ai/tools/marketplace/get_price_comparison/` | Perbandingan harga ERP Pricelist/Wholesale vs listing marketplace (`include_live=false` default, maks 20 SKU saat live; butuh `account_id` atau `pricelist_id`) |
| `get_fee_and_escrow_anomalies` | `POST /ai/tools/marketplace/get_fee_and_escrow_anomalies/` | Deteksi `negative_fee`, `high_fee`, `zero_income`, dan selisih total ERP vs marketplace |
| `get_deduction_and_settlement_summary` | `POST /ai/tools/marketplace/get_deduction_and_settlement_summary/` | Rekap potongan fee per kategori kanonik, settlement belum ter-posting ke `JournalEntry`, dan validasi selisih gross vs nett |
| `get_cashflow_and_balance_summary` | `POST /ai/tools/marketplace/get_cashflow_and_balance_summary/` | Ringkasan `MarketplaceBalanceSnapshot`, mutasi `MarketplaceCashflowLog`, dan opsi `include_live=true` read-only untuk Shopee & TikTok Shop |
| `get_competitor_insights` | `POST /ai/tools/marketplace/get_competitor_insights/` | Analisis penurunan harga kompetitor, velocity `delta_sold_qty`, resolusi BOM varian, dan audit stale |

Kode error khusus domain `marketplace`:
- `MISSING_PRICELIST_OR_ACCOUNT` / `MISSING_PRICELIST_ID` (saat `get_price_comparison` tidak memiliki `account_id` maupun `pricelist_id`)
- `INVALID_COMPANY_IDS` (format `company_ids` bukan list integer)
- `ACCOUNT_NOT_FOUND` / `PRICELIST_NOT_FOUND`

---

### 6.4 Domain: FastPOS (`is_write: false`)

Domain `fastpos` mengekspos data analitik produk, klasifikasi BCG, COGS/HPP harian, ITO, Power, alert stok, antrean List Order, dan laporan Stock Opname dari `modules/fastpos` di Django ERP. Semua tool bersifat read-only dan mendukung parameter opsional `company_ids: list[int]`.

| Tool Name | Endpoint | Sumber Model Django ERP | Fungsi Utama |
|---|---|---|---|
| `get_product_performance` | `POST /ai/tools/fastpos/get_product_performance/` | `ProductStockRecap`, `ProductRecommendationRecap`, `FastposIndex`, `CogsDailySnapshot`, `BcgHistory`, `ProductAnalyticsService` | Profil performa 1 produk atau top-N produk: tren penjualan 6 bulan (`sales_m1..m6`, rata-rata, top/middle, termasuk BOM explosion MTH), stok & cover days, BCG + histori perpindahan BCG, COGS/HPP harian & gross margin, ITO, Power, Index global/kategori, serta `attention_score` (`CRITICAL`/`HIGH`/`ATTENTION`/`NORMAL`). |
| `get_stock_alerts` | `POST /ai/tools/fastpos/get_stock_alerts/` | `ProductStockRecap`, `ProductRecommendationRecap`, `stockKosongView`, `zeroReportView`, `overStockView`, `OverStockSixMonthView` | Daftar & ringkasan alert FastPOS: `stok_kosong`, `nol_sales` (dead stock), `overstock`, `overstock_6m`, dan rekomendasi kuantitas restock (`recommended_min_qty`, `recommended_stock_qty`) untuk `branch`, `jd`, maupun `mentah`. |
| `get_list_orders` | `POST /ai/tools/fastpos/get_list_orders/` | `ListOrder`, `PreListOrderItem`, `RawExpeditionHistory`, `supplierFastpos` | Monitoring antrean `ListOrder` restock (Branch/JD/MTH): filter status `nota` (`YES`/`NO`), `urgent_only`, `supplier`, usia order, estimasi waktu, dan relasi PO/MO/Interbranch. |
| `get_stock_opname` | `POST /ai/tools/fastpos/get_stock_opname/` | `StockOpname`, `LogStockOpname`, `ProblemOpname`, `WarehouseOpname` | Laporan hasil Stock Opname FastPOS per gudang/produk: qty sistem vs qty akurat, selisih, status problem, solusi, dan flag `is_done`. |

#### Kontrak Parameter & Payload Utama `fastpos`

- **`fastpos.get_product_performance`**:
  - Params: `company_id`, `company_ids`, `product_id`, `sku`, `search`, `recap_type` (`"all"|"branch"|"jd"|"mentah"`), `bcg_order` (`1..4`), `attention_level` (`"all"|"CRITICAL"|"HIGH"|"ATTENTION"|"NORMAL"`), `sort_by` (`"revenue_6m"|"power"|"ito"|"attention_score"|"cogs_pct"`), `include_bcg_history` (`false`), `include_cogs_7d` (`false`), `date_from`, `date_to`, `limit` (`1..100`, default `20`), `offset`.
  - Response `data`: `summary` (`products_analyzed`, `total_revenue_6m`, `total_inventory_value`, `critical_count`, `high_attention_count`), `total_count`, `products[]` (`product_id`, `sku`, `name`, `company_name`, `recap_type`, `bcg_order`, `sales_6m_qty`, `revenue_6m`, `sales_m1..sales_m6`, `average_sales`, `current_stock`, `indent`, `dlv_stock`, `stock_cover_days`, `cogs_per_unit`, `cogs_fp_pct`, `gross_margin_pct`, `ito`, `power`, `index_global`, `index_category`, `attention_score`, `attention_level`, `bcg_history[]`, `cogs_daily_7d[]`).
- **`fastpos.get_stock_alerts`**:
  - Params: `company_id`, `company_ids`, `alert_type` (`"all"|"stok_kosong"|"nol_sales"|"overstock"|"overstock_6m"|"restock"`), `recap_type` (`"all"|"branch"|"jd"|"mentah"`), `sku`, `search`, `bcg_order` (`1..4`), `limit` (`1..100`, default `20`), `offset`.
  - Response `data`: `summary` (`total_alerts`, `stok_kosong_count`, `nol_sales_count`, `overstock_count`, `restock_count`), `total_count`, `alerts[]` (`sku`, `name`, `company_name`, `recap_type`, `alert_type`, `current_stock`, `indent`, `dlv_stock`, `average_sales`, `stock_cover_days`, `recommended_min_qty`, `recommended_stock_qty`, `bcg_order`).
- **`fastpos.get_list_orders`**:
  - Params: `company_id`, `company_ids`, `order_type` (`"all"|"branch"|"jd"|"mth"`), `sku`, `search`, `supplier`, `nota` (`"all"|"YES"|"NO"`), `urgent_only` (`false`), `date_from`, `date_to`, `limit` (`1..100`, default `20`), `offset`.
  - Response `data`: `summary` (`total_orders`, `urgent_count`, `pending_nota_count`, `total_qty`, `total_amount`), `total_count`, `orders[]` (`id`, `tgl`, `company_name`, `sku`, `product_name`, `supplier`, `qty`, `satuan`, `harga`, `total`, `bcg`, `nota`, `checklist_nota`, `urgent`, `usia`, `qc_note`).
- **`fastpos.get_stock_opname`**:
  - Params: `company_id`, `company_ids`, `sku`, `search`, `warehouse_name`, `status`, `problem_only` (`false`), `is_done` (`null|true|false`), `date_from`, `date_to`, `limit` (`1..100`, default `20`), `offset`.
  - Response `data`: `summary` (`total_items`, `matched_count`, `discrepancy_count`, `open_problem_count`), `total_count`, `opnames[]` (`id`, `date`, `company_name`, `warehouse_name`, `sku`, `product_name`, `quantity_stock`, `quantity_accurate`, `difference`, `status`, `status_problem`, `solution`, `is_done`).

---

### 6.5 Domain: Helpdesk (`is_write: false`)

Domain `helpdesk` mengekspos data tiket komplain pelanggan (`modules/helpdesk/models/tiketModel.py` & 4 laporan agregat `reportingView.py`) serta data retur & refund produk lintas modul. Semua tool bersifat read-only dan mendukung `company_ids: list[int]`.

| Tool Name | Endpoint | Sumber Model Django ERP | Fungsi Utama |
|---|---|---|---|
| `get_complaints` | `POST /ai/tools/helpdesk/get_complaints/` | `helpdesk.Tiket` (exclude `Q_SERVICE_TICKETS` default), `reportingView.py` | Daftar & ringkasan tiket komplain pelanggan per produk/cabang/periode: kendala produk, stage (`1=Masuk`, `2=Pengecekan`, `3=Selesai`), durasi penanganan, rasio komplain vs penjualan, dan kerugian (`harga_produk + ongkir_komplain`). |
| `get_product_returns` | `POST /ai/tools/helpdesk/get_product_returns/` | `inventory.ReturnTransfer`, `helpdesk.Tiket` (pengecekan retur), `accounting.JournalEntry` + `RefundsLine` (`refund customer`) | Rekap & rincian transaksi retur produk lintas sumber: total qty retur, nilai refund, alasan/kendala, hasil pengecekan retur, dan referensi SO asal. |

#### Kontrak Parameter & Payload Utama `helpdesk`

- **`helpdesk.get_complaints`**:
  - Params: `company_id`, `company_ids`, `sku`, `search`, `order_ref`, `customer_name`, `issue_category`, `stage` (`"all"|"open"|"checking"|"done"`), `include_service_stempel` (`false`), `report_view` (`"list"|"summary"|"by_branch"|"by_issue"|"vs_sales"|"loss"`), `date_from` (default YTD), `date_to` (default hari ini), `limit` (`1..100`, default `20`), `offset`.
  - Response `data`: `date_window`, `summary` (`total_tickets`, `open_count`, `checking_count`, `done_count`, `total_qty`, `total_product_loss`, `total_shipping_loss`, `total_loss`, `complaint_vs_sales_pct`), `by_issue[]`, `by_branch[]`, `total_count`, `complaints[]` (`ticket_id`, `name`, `created_at`, `company_name`, `so_number`, `customer_name`, `sku`, `product_name`, `qty`, `kendala_produk`, `tindakan_lanjutan`, `stage_label`, `harga_produk`, `ongkir_komplain`, `durasi`).
- **`helpdesk.get_product_returns`**:
  - Params: `company_id`, `company_ids`, `sku`, `search`, `order_ref`, `source` (`"all"|"return_transfer"|"helpdesk"|"refund"`), `date_from` (default YTD), `date_to` (default hari ini), `limit` (`1..100`, default `20`), `offset`.
  - Response `data`: `date_window`, `summary` (`total_return_events`, `total_qty_returned`, `total_refund_amount`, `helpdesk_return_count`, `transfer_return_count`, `refund_count`, `return_rate_pct`), `by_reason[]`, `total_count`, `returns[]` (`date`, `source`, `reference`, `so_number`, `company_name`, `sku`, `product_name`, `qty`, `refund_amount`, `reason`, `inspection_result`, `status`).

---

### 6.6 Tool Tambahan Domain Inventory: `inventory.get_price_history` (`is_write: false`)

- **Endpoint**: `POST /ai/tools/inventory/get_price_history/`
- **Sumber Model Django ERP**:
  1. `ess.LogEntry` (`model='PricelistLineModel'`, di-log oleh `PricelistService.update_pricelist_lines`)
  2. `marketplace.PriceUpdateLog` (`status='SUCCESS'` default, atau semua jika `include_failed=true`)
  3. `inventory.InventoryUpdatePriceLog` (`service_type='sales_pricelist'` untuk sinkronisasi PCLS)
  4. `sales.PricelistLineModel` (snapshot harga jual terkini per pricelist & tier qty)
- **Parameter**:
  | Field | Tipe | Default | Keterangan |
  |---|---|---|---|
  | `company_id` / `company_ids` | integer / list[int] | active company | Cakupan perusahaan / multi-cabang |
  | `product_id` / `sku` / `search` | integer / string | `null` | Identitas produk (ID, SKU exact, atau keyword nama) |
  | `source` | `"all"\|"erp_pricelist"\|"marketplace"\|"pcls"` | `"all"` | Filter sumber riwayat perubahan harga |
  | `marketplace` | string | `null` | Filter platform (`Shopee`, `TikTok Shop`, `Tokopedia`, `Lazada`) |
  | `pricelist_name` | string | `null` | Filter nama pricelist ERP (misal `Jawa`, `Luar Jawa`, `Medan`) |
  | `include_failed` | boolean | `false` | Sertakan entri log `FAILED` selain `SUCCESS` |
  | `date_from` / `date_to` | string (`YYYY-MM-DD`) | 90 hari terakhir s/d hari ini | Rentang waktu kronologi perubahan harga |
  | `limit` / `offset` | integer | `20` / `0` | Paginasi (`limit` maks `100`) |
- **Response `data`**:
  ```json
  {
    "date_window": {"date_from": "2026-08-28", "date_to": "2026-11-26"},
    "current_prices": [
      {"pricelist_name": "Pricelist FP Jawa", "min_qty": 1, "price": 45000.0, "region": "jawa"}
    ],
    "summary": {
      "total_changes": 3,
      "latest_change_date": "2026-11-18",
      "min_price": 42000.0,
      "max_price": 45000.0,
      "net_change_idr": 3000.0,
      "net_change_pct": 7.14
    },
    "total_count": 3,
    "changes": [
      {
        "changed_at": "2026-11-18 10:15:00",
        "source": "erp_pricelist",
        "channel_or_pricelist": "Pricelist FP Jawa",
        "sku": "SKU-A01",
        "product_name": "Barang A",
        "qty_tier": "1",
        "old_price": 42000.0,
        "new_price": 45000.0,
        "delta_idr": 3000.0,
        "delta_pct": 7.14,
        "updated_by": "Admin Pricing",
        "status": "SUCCESS"
      }
    ]
  }
  ```

---

## 9. Cara Menambahkan Tool Baru di Django ERP

Untuk memperluas katalog tools di masa depan:
1. Buat class baru di `ai_tools/tools/<domain>/<tool_name>.py` mewarisi `BaseTool`.
2. Tentukan `domain`, `name`, `description`, `is_write`, dan `params_schema`.
3. Implementasikan `validate_params(params)` dan `execute(user, company, params)`.
4. Daftarkan di `ai_tools/tools/<domain>/registry.py`.
5. Tool baru akan langsung otomatis muncul di `GET /ai/tools/` tanpa perlu restart atau ubah kode di sisi client agent (jika menggunakan dynamic loader).
