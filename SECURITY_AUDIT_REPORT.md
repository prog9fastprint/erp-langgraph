# Laporan Audit Keamanan: Core Logic FastPrint ERP AI Orchestrator

**Target Repository**: `C:\laragon\www\erp-langgraph`  
**Tanggal Audit**: 21 November 2026  
**Auditor Bot**: Security Check (Orca)  
**Status**: Review Ready  

---

## 1. Ringkasan Eksekutif (Executive Summary)

Audit keamanan berfokus pada **Core Logic** arsitektur *ERP AI Orchestrator* (`app/main.py`, `app/core/`, `app/graph/`, `app/services/`, dan `app/schemas/`). 

Secara umum, arsitektur telah menerapkan beberapa praktik perlindungan yang baik seperti sanitasi token pada *Server-Sent Events (SSE)* dan pembersihan metadata sensitif (*Django ORM audit fields*) sebelum data diproses LLM. Namun, ditemukan **5 temuan keamanan signifikan**—termasuk satu celah berisiko **Tinggi/Kritis (*Privilege Escalation*)** yang memungkinkan client memanipulasi hak akses otorisasi (*Role Spoofing*).

### Matriks Temuan Keamanan

| ID | Celah Keamanan | Kategori | Tingkat Risiko | Status |
| :--- | :--- | :--- | :---: | :---: |
| **SEC-01** | Manipulasi Hak Akses via Body Payload (`user_roles`) | Broken Access Control | **Tinggi (High)** | Perlu Perbaikan |
| **SEC-02** | Hardcoded Secret & Hashing MD5 Prediktif untuk Login ERP | Cryptographic Failure / Secret Leak | **Tinggi (High)** | Perlu Perbaikan |
| **SEC-03** | Default Autentikasi Longgar & Fallback ke Akun Sistem Master | Authentication Bypass | **Sedang-Tinggi** | Perlu Perbaikan |
| **SEC-04** | Potensi Cross-Tenant IDOR pada `company_id` | Broken Object Level Auth | **Sedang (Medium)** | Perlu Perbaikan |
| **SEC-05** | Eksposur Swagger `/docs` & Bocoran Detail Exception | Information Disclosure | **Rendah-Sedang** | Perlu Perbaikan |

---

## 2. Rincian Temuan Keamanan (Detailed Findings)

### SEC-01: Privilege Escalation via Untrusted Client Roles (*Broken Function Level Authorization*)
- **Lokasi Kode**:
  - `app/schemas/chat.py` (Baris 31): `user_roles: Optional[List[str]] = Field(...)`
  - `app/main.py` (Baris 235): Meneruskan `payload.user_roles` langsung ke *workflow*
  - `app/services/tool_registry_service.py` (Baris 478–479 & 574):
    ```python
    is_admin = bool(user_roles and "admin" in user_roles)
    ```
    ```sql
    AND ($3::text[] IS NULL AND required_permission IS NULL 
         OR $3::text[] IS NOT NULL AND (required_permission IS NULL OR required_permission = ANY($3) OR 'admin' = ANY($3)))
    ```
- **Dampak / Eksploitasi**:
  Client atau penyerang dapat mengirimkan request HTTP POST ke `/api/v1/chat` dengan menyisipkan JSON body:
  ```json
  {
    "query": "Tampilkan laporan keuangan laba rugi dan neraca",
    "user_roles": ["admin", "accounting.view_financial_reports"]
  }
  ```
  Sistem mempercayai parameter ini tanpa memverifikasi signature atau klaim peran (*roles claim*) dari token JWT resmi. Penyerang langsung mendapatkan izin admin/manajerial untuk menjalankan semua tool ERP sensitif.
- **Rekomendasi Perbaikan**:
  1. Hapus parameter `user_roles` dari skema input `ChatRequest`.
  2. Dekode dan verifikasi tanda tangan `user_token` (JWT) di level middleware/endpoint, lalu ekstrak `roles` / `permissions` dari payload JWT yang valid.

---

### SEC-02: Hardcoded Secret & Predictable MD5 Token Generation
- **Lokasi Kode**:
  - `app/services/django_client.py` (Baris 252–256)
  - `app/services/erp_client.py` (Baris 51–55)
  ```python
  def _generate_secret_key(self) -> str:
      now = get_current_erp_datetime().strftime("%Y-%m-%d-%H:%M")
      raw_string = f"erp_master_nexus-{now}"
      return hashlib.md5(raw_string.encode()).hexdigest()
  ```
- **Dampak / Eksploitasi**:
  Secret key `"erp_master_nexus-"` ditulis *hardcoded* di dalam repositori dan menggunakan fungsi hash MD5 yang rentan. Siapa pun yang memiliki akses ke kode sumber (atau hasil kompilasi) dapat menghitung header `key` secara deterministik pada menit berjalan untuk memanggil endpoint `/external_api/get_token/` dan membobol autentikasi ERP.
- **Rekomendasi Perbaikan**:
  1. Hapus string statis `"erp_master_nexus"`.
  2. Gunakan mekanisme autentikasi standar: API Key berkredensial tinggi yang disimpan di environment variable (`.env`), tanda tangan HMAC-SHA256, atau token OAuth2 / Service Account JWT.

---

### SEC-03: Default Autentikasi Longgar & Fallback ke Akun Sistem Master
- **Lokasi Kode**:
  - `app/core/config.py` (Baris 23–24): `REQUIRE_USER_TOKEN: bool = False`, `REQUIRE_SERVICE_KEY: bool = False`
  - `app/main.py` (Baris 204–212): Pemeriksaan token hanya dipaksa jika `APP_ENV == "production"`
  - `app/services/django_client.py` (Baris 362–364):
    ```python
    active_token = clean_user_token or self.token
    if not active_token:
        active_token = await self.login()
    ```
- **Dampak / Eksploitasi**:
  Jika service dijalankan di server staging, testing publik, atau lupa diset ke `production`, pengguna tanpa token autentikasi tetap dapat menggunakan bot. Sistem secara otomatis menggunakan token service/master (`self.token`), mengeksekusi aksi dengan kredensial superuser orchestrator.
- **Rekomendasi Perbaikan**:
  1. Setel `REQUIRE_USER_TOKEN: bool = True` sebagai nilai default.
  2. Cegah fallback otomatis ke master token jika konteks request berasal dari user chat biasa.

---

### SEC-04: Potensi Cross-Tenant IDOR pada `company_id`
- **Lokasi Kode**:
  - `app/schemas/chat.py` (Baris 26–27): `company_id` & `company_ids` diterima bebas dari request body
  - `app/graph/workflow.py` (Baris 345–385): Resolusi otomatis cabang dari teks atau payload
- **Dampak / Eksploitasi**:
  Pengguna dari Cabang A dapat menyisipkan `company_id: 6` (misal: Cabang Bandung) atau mengetik *"cek stok di cabang bandung"* atau *"semua cabang"*. Orchestrator saat ini belum memvalidasi apakah user yang bersangkutan memiliki izin operasional ke cabang tersebut. Jika request diproses menggunakan master/service token, data cabang lain dapat bocor (*Cross-Tenant Data Leakage*).
- **Rekomendasi Perbaikan**:
  Validasi nilai `company_id` terhadap daftar cabang yang diizinkan untuk akun pengguna (misal dari klaim `allowed_companies` dalam JWT).

---

### SEC-05: Informasi Debug Internal & Endpoint Swagger Terbuka
- **Lokasi Kode**:
  - `app/main.py` (Baris 100–101): `docs_url="/docs"`, `redoc_url="/redoc"` aktif tanpa proteksi
  - `app/main.py` (Baris 298–304): `detail = f"Chat workflow failed: {str(exc)}"` saat non-prod
- **Dampak / Eksploitasi**:
  - Dokumentasi API dan skema internal dapat dibaca siapa saja di jaringan tanpa login.
  - Pesan exception mentah pada lingkungan non-prod dapat menampilkan path sistem lokal (`C:\laragon\www\...`), konfigurasi database, atau library internal saat terjadi error.
- **Rekomendasi Perbaikan**:
  Matikan dokumentasi interaktif di lingkungan production:
  ```python
  docs_url="/docs" if settings.APP_ENV != "production" else None
  redoc_url=None
  ```
  Gunakan error ID acak (`X-Request-ID`) dan log detail exception ke server internal, bukan mengembalikannya ke client.

---

## 3. Aspek Positif yang Sudah Diimplementasikan (Good Practices)

- **Sanitasi Kredensial pada Event SSE**: `app/core/context.py` secara aktif menghapus token, password, dan service key sebelum di-stream ke browser client (`sanitize_tool_params`).
- **Pruning Audit Trail Django ORM**: `sanitize_erp_payload` membersihkan field audit seperti `create_uid`, `access_token`, dan `audit_trail` agar tidak dikirim ke LLM model pihak ketiga.
- **Constant-Time Comparison**: Pengecekan service key menggunakan `hmac.compare_digest` untuk memitigasi serangan *timing attack*.

---

## 4. Langkah Remediasi yang Disarankan

1. **Prioritas 1 (Kritis)**: Hapus field `user_roles` dari `ChatRequest` di `app/schemas/chat.py`. Buat dependensi FastAPI untuk mengekstrak dan memvalidasi role dari token JWT.
2. **Prioritas 1 (Kritis)**: Pindahkan `erp_master_nexus` ke environment variable rahasia dan beralih ke format hash HMAC yang aman.
3. **Prioritas 2**: Tambahkan middleware validasi multi-tenant (`company_id`) berdasarkan token JWT.
4. **Prioritas 3**: Nonaktifkan `/docs` pada production mode dan samarkan error trace.
