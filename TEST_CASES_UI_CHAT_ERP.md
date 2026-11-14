# Panduan & Matriks Test Case Pengujian UI Chat FastPrint ERP

Dokumen ini merupakan panduan skenario pengujian manual dan otomatis untuk antarmuka **UI Chat FastPrint ERP AI Orchestrator**. Seluruh skenario telah diselaraskan dengan data operasional riil yang tersimpan pada database ERP.

---

## 1. Referensi Data Riil Database ERP

Gunakan kode SKU, nama produk, dan periode pembukuan berikut saat melakukan pengujian di UI Chat:

| Kode SKU / Barcode | Nama Produk di Katalog ERP | Stok Fisik (Pusat / HEAD) | Harga Satuan (Rp) | Catatan Pengujian |
| :--- | :--- | :--- | :--- | :--- |
| `91010600` | FP KARET STEMPEL FLASH BULAT 42 | **30 unit** | **Rp 1.087,55** | Uji coba happy path stok & harga |
| `91030300` | FP KARET STEMPEL FLASH KOTAK 18x55 | **18 unit** | **Rp 323,49** | Uji coba batch multi-produk |
| `1000020201` | FP CANON DYE ANTI UV 1000ML - BLACK | - | **Rp 116.050,00** | Uji coba harga jual produk nominal besar |
| `100040202` | FP BROTHER DYE ANTI UV 100ML - CYAN | - | **Rp 6.948,52** | Uji coba harga via pencarian nama |
| `1000040207` | FP BROTHER DYE ANTI UV 1000ML 1 SET 4 WARNA | **0 unit** (Nihil) | Rp 0,00 | Uji coba penanganan stok kosong / habis |
| `SKU-UNKNOWN-999` | *(Tidak Terdaftar)* | - | - | Uji coba produk tidak ditemukan (Negative) |

> **Catatan Pembukuan Akuntansi**: Data transaksi dan jurnal keuangan aktif berada pada periode tahun berjalan **2026** (Pendapatan Usaha Konsolidasi: ~Rp 115,44 Miliar).

---

## 2. Matriks Skenario Pengujian

### Kategori A: Tanya Stok Barang (`inventory.get_stock_level`)

| ID Test Case | Jenis Skenario | Prompt Siap Salin ke UI Chat | Pengaturan UI | Ekspektasi Respon UI | Kriteria Lolos (Pass) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TC-STK-01** | Happy Path (SKU) | `Berapa stok untuk SKU 91010600?` | Default Cabang (Pusat / 1) | Tabel informasi stok memuat SKU `91010600`, nama `FP KARET STEMPEL FLASH BULAT 42`, dan Qty Fisik **30 unit**. | Tabel muncul rapi, kuantitas 30 unit. |
| **TC-STK-02** | Happy Path (Nama Produk) | `Cek stok Karet Stempel Flash Kotak 18x55` | Default Cabang | AI mencocokkan nama dengan SKU `91030300` dan menampilkan stok **18 unit**. | Data produk dan stok ditemukan tanpa error. |
| **TC-STK-03** | Breakdown Lokasi/Gudang | `Tampilkan stok 91010600 lengkap dengan rincian gudang dan rak` | Default Cabang | Muncul sub-tabel `Rincian Stok per Lokasi / Rak` berisi Gudang `HEAD`, Lokasi `Stock`, dan Kuantitas `30 unit`. | Sub-tabel breakdown lokasi tampil lengkap. |
| **TC-STK-04** | Stok Kosong (Zero Stock) | `Berapa sisa stok untuk SKU 1000040207?` | Default Cabang | AI mengonfirmasi produk terdaftar (`FP BROTHER DYE ANTI UV 1000ML 1 SET 4 WARNA`) namun stok saat ini **0 / kosong**. | Menyatakan stok 0 tanpa mengarang angka. |
| **TC-STK-05** | Negatif (SKU Tidak Terdaftar) | `Cek stok barang SKU-UNKNOWN-999` | Default Cabang | Status verifikasi `EMPTY`. AI menginformasikan bahwa produk tidak ditemukan dalam katalog tanpa error 500. | Respon sopan produk tidak ditemukan. |
| **TC-STK-06** | Multi-Produk Batch | `Cek ketersediaan stok untuk 91010600 dan 91030300` | Default Cabang | Menampilkan kedua SKU dalam 1 tabel perbandingan stok. | Kedua SKU terangkum dalam 1 tabel. |

---

### Kategori B: Tanya Nilai Laba Rugi Multi-Cabang + HO (`accounting.get_profit_and_loss`)

| ID Test Case | Jenis Skenario | Prompt Siap Salin ke UI Chat | Pengaturan UI | Ekspektasi Respon UI | Kriteria Lolos (Pass) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TC-PL-01** | Happy Path (Checkbox UI) | `Berapa nilai laba rugi tahun 2026?` | **Centang Semua Cabang + HO** (`company_ids: [1..17]`) | Menampilkan tabel `Laporan Laba Rugi Multi-Perusahaan (Combined Profit & Loss)` dengan kolom: No, Perusahaan/Cabang, Pendapatan Usaha, HPP, Laba Kotor, Laba Bersih, dan baris **TOTAL GABUNGAN**. | Tabel multi-cabang tampil dengan baris TOTAL GABUNGAN. |
| **TC-PL-02** | Happy Path (Prompt Teks Murni) | `Tampilkan laporan laba rugi tahun 2026 untuk semua cabang dan HO` | **Single Company** (Tanpa centang checkbox) | AI mengenali frasa *"semua cabang dan HO"*, otomatis mengeksekusi multi-company rollup ke seluruh cabang. | Tabel konsolidasi multi-cabang tampil lengkap. |
| **TC-PL-03** | Rincian Akun (Hierarki) | `Tampilkan laba rugi tahun 2026 dengan rincian akun untuk semua cabang` | Centang Semua Cabang + HO ATAU Teks | Menampilkan tabel ringkasan konsolidasi per cabang dan sub-tabel rincian akun beban pokok serta beban operasional. | Rincian akun hierarki tampil terstruktur. |
| **TC-PL-04** | Komparasi 2 Cabang Spesifik | `Bandingkan laba rugi cabang Surabaya dan Jakarta tahun 2026` | Bebas | Respon tabel perbandingan khusus cabang Surabaya (`company_id: 7`) dan Jakarta (`company_id: 8`) serta baris total. | Hanya 2 cabang terkait yang diperbandingkan. |
| **TC-PL-05** | Hak Akses Terbatas (Partial 403) | `Laba rugi seluruh cabang tahun 2026` *(login sebagai user tanpa izin cabang tertentu)* | Centang Semua Cabang + HO | Cabang yang berizin menampilkan angka normal, cabang yang dibatasi menampilkan `Akses Dibatasi (HTTP 403)`, dan TOTAL GABUNGAN tetap valid. | Sistem tidak crash 500; status pembatasan izin terisolasi. |

---

### Kategori C: Tanya Harga Barang (`inventory.get_product_info`)

| ID Test Case | Jenis Skenario | Prompt Siap Salin ke UI Chat | Pengaturan UI | Ekspektasi Respon UI | Kriteria Lolos (Pass) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TC-PRC-01** | Happy Path (SKU) | `Berapa harga jual barang dengan SKU 1000020201?` | Default Cabang | Menampilkan informasi master produk `FP CANON DYE ANTI UV 1000ML - BLACK` dan `Harga Jual: Rp 116.050`. | Format harga berlabel `Rp` dengan pemisah ribuan. |
| **TC-PRC-02** | Happy Path (Nama Produk) | `Berapa harga produk FP Brother Dye Anti UV 100ml Cyan?` | Default Cabang | AI mencocokkan produk dengan SKU `100040202` dan menampilkan `Harga Jual: Rp 6.948,52`. | Harga produk ditemukan sesuai nama. |
| **TC-PRC-03** | Harga Jual & Estimasi HPP | `Berapa harga jual dan estimasi HPP untuk SKU 91010600?` | Default Cabang | Menampilkan `FP KARET STEMPEL FLASH BULAT 42` beserta `Estimasi HPP: Rp 1.087,55`. | Informasi harga jual dan biaya/HPP tampil. |
| **TC-PRC-04** | Negatif (Produk Tidak Ada) | `Berapa harga produk Kertas HVS Alien 500gsm?` | Default Cabang | AI merespon dengan sopan bahwa produk tidak ditemukan dalam katalog tanpa mengarang angka fiktif. | Tidak menampilkan harga halusinasi. |

---

### Kategori D: Guardrails & Non-ERP Intent

| ID Test Case | Jenis Skenario | Prompt Siap Salin ke UI Chat | Pengaturan UI | Ekspektasi Respon UI | Kriteria Lolos (Pass) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TC-GRD-01** | Greeting / Percakapan | `Halo selamat pagi, kamu bisa bantu apa saja di ERP?` | Bebas | Respon ramah memaparkan kapabilitas sistem (cek stok, laba rugi, katalog produk) tanpa melakukan query database. | Menjawab ramah tanpa query tool ERP. |
| **TC-GRD-02** | Out of Scope / Non-Bisnis | `Siapa juara piala dunia tahun 2022?` | Bebas | AI menolak secara sopan dan mengarahkan kembali ke topik pengelolaan operasional FastPrint ERP. | Menolak secara halus terarah ke konteks ERP. |

---

## 3. Checklist Evaluasi & Validasi UI

Periksa elemen antarmuka berikut pada setiap respon di UI Chat ERP:

1. **Format Mata Uang & Angka**:
   - Seluruh nominal moneter diawali `Rp` dengan format pemisah ribuan standar Indonesia (contoh: `Rp 1.087,55` atau `Rp 115.440.844.730,82`).
   - Kuantitas stok menyertakan satuan yang jelas (contoh: `30 unit`, `18 unit`).
2. **Kerapian Tabel Markdown**:
   - Kolom tabel selaras (header dan baris isi lurus).
   - Baris `TOTAL GABUNGAN` dicetak tebal (**bold**).
3. **Chip Rekomendasi Pertanyaan Lanjutan (Suggested Follow-ups)**:
   - Di bawah balon jawaban AI, pastikan tampil tombol chip pertanyaan lanjutan yang kontekstual (contoh: *"Lihat rincian beban operasional"*, *"Cek sisa stok di gudang lain"*).
4. **Indikator Loading**:
   - UI menampilkan indikator animasi pengetikan (*typing indicator*) selama backend memproses data.

---

## 4. Eksekusi Pengujian Otomatis

Untuk menjalankan seluruh rangkaian pengujian unit dan integrasi multi-cabang secara otomatis di terminal:

```powershell
# Jalankan test suite multi-company
.\.venv\Scripts\python.exe -m unittest tests/test_multi_company.py

# Jalankan seluruh test suite orchestrator
.\.venv\Scripts\python.exe -m unittest discover tests
```
