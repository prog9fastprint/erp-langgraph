# Handover Stabilisasi Chat ERP untuk Antigravity

## Cara melanjutkan

Lanjutkan dari dua worktree berikut. Perubahan masih berupa diff lokal dan belum di-commit. Periksa `git status --short --branch`, `git diff`, dan file tes untracked sebelum mengedit. Checkout asal tetap menjadi lokasi [rencana yang direview](2026-12-27-erp-chat-stabilization.md).

| Repo | Worktree | Branch | Base |
|---|---|---|---|
| LangGraph | `/Users/programmer/fastprint-project/erp-langgraph-chat-stabilization` | `codex/chat-stabilization` | `183b659` |
| Django ERP | `/Users/programmer/fastprint-project/ERP-chat-stabilization` | `codex/chat-stabilization` | `897c3a240` |

Seluruh pengujian sejauh ini memakai fixture atau mock. Tidak ada migrasi, perubahan konfigurasi staging, deploy, maupun akses ke database aktif. Pertahankan operasi ERP read-only, cabang yang diotorisasi, format kolom tabel yang sudah dipakai, dan kontrak API/SSE. Jalankan replay di staging dengan akun, thread, dan data uji terisolasi.

## Perubahan yang sudah ada

- **LangGraph.** Memperbarui tes katalog 37 tool inti dan golden case PDF. Menambahkan fallback klarifikasi untuk rencana tool kosong, jawaban cadangan untuk output kosong, penyaringan chip placeholder, dan pemakaian laporan sebelumnya hanya dalam cakupan cabang yang sesuai. Mempertahankan tanggal invoice saat ganti cabang, menahan retry yang akan memperlebar pencarian invoice belum dibayar, dan mengambil nomor jurnal dari klausa pengguna. Menambahkan tabel angka rekonsiliasi lintas cabang dan deduplikasi permintaan baca identik.
- **Pencarian jurnal.** LangGraph mengirim `entry_number` dan `search` untuk kompatibilitas. Django memakai filter `number__iexact` sebelum count dan pagination ketika `entry_number` ada. Fallback dari HQ dibatasi ke cabang yang diotorisasi. Daftar cabang eksplisit dan cabang yang dipilih lewat chat membatasi fallback. Sumber jurnal tampil di badge respons, tautan, dan metadata tabel tanpa menggeser cabang sesi.
- **Django.** Alias `posted` memetakan ke `post`. Filter `unpaid` memilih invoice/bill aktif berstatus `draft`, `post`, atau `partial`, dengan scope perusahaan dan periode tetap berlaku. Pencarian saldo mencocokkan setiap kata terhadap kode, alias, atau nama akun. Penyimpanan balasan SSE, pembuatan pesan pengguna, dan Undo mengunci thread yang sama. Balasan terlambat dilewati jika pesan pemicu sudah dihapus atau digantikan.

## Verifikasi terakhir

- **LangGraph.** Suite penuh lulus **882 tes** sebelum perubahan hasil review terakhir. Setelah perubahan scope, periode, dan nomor jurnal, **55 tes terfokus** lulus. Jalankan suite penuh lagi dari diff terkini. Satu tes ingest membuat fixture sementara di `tests/` sehingga sandbox read-only perlu izin tulis untuk suite penuh.
- **Django.** **115 tes mock-only** terkait lulus dengan backend database dummy. Tes mencakup filter jurnal, SSE terputus, dan Undo. Run opsional yang lebih luas menemukan `test_marketplace_redirect_id_fields_present_in_tool_outputs` gagal dengan `KeyError: 'items'`. Kegagalan yang sama direproduksi di checkout Django asal.
- Pemeriksaan diff terakhir lulus di kedua worktree. Di LangGraph, gunakan `git -c core.whitespace=cr-at-eol diff --check` karena sebagian file memakai CRLF.

Dari worktree LangGraph, jalankan suite dengan konfigurasi tes CI.

```bash
source ../erp-langgraph/.venv/bin/activate
APP_ENV=test FASTPRINT_TEST_ENV=true OPENAI_API_KEY=dummy-key-for-testing \
ERP_BASE_URL=http://127.0.0.1:8000 PYTHONDONTWRITEBYTECODE=1 \
python -m unittest discover tests
```

Dari worktree Django, gunakan perintah berikut untuk mengulang 115 tes. Pengaturan proses dan `.env` diabaikan, dan koneksi database memakai backend dummy.

```bash
../ERP/venv/bin/python -B - <<'PY'
import os
import unittest
from unittest.mock import patch
import decouple
import django
from django.conf import settings

overrides = {
    "SECRET_KEY": "local-unit-test-key",
    "BASE_URL": "http://testserver",
    "DB_NAME": "unit-test",
    "DB_USER": "unit-test",
    "DB_PASSWORD": "unit-test",
    "DB_HOST": "",
    "DB_PORT": "",
}
with patch.dict(os.environ, overrides, clear=True):
    decouple.config = decouple.Config(decouple.RepositoryEmpty())
    from erp import settings as project_settings

    config = {key: getattr(project_settings, key) for key in dir(project_settings) if key.isupper()}
    config["DATABASES"] = {"default": {"ENGINE": "django.db.backends.dummy"}}
    settings.configure(**config)
    django.setup()

    from django.urls import set_urlconf
    from ess.urls import urlpatterns

    set_urlconf(tuple(urlpatterns))
    suite = unittest.defaultTestLoader.loadTestsFromNames([
        "modules.aiassistant.test_proxy_unit", "ai_tools.tests",
    ])
    result = unittest.TextTestRunner(verbosity=1).run(suite)
raise SystemExit(not result.wasSuccessful())
PY
```

Dalam sandbox read-only, jalankan isi blok Python dengan argumen `-c` jika shell memerlukan file temporer untuk heredoc.

## Pekerjaan berikutnya

1. **Mulai dari tes routing yang belum ditulis.** Percobaan menambahkan tes ini dihentikan sebelum patch diterapkan. Di `legacy_regex`, prompt audit #13 tentang invoice belum dibayar jatuh ke `general`, sedangkan #15 yang menyebut nomor BSH dan jurnal bisa memilih `fetch_vendor_invoices`. Reproduksi lewat `arouter_node` lalu perbaiki routing agar kedua prompt mencapai `get_journal_entries`. Pertahankan guardrail tagihan pelanggan eksplisit dan alur tagihan vendor.
2. **Ulangi urutan audit #13 → #14 → #15.** Saat #14 mengunci sesi ke Jakarta, #15 dapat mewarisi cabang Jakarta dan tidak memicu pencarian dari HQ. Bedakan permintaan nomor jurnal baru dari follow-up yang sengaja mempertahankan cabang. Jangan abaikan cabang eksplisit di permintaan baru. Jika nomor yang sama ditemukan di beberapa cabang, implementasi saat ini memilih kecocokan pertama berdasarkan urutan otorisasi. Uji kebutuhan disambiguasi sebelum mengandalkan hasil itu.
3. **Lakukan verifikasi akhir kedua repo.** Jalankan suite penuh LangGraph dan tes Django terdampak dengan backend dummy. Tambahkan cakupan checkpoint untuk pergantian cabang dan pencabutan otorisasi. Pastikan replay tabel tidak mengarang angka per cabang per bulan saat payload Django hanya menyediakan total per cabang dan total per bulan.
4. **Replay 19 turn di staging terisolasi.** Catat konfigurasi aktif sebagai baseline sebelum mencoba model router yang lebih cepat atau `SYNTHESIZER_FAST_DETERMINISTIC`. Syarat rilis tetap semua P0 tertutup, minimal **18 dari 19** turn lulus, dan median **maksimal 10 detik**. Catat p95. Status sekarang belum memenuhi verifikasi rilis. [Audit staging](../../archive/ai_chat_audit_2026_11_27.md)
