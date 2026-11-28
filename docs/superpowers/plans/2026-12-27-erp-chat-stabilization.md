# Rencana Stabilisasi Chat ERP

**Draf untuk review.** Waktu menuju rilis kurang dari dua minggu. Fokus fase ini adalah menutup kegagalan audit yang masih muncul di LangGraph dan Django dengan alur 7 node saat ini.

## Langkah pertama

Ulangi 19 turn audit lewat UI Django di staging sebelum mengubah kode atau konfigurasi. Pertahankan urutan percakapan, konteks cabang, dan periode laporan. Catat versi kedua layanan, mode router, model aktif, hasil yang diharapkan dari Django, serta latensi median dan p95. Audit sebelumnya mencatat 12 turn gagal dan median 20,1 detik. [Audit staging](../../archive/ai_chat_audit_2026_11_27.md#1-executive-summary--data-reconciliation)

## Urutan pengerjaan

1. **Pulihkan baseline tes.** Suite lokal menjalankan 868 tes dengan 3 kegagalan. Selaraskan tes katalog dan formatter dengan tool inti FastPrint, lalu tambahkan kasus routing dengan lampiran PDF ke golden dataset.
2. **Tutup P0 respons dan latensi.** Pastikan follow-up tanpa rencana tool menghasilkan jawaban kontekstual atau klarifikasi. Di proxy Django, simpan satu status interupsi jika stream putus sebelum jawaban tersimpan. Uji perubahan konfigurasi model router dan `SYNTHESIZER_FAST_DETERMINISTIC` satu per satu di staging dan ukur dampaknya.
3. **Tutup P0 akurasi accounting.** Tampilkan angka rekonsiliasi per cabang dari payload Django. Untuk pencarian nomor jurnal yang persis, batasi pencarian lintas cabang ke cabang yang diotorisasi. Perbaiki validasi COA, pemetaan status jurnal, dan resolusi tanggal relatif. Pertahankan periode dan cabang pada follow-up. Hilangkan chip placeholder dan panggilan tool duplikat. [Urutan remediasi audit](../../archive/ai_chat_audit_2026_11_27.md#5-recommended-remediation-sequence-for-follow-up-implementation)

## Syarat rilis

- Seluruh P0 audit tertutup, minimal **18 dari 19 turn lulus**, dan **median respons maksimal 10 detik**. Cocokkan angka, periode, dan cabang dengan Django. Catat p95 dan hitung turn tanpa respons sebagai gagal.
- Suite LangGraph dan tes Django yang terdampak lulus. Verifikasi SSE, rekonsiliasi PDF, akses cabang terbatas, `entity_links`, serta fitur buka data lengkap melalui `table_query_meta`.
- Pertahankan judul dan urutan kolom tabel yang sudah dipakai, serta kontrak API dan SSE saat ini.

## Batas fase

Tunda Thin Agent V2, penghapusan router legacy, dan ekspansi FastPOS serta Helpdesk hingga rilis inti lulus QA. [Checklist arsitektur](../../ARCHITECTURE_MAP.md#7-checklist-keputusan-langkah-berikutnya)

Baseline di atas berasal dari `unittest` lokal dan pembacaan kode. CodeGraph tidak tersedia saat peninjauan. Konfigurasi aktif dan hasil terbaru di staging perlu dikonfirmasi lewat replay.
