"""
Dynamic Semantic Tool Registry Service for FastPrint ERP LangGraph Orchestrator (TASK-5.1).
Provides semantic similarity tool retrieval using PostgreSQL with pgvector (HNSW index)
and Google Gemini embeddings, RBAC permission filtering, and resilient in-memory fallback.
"""
import asyncio
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

from app.core.config import settings
from app.schemas.tools import (
    AccountBalanceFilterSchema,
    JournalEntriesFilterSchema,
    GeneralLedgerFilterSchema,
    ProfitAndLossFilterSchema,
    BalanceSheetFilterSchema,
    VendorBillsFilterSchema,
    BankReconciliationFilterSchema,
    StockLevelFilterSchema,
    ProductInfoFilterSchema,
    LocationsFilterSchema,
    StockHistoryFilterSchema,
    TransfersFilterSchema,
    LowStockAlertsFilterSchema,
    StockAdjustmentsFilterSchema,
    ScrapRecordsFilterSchema,
    WarehousesFilterSchema,
    ProductCategoriesFilterSchema,
    InventoryVelocityForecastFilterSchema,
    PriceHistoryFilterSchema,
    TopSellingProductsFilterSchema,
    SalesOrdersFilterSchema,
    SalesOrderDetailFilterSchema,
    SalesSummaryFilterSchema,
    ContactsFilterSchema,
    VendorInvoiceFilterSchema,
    AccountHealthFilterSchema,
    MarketplaceOrdersFilterSchema,
    KilatOrderDiagnosticsFilterSchema,
    ProductMappingStatusFilterSchema,
    StockSyncSimulationFilterSchema,
    SyncExecutionLogsFilterSchema,
    PriceComparisonFilterSchema,
    FeeAndEscrowAnomaliesFilterSchema,
    DeductionAndSettlementSummaryFilterSchema,
    CashflowAndBalanceSummaryFilterSchema,
    CompetitorInsightsFilterSchema,
)
from app.services.vector_service import VectorService, vector_service as global_vector_service

logger = logging.getLogger("tool_registry_service")


# Canonical Core Tool Definitions for FastPrint ERP
CORE_TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "tool_name": "fetch_vendor_invoices",
        "domain": "accounting",
        "description": "Pencarian rincian baris jurnal faktur pembelian vendor atau referensi dokumen voucher pembelian (seperti PC/2026/...) beserta detail line item akuntansi.",
        "schema_cls": VendorInvoiceFilterSchema,
        "keywords": ["faktur pembelian", "invoice pembelian", "voucher pembelian", "detail faktur vendor", "baris invoice vendor"],
        "sample_queries": [
            "cek detail baris faktur pembelian PC/2026/00002",
            "tampilkan rincian line item invoice pembelian PT Sumber Makmur",
            "cari voucher faktur vendor nomor PC/2026/00115",
            "lihat detail jurnal faktur pembelian supplier bulan ini",
        ],
        "negative_hints": "Bukan untuk tagihan/piutang pelanggan (customer), sales order, atau daftar hutang usaha/AP aging jatuh tempo pemasok (gunakan get_vendor_bills untuk daftar tagihan & hutang vendor umum).",
        "required_permission": "accounting.view_invoices",
        "is_write": False,
    },
    {
        "tool_name": "get_journal_entries",
        "domain": "accounting",
        "description": "Pencarian dan penelusuran buku jurnal umum akuntansi, entri voucher debit/kredit, nomor dokumen transaksi, dan baris pembukuan keuangan.",
        "schema_cls": JournalEntriesFilterSchema,
        "keywords": ["jurnal", "journal", "voucher", "memorial", "debit kredit"],
        "sample_queries": [
            "cek jurnal umum JE/2026/00012",
            "tampilkan daftar entri jurnal akuntansi minggu ini",
            "lihat voucher debit kredit transaksi tanggal 10 Maret",
            "cari jurnal memorial penyesuaian bulan lalu",
        ],
        "negative_hints": "Bukan untuk laporan buku besar saldo berjalan per akun (gunakan get_general_ledger) atau ringkasan saldo akhir akun COA saja (gunakan get_account_balance).",
        "required_permission": "accounting.view_ledger",
        "is_write": False,
    },
    {
        "tool_name": "get_account_balance",
        "domain": "accounting",
        "description": "Pemeriksaan ringkasan saldo akun kas, bank, piutang, hutang, aset, atau akun buku besar lainnya dalam Chart of Accounts (COA).",
        "schema_cls": AccountBalanceFilterSchema,
        "keywords": ["saldo", "kas", "bank", "balance", "coa", "chart of accounts"],
        "sample_queries": [
            "berapa saldo kas utama hari ini",
            "cek saldo rekening Bank BCA per akhir bulan",
            "lihat ringkasan saldo akun piutang dan hutang COA",
            "cek balance akun 1101 cabang Surabaya",
        ],
        "negative_hints": "Bukan untuk saldo dompet seller Shopee/TikTok/Tokopedia (gunakan get_cashflow_and_balance_summary) atau rincian mutasi transaksi per baris (gunakan get_general_ledger).",
        "required_permission": None,
        "is_write": False,
    },
    {
        "tool_name": "get_general_ledger",
        "domain": "accounting",
        "description": "Laporan buku besar (General Ledger) lengkap, mutasi transaksi debit kredit per akun, saldo awal, dan saldo akhir periode.",
        "schema_cls": GeneralLedgerFilterSchema,
        "keywords": ["buku besar", "general ledger", "gl", "mutasi akun", "running balance"],
        "sample_queries": [
            "tampilkan buku besar akun kas kecil bulan ini",
            "cek mutasi GL akun 4101 penjualan Surabaya",
            "lihat rincian transaksi debit kredit akun beban operasional",
            "cek running balance buku besar bank mandiri",
        ],
        "negative_hints": "Bukan untuk mutasi keluar-masuk stok barang gudang (gunakan get_stock_history) atau mutasi penarikan dana dompet marketplace (gunakan get_cashflow_and_balance_summary).",
        "required_permission": "accounting.view_ledger",
        "is_write": False,
    },
    {
        "tool_name": "get_profit_and_loss",
        "domain": "accounting",
        "description": "Laporan laba rugi (Income Statement / P&L), rincian pendapatan operasional, harga pokok penjualan (HPP/COGS), beban operasional, dan laba bersih perusahaan.",
        "schema_cls": ProfitAndLossFilterSchema,
        "keywords": ["laba rugi", "profit and loss", "p&l", "pendapatan", "hpp", "cogs", "laba bersih"],
        "sample_queries": [
            "cek laporan laba rugi bulan ini",
            "berapa net profit dan HPP cabang Mainstore Q1",
            "tampilkan P&L lengkap dengan breakdown beban operasional",
            "bandingkan laba bersih semua cabang bulan lalu",
        ],
        "negative_hints": "Bukan untuk posisi aset/liabilitas/ekuitas neraca (gunakan get_balance_sheet) atau rekap omzet per channel sales order (gunakan get_sales_summary).",
        "required_permission": "accounting.view_financial_reports",
        "is_write": False,
    },
    {
        "tool_name": "get_balance_sheet",
        "domain": "accounting",
        "description": "Laporan neraca keuangan (Balance Sheet / Statement of Financial Position), posisi total aset, kewajiban liabilitas, dan ekuitas modal.",
        "schema_cls": BalanceSheetFilterSchema,
        "keywords": ["neraca", "balance sheet", "aset", "liabilitas", "ekuitas", "kewajiban"],
        "sample_queries": [
            "tampilkan laporan neraca keuangan per hari ini",
            "cek posisi total aset, hutang, dan modal perusahaan",
            "lihat balance sheet cabang Bandung akhir bulan lalu",
            "berapa total aset lancar dan ekuitas di neraca",
        ],
        "negative_hints": "Bukan untuk laporan pendapatan, HPP, beban, atau laba rugi periode berjalan (gunakan get_profit_and_loss).",
        "required_permission": "accounting.view_financial_reports",
        "is_write": False,
    },
    {
        "tool_name": "get_vendor_bills",
        "domain": "accounting",
        "description": "Daftar tagihan dan faktur hutang usaha kepada vendor atau pemasok (Vendor Bills / Accounts Payable), status pembayaran, tanggal jatuh tempo (overdue/due soon), dan nilai sisa tagihan supplier.",
        "schema_cls": VendorBillsFilterSchema,
        "keywords": ["vendor bills", "tagihan vendor", "hutang vendor", "hutang usaha", "ap aging", "tagihan pemasok", "tagihan supplier", "faktur supplier"],
        "sample_queries": [
            "cek daftar tagihan vendor yang belum dibayar",
            "tagihan supplier PT Sumber Makmur yang jatuh tempo minggu ini",
            "lihat hutang usaha pemasok overdue bulan ini",
            "berapa sisa tagihan AP vendor di cabang Surabaya",
        ],
        "negative_hints": "Bukan untuk tagihan/piutang pelanggan (customer), sales order, atau rincian baris jurnal memorial spesifik (gunakan fetch_vendor_invoices hanya untuk detail line item voucher pembelian seperti PC/...).",
        "required_permission": "accounting.view_bills",
        "is_write": False,
    },
    {
        "tool_name": "get_bank_reconciliation_summary",
        "domain": "accounting",
        "description": "Ringkasan rekonsiliasi mutasi rekening bank dan rekening koran, saldo menurut buku vs bank, dan transaksi belum terekonsiliasi.",
        "schema_cls": BankReconciliationFilterSchema,
        "keywords": ["rekonsiliasi", "rekening koran", "bank recon", "buku bank"],
        "sample_queries": [
            "cek status rekonsiliasi Bank BCA bulan ini",
            "ada transaksi rekening koran Mandiri yang belum cocok dengan buku?",
            "ringkasan bank recon dan selisih saldo buku vs bank",
            "lihat mutasi bank belum terekonsiliasi minggu ini",
        ],
        "negative_hints": "Bukan untuk sekadar cek saldo akhir akun kas/bank COA (gunakan get_account_balance) atau settlement dana dompet marketplace (gunakan get_cashflow_and_balance_summary).",
        "required_permission": "accounting.view_bank",
        "is_write": False,
    },
    {
        "tool_name": "get_stock_level",
        "domain": "inventory",
        "description": "Pengecekan kuantitas jumlah stok fisik barang dan kuantitas tersedia (available stock) berdasarkan nama barang, SKU produk, barcode, gudang, atau lokasi rak.",
        "schema_cls": StockLevelFilterSchema,
        "keywords": ["stok", "stock", "kuantitas", "sisa barang", "stok fisik", "available stock"],
        "sample_queries": [
            "cek stok tinta sublim Epson di gudang Surabaya",
            "berapa sisa qty available SKU HVS-A4-80G semua cabang",
            "stok fisik kertas foto glossy yang ready di Mainstore",
            "cek ketersediaan stok barang di rak gudang Bandung",
        ],
        "negative_hints": "Bukan untuk riwayat keluar-masuk stok (gunakan get_stock_history), simulasi hitung push stok marketplace (gunakan get_stock_sync_simulation), atau spesifikasi/harga katalog produk saja (gunakan get_product_info).",
        "required_permission": None,
        "is_write": False,
    },
    {
        "tool_name": "get_product_info",
        "domain": "inventory",
        "description": "Informasi detail master produk, spesifikasi teknis barang, kategori, satuan ukuran (UoM), harga jual, dan status aktif produk.",
        "schema_cls": ProductInfoFilterSchema,
        "keywords": ["detail produk", "spesifikasi barang", "harga produk", "uom", "master produk"],
        "sample_queries": [
            "berapa harga jual kertas stiker vinyl A4",
            "cek detail master produk dan SKU tinta DTF putih",
            "lihat spesifikasi, kategori, dan satuan UoM produk laminasi",
            "cari produk berdasarkan barcode atau nama barang di katalog",
        ],
        "negative_hints": "Bukan untuk jumlah kuantitas stok fisik gudang (gunakan get_stock_level), histori naik-turun harga (gunakan get_price_history), atau perbandingan harga ke Shopee/Tokopedia (gunakan get_price_comparison).",
        "required_permission": None,
        "is_write": False,
    },
    {
        "tool_name": "get_locations",
        "domain": "inventory",
        "description": "Daftar lokasi penyimpanan gudang, lorong, dan rak internal penyimpanan barang di gudang inventori.",
        "schema_cls": LocationsFilterSchema,
        "keywords": ["lokasi", "rak", "lorong", "bin", "aisle", "daftar lokasi"],
        "sample_queries": [
            "tampilkan daftar lokasi rak internal gudang Mainstore",
            "cek lokasi bin penyimpanan barang di gudang Surabaya",
            "daftar sub-lokasi dan lorong inventori yang aktif",
        ],
        "negative_hints": "Bukan untuk daftar gudang fisik/cabang level atas (gunakan get_warehouses) atau jumlah kuantitas stok per barang (gunakan get_stock_level).",
        "required_permission": None,
        "is_write": False,
    },
    {
        "tool_name": "get_stock_history",
        "domain": "inventory",
        "description": "Riwayat mutasi perpindahan stok barang (Stock Moves), pergerakan keluar masuk barang, log kartu stok per periode.",
        "schema_cls": StockHistoryFilterSchema,
        "keywords": ["riwayat stok", "stock move", "kartu stok", "pergerakan barang", "mutasi stok"],
        "sample_queries": [
            "cek kartu stok tinta art paper bulan ini",
            "lihat riwayat keluar masuk barang SKU TINTA-CMYK minggu lalu",
            "mutasi pergerakan stok produk kertas sublim di Surabaya",
            "tampilkan log stock move barang masuk dan keluar kemarin",
        ],
        "negative_hints": "Bukan untuk mutasi akun keuangan/buku besar (gunakan get_general_ledger) atau daftar dokumen surat jalan transfer antar gudang (gunakan get_transfers).",
        "required_permission": "inventory.view_moves",
        "is_write": False,
    },
    {
        "tool_name": "get_transfers",
        "domain": "inventory",
        "description": "Daftar dokumen transfer pergerakan barang antar gudang atau antar cabang, surat jalan pengiriman, dan penerimaan internal (Internal Transfers).",
        "schema_cls": TransfersFilterSchema,
        "keywords": ["transfer", "surat jalan", "internal transfer", "kirim antar gudang", "mutasi cabang"],
        "sample_queries": [
            "cek surat jalan WH/OUT/00123",
            "daftar internal transfer barang antar cabang bulan ini",
            "lihat pengiriman stok antar gudang yang masih menunggu diterima",
            "status dokumen mutasi cabang dari Mainstore ke Bandung",
        ],
        "negative_hints": "Bukan untuk kartu stok keluar-masuk per produk (gunakan get_stock_history) atau tracking resi pengiriman order pelanggan/marketplace (gunakan get_sales_order_detail / get_marketplace_orders).",
        "required_permission": "inventory.view_transfers",
        "is_write": False,
    },
    {
        "tool_name": "get_low_stock_alerts",
        "domain": "inventory",
        "description": "Peringatan dini barang menipis atau habis, daftar produk dengan stok di bawah batas minimum (reorder point), dan rekomendasi pemesanan kembali.",
        "schema_cls": LowStockAlertsFilterSchema,
        "keywords": ["stok menipis", "stok kritis", "low stock", "habis", "reorder point"],
        "sample_queries": [
            "barang apa saja yang stoknya menipis di cabang Surabaya",
            "cek alert low stock di bawah batas minimum reorder point",
            "daftar produk stok kritis atau hampir habis hari ini",
            "produk apa yang stoknya kosong dan perlu segera dipesan",
        ],
        "negative_hints": "Bukan untuk cek stok satu SKU spesifik (gunakan get_stock_level) atau analisis kecepatan jual fast/slow moving dan dead stock (gunakan get_inventory_velocity_forecast).",
        "required_permission": None,
        "is_write": False,
    },
    {
        "tool_name": "get_stock_adjustments",
        "domain": "inventory",
        "description": "Dokumen penyesuaian stok opname inventori (Stock Adjustments), koreksi selisih fisik vs sistem, dan pencatatan inventaris berkala.",
        "schema_cls": StockAdjustmentsFilterSchema,
        "keywords": ["penyesuaian stok", "stok opname", "stock adjustment", "koreksi stok", "selisih fisik"],
        "sample_queries": [
            "cek daftar dokumen stok opname bulan ini",
            "riwayat stock adjustment koreksi selisih fisik vs sistem",
            "lihat penyesuaian stok inventori terakhir di Mainstore",
        ],
        "negative_hints": "Bukan untuk pencatatan barang rusak/cacat/expired/pemusnahan (gunakan get_scrap_records) atau surat jalan transfer antar gudang (gunakan get_transfers).",
        "required_permission": "inventory.manage_adjustments",
        "is_write": False,
    },
    {
        "tool_name": "get_scrap_records",
        "domain": "inventory",
        "description": "Catatan barang rusak, cacat, expired, atau pemusnahan barang (Scrap Records) dari inventori gudang.",
        "schema_cls": ScrapRecordsFilterSchema,
        "keywords": ["scrap", "barang rusak", "cacat", "expired", "pemusnahan"],
        "sample_queries": [
            "cek daftar barang rusak atau cacat bulan ini",
            "lihat catatan scrap dan barang expired di gudang Surabaya",
            "riwayat pemusnahan stok reject kuartal ini",
        ],
        "negative_hints": "Bukan untuk selisih hasil stok opname reguler (gunakan get_stock_adjustments) atau peringatan stok menipis (gunakan get_low_stock_alerts).",
        "required_permission": "inventory.manage_scrap",
        "is_write": False,
    },
    {
        "tool_name": "get_warehouses",
        "domain": "inventory",
        "description": "Daftar seluruh gudang fisik perusahaan, kode gudang, alamat, dan cabang yang menaungi.",
        "schema_cls": WarehousesFilterSchema,
        "keywords": ["gudang", "warehouse", "depo", "daftar gudang"],
        "sample_queries": [
            "tampilkan daftar semua gudang fisik cabang FastPrint",
            "cek kode warehouse dan alamat gudang di Jakarta",
            "gudang apa saja yang terdaftar di perusahaan ini",
        ],
        "negative_hints": "Bukan untuk daftar rak/lorong detail di dalam gudang (gunakan get_locations) atau jumlah kuantitas stok barang (gunakan get_stock_level).",
        "required_permission": None,
        "is_write": False,
    },
    {
        "tool_name": "get_product_categories",
        "domain": "inventory",
        "description": "Daftar hierarki kategori produk, kelompok barang, dan klasifikasi master inventori.",
        "schema_cls": ProductCategoriesFilterSchema,
        "keywords": ["kategori", "kelompok barang", "category", "klasifikasi produk"],
        "sample_queries": [
            "tampilkan semua kategori produk yang ada di ERP",
            "cari kategori barang yang berhubungan dengan tinta",
            "lihat klasifikasi kelompok produk inventori",
        ],
        "negative_hints": "Bukan untuk mencari daftar barang/SKU beserta harga atau stok di dalam kategori (gunakan get_product_info atau get_stock_level).",
        "required_permission": None,
        "is_write": False,
    },
    {
        "tool_name": "get_inventory_velocity_forecast",
        "domain": "inventory",
        "description": "Analisis kecepatan perputaran stok (fast-moving vs slow-moving/dead stock), days of supply, dan rekomendasi restock berdasarkan laju penjualan dan lead time.",
        "schema_cls": InventoryVelocityForecastFilterSchema,
        "keywords": ["fast moving", "slow moving", "dead stock", "forecast stok", "restock", "rekomendasi order", "perputaran stok"],
        "sample_queries": [
            "produk apa yang masuk dead stock atau slow moving 90 hari terakhir",
            "cek analisis fast moving dan rekomendasi restock bulan ini",
            "berapa days of supply stok kertas sublim saat ini",
            "forecast kebutuhan restock barang berdasarkan kecepatan penjualan",
        ],
        "negative_hints": "Bukan untuk ranking produk terlaris berdasarkan total omzet/qty penjualan semata (gunakan get_top_selling_products) atau alert batas minimum statis (gunakan get_low_stock_alerts).",
        "required_permission": None,
        "is_write": False,
    },
    {
        "tool_name": "get_price_history",
        "domain": "inventory",
        "description": "Riwayat kronologis perubahan harga produk (harga lama ke harga baru, selisih nominal dan persentase, tier qty, dan pengubah) gabungan dari ERP Pricelist (LogEntry), Marketplace (PriceUpdateLog), dan PCLS (InventoryUpdatePriceLog) beserta snapshot harga terkini.",
        "schema_cls": PriceHistoryFilterSchema,
        "keywords": ["perubahan harga", "riwayat harga", "histori harga", "kenaikan harga", "penurunan harga", "price history", "pergerakan harga"],
        "sample_queries": [
            "cek riwayat perubahan harga produk tinta DTF 1 liter",
            "kapan terakhir harga kertas foto A4 naik dan siapa yang ubah",
            "histori pergerakan harga SKU PCLS-001 dari pricelist dan marketplace",
            "lihat log kenaikan atau penurunan harga barang bulan ini",
        ],
        "negative_hints": "Bukan untuk perbandingan harga ERP vs listing marketplace saat ini (gunakan get_price_comparison), pergerakan harga kompetitor (gunakan get_competitor_insights), atau diagnosa status gagal job Celery sync (gunakan get_sync_execution_logs).",
        "required_permission": None,
        "is_write": False,
    },
    {
        "tool_name": "get_top_selling_products",
        "domain": "sales",
        "description": "Laporan produk paling laris (Best Selling / Top Products), rangking volume penjualan produk, dan nilai total omzet penjualan per cabang.",
        "schema_cls": TopSellingProductsFilterSchema,
        "keywords": ["terlaris", "top selling", "best seller", "omzet produk", "penjualan tertinggi"],
        "sample_queries": [
            "10 produk paling laris bulan ini di Mainstore",
            "cek top selling products berdasarkan omzet minggu lalu",
            "barang best seller cabang Surabaya tahun ini",
            "ranking produk dengan volume penjualan tertinggi",
        ],
        "negative_hints": "Bukan untuk analisis perputaran stok/dead stock/rekomendasi restock gudang (gunakan get_inventory_velocity_forecast) atau ringkasan total omzet keseluruhan tanpa ranking produk (gunakan get_sales_summary).",
        "required_permission": None,
        "is_write": False,
    },
    {
        "tool_name": "get_sales_orders",
        "domain": "sales",
        "description": "Pencarian dan penelusuran daftar pesanan penjualan (Sales Orders / SO) dan penawaran (Quotations), filter status transaksi, tanggal, pelanggan, dan channel penjualan.",
        "schema_cls": SalesOrdersFilterSchema,
        "keywords": ["sales order", "so", "pesanan penjualan", "daftar order", "order penjualan", "quotation", "penawaran penjualan"],
        "sample_queries": [
            "tampilkan daftar sales order hari ini di Mainstore",
            "cek pesanan penjualan atas nama pelanggan PT Maju Jaya",
            "daftar quotation penawaran yang masih draft minggu ini",
            "cari list SO terbaru dengan nilai terbesar bulan ini",
        ],
        "negative_hints": "Bukan untuk rincian baris item satu nomor SO spesifik (gunakan get_sales_order_detail) atau monitoring SLA batas kirim, resi AWB kosong, dan status mismatch khusus platform marketplace (gunakan get_marketplace_orders).",
        "required_permission": None,
        "is_write": False,
    },
    {
        "tool_name": "get_sales_order_detail",
        "domain": "sales",
        "description": "Rincian detail pesanan penjualan (Sales Order Detail), baris item produk, kuantitas, harga, diskon, kurir pengiriman, resi AWB, dan metadata invoice marketplace.",
        "schema_cls": SalesOrderDetailFilterSchema,
        "keywords": ["detail so", "rincian order", "cek order", "status pesanan", "detail pesanan", "resi so", "resi pesanan", "nomor so"],
        "sample_queries": [
            "cek detail pesanan MS/SO/0585308",
            "lihat rincian item, diskon, dan resi AWB untuk SO/2026/00912",
            "lacak detail order berdasarkan nomor resi SPXID04819283",
            "tampilkan isi barang dan kurir dari invoice INV/2026/04/123",
        ],
        "negative_hints": "Bukan untuk daftar banyak pesanan penjualan dalam satu periode (gunakan get_sales_orders) atau diagnosa khusus reservasi Pesanan Kilat Shopee (gunakan get_kilat_order_diagnostics).",
        "required_permission": None,
        "is_write": False,
    },
    {
        "tool_name": "get_sales_summary",
        "domain": "sales",
        "description": "Laporan ringkasan eksekutif penjualan, total omzet terealisasi, volume transaksi, nilai rata-rata pesanan (AOV), potensi pipeline penawaran, dan performa per channel penjualan.",
        "schema_cls": SalesSummaryFilterSchema,
        "keywords": ["omzet", "omset", "total penjualan", "ringkasan penjualan", "performa sales", "summary sales", "laporan omzet"],
        "sample_queries": [
            "berapa total omzet penjualan semua cabang bulan ini",
            "cek ringkasan performa sales dan AOV minggu lalu",
            "bandingkan omset Shopee, Tokopedia, dan offline di Surabaya",
            "lihat summary penjualan dan pipeline quotation hari ini",
        ],
        "negative_hints": "Bukan untuk laporan akuntansi laba rugi/HPP/beban operasional (gunakan get_profit_and_loss) atau ranking produk terlaris (gunakan get_top_selling_products).",
        "required_permission": None,
        "is_write": False,
    },
    {
        "tool_name": "get_contacts",
        "domain": "sales",
        "description": "Pencarian dan detail kontak pelanggan / customer / buyer maupun vendor (nama, nomor HP/WA, email, alamat, kota, channel transaksi, sales HP, dan ringkasan quotation/order).",
        "schema_cls": ContactsFilterSchema,
        "keywords": ["contact", "kontak", "detail contact", "detail kontak", "nomor hp", "nomor wa", "pelanggan", "kontak pembeli", "buyer"],
        "sample_queries": [
            "cari kontak dan nomor WA pelanggan bernama Hendra",
            "cek detail contact PT Cahaya Grafika dan sales PIC-nya",
            "siapa pemilik nomor HP 08123456789 di master kontak",
            "lihat alamat dan ringkasan quotation customer Budi",
        ],
        "negative_hints": "Bukan untuk mengecek status hutang/tagihan faktur vendor (gunakan get_vendor_bills) atau daftar sales order per periode (gunakan get_sales_orders).",
        "required_permission": None,
        "is_write": False,
    },
    {
        "tool_name": "get_account_health",
        "domain": "marketplace",
        "description": "Pengecekan kesehatan akun toko marketplace, masa berlaku token OAuth (expired/expiring soon tanpa membocorkan secret), status moderasi toko, kelengkapan konfigurasi AccountOrder dan AccountProperties, serta log error fetch order Celery di Shopee, TikTok Shop, Tokopedia, dan Lazada.",
        "schema_cls": AccountHealthFilterSchema,
        "keywords": ["kesehatan akun", "account health", "token expired", "token marketplace", "gagal fetch order", "toko marketplace", "oauth marketplace"],
        "sample_queries": [
            "cek kesehatan akun toko Shopee dan TikTok yang tokennya mau expired",
            "kenapa akun Tokopedia gagal fetch order di Celery",
            "lihat status koneksi OAuth dan konfigurasi toko marketplace",
            "toko marketplace mana saja yang bermasalah hari ini",
        ],
        "negative_hints": "Bukan untuk cek saldo dompet seller toko (gunakan get_cashflow_and_balance_summary) atau daftar pesanan marketplace telat kirim (gunakan get_marketplace_orders).",
        "required_permission": None,
        "is_write": False,
    },
    {
        "tool_name": "get_marketplace_orders",
        "domain": "marketplace",
        "description": "Pemeriksaan order marketplace untuk monitoring batas kirim SLA (overdue_shipping, due_soon), resi AWB kosong, alasan pembatalan, dan selisih status antara Sales Order ERP dan status platform Shopee/TikTok/Tokopedia.",
        "schema_cls": MarketplaceOrdersFilterSchema,
        "keywords": ["order marketplace", "sla pengiriman", "overdue shipping", "belum ada resi", "missing awb", "status mismatch", "batas kirim"],
        "sample_queries": [
            "cek order Shopee yang lewat batas kirim overdue SLA",
            "pesanan TikTok Shop mana yang belum ada resi AWB",
            "daftar order marketplace batal dan alasan pembatalannya minggu ini",
            "cek order marketplace yang statusnya beda antara ERP dan platform",
        ],
        "negative_hints": "Bukan untuk daftar Sales Order ERP umum/offline/quotation (gunakan get_sales_orders) atau diagnosa khusus reservasi Pesanan Kilat Shopee (gunakan get_kilat_order_diagnostics).",
        "required_permission": None,
        "is_write": False,
    },
    {
        "tool_name": "get_kilat_order_diagnostics",
        "domain": "marketplace",
        "description": "Diagnosa read-only Pesanan Kilat Shopee untuk melacak nomor reservasi yang invoice-nya belum ter-update (invoice_updated=false), deteksi duplikasi QuotationMarketplace/Draft, serta kandidat SO dari LogEntry dan catatan quotation.",
        "schema_cls": KilatOrderDiagnosticsFilterSchema,
        "keywords": ["pesanan kilat", "order kilat", "no reservasi", "reservasi kilat", "invoice_updated", "kilat shopee"],
        "sample_queries": [
            "cek pesanan kilat Shopee yang nyangkut hari ini",
            "reservasi kilat mana yang invoice_updated masih false",
            "deteksi duplikat draft atau quotation pesanan kilat",
            "diagnosa no reservasi kilat yang belum cocok ke SO",
        ],
        "negative_hints": "Bukan untuk order marketplace reguler non-kilat (gunakan get_marketplace_orders) dan hanya read-only (tidak memproses/merge/hapus order kilat).",
        "required_permission": None,
        "is_write": False,
    },
    {
        "tool_name": "get_product_mapping_status",
        "domain": "marketplace",
        "description": "Audit status mapping produk marketplace ke master ERP, daftar SKU yang belum ter-mapping (unmapped), aturan kelipatan paket (qty_multiplier), duplikat mapping aktif, dan daftar pengecualian ExceptionProduct.",
        "schema_cls": ProductMappingStatusFilterSchema,
        "keywords": ["mapping produk", "unmapped sku", "belum mapping", "qty_multiplier", "exception product", "duplikat mapping"],
        "sample_queries": [
            "produk Shopee mana saja yang SKU-nya belum di-mapping ke ERP",
            "cek daftar duplikat mapping SKU aktif di Tokopedia",
            "lihat aturan qty_multiplier dan daftar ExceptionProduct marketplace",
            "audit status mapping produk toko TikTok Shop",
        ],
        "negative_hints": "Bukan untuk simulasi angka stok push marketplace (gunakan get_stock_sync_simulation) atau inspeksi log gagal sinkronisasi stok/harga (gunakan get_sync_execution_logs).",
        "required_permission": None,
        "is_write": False,
    },
    {
        "tool_name": "get_stock_sync_simulation",
        "domain": "marketplace",
        "description": "Simulasi dry-run perhitungan stok sinkronisasi marketplace (Onhand, Forecast, Split Onhand, Split Forecast, Forecast Sales) beserta pooling shared_stock_companies, konversi qty_multiplier, dan perbandingan dengan StockUpdateLog terakhir.",
        "schema_cls": StockSyncSimulationFilterSchema,
        "keywords": ["simulasi stok", "push stok", "determine stock", "shared stock", "selisih stok marketplace", "stock sync"],
        "sample_queries": [
            "simulasi hitung stok push marketplace untuk SKU KRT-A4",
            "kenapa stok di Shopee beda dengan stok fisik ERP",
            "cek perhitungan shared stock dan qty_multiplier sebelum sync",
            "dry run determine stock toko FastPrint Official",
        ],
        "negative_hints": "Bukan untuk mengeksekusi push/update stok live ke marketplace (read-only) dan bukan untuk cek riwayat log job push yang sudah berjalan (gunakan get_sync_execution_logs).",
        "required_permission": None,
        "is_write": False,
    },
    {
        "tool_name": "get_sync_execution_logs",
        "domain": "marketplace",
        "description": "Inspeksi log eksekusi update stok (StockUpdateLog), update harga (PriceUpdateLog), dan harga PCLS (InventoryUpdatePriceLog) termasuk status gagal/skip, trigger Celery Beat vs manual, serta kode error marketplace yang sudah dinormalisasi.",
        "schema_cls": SyncExecutionLogsFilterSchema,
        "keywords": ["log update stok", "log update harga", "sync log", "celery gagal", "priceupdatelog", "stockupdatelog", "pcls price"],
        "sample_queries": [
            "cek log update stok marketplace yang gagal hari ini",
            "kenapa sync harga PCLS atau PriceUpdateLog gagal di Celery",
            "lihat riwayat StockUpdateLog yang statusnya FAILED atau SKIPPED",
            "inspeksi error sinkronisasi stok dan harga Shopee terakhir",
        ],
        "negative_hints": "Bukan untuk histori perubahan nominal harga jual produk (gunakan get_price_history) atau simulasi perhitungan stok sebelum sync (gunakan get_stock_sync_simulation).",
        "required_permission": None,
        "is_write": False,
    },
    {
        "tool_name": "get_price_comparison",
        "domain": "marketplace",
        "description": "Perbandingan harga jual ERP Pricelist dan tier grosir WholesaleMP terhadap harga listing di Shopee, TikTok Shop, dan Tokopedia untuk mendeteksi selisih margin atau SKU tanpa harga, dengan opsi live read-only maksimal 20 SKU.",
        "schema_cls": PriceComparisonFilterSchema,
        "keywords": ["perbandingan harga", "bandingkan harga", "selisih harga", "pricelist marketplace", "harga shopee vs erp", "wholesale mp"],
        "sample_queries": [
            "bandingkan harga ERP pricelist vs harga tayang di Shopee",
            "cek produk yang selisih harga marketplace-nya di atas 10%",
            "lihat perbandingan harga WholesaleMP vs listing Tokopedia",
            "SKU mana yang belum punya harga tayang di TikTok Shop",
        ],
        "negative_hints": "Bukan untuk membandingkan harga toko kita dengan toko kompetitor/pesaing (gunakan get_competitor_insights) atau riwayat naik-turun harga historis (gunakan get_price_history).",
        "required_permission": None,
        "is_write": False,
    },
    {
        "tool_name": "get_fee_and_escrow_anomalies",
        "domain": "marketplace",
        "description": "Deteksi anomali potongan fee dan escrow pada order marketplace: negative_fee, high_fee di atas persentase ambang batas, zero_income saat mp_total positif, dan erp_mp_mismatch antara total SO ERP dan mp_total.",
        "schema_cls": FeeAndEscrowAnomaliesFilterSchema,
        "keywords": ["anomali fee", "anomali potongan", "high fee", "negative fee", "zero income", "escrow anomaly", "potongan di atas"],
        "sample_queries": [
            "cek order Shopee yang potongan fee-nya di atas 20%",
            "deteksi anomali escrow, negative fee, dan zero income minggu ini",
            "order marketplace mana yang total SO ERP beda dengan mp_total",
            "cari anomali potongan biaya admin yang tidak wajar",
        ],
        "negative_hints": "Bukan untuk rekap total pendapatan kotor, total potongan per kategori, dan settlement keseluruhan (gunakan get_deduction_and_settlement_summary).",
        "required_permission": None,
        "is_write": False,
    },
    {
        "tool_name": "get_deduction_and_settlement_summary",
        "domain": "marketplace",
        "description": "Rekapitulasi pendapatan kotor (gross), potongan biaya marketplace per kategori kanonik (Komisi Platform, Biaya Layanan, Ongkir, Voucher Penjual, Pajak), pendapatan bersih (nett), settlement belum ter-posting ke jurnal, dan validasi aritmatika.",
        "schema_cls": DeductionAndSettlementSummaryFilterSchema,
        "keywords": ["potongan fee", "rekap deduction", "komisi marketplace", "settlement belum posting", "unposted journal", "revenue history"],
        "sample_queries": [
            "rekap potongan komisi, biaya layanan, dan voucher Shopee bulan ini",
            "cek settlement marketplace yang belum ter-posting ke jurnal",
            "berapa pendapatan gross vs nett setelah deduction Tokopedia",
            "ringkasan biaya admin dan settlement TikTok Shop minggu lalu",
        ],
        "negative_hints": "Bukan untuk mencari order individual dengan anomali fee negatif/tinggi (gunakan get_fee_and_escrow_anomalies) atau cek saldo dompet dan penarikan dana (gunakan get_cashflow_and_balance_summary).",
        "required_permission": None,
        "is_write": False,
    },
    {
        "tool_name": "get_cashflow_and_balance_summary",
        "domain": "marketplace",
        "description": "Ringkasan saldo dompet seller marketplace (available_balance vs pending_balance escrow), kesegaran snapshot saldo, mutasi kas masuk settlement dan penarikan dana withdrawal (MarketplaceCashflowLog), dengan opsi pengecekan live read-only.",
        "schema_cls": CashflowAndBalanceSummaryFilterSchema,
        "keywords": ["saldo dompet", "pending escrow", "wallet balance", "withdrawal marketplace", "cashflow marketplace", "balance snapshot"],
        "sample_queries": [
            "berapa saldo dompet seller dan pending escrow di Shopee",
            "cek riwayat penarikan dana withdrawal marketplace minggu ini",
            "lihat cashflow kas masuk settlement dompet TikTok Shop",
            "cek wallet balance live semua toko marketplace",
        ],
        "negative_hints": "Bukan untuk saldo akun kas/bank COA ERP (gunakan get_account_balance) atau rekonsiliasi rekening koran bank (gunakan get_bank_reconciliation_summary).",
        "required_permission": None,
        "is_write": False,
    },
    {
        "tool_name": "get_competitor_insights",
        "domain": "marketplace",
        "description": "Analisis pergerakan harga kompetitor, kecepatan penjualan (delta_sold_qty velocity), penurunan harga kompetitor, resolusi BOM varian produk, serta daftar kompetitor yang audit hariannya sudah usang (stale).",
        "schema_cls": CompetitorInsightsFilterSchema,
        "keywords": ["kompetitor", "competitor", "pesaing", "penurunan harga kompetitor", "velocity kompetitor", "audit stale"],
        "sample_queries": [
            "kompetitor mana saja yang menurunkan harga minggu ini",
            "cek kecepatan penjualan velocity produk pesaing di Shopee",
            "daftar pantauan kompetitor yang audit hariannya sudah stale",
            "bandingkan pergerakan harga kompetitor untuk produk tinta",
        ],
        "negative_hints": "Bukan untuk perbandingan harga internal ERP pricelist terhadap listing toko kita sendiri (gunakan get_price_comparison) atau histori perubahan harga internal kita (gunakan get_price_history).",
        "required_permission": None,
        "is_write": False,
    },
]


def build_tool_search_text(tool: Dict[str, Any]) -> str:
    """
    Construct canonical enriched tool card text for PGVector embeddings and FTS indexing.
    Combines tool identity, description, keywords, sample queries, and negative hints.
    """
    tool_name = str(tool.get("tool_name") or "").strip()
    domain = str(tool.get("domain") or "").strip()
    description = str(tool.get("description") or "").strip()
    keywords = [str(k).strip() for k in (tool.get("keywords") or []) if str(k).strip()]
    sample_queries = [str(q).strip() for q in (tool.get("sample_queries") or []) if str(q).strip()]
    negative_hints = str(tool.get("negative_hints") or "").strip()

    lines = [
        f"Tool: {tool_name} ({domain})",
        f"Deskripsi: {description}",
        f"Kata Kunci: {', '.join(keywords)}",
    ]
    if sample_queries:
        lines.append(f"Contoh Pertanyaan: {' | '.join(sample_queries)}")
    if negative_hints:
        lines.append(f"Bukan Untuk: {negative_hints}")
    return "\n".join(lines)


FALLBACK_STOP_WORDS: Set[str] = {
    "dan", "atau", "yang", "ini", "itu", "untuk", "dari", "pada", "ke", "di",
    "ada", "apa", "bagaimana", "cara", "alur", "rencana", "bahan", "barang",
    "informasi", "perusahaan", "kantor", "toko", "bisa", "tolong", "cek", "lihat",
    "saya", "kami", "kita", "kamu", "anda", "dia", "mereka", "dengan", "secara",
    "terkait", "tentang", "mengenai", "atas", "bawah", "dalam", "luar", "semua",
    "tampilkan", "cari", "berapa", "mana", "saja", "dong", "ya", "mohon", "bantu",
}

SHORT_ERP_ACRONYMS: Set[str] = {
    "so", "gl", "ap", "ar", "hp", "wa", "po", "do", "mp", "je", "pc", "pl",
}

RRF_K_CONSTANT: float = 60.0


def _extract_query_tokens(query: str) -> List[str]:
    """Extract normalized search tokens preserving short ERP acronyms (so, gl, ap, ar, etc.)."""
    cleaned = (query or "").lower().strip()
    if not cleaned:
        return []
    raw_tokens = re.findall(r"[a-z0-9]+", cleaned)
    tokens: List[str] = []
    seen: Set[str] = set()
    for tok in raw_tokens:
        if tok in FALLBACK_STOP_WORDS:
            continue
        if len(tok) <= 2 and tok not in SHORT_ERP_ACRONYMS:
            continue
        if tok not in seen:
            seen.add(tok)
            tokens.append(tok)
    return tokens


def _extract_short_acronyms(query: str) -> List[str]:
    """Return distinct short ERP acronyms present in query for bounded FTS/regex matching."""
    cleaned = (query or "").lower().strip()
    if not cleaned:
        return []
    raw_tokens = re.findall(r"[a-z0-9]+", cleaned)
    found: List[str] = []
    for tok in raw_tokens:
        if tok in SHORT_ERP_ACRONYMS and tok not in found:
            found.append(tok)
    if re.search(r"\bp\s*&\s*l\b", cleaned) and "pl" not in found:
        found.append("pl")
    return found


def _resolve_rbac_roles(
    user_roles: Optional[List[str]],
    enforce_rbac: Optional[bool] = None,
) -> Tuple[bool, List[str]]:
    """
    Resolve Conditional RBAC policy:
    - Default (enforce_rbac=None): enforce required_permission only when verified user_roles is non-empty.
      When user_roles is None or empty, return all read-only (is_write=False) tools and defer final
      authorization to Django Core JWT (HTTP 403).
    """
    cleaned_roles = [str(r).strip() for r in (user_roles or []) if str(r).strip()]
    should_enforce = bool(cleaned_roles) if enforce_rbac is None else bool(enforce_rbac)
    return should_enforce, cleaned_roles


def merge_rrf_candidates(
    candidate_lists: List[List[Dict[str, Any]]],
    k: float = RRF_K_CONSTANT,
    max_results: int = 12,
) -> List[Dict[str, Any]]:
    """
    Deduplicate and fuse multiple ranked tool candidate lists (e.g. per-clause retrieval)
    using Reciprocal Rank Fusion (RRF), retaining highest-scoring metadata per tool_name.
    """
    merged: Dict[str, Dict[str, Any]] = {}
    rrf_scores: Dict[str, float] = {}
    best_distance: Dict[str, Optional[float]] = {}

    for cand_list in candidate_lists:
        for rank_idx, cand in enumerate(cand_list or [], start=1):
            t_name = cand.get("tool_name")
            if not t_name:
                continue
            base_rrf = cand.get("rrf_score")
            contrib = float(base_rrf) if isinstance(base_rrf, (int, float)) and base_rrf > 0 else (1.0 / (k + rank_idx))
            rrf_scores[t_name] = rrf_scores.get(t_name, 0.0) + contrib

            dist = cand.get("distance")
            prev_dist = best_distance.get(t_name)
            if isinstance(dist, (int, float)):
                if prev_dist is None or dist < prev_dist:
                    best_distance[t_name] = float(dist)
            elif t_name not in best_distance:
                best_distance[t_name] = None

            if t_name not in merged:
                merged[t_name] = dict(cand)

    ranked_names = sorted(
        merged.keys(),
        key=lambda name: (
            -rrf_scores[name],
            best_distance[name] if best_distance[name] is not None else 1.0,
            name,
        ),
    )

    results: List[Dict[str, Any]] = []
    for name in ranked_names[:max_results]:
        item = dict(merged[name])
        item["rrf_score"] = round(rrf_scores[name], 6)
        item["distance"] = best_distance[name]
        results.append(item)
    return results


class ToolRegistryService:
    """
    Central service for managing and searching ERP tool metadata.
    Supports PGVector cosine similarity search, RBAC filtering, and in-memory fallback.
    """

    def __init__(
        self,
        vector_service: Optional[VectorService] = None,
        table_name: Optional[str] = None,
        default_definitions: Optional[List[Dict[str, Any]]] = None,
    ):
        self.vector_service = vector_service or global_vector_service
        resolved_table = (table_name or settings.TOOL_CATALOG_TABLE_NAME or "").strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", resolved_table):
            raise ValueError(f"Invalid SQL identifier for table_name: {resolved_table!r}")
        self.table_name = resolved_table
        self.definitions = default_definitions or CORE_TOOL_DEFINITIONS
        self._in_memory_catalog: Dict[str, Dict[str, Any]] = {}
        self._build_in_memory_catalog()

    def _build_in_memory_catalog(self) -> None:
        """Construct local in-memory catalog from core tool definitions."""
        for tool in self.definitions:
            schema_cls = tool.get("schema_cls")
            params_schema = (
                schema_cls.model_json_schema()
                if schema_cls and hasattr(schema_cls, "model_json_schema")
                else tool.get("params_schema", {})
            )
            self._in_memory_catalog[tool["tool_name"]] = {
                "tool_name": tool["tool_name"],
                "domain": tool["domain"],
                "description": tool["description"],
                "params_schema": params_schema,
                "required_permission": tool.get("required_permission"),
                "is_write": bool(tool.get("is_write", False)),
                "keywords": [k.lower() for k in tool.get("keywords", [])],
                "sample_queries": [str(q).strip() for q in tool.get("sample_queries", []) if str(q).strip()],
                "negative_hints": str(tool.get("negative_hints") or "").strip(),
                "search_text": build_tool_search_text(tool),
            }

    async def initialize_catalog(
        self,
        force_update: bool = False,
        domain_filter: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Create table & HNSW index if needed, and incrementally seed tools.

        Args:
            force_update: If True, updates existing tools and recalculates embeddings.
            domain_filter: Optional domain filter to only seed specific domain tools.

        Returns:
            Dict containing counts of 'created', 'updated', 'skipped'.
        """
        pool = await self.vector_service.get_pool()
        if pool is None:
            logger.warning("PGVector connection pool unavailable. Tool catalog table cannot be initialized.")
            return {"created": 0, "updated": 0, "skipped": 0, "error": 1}

        # 1. Create table, idempotent search_text/search_tsv columns, HNSW & GIN indexes
        ddl_sql = f"""
            CREATE EXTENSION IF NOT EXISTS vector;

            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id SERIAL PRIMARY KEY,
                tool_name VARCHAR(64) UNIQUE NOT NULL,
                domain VARCHAR(32) NOT NULL,
                description TEXT NOT NULL,
                params_schema JSONB NOT NULL,
                required_permission VARCHAR(64),
                is_write BOOLEAN DEFAULT FALSE,
                search_text TEXT,
                search_tsv tsvector GENERATED ALWAYS AS (
                    to_tsvector('simple', coalesce(search_text, ''))
                ) STORED,
                embedding vector(768),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );

            ALTER TABLE {self.table_name}
                ADD COLUMN IF NOT EXISTS search_text TEXT;

            ALTER TABLE {self.table_name}
                ADD COLUMN IF NOT EXISTS search_tsv tsvector GENERATED ALWAYS AS (
                    to_tsvector('simple', coalesce(search_text, ''))
                ) STORED;

            CREATE INDEX IF NOT EXISTS idx_{self.table_name}_embedding
            ON {self.table_name} USING hnsw (embedding vector_cosine_ops);

            CREATE INDEX IF NOT EXISTS idx_{self.table_name}_search_tsv
            ON {self.table_name} USING gin (search_tsv);
        """
        async with pool.acquire() as conn:
            await conn.execute(ddl_sql)

        # 2. Check existing tools in DB (track description & search_text for incremental sync)
        async with pool.acquire() as conn:
            rows = await conn.fetch(f"SELECT tool_name, description, search_text FROM {self.table_name};")
            existing_tools = {
                r["tool_name"]: {
                    "description": r.get("description") if hasattr(r, "get") else r["description"],
                    "search_text": r.get("search_text") if hasattr(r, "get") else None,
                }
                for r in rows
            }

        created = 0
        updated = 0
        skipped = 0

        for tool in self.definitions:
            t_name = tool["tool_name"]
            t_domain = tool["domain"]
            if domain_filter and t_domain != domain_filter:
                continue

            t_desc = tool["description"]
            t_search_text = build_tool_search_text(tool)
            schema_cls = tool.get("schema_cls")
            params_schema = (
                schema_cls.model_json_schema()
                if schema_cls and hasattr(schema_cls, "model_json_schema")
                else tool.get("params_schema", {})
            )
            req_perm = tool.get("required_permission")
            is_write = bool(tool.get("is_write", False))

            if t_name in existing_tools:
                existing = existing_tools[t_name]
                if (
                    not force_update
                    and existing["description"] == t_desc
                    and existing["search_text"] == t_search_text
                ):
                    skipped += 1
                    continue

                # Update existing tool (forced, legacy NULL search_text, or enriched card changed)
                embedding = await self.vector_service.get_embedding(t_search_text)
                emb_vec = [float(v) for v in embedding] if embedding else None
                update_sql = f"""
                    UPDATE {self.table_name}
                    SET domain = $2,
                        description = $3,
                        params_schema = $4::jsonb,
                        required_permission = $5,
                        is_write = $6,
                        search_text = $7,
                        embedding = $8,
                        updated_at = NOW()
                    WHERE tool_name = $1;
                """
                async with pool.acquire() as conn:
                    await conn.execute(
                        update_sql,
                        t_name,
                        t_domain,
                        t_desc,
                        json.dumps(params_schema),
                        req_perm,
                        is_write,
                        t_search_text,
                        emb_vec,
                    )
                updated += 1
            else:
                # Incremental Insert (New tool)
                embedding = await self.vector_service.get_embedding(t_search_text)
                emb_vec = [float(v) for v in embedding] if embedding else None
                insert_sql = f"""
                    INSERT INTO {self.table_name}
                    (tool_name, domain, description, params_schema, required_permission, is_write, search_text, embedding)
                    VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7, $8)
                    ON CONFLICT (tool_name) DO NOTHING;
                """
                async with pool.acquire() as conn:
                    await conn.execute(
                        insert_sql,
                        t_name,
                        t_domain,
                        t_desc,
                        json.dumps(params_schema),
                        req_perm,
                        is_write,
                        t_search_text,
                        emb_vec,
                    )
                created += 1

        logger.info(
            f"Tool catalog sync completed: {created} created, {updated} updated, {skipped} skipped."
        )
        return {"created": created, "updated": updated, "skipped": skipped}

    def _hydrate_db_record(self, r: Any) -> Dict[str, Any]:
        """Convert a PGVector record into a normalized candidate tool dictionary enriched with local metadata."""
        ps = r["params_schema"]
        if isinstance(ps, str):
            try:
                ps = json.loads(ps)
            except Exception:
                pass
        t_name = r["tool_name"]
        cached = self._in_memory_catalog.get(t_name) or {}
        raw_dist = r.get("distance") if hasattr(r, "get") else r["distance"]
        raw_rrf = r.get("rrf_score") if hasattr(r, "get") else None
        item: Dict[str, Any] = {
            "tool_name": t_name,
            "domain": r["domain"],
            "description": r["description"],
            "params_schema": ps,
            "required_permission": r["required_permission"],
            "is_write": bool(r["is_write"]),
            "negative_hints": cached.get("negative_hints", ""),
            "sample_queries": list(cached.get("sample_queries", [])),
            "distance": float(raw_dist) if raw_dist is not None else None,
        }
        if raw_rrf is not None:
            item["rrf_score"] = round(float(raw_rrf), 6)
        return item

    async def search_tools(
        self,
        query: str,
        domain: Optional[str] = None,
        user_roles: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        distance_threshold: Optional[float] = None,
        enforce_rbac: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Dynamically retrieve top relevant tools using PGVector cosine similarity and Conditional RBAC.

        Args:
            query: User input query text.
            domain: Optional ERP domain filter ('accounting', 'inventory', 'sales', 'marketplace').
            user_roles: Optional list of verified user roles/permissions.
            top_k: Maximum candidate tools to return (defaults to settings.TOOL_RETRIEVAL_TOP_K).
            distance_threshold: Maximum cosine distance cutoff (defaults to settings.TOOL_RETRIEVAL_DISTANCE_THRESHOLD).
            enforce_rbac: Optional override for Conditional RBAC (default: True iff user_roles is non-empty).

        Returns:
            List of candidate tool dicts sorted by cosine distance ascending.
        """
        limit = top_k if top_k is not None else settings.TOOL_RETRIEVAL_TOP_K
        max_dist = (
            distance_threshold
            if distance_threshold is not None
            else settings.TOOL_RETRIEVAL_DISTANCE_THRESHOLD
        )

        # Check feature flag or fallback condition
        if not settings.DYNAMIC_TOOL_RETRIEVAL_ENABLED:
            return self.search_tools_fallback(
                query=query,
                domain=domain,
                user_roles=user_roles,
                top_k=limit,
                enforce_rbac=enforce_rbac,
            )

        pool = await self.vector_service.get_pool()
        if pool is None:
            logger.info("Vector DB pool unavailable. Using in-memory tool retrieval fallback.")
            return self.search_tools_fallback(
                query=query,
                domain=domain,
                user_roles=user_roles,
                top_k=limit,
                enforce_rbac=enforce_rbac,
            )

        # 1. Embed user query via Gemini
        query_vector = await self.vector_service.get_embedding(query)
        if query_vector is None:
            logger.info("Query embedding failed or unavailable. Using in-memory tool retrieval fallback.")
            return self.search_tools_fallback(
                query=query,
                domain=domain,
                user_roles=user_roles,
                top_k=limit,
                enforce_rbac=enforce_rbac,
            )

        try:
            vector_arr = [float(v) for v in query_vector]
            should_enforce, roles_param = _resolve_rbac_roles(user_roles, enforce_rbac=enforce_rbac)

            sql = f"""
                SELECT
                    tool_name,
                    domain,
                    description,
                    params_schema,
                    required_permission,
                    is_write,
                    (embedding <=> $1) AS distance
                FROM {self.table_name}
                WHERE ($2::text IS NULL OR domain = $2)
                  AND (
                      (NOT $3::boolean AND is_write = FALSE)
                      OR (
                          $3::boolean
                          AND (
                              required_permission IS NULL
                              OR required_permission = ANY($4::text[])
                              OR 'admin' = ANY($4::text[])
                          )
                      )
                  )
                  AND (embedding <=> $1) <= $5
                ORDER BY distance ASC
                LIMIT $6;
            """
            async with pool.acquire() as conn:
                records = await conn.fetch(sql, vector_arr, domain, should_enforce, roles_param, max_dist, limit)

            return [self._hydrate_db_record(r) for r in records]

        except Exception as exc:
            logger.warning(f"Error querying PGVector for tool catalog: {exc}. Falling back to in-memory catalog.")
            return self.search_tools_fallback(
                query=query,
                domain=domain,
                user_roles=user_roles,
                top_k=limit,
                enforce_rbac=enforce_rbac,
            )

    async def search_tools_hybrid(
        self,
        query: str,
        domain: Optional[str] = None,
        user_roles: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        distance_threshold: Optional[float] = None,
        enforce_rbac: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid semantic + lexical tool retrieval combining PGVector cosine distance and
        PostgreSQL FTS ('simple') + ERP acronym matching via Reciprocal Rank Fusion (k=60),
        scoped by Conditional RBAC with automatic in-memory fallback when offline.
        """
        limit = top_k if top_k is not None else settings.TOOL_RETRIEVAL_TOP_K
        max_dist = (
            distance_threshold
            if distance_threshold is not None
            else settings.TOOL_RETRIEVAL_DISTANCE_THRESHOLD
        )

        if not settings.DYNAMIC_TOOL_RETRIEVAL_ENABLED:
            return self.search_tools_fallback(
                query=query,
                domain=domain,
                user_roles=user_roles,
                top_k=limit,
                enforce_rbac=enforce_rbac,
            )

        pool = await self.vector_service.get_pool()
        if pool is None:
            logger.info("Vector DB pool unavailable for hybrid search. Using in-memory fallback.")
            return self.search_tools_fallback(
                query=query,
                domain=domain,
                user_roles=user_roles,
                top_k=limit,
                enforce_rbac=enforce_rbac,
            )

        query_vector = await self.vector_service.get_embedding(query)
        if query_vector is None:
            logger.info("Query embedding unavailable for hybrid search. Using in-memory fallback.")
            return self.search_tools_fallback(
                query=query,
                domain=domain,
                user_roles=user_roles,
                top_k=limit,
                enforce_rbac=enforce_rbac,
            )

        try:
            vector_arr = [float(v) for v in query_vector]
            should_enforce, roles_param = _resolve_rbac_roles(user_roles, enforce_rbac=enforce_rbac)
            cleaned_query = (query or "").strip()
            tokens = _extract_query_tokens(cleaned_query)
            or_tsquery = " | ".join(tokens[:16]) if tokens else ""
            acronyms = _extract_short_acronyms(cleaned_query)
            acronym_regex = rf"\m({'|'.join(re.escape(a) for a in acronyms)})\M" if acronyms else ""

            sql = f"""
                WITH base AS (
                    SELECT
                        tool_name,
                        domain,
                        description,
                        params_schema,
                        required_permission,
                        is_write,
                        search_text,
                        search_tsv,
                        embedding
                    FROM {self.table_name}
                    WHERE ($2::text IS NULL OR domain = $2)
                      AND (
                          (NOT $3::boolean AND is_write = FALSE)
                          OR (
                              $3::boolean
                              AND (
                                  required_permission IS NULL
                                  OR required_permission = ANY($4::text[])
                                  OR 'admin' = ANY($4::text[])
                              )
                          )
                      )
                ),
                vec_ranked AS (
                    SELECT
                        tool_name,
                        (embedding <=> $1) AS distance,
                        ROW_NUMBER() OVER (ORDER BY (embedding <=> $1) ASC, tool_name ASC) AS vec_rank
                    FROM base
                    WHERE embedding IS NOT NULL
                      AND (embedding <=> $1) <= $5
                    ORDER BY distance ASC, tool_name ASC
                    LIMIT 15
                ),
                fts_scored AS (
                    SELECT
                        tool_name,
                        (
                            CASE
                                WHEN $6::text <> '' AND search_tsv IS NOT NULL
                                     AND search_tsv @@ websearch_to_tsquery('simple', $6)
                                THEN ts_rank_cd(search_tsv, websearch_to_tsquery('simple', $6)) * 2.0
                                ELSE 0.0
                            END
                            + CASE
                                WHEN $7::text <> '' AND search_tsv IS NOT NULL
                                     AND search_tsv @@ to_tsquery('simple', $7)
                                THEN ts_rank_cd(search_tsv, to_tsquery('simple', $7))
                                ELSE 0.0
                            END
                            + CASE
                                WHEN $8::text <> '' AND coalesce(search_text, description) ~* $8
                                THEN 0.25
                                ELSE 0.0
                            END
                        ) AS fts_score
                    FROM base
                    WHERE (
                        ($6::text <> '' AND search_tsv IS NOT NULL AND search_tsv @@ websearch_to_tsquery('simple', $6))
                        OR ($7::text <> '' AND search_tsv IS NOT NULL AND search_tsv @@ to_tsquery('simple', $7))
                        OR ($8::text <> '' AND coalesce(search_text, description) ~* $8)
                    )
                ),
                fts_ranked AS (
                    SELECT
                        tool_name,
                        fts_score,
                        ROW_NUMBER() OVER (ORDER BY fts_score DESC, tool_name ASC) AS fts_rank
                    FROM fts_scored
                    ORDER BY fts_score DESC, tool_name ASC
                    LIMIT 15
                ),
                combined AS (
                    SELECT
                        COALESCE(v.tool_name, f.tool_name) AS tool_name,
                        v.distance,
                        (
                            COALESCE(1.0 / (60.0 + v.vec_rank), 0.0)
                            + COALESCE(1.0 / (60.0 + f.fts_rank), 0.0)
                        ) AS rrf_score
                    FROM vec_ranked v
                    FULL OUTER JOIN fts_ranked f ON v.tool_name = f.tool_name
                )
                SELECT
                    b.tool_name,
                    b.domain,
                    b.description,
                    b.params_schema,
                    b.required_permission,
                    b.is_write,
                    c.distance,
                    c.rrf_score
                FROM combined c
                JOIN base b ON b.tool_name = c.tool_name
                ORDER BY c.rrf_score DESC, COALESCE(c.distance, 1.0) ASC, b.tool_name ASC
                LIMIT $9;
            """
            async with pool.acquire() as conn:
                records = await conn.fetch(
                    sql,
                    vector_arr,
                    domain,
                    should_enforce,
                    roles_param,
                    max_dist,
                    cleaned_query,
                    or_tsquery,
                    acronym_regex,
                    limit,
                )

            return [self._hydrate_db_record(r) for r in records]

        except Exception as exc:
            logger.warning(f"Error executing hybrid PGVector/FTS tool search: {exc}. Falling back to in-memory catalog.")
            return self.search_tools_fallback(
                query=query,
                domain=domain,
                user_roles=user_roles,
                top_k=limit,
                enforce_rbac=enforce_rbac,
            )

    def _run_async_search_sync(
        self,
        coro_fn,
        query: str,
        domain: Optional[str],
        user_roles: Optional[List[str]],
        top_k: Optional[int],
        distance_threshold: Optional[float],
        enforce_rbac: Optional[bool],
        label: str,
    ) -> List[Dict[str, Any]]:
        """Shared safe sync wrapper around async retrieval coroutines with offline fallback."""
        limit = top_k if top_k is not None else settings.TOOL_RETRIEVAL_TOP_K
        is_test = "unittest" in sys.modules or os.environ.get("FASTPRINT_TEST_ENV") == "true" or settings.APP_ENV == "test"
        if is_test or not settings.DYNAMIC_TOOL_RETRIEVAL_ENABLED or not settings.VECTOR_DB_URL:
            return self.search_tools_fallback(
                query=query,
                domain=domain,
                user_roles=user_roles,
                top_k=limit,
                enforce_rbac=enforce_rbac,
            )

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            kwargs = {
                "query": query,
                "domain": domain,
                "user_roles": user_roles,
                "top_k": limit,
                "distance_threshold": distance_threshold,
                "enforce_rbac": enforce_rbac,
            }
            if loop is not None and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    return executor.submit(asyncio.run, coro_fn(**kwargs)).result(
                        timeout=settings.REQUEST_TIMEOUT_SEC
                    )
            return asyncio.run(coro_fn(**kwargs))
        except Exception as exc:
            logger.warning(f"{label} execution failed: {exc}. Using in-memory fallback.")
            return self.search_tools_fallback(
                query=query,
                domain=domain,
                user_roles=user_roles,
                top_k=limit,
                enforce_rbac=enforce_rbac,
            )

    def search_tools_sync(
        self,
        query: str,
        domain: Optional[str] = None,
        user_roles: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        distance_threshold: Optional[float] = None,
        enforce_rbac: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Synchronous wrapper around search_tools with automatic fallback to in-memory matching.
        Safe for use inside synchronous nodes or test suites.
        """
        return self._run_async_search_sync(
            coro_fn=self.search_tools,
            query=query,
            domain=domain,
            user_roles=user_roles,
            top_k=top_k,
            distance_threshold=distance_threshold,
            enforce_rbac=enforce_rbac,
            label="search_tools_sync",
        )

    def search_tools_hybrid_sync(
        self,
        query: str,
        domain: Optional[str] = None,
        user_roles: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        distance_threshold: Optional[float] = None,
        enforce_rbac: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Synchronous wrapper around search_tools_hybrid with automatic fallback to in-memory matching.
        """
        return self._run_async_search_sync(
            coro_fn=self.search_tools_hybrid,
            query=query,
            domain=domain,
            user_roles=user_roles,
            top_k=top_k,
            distance_threshold=distance_threshold,
            enforce_rbac=enforce_rbac,
            label="search_tools_hybrid_sync",
        )

    def search_tools_fallback(
        self,
        query: str,
        domain: Optional[str] = None,
        user_roles: Optional[List[str]] = None,
        top_k: int = 8,
        enforce_rbac: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        In-memory hybrid keyword, sample-query, description, and Conditional RBAC fallback scorer
        used when PostgreSQL/PGVector or embeddings are offline or inside unit tests.
        """
        cleaned_query = (query or "").lower().strip()
        if not cleaned_query:
            return []

        tokens = _extract_query_tokens(cleaned_query)
        token_set = set(tokens)
        should_enforce, cleaned_roles = _resolve_rbac_roles(user_roles, enforce_rbac=enforce_rbac)
        is_admin = "admin" in cleaned_roles
        roles_set = set(cleaned_roles)

        candidates: List[Tuple[float, Dict[str, Any]]] = []

        for tool in self._in_memory_catalog.values():
            # Domain check
            if domain and tool["domain"] != domain:
                continue

            # Conditional RBAC check
            if should_enforce:
                req_perm = tool.get("required_permission")
                if req_perm is not None and not is_admin and req_perm not in roles_set:
                    continue
            elif tool.get("is_write", False):
                continue

            score = 0.0
            desc_lower = tool["description"].lower()
            desc_tokens = set(re.findall(r"[a-z0-9]+", desc_lower))
            name_lower = tool["tool_name"].lower()
            name_parts = set(name_lower.split("_"))
            keywords = tool.get("keywords", [])
            sample_queries = tool.get("sample_queries", [])

            # 1. Exact keyword phrase matches
            for kw in keywords:
                if kw and re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", cleaned_query):
                    score += 3.5

            # 2. Sample query overlap scoring (captures natural phrasing & slang)
            best_sample_overlap = 0.0
            all_sample_tokens: Set[str] = set()
            for sq in sample_queries:
                sq_low = sq.lower()
                sq_tokens = set(_extract_query_tokens(sq_low))
                all_sample_tokens.update(sq_tokens)
                if not token_set or not sq_tokens:
                    continue
                overlap = len(token_set & sq_tokens)
                if overlap >= 2:
                    sample_score = min(4.5, overlap * 1.5)
                    if sample_score > best_sample_overlap:
                        best_sample_overlap = sample_score
                elif overlap == 1 and len(token_set) == 1:
                    best_sample_overlap = max(best_sample_overlap, 1.5)
            score += best_sample_overlap

            # 3. Individual token scoring against tool_name, description, and sample vocabulary
            for token in tokens:
                if token in name_parts or (len(token) >= 4 and token in name_lower):
                    score += 3.0
                elif token in desc_tokens or (len(token) >= 4 and token in desc_lower):
                    score += 1.0
                elif token in all_sample_tokens:
                    score += 0.75

            if score >= 3.0:
                candidates.append((score, tool))

        candidates.sort(key=lambda x: (-x[0], x[1]["tool_name"]))

        selected: List[Dict[str, Any]] = []
        for rank_idx, (score, tool) in enumerate(candidates[:top_k], start=1):
            item = dict(tool)
            item["fallback_score"] = round(score, 4)
            item["rrf_score"] = round(1.0 / (RRF_K_CONSTANT + rank_idx), 6)
            item["distance"] = None
            selected.append(item)
        return selected

    def get_tool_by_name(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve tool metadata by tool_name from in-memory catalog."""
        return self._in_memory_catalog.get(tool_name)


# Global singleton instance
_tool_registry_service: Optional[ToolRegistryService] = None


def get_tool_registry_service() -> ToolRegistryService:
    """Access global ToolRegistryService singleton."""
    global _tool_registry_service
    if _tool_registry_service is None:
        _tool_registry_service = ToolRegistryService()
    return _tool_registry_service
