"""
Regex intent patterns and disambiguation term tables for FastPrint ERP Router.
"""
import re
from typing import Optional


# Disambiguation term sets
CUSTOMER_TERMS = {"pelanggan", "customer", "piutang", "buyer", "klien"}
VENDOR_TERMS = {"vendor", "supplier", "pemasok", "hutang"}
PURCHASE_INVOICE_TERMS = ("faktur pembelian", "tagihan pembelian", "invoice pembelian")
CUSTOMER_BILLING_TERMS = (
    "tagihan",
    "faktur",
    "invoice",
    "bill",
    "piutang",
    "hutang",
    "pembayaran",
    "bayar",
)

ACCOUNTING_PROFIT_LOSS_PATTERNS = [
    r"\b(laba\s*(?:dan\s*)?rugi(?:nya)?|rugi\s*laba(?:nya)?|p&l|profit\s*(?:and|&)\s*loss|income\s*statement|omzet(?:nya)?|omset(?:nya)?|gross\s*margin|laba\s*kotor|laba\s*bersih)\b",
]

ACCOUNTING_BALANCE_SHEET_PATTERNS = [
    r"\b(neraca(?:\s*keuangan)?|balance\s*sheet|posisi\s*keuangan|total\s*aset|aktiva\s*(?:dan\s*)?pasiva|keseimbangan\s*neraca)\b",
]

ACCOUNTING_GENERAL_LEDGER_PATTERNS = [
    r"\b(buku\s*besar|general\s*ledger|ledger\s*akun|kartu\s*buku\s*besar|running\s*balance|mutasi\s*rekening)\b",
]

ACCOUNTING_BANK_RECON_PATTERNS = [
    r"\b(rekonsiliasi\s*bank|bank\s*reconciliation|rekon\s*bank|selisih\s*(?:rekening\s*koran|bank)|statement\s*balance)\b",
]

ACCOUNTING_VENDOR_BILLS_PATTERNS = [
    r"\b(vendor\s*bills?|ap\s*aging|umur\s*hutang|jatuh\s*tempo|overdue|due\s*soon)\b",
]

ACCOUNTING_BALANCE_PATTERNS = [
    r"\bsaldo(nya)?\b",
    r"\bbalance\b",
    r"\bkas\b",
    r"\bkas kecil\b",
    r"\bbuku kas\b",
    r"\bsaldo akun\b",
]

ACCOUNTING_JOURNAL_PATTERNS = [
    r"\bjurnal(nya)?\b",
    r"\bjournal\b",
    r"\bentri jurnal\b",
    r"\bdebit kredit\b",
]

INVENTORY_STOCK_PATTERNS = [
    r"\bstok(nya)?\b",
    r"\bstock(nya)?\b",
    r"\bsisa barang(nya)?\b",
    r"\bsisa stok(nya)?\b",
    r"\bkuantitas(nya)?\b",
    r"\bqty\b",
    r"\bon hand\b",
]

INVENTORY_STOCK_HISTORY_PATTERNS = [
    r"\bkartu\s+stok\b",
    r"\briwayat\s+(mutasi|stok|pergerakan)\b",
    r"\bmutasi\s+stok\b",
    r"\bpergerakan\s+(stok|barang)\b",
    r"\bbuku\s+besar\s+stok\b",
    r"\bstock\s+history\b",
    r"\bhistory\s+stok\b",
    r"\bkartu\s+persediaan\b",
]

INVENTORY_TRANSFERS_PATTERNS = [
    r"\bsurat\s+jalan\b",
    r"\btransfer\s+(barang|gudang|stok|internal)\b",
    r"\bmutasi\s+(antar\s+gudang|barang)\b",
    r"\bpengiriman\s+barang\b",
    r"\bpenerimaan\s+barang\b",
    r"\bdelivery\s+order\b",
    r"\bwh/(?:in|out)/\d+\b",
    r"\bdokumen\s+transfer\b",
]

INVENTORY_LOW_STOCK_PATTERNS = [
    r"\b(?:stok|stock)(?:nya)?\s+(?:paling\s+)?(?:menipis|habis|kritis|minimum|kurang|minus|sedikit|tersedikit|terkecil|terendah)\b",
    r"\b(?:stok|stock)(?:nya)?\s+.*?(?:menipis|habis|kritis|minimum|tersedikit|terkecil|terendah|paling\s+sedikit)\b",
    r"\b(?:produk|barang|item)\s+(?:yang\s+)?(?:stok|stock)(?:nya)?\s+(?:paling\s+)?(?:tersedikit|sedikit|terkecil|terendah|menipis|habis)\b",
    r"\blow\s+stock\b",
    r"\balert\s+stok\b",
    r"\bstok\s+alert\b",
    r"\bdi\s*bawah\s+minimum\b",
    r"\bperingatan\s+stok\b",
]

INVENTORY_ADJUSTMENTS_PATTERNS = [
    r"\bstock\s+opname\b",
    r"\bstok\s+opname\b",
    r"\bpenyesuaian\s+stok\b",
    r"\badjustment\s+stok\b",
    r"\bselisih\s+stok\b",
    r"\bkoreksi\s+stok\b",
    r"\bhasil\s+opname\b",
]

INVENTORY_SCRAP_PATTERNS = [
    r"\bbarang\s+(rusak|cacat|scrap)\b",
    r"\bscrap\s*(order|record)?\b",
    r"\bpemusnahan\s+barang\b",
    r"\bproduk\s+rusak\b",
    r"\bcatatan\s+scrap\b",
]

INVENTORY_WAREHOUSES_PATTERNS = [
    r"\bmaster\s+gudang\b",
    r"\b(daftar|list)\s+master\s+gudang\b",
    r"\bhierarki\s+gudang\b",
    r"\bmaster\s+warehouses?\b",
    r"\binfo\s+master\s+gudang\b",
]

INVENTORY_CATEGORIES_PATTERNS = [
    r"\bkategori\s+(produk|barang)\b",
    r"\bmaster\s+kategori\b",
    r"\bdaftar\s+kategori\b",
    r"\bproduct\s+categor(?:y|ies)\b",
]

INVENTORY_LOCATION_PATTERNS = [
    r"\b(daftar|list|semua|info|data)\s+(gudang|lokasi|rak|warehouse)\b",
    r"\bada\s+(gudang|lokasi|rak|warehouse)\b",
    r"\b(gudang|lokasi|rak|warehouse)\s+(apa\s+saja|tersedia|aktif)\b",
    r"^(gudang|lokasi|rak|warehouse)\b",
]

SALES_TOP_PRODUCTS_PATTERNS = [
    r"\b(?:produk|barang|item)\s+(?:dengan\s+)?(?:pendapatan|omset|omzet|penjualan|laba)\s+(?:tertinggi|terbanyak|terbesar|terlaris|teratas|top)\b",
    r"\b(?:top|peringkat|daftar)\s+\d*\s*(?:produk|barang|item)\s+(?:terlaris|terbanyak|terbesar|omset|pendapatan)\b",
    r"\b(?:pendapatan|omset|omzet|penjualan)\s+(?:tertinggi|terbesar)\s+(?:per\s+|tiap\s+)?(?:produk|barang|sku)\b",
    r"\bproduk\s+terlaris\b",
    r"\bproduk\s+paling\s+laku\b",
    r"\bpenjualan\s+(?:terbanyak|tertinggi|terbesar)\s+(?:di\s*mana|cabang\s+mana|toko\s+mana|outlet\s+mana)\b",
    r"\b(?:omzet|omset)\s+penjualan\s+sku\b",
]

SALES_SUMMARY_PATTERNS = [
    r"\b(?:ringkasan|summary|performa|laporan)\s+(?:penjualan|sales|omzet|omset)\b",
    r"\b(?:total|berapa)\s+(?:omzet|omset|penjualan)\b",
    r"\b(?:omzet|omset)\s+(?:penjualan|bulan\s+ini|tahun\s+ini|hari\s+ini|kemarin|periode)\b",
    r"\b(?:omzet|omset)\s+(?:cabang|perusahaan|toko)\b",
    r"\b(?:omzet|omset|penjualan|sales)\s+(?:q[1-4]|kuartal\s*[1-4]|triwulan\s*[1-4]|semester\s*[12])\b",
    r"\b(?:target|proyeksi|run[\s\-]?rate|pencapaian|tembus)\s+.*?(?:omzet|omset|penjualan|sales|target)\b",
    r"\b(?:omzet|omset|penjualan|sales)\s+.*?(?:target|tembus|mencapai|proyeksi|run[\s\-]?rate|impact|dampak)\b",
    r"\b(?:impact|dampak|pengaruh)(?:nya)?(?:\s+[a-z0-9\-]+){0,3}\s+(?:ke|terhadap)\s+(?:omzet|omset|penjualan)\b",
    r"\b(?:rata-rata|aov)\s+(?:nilai\s+order|pesanan|penjualan)\b",
]

SALES_ORDER_DETAIL_PATTERNS = [
    r"\b(?:detail|rincian|status|info|cek)\s+(?:so|sales\s*order|quotation|penawaran|pesanan|order)\s+(?:nomor\s+|no\s*)?([a-zA-Z0-9/\-_]+\d+[a-zA-Z0-9/\-_]*)\b",
    r"\b(?:detail|rincian)\s+(?:so|sales\s*order|pesanan|order)\b",
    r"\b(?:lacak|resi|awb)\s+(?:so|pesanan|order|pengiriman)\b",
    r"\b(?:so|sales\s*order|quotation)\s*(?:nomor\s+|no\s*)?[a-zA-Z0-9/\-_]+\s+(?:status|detail|rincian)\b",
    r"\bso/\d{4}/\d+\b",
    r"\b[a-z]{2,5}/so/\d+\b",
]

SALES_CONTACTS_PATTERNS = [
    r"\b(?:detail|info|data|cek|cari|lihat|daftar|list)\s+(?:contact|kontak)(?:nya)?\b",
    r"\b(?:contact|kontak)(?:nya)?\s+(?:siapa(?:\s+saja)?|mana(?:\s+saja)?|apa(?:\s+saja)?)\b",
    r"\bsiapa\s+(?:saja\s+)?(?:contact|kontak|pelanggan|customer|pembeli|buyer)(?:nya)?\b",
    r"\b(?:daftar|list|top|peringkat(?:\s+teratas)?|rangking|ranking)\s+\d*\s*(?:contact|kontak|pelanggan|customer|pembeli|buyer)\b",
    r"\b\d*\s*(?:pelanggan|customer|pembeli|buyer)\s+(?:dengan\s+)?(?:pembelian|belanja|order|transaksi|nilai)\s+(?:terbanyak|tertinggi|terbesar|teratas|paling\s+(?:banyak|besar|tinggi))\b",
    r"\b(?:pelanggan|customer|pembeli|buyer)\s+.*?(?:paling\s+banyak\s+beli|paling\s+sering\s+beli|pembelian\s+terbanyak|belanja\s+terbesar|top\s+spender)\b",
    r"\bsiapa\s+(?:saja\s+)?(?:(?:pelanggan|customer|pembeli|buyer)\s+)?(?:yang|yg)\s+paling\s+(?:banyak|sering)\s+(?:beli|belanja|order|transaksi)\b",
    r"\b(?:nomor|no\.?)\s*(?:hp|wa|whatsapp|telp|telepon)\s+(?:contact|kontak|pelanggan|customer|pembeli)\b",
    r"\b(?:contact|kontak)\s+(?:dari\s+|di\s+)?(?:wa|whatsapp|shopee|tiktok|tokopedia|lazada)\b",
    r"\b(?:contact|kontak)\s+[a-zA-Z0-9]",
]

SALES_ORDERS_PATTERNS = [
    r"\b(?:daftar|list|tampilkan|cari|cek|lihat|detail|rincian)\s+(?:so|sales\s*order|order\s+penjualan|pesanan\s+penjualan|quotation|penawaran)\b",
    r"\b(?:so|sales\s*order|order\s+penjualan|pesanan|quotation|penawaran)\s+(?:terbaru|hari\s+ini|bulan\s+ini|kemarin|minggu\s+ini|pending|draft|cancel)\b",
    r"\b(?:penjualan|omzet|omset|transaksi|nominal)\s+(?:ter\s*banyak|ter\s*besar|ter\s*tinggi|paling\s+(?:banyak|besar|tinggi))\s+(?:di\s+)?(?:quotation|penawaran|so|sales\s*order)\b",
    r"\b(?:quotation|penawaran)\s+(?:mana|apa)\s+(?:yang\s+)?(?:penjualan(?:nya)?\s+|nominal(?:nya)?\s+|total(?:nya)?\s+)?(?:ter\s*banyak|ter\s*besar|ter\s*tinggi|paling\s+(?:banyak|besar|tinggi))\b",
    r"\b(?:penjualan|order|pesanan|transaksi)\s+(?:di\s+|via\s+|lewat\s+|channel\s+)(?:wa|whatsapp|cust\s+datang|website\s+fp|goshop)\b",
    r"\b(?:ada\s+berapa|daftar)\s+(?:pesanan|order)\s+(?:dari|customer|pelanggan)\b",
    r"\bdaftar\s+pesanan\b",
    r"\bpesanan\s+penjualan\b",
]

MARKETPLACE_WRITE_PATTERNS = [
    r"\b(update|push|sinkron(?:kan)?|ubah|edit|naik(?:kan)?|turun(?:kan)?)\s+(?:data\s+)?(stok|stock|harga|price)\s+(?:ke\s+|di\s+)?(shopee|tiktok(?:\s*shop)?|tokopedia|lazada|marketplace)\b",
    r"\b(proses|gabung|merge|perbaiki|fix|hapus)\s+(?:order\s*kilat|pesanan\s*kilat|reservasi\s*kilat|handling\s+kilat)\b",
]

MARKETPLACE_ACCOUNT_HEALTH_PATTERNS = [
    r"\b(kesehatan\s+akun|account\s+health|cek\s+akun\s+marketplace|status\s+(?:akun|toko)\s+(?:marketplace|shopee|tiktok(?:\s*shop)?|tokopedia|lazada))\b",
    r"\b(token\s+(?:oauth\s+)?(?:marketplace|shopee|tiktok(?:\s*shop)?|tokopedia|lazada)|token\s+(?:expired|kadaluarsa|habis|akan\s+habis|expiring))\b",
    r"\b(gagal\s+fetch\s+order|fetch\s+order\s+error|moderasi\s+toko|accountorder|accountproperties)\b",
    r"\b(akun\s+(?:marketplace|shopee|tiktok(?:\s*shop)?|tokopedia|lazada)\s+(?:mana\s+yang|yang\s+)?(?:bermasalah|error|expired|nonaktif|inactive|gagal))\b",
    r"\b(penarikan\s+(?:data\s+|order\s+)?(?:marketplace|shopee|tiktok(?:\s*shop)?|tokopedia|lazada)|(?:nota|order|pesanan)\s+(?:marketplace|shopee|tiktok(?:\s*shop)?|tokopedia|lazada)\s+.*?(?:gagal|error|tertarik|di\s*tarik|di\s*proses))\b",
    r"\b(?:gagal|error|bermasalah)\s+(?:di\s*)?(?:tarik|penarikan|proses|fetch|sync|sinkron)\s*.*?(?:marketplace|shopee|tiktok(?:\s*shop)?|tokopedia|lazada)\b",
    r"\b(?:marketplace|shopee|tiktok(?:\s*shop)?|tokopedia|lazada)\s+.*?(?:gagal\s+(?:di\s*)?(?:tarik|proses|fetch)|penarikan\s+.*?gagal)\b",
]

MARKETPLACE_KILAT_PATTERNS = [
    r"\b(pesanan\s+kilat|order\s+kilat|kilat\s+shopee|reservasi\s+kilat|no\.?\s*reservasi|no_reservasi|invoice_updated)\b",
]

MARKETPLACE_ORDERS_PATTERNS = [
    r"\b(overdue\s+shipping|overdue\s+sla|lewat(?:i)?\s+batas\s+kirim|melewati\s+batas\s+kirim|batas\s+kirim|due\s+soon|missing\s+awb|belum\s+(?:punya|ada)\s+resi(?:\s+awb)?|tanpa\s+resi|status\s+mismatch|selisih\s+status\s+(?:erp|marketplace))\b",
    r"\b(?:order|pesanan)\s+(?:marketplace|shopee|tiktok(?:\s*shop)?|tokopedia|lazada)\b",
]

MARKETPLACE_MAPPING_PATTERNS = [
    r"\b(mapping\s+(?:produk|sku|marketplace)|product\s+mapping|produk\s+(?:marketplace\s+)?(?:yang\s+)?(?:belum|sudah)\s+ter-?mapping|unmapped(?:\s+sku)?|belum\s+(?:di|ter)-?mapping|qty_multiplier|duplikat\s+mapping|exception\s*product|masuk\s+exception)\b",
]

MARKETPLACE_STOCK_SYNC_SIM_PATTERNS = [
    r"\b(simulasi\s+(?:update\s+|sinkronisasi\s+|push\s+)?stok|simulasi\s+stok\s+(?:marketplace|shopee|tiktok(?:\s*shop)?|tokopedia|lazada)|determine_stock|shared_stock|shared\s+stock|stok\s+(?:yang\s+)?(?:di-?push|dikirim)\s+ke\s+(?:marketplace|shopee|tiktok(?:\s*shop)?|tokopedia|lazada)|selisih\s+(?:kalkulasi\s+)?stok\s+(?:erp|gudang)\s+(?:vs|dan|dengan)\s+(?:marketplace|shopee|tiktok|log))\b",
]

MARKETPLACE_SYNC_LOGS_PATTERNS = [
    r"\b(stockupdatelog|priceupdatelog|inventoryupdatepricelog|pcls\s+price)\b",
    r"\b(log\s+(?:gagal|error|skip|sukses|eksekusi|update|sinkron(?:isasi)?)\s+(?:stok|stock|harga|price|marketplace|celery))\b",
    r"\b(?:kenapa|mengapa|cek)\s+(?:update|sinkron(?:isasi)?)\s+(?:stok|stock|harga|price)\s+(?:atau\s+(?:harga|stok)\s+)?(?:marketplace|shopee|tiktok(?:\s*shop)?|tokopedia|lazada)?\s*(?:gagal|error|skip|tidak\s+update)\b",
    r"\b(?:gagal|error)\s+(?:update|sinkron(?:isasi)?)\s+(?:stok|harga)\s+(?:marketplace|shopee|tiktok(?:\s*shop)?|tokopedia|lazada|celery)\b",
]

MARKETPLACE_PRICE_COMPARISON_PATTERNS = [
    r"\b(?:bandingkan|perbandingan|selisih|compare)\s+harga\b.*?\b(?:marketplace|shopee|tiktok(?:\s*shop)?|tokopedia|lazada|pricelist|erp)\b",
    r"\b(?:cek\s+|lihat\s+|berapa\s+)?harga\b.*?\bdi\s+(?:shopee|tiktok(?:\s*shop)?|tokopedia|lazada|marketplace)\b",
    r"\b(?:harga\s+erp\s+vs\s+(?:harga\s+)?(?:shopee|tiktok(?:\s*shop)?|tokopedia|lazada|marketplace)|wholesalemp|harga\s+grosir\s+marketplace)\b",
]

MARKETPLACE_FEE_ANOMALIES_PATTERNS = [
    r"\b(anomali\s+(?:potongan|fee|biaya|escrow)|high_fee|negative_fee|zero_income|erp_mp_mismatch|seller_expected_income|expected\s+income\s*(?:nol|0|minus|negatif))\b",
    r"\b(potongan\s+(?:marketplace|shopee|tiktok(?:\s*shop)?|tokopedia|lazada)\s+(?:di\s*atas|lebih\s+dari|>=?|>)\s*\d+\s*%)\b",
]

MARKETPLACE_DEDUCTION_SETTLEMENT_PATTERNS = [
    r"\b(?:rekap(?:itulasi)?|ringkasan|total|laporan|berapa|cek|lihat)\s+(?:potongan(?:\s+fee|\s+biaya)?|deduction|komisi|biaya\s+(?:layanan|admin|penanganan)|fee)\s+(?:marketplace|shopee|tiktok(?:\s*shop)?|tokopedia|lazada)\b",
    r"\b(?:potongan(?:\s+fee|\s+biaya)?|deduction|biaya\s+(?:layanan|admin)|fee)\s+(?:dari\s+)?(?:marketplace|shopee|tiktok(?:\s*shop)?|tokopedia|lazada)\b",
    r"\b(?:impact|dampak|pengaruh)(?:nya)?\s+.*?(?:potongan|deduction|fee|biaya\s+layanan|komisi|gagal\s+tarik|penarikan)\b",
    r"\b(settlement\s+(?:marketplace\s+)?(?:mana\s+yang\s+)?belum\s+(?:ter-?)?posting|unposted\s+journal|revenue\s+history|gross\s+vs\s+nett)\b",
]

MARKETPLACE_CASHFLOW_BALANCE_PATTERNS = [
    r"\b(saldo\s+(?:dompet|wallet|seller|escrow)|pending\s+escrow|pending\s+balance|available\s+balance|balance\s+snapshot|penarikan\s+dana|withdrawal\s+(?:marketplace|shopee|tiktok)|cashflow\s+(?:marketplace|shopee|tiktok))\b",
    r"\b(?:saldo|dompet)\s+(?:toko\s+|akun\s+)?(?:shopee|tiktok(?:\s*shop)?|tokopedia|lazada)\b",
]

MARKETPLACE_COMPETITOR_PATTERNS = [
    r"\b(kompetitor|competitor|pesaing|inspect\s+kompetitor|velocity\s+(?:penjualan\s+)?kompetitor|audit\s+kompetitor|belum\s+di-?audit|stale\s+audit|delta_sold_qty)\b",
]

ALL_MARKETPLACE_PATTERN_GROUPS = (
    MARKETPLACE_ACCOUNT_HEALTH_PATTERNS
    + MARKETPLACE_KILAT_PATTERNS
    + MARKETPLACE_ORDERS_PATTERNS
    + MARKETPLACE_MAPPING_PATTERNS
    + MARKETPLACE_STOCK_SYNC_SIM_PATTERNS
    + MARKETPLACE_SYNC_LOGS_PATTERNS
    + MARKETPLACE_PRICE_COMPARISON_PATTERNS
    + MARKETPLACE_FEE_ANOMALIES_PATTERNS
    + MARKETPLACE_DEDUCTION_SETTLEMENT_PATTERNS
    + MARKETPLACE_CASHFLOW_BALANCE_PATTERNS
    + MARKETPLACE_COMPETITOR_PATTERNS
)

INVENTORY_VELOCITY_FORECAST_PATTERNS = [
    r"\b(fast\s*[-]?\s*moving|slow\s*[-]?\s*moving|low\s*[-]?\s*moving|dead\s+stock)\b",
    r"\b(forecast(?:ing)?)\s+.*?(?:stok|stock|order|produk|barang|restock|penjualan|permintaan)\b",
    r"\b(?:stok|stock|produk|barang)\s+.*?(?:fast\s*[-]?\s*moving|slow\s*[-]?\s*moving|low\s*[-]?\s*moving|forecast(?:ing)?|perputaran|turnover)\b",
    r"\b(?:mana\s+yang\s+|produk\s+(?:mana\s+)?(?:yang\s+)?)?(?:harus\s+(?:segera\s+)?di\s*[-]?\s*order|segera\s+di\s*[-]?\s*restock|rekomendasi\s+(?:restock|order))\b",
    r"\bperputaran\s+(?:stok|stock|barang|produk)\b",
]

INVENTORY_PRODUCT_PATTERNS = [
    r"\bproduk(nya)?\b",
    r"\bproduct\b",
    r"\bharga(nya)?\b",
    r"\bdetail barang\b",
    r"\bsku\b",
    r"\bbarcode\b",
    r"\bvarian\b",
]

KNOWLEDGE_RAG_PATTERNS = [
    r"\bsop\b",
    r"\bkebijakan\b",
    r"\bprosedur\b",
    r"\baturan\b",
    r"\b(per)?syarat(an)?\b",
    r"\b(ke)?tentuan\b",
    r"\b(di|me)?retur(kan)?(nya)?\b",
    r"\b(ber)?garansi\b",
    r"\bspesifikasi\b",
    r"\b(gsm|gramasi|ketebalan)\b",
    r"\bprofil\s+(vendor|supplier|customer|pelanggan)\b",
    r"\btermin\s+(pembayaran|kredit)\b",
    r"\bcara\s+(klaim|retur|pengajuan)\b",
    r"\b(di)?klaim\b",
]

GREETING_PATTERNS = [
    r"^(halo|hai|hi|hey|hei|helo|hello)\b",
    r"^(selamat\s+(pagi|siang|sore|malam|petang))\b",
    r"^(assalamu\s*['’]?alaikum|sampurasun|kulonuwun)\b",
    r"^(ping|p\b|tes|test|testing)\b",
    r"^(terima\s*kasih|makasih|thanks|thank\s*you|matur\s*nuwun)\b",
]

GENERAL_INQUIRY_PATTERNS = [
    r"^(bisa\s+(?:bantu\s+apa|apa\s+aja|melakukan\s+apa)|kamu\s+(?:siapa|bisa\s+apa(?:\s+aja)?)|anda\s+bisa\s+apa(?:\s+aja)?|fitur\s+apa(?:\s+aja)?|menu|help|bantuan|siapa\s+kamu|apa\s+yang\s+bisa\s+(?:kamu|anda)\s+(?:lakukan|bantu))\b",
]

_META_QUESTION_RE = re.compile(
    r"^(?:(?:jadi|terus|lalu|nah|oh\s+ya)\s+)?(?:"
    r"(?:periode|rentang\s+(?:waktu|tanggal)|tanggal)\s+(?:data\s+)?(?:yang|yg)\s+(?:kamu\s+|anda\s+)?(?:pakai|gunakan|dicek|cek|tampilkan|ditampilkan)\s+(?:kapan|berapa|mana)"
    r"|kapan\s+periode\s+(?:data\s+)?(?:yang|yg)\s+(?:kamu\s+|anda\s+)?(?:pakai|gunakan)"
    r"|(?:company|perusahaan|cabang)\s+(?:mana\s+(?:aja|saja)|apa\s+(?:aja|saja))(?:\s+(?:yang|yg)\s+(?:kamu\s+|anda\s+)?(?:pakai|gunakan|aktif|terdaftar|ada|tersedia|dicek))?"
    r"|(?:yang|yg)\s+(?:kamu\s+|anda\s+)?pakai\s+(?:company|perusahaan|cabang)\s+(?:mana|apa)"
    r"|(?:data|database|erp|sistem)\s+(?:terbaru|terakhir|paling\s+baru|cutoff|update)(?:\s+(?:tanggal|tgl|sampai|sejak|dari))?(?:\s+(?:kapan|berapa))?"
    r"|(?:tanggal|tgl)\s+(?:data|update)\s+(?:terbaru|terakhir)(?:\s+(?:kapan|berapa))?"
    r"|sampai\s+tanggal\s+berapa\s+data(?:\s+erp)?(?:\s+(?:tersedia|ada|terbaru))?"
    r")$",
    re.IGNORECASE,
)

_CASUAL_NON_ERP_INTERROGATIVE_RE = re.compile(
    r"\b(?:(?:kamu|anda|lu|elo)\s+(?:tau|tahu|kenal|suka|pernah|bisa\s+masak|nonton)|(?:tau|tahu)\s+(?:enggak|nggak|gak|ga)\s+(?:tentang|soal)?)\b",
    re.IGNORECASE,
)


def is_meta_question(text: str) -> bool:
    """Return True if user asks about active session period, company scope, or latest ERP date."""
    cleaned = re.sub(r"[?!.,;:'\"]+$", "", (text or "").strip())
    return bool(_META_QUESTION_RE.match(cleaned))

OUT_OF_SCOPE_PATTERNS = [
    r"\b(system\s*design|arsitektur\s*(sistem)?|source\s*code|prompt\s*(kamu|anda)|system\s*prompt)\b",
    r"\b(buatkan\s+(puisi|cerita|kode|script)|ceritakan\s+(lelucon|dongeng)|siapa\s+presiden|ramalan\s+cuaca)\b",
]

RETRY_PATTERNS = [
    r"^(?:(?:bisa\s+)?tolong\s+|mohon\s+|silakan\s+|bisa\s+)?(?:coba\s+(?:(?:di)?cek\s+|(?:di)?periksa\s+|(?:di)?cari\s+|(?:di)?lihat\s+)?(?:lagi|ulang)|(?:coba\s+)?(?:di)?cek\s+(?:sekali\s+)?(?:lagi|ulang)|(?:coba\s+)?(?:di)?periksa\s+(?:sekali\s+)?(?:lagi|ulang)|(?:coba\s+)?(?:di)?cari\s+(?:sekali\s+)?(?:lagi|ulang)|ulang(i)?|coba\s+ulang|ulangi\s+lagi|sekali\s+lagi|coba\s+sekali\s+lagi|retry|tes\s+lagi)(?:\s+dong|\s+ya|\s+deh|\s+bro|\s+kak|\s+mas|\s+mbak|\s+min|\s+gan|\s+pak|\s+bu)?$",
]


def _match_any(patterns: list, text: str) -> bool:
    """Check if any regex pattern matches the text."""
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def _find_first_match(patterns: list, text: str) -> Optional[re.Match]:
    """Find the earliest regex pattern match in text."""
    best: Optional[re.Match] = None
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            if best is None or m.start() < best.start():
                best = m
    return best

