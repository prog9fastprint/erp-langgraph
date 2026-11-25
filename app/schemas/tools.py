"""
Pydantic Schemas for AI Tools Layer (FastPrint ERP Native Tools).
Provides strict input validation for LLMs and response contract validation for Django Core.
"""
from datetime import datetime
from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar, Union
from pydantic import BaseModel, Field, field_validator, model_validator

T = TypeVar("T")


def _validate_date_str(value: Optional[str], field_name: str) -> Optional[str]:
    """Validate that string adheres to YYYY-MM-DD date format."""
    if value is None or value == "":
        return None
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return value
    except ValueError:
        raise ValueError(f"Invalid date format for '{field_name}': '{value}'. Must be YYYY-MM-DD.")


# ==============================================================================
# 1. Input Schemas (LLM Tool Calling Filters)
# ==============================================================================

class AccountBalanceFilterSchema(BaseModel):
    """Input filter schema for accounting.get_account_balance."""
    company_id: Optional[int] = Field(None, description="Company ID (defaults to default company if omitted)")
    account_id: Optional[int] = Field(None, description="Specific chart of account ID")
    account_code: Optional[str] = Field(None, description="Account code string (e.g. '111100')")
    date_from: Optional[str] = Field(None, description="Start date filter in YYYY-MM-DD format")
    date_to: Optional[str] = Field(None, description="End date filter in YYYY-MM-DD format")
    include_breakdown: Optional[bool] = Field(None, description="Tampilkan rincian daftar akun (default True di backend). Jika False, hanya kembalikan blok totals")

    @field_validator("date_from")
    @classmethod
    def validate_date_from(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_from")

    @field_validator("date_to")
    @classmethod
    def validate_date_to(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_to")


class GeneralLedgerFilterSchema(BaseModel):
    """Input filter schema for accounting.get_general_ledger."""
    company_id: Optional[int] = Field(None, description="Company ID")
    account_id: Optional[int] = Field(None, description="Filter ID akun tunggal")
    account_code: Optional[str] = Field(None, description="Filter kode akun spesifik (misal '111100')")
    date_from: Optional[str] = Field(None, description="Batas awal tanggal mutasi (YYYY-MM-DD)")
    date_to: Optional[str] = Field(None, description="Batas akhir tanggal mutasi (YYYY-MM-DD)")
    journal_id: Optional[int] = Field(None, description="Filter buku jurnal tertentu")
    status: Optional[str] = Field("post", description="Filter status jurnal: 'post', 'draft', 'all', 'cancel'")
    limit: int = Field(50, ge=1, le=200, description="Batas baris transaksi per akun (default 50, maks 200)")
    offset: int = Field(0, ge=0, description="Offset paginasi transaksi")

    @field_validator("date_from")
    @classmethod
    def validate_date_from(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_from")

    @field_validator("date_to")
    @classmethod
    def validate_date_to(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_to")


class ProfitAndLossFilterSchema(BaseModel):
    """Input filter schema for accounting.get_profit_and_loss."""
    company_id: Optional[int] = Field(None, description="Company ID")
    date_from: Optional[str] = Field(None, description="Tanggal awal periode pelaporan (YYYY-MM-DD)")
    date_to: Optional[str] = Field(None, description="Tanggal akhir periode pelaporan (YYYY-MM-DD)")
    include_breakdown: bool = Field(False, description="Tampilkan rincian hierarki grup, tipe, dan akun (default False)")

    @field_validator("date_from")
    @classmethod
    def validate_date_from(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_from")

    @field_validator("date_to")
    @classmethod
    def validate_date_to(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_to")


class BalanceSheetFilterSchema(BaseModel):
    """Input filter schema for accounting.get_balance_sheet."""
    company_id: Optional[int] = Field(None, description="Company ID")
    as_of_date: Optional[str] = Field(None, description="Tanggal cut-off saldo kumulatif (YYYY-MM-DD)")
    date_to: Optional[str] = Field(None, description="Alias untuk as_of_date (YYYY-MM-DD)")
    include_breakdown: bool = Field(False, description="Tampilkan hierarki Aset, Liabilitas, dan Ekuitas (default False)")

    @field_validator("as_of_date")
    @classmethod
    def validate_as_of_date(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "as_of_date")

    @field_validator("date_to")
    @classmethod
    def validate_date_to(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_to")

    @model_validator(mode="after")
    def sync_as_of_date(self) -> "BalanceSheetFilterSchema":
        if not self.as_of_date and self.date_to:
            self.as_of_date = self.date_to
        return self


class VendorBillsFilterSchema(BaseModel):
    """Input filter schema for accounting.get_vendor_bills."""
    company_id: Optional[int] = Field(None, description="Company ID")
    vendor_id: Optional[int] = Field(None, description="Filter ID vendor spesifik")
    vendor_name: Optional[str] = Field(None, description="Filter nama vendor")
    status: Optional[str] = Field(None, description="Filter status faktur: 'draft', 'post', 'paid', 'cancel'")
    due_status: Optional[Literal["all", "overdue", "due_soon"]] = Field("all", description="Filter jatuh tempo: 'all', 'overdue', 'due_soon'")
    date_from: Optional[str] = Field(None, description="Batas awal tanggal faktur (YYYY-MM-DD)")
    date_to: Optional[str] = Field(None, description="Batas akhir tanggal faktur (YYYY-MM-DD)")
    limit: int = Field(50, ge=1, le=200, description="Jumlah baris data (default 50, maks 200)")
    offset: int = Field(0, ge=0, description="Offset paginasi (default 0)")

    @field_validator("date_from")
    @classmethod
    def validate_date_from(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_from")

    @field_validator("date_to")
    @classmethod
    def validate_date_to(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_to")


class BankReconciliationFilterSchema(BaseModel):
    """Input filter schema for accounting.get_bank_reconciliation_summary."""
    company_id: Optional[int] = Field(None, description="Company ID")
    journal_id: Optional[int] = Field(None, description="ID buku jurnal bank (opsional)")
    date_from: Optional[str] = Field(None, description="Tanggal awal rentang rekonsiliasi (YYYY-MM-DD)")
    date_to: Optional[str] = Field(None, description="Tanggal akhir rentang rekonsiliasi (YYYY-MM-DD)")

    @field_validator("date_from")
    @classmethod
    def validate_date_from(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_from")

    @field_validator("date_to")
    @classmethod
    def validate_date_to(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_to")


class JournalEntriesFilterSchema(BaseModel):
    """Input filter schema for accounting.get_journal_entries."""
    company_id: Optional[int] = Field(None, description="Company ID (defaults to default company)")
    journal_id: Optional[int] = Field(None, description="Filter by specific journal book ID")
    entry_number: Optional[str] = Field(None, description="Exact or partial journal entry number (e.g. 'JE/2026/001')")
    vendor_id: Optional[int] = Field(None, description="Filter by specific vendor ID")
    vendor_name: Optional[str] = Field(None, description="Filter by vendor or supplier name")
    status: Optional[str] = Field(None, description="Status filter: 'draft', 'posted', 'cancel', etc.")
    date_from: Optional[str] = Field(None, description="Filter entries on or after YYYY-MM-DD")
    date_to: Optional[str] = Field(None, description="Filter entries on or before YYYY-MM-DD")
    account_id: Optional[int] = Field(None, description="Filter entries containing items for this account ID")
    search: Optional[str] = Field(None, description="Keyword search matching document number or description")
    include_lines: Optional[bool] = Field(None, description="Include line items detail for each journal entry")
    limit: int = Field(50, ge=1, le=200, description="Page limit (1-200)")
    offset: int = Field(0, ge=0, description="Pagination offset")

    @field_validator("date_from")
    @classmethod
    def validate_date_from(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_from")

    @field_validator("date_to")
    @classmethod
    def validate_date_to(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_to")


class StockLevelFilterSchema(BaseModel):
    """Input filter schema for inventory.get_stock_level."""
    company_id: Optional[int] = Field(None, description="Company ID")
    product_id: Optional[int] = Field(None, description="Product Variant ID")
    code: Optional[str] = Field(None, description="Product SKU code, barcode, or product variant name")
    codes: Optional[List[str]] = Field(None, description="Daftar kode SKU untuk pengecekan batch multi-produk")
    product_ids: Optional[List[int]] = Field(None, description="Daftar ProductVariant ID untuk pengecekan batch")
    warehouse_id: Optional[int] = Field(None, description="Filter stok berdasarkan ID Gudang spesifik")
    location_id: Optional[int] = Field(None, description="Optional inventory location ID filter")
    location_name: Optional[str] = Field(None, description="Optional warehouse or location name (e.g. 'mainstore', 'gudang utama')")
    include_breakdown: Optional[bool] = Field(None, description="Menyertakan array rincian stok per lokasi/gudang")
    include_cost: Optional[bool] = Field(None, description="Menyertakan perhitungan HPP dan total valuasi nilai stok")
    limit: int = Field(20, ge=1, le=100, description="Max records to return")
    offset: int = Field(0, ge=0, description="Offset for pagination")

    @model_validator(mode="after")
    def validate_identifier(self) -> "StockLevelFilterSchema":
        if (
            self.product_id is None
            and not self.code
            and not self.codes
            and not self.product_ids
            and self.warehouse_id is None
            and self.location_id is None
            and not self.location_name
        ):
            raise ValueError("Either 'product_id' or 'code' (SKU/name), 'warehouse_id', 'location_id', or batch list must be provided for stock check.")
        return self


class LocationsFilterSchema(BaseModel):
    """Input filter schema for inventory.get_locations."""
    company_id: Optional[int] = Field(None, description="Company ID (defaults to default company)")
    search: Optional[str] = Field(None, description="Keyword search matching warehouse or location name")
    warehouse_id: Optional[int] = Field(None, description="Filter lokasi yang berada di bawah gudang tertentu")
    type: Optional[str] = Field(None, description="Location type filter: 'internal', 'customer', 'vendor'")
    category: Optional[str] = Field(None, description="Kategori operasional lokasi: 'inventory', 'production', 'scrap', 'retur', 'transit', 'adjustment'")
    limit: int = Field(20, ge=1, le=100, description="Max results (1-100)")


class ProductInfoFilterSchema(BaseModel):
    """Input filter schema for inventory.get_product_info."""
    company_id: Optional[int] = Field(None, description="Company ID")
    product_id: Optional[int] = Field(None, description="Specific Product ID")
    code: Optional[str] = Field(None, description="Exact product code / SKU")
    search: Optional[str] = Field(None, description="Keyword search matching product code, name, or reference")
    category_id: Optional[int] = Field(None, description="Filter berdasarkan ID ProductCategory")
    can_be_sold: Optional[bool] = Field(None, description="Filter produk yang dapat dijual")
    can_be_purchased: Optional[bool] = Field(None, description="Filter produk yang dapat dibeli")
    include_variants: bool = Field(True, description="Whether to include variant details")
    include_cost: Optional[bool] = Field(None, description="Jika true, sertakan estimasi HPP (default_cost)")
    limit: int = Field(10, ge=1, le=50, description="Max results (1-50)")
    offset: Optional[int] = Field(None, description="Offset pagination")


class StockHistoryFilterSchema(BaseModel):
    """Input filter schema for inventory.get_stock_history (TASK-4.2 / Phase 2)."""
    company_id: Optional[int] = Field(None, description="Company ID")
    code: Optional[str] = Field(None, description="Kode SKU produk")
    product_id: Optional[int] = Field(None, description="ID produk")
    warehouse_id: Optional[int] = Field(None, description="Filter pergerakan di gudang tertentu")
    location_id: Optional[int] = Field(None, description="Filter pergerakan di lokasi/rak tertentu")
    operation_type: Optional[str] = Field(None, description="Tipe operasi (e.g. Receipts, Delivery Orders, Scrap)")
    date_from: Optional[str] = Field(None, description="Tanggal awal YYYY-MM-DD")
    date_to: Optional[str] = Field(None, description="Tanggal akhir YYYY-MM-DD")
    source_ref: Optional[str] = Field(None, description="Nomor referensi dokumen transfer asal")
    limit: int = Field(20, ge=1, le=100, description="Max results (1-100)")
    offset: int = Field(0, ge=0, description="Offset pagination")

    @field_validator("date_from")
    @classmethod
    def validate_date_from(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_from")

    @field_validator("date_to")
    @classmethod
    def validate_date_to(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_to")


class TransfersFilterSchema(BaseModel):
    """Input filter schema for inventory.get_transfers (TASK-4.2 / Phase 2)."""
    company_id: Optional[int] = Field(None, description="Company ID")
    ref: Optional[str] = Field(None, description="Nomor dokumen transfer (e.g. WH/IN/0001)")
    source_document: Optional[str] = Field(None, description="Nomor PO atau SO asal")
    status: Optional[str] = Field(None, description="Status dokumen transfer ('Draft', 'Waiting', 'In Progress', 'Done', 'Canceled')")
    operation_type: Optional[str] = Field(None, description="Nama atau jenis operasi transfer")
    contact_id: Optional[int] = Field(None, description="ID vendor atau customer terkait")
    date_from: Optional[str] = Field(None, description="Tanggal awal YYYY-MM-DD")
    date_to: Optional[str] = Field(None, description="Tanggal akhir YYYY-MM-DD")
    include_lines: bool = Field(True, description="Sertakan rincian daftar baris barang (TransferLine)")
    limit: int = Field(20, ge=1, le=50, description="Max results (1-50)")

    @field_validator("date_from")
    @classmethod
    def validate_date_from(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_from")

    @field_validator("date_to")
    @classmethod
    def validate_date_to(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_to")


class LowStockAlertsFilterSchema(BaseModel):
    """Input filter schema for inventory.get_low_stock_alerts (Phase 3)."""
    company_id: Optional[int] = Field(None, description="Company ID")
    warehouse_id: Optional[int] = Field(None, description="Filter gudang tertentu")
    category_id: Optional[int] = Field(None, description="Filter kategori produk")
    search: Optional[str] = Field(None, description="Kata kunci pencarian SKU atau nama produk")
    threshold: float = Field(0.0, description="Ambang batas stok (default 0 untuk stok habis/minus)")
    exclude_zero: Optional[bool] = Field(None, description="Jika true, hanya tampilkan produk dengan stok positif (> 0) terendah")
    limit: int = Field(50, ge=1, le=100, description="Batas jumlah record (1-100)")
    offset: int = Field(0, ge=0, description="Offset paginasi")


class StockAdjustmentsFilterSchema(BaseModel):
    """Input filter schema for inventory.get_stock_adjustments (Phase 3)."""
    company_id: Optional[int] = Field(None, description="Company ID")
    product_id: Optional[int] = Field(None, description="ID produk")
    location_id: Optional[int] = Field(None, description="ID lokasi")
    status: Optional[str] = Field(None, description="Status adjustment ('accepted', 'pending', 'cancelled')")
    date_from: Optional[str] = Field(None, description="Tanggal awal YYYY-MM-DD")
    date_to: Optional[str] = Field(None, description="Tanggal akhir YYYY-MM-DD")
    limit: int = Field(20, ge=1, le=100, description="Max results (1-100)")

    @field_validator("date_from")
    @classmethod
    def validate_date_from(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_from")

    @field_validator("date_to")
    @classmethod
    def validate_date_to(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_to")


class ScrapRecordsFilterSchema(BaseModel):
    """Input filter schema for inventory.get_scrap_records (Phase 3)."""
    company_id: Optional[int] = Field(None, description="Company ID")
    product_id: Optional[int] = Field(None, description="ID produk")
    source_location_id: Optional[int] = Field(None, description="ID lokasi asal barang rusak")
    date_from: Optional[str] = Field(None, description="Tanggal awal YYYY-MM-DD")
    date_to: Optional[str] = Field(None, description="Tanggal akhir YYYY-MM-DD")
    limit: int = Field(20, ge=1, le=100, description="Max results (1-100)")

    @field_validator("date_from")
    @classmethod
    def validate_date_from(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_from")

    @field_validator("date_to")
    @classmethod
    def validate_date_to(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_to")


class WarehousesFilterSchema(BaseModel):
    """Input filter schema for inventory.get_warehouses (Phase 3)."""
    company_id: Optional[int] = Field(None, description="Company ID")
    search: Optional[str] = Field(None, description="Pencarian nama gudang")
    include_locations: bool = Field(True, description="Menyertakan rincian lokasi internal di dalam gudang")


class ProductCategoriesFilterSchema(BaseModel):
    """Input filter schema for inventory.get_product_categories (Phase 3)."""
    company_id: Optional[int] = Field(None, description="Company ID")
    search: Optional[str] = Field(None, description="Pencarian nama kategori produk")


class TopSellingProductsFilterSchema(BaseModel):
    """Input filter schema for sales.get_top_selling_products."""
    company_id: Optional[int] = Field(None, description="Company ID")
    company_ids: Optional[List[int]] = Field(None, description="Optional list of authorized company IDs")
    date_from: Optional[str] = Field(None, description="Tanggal awal transaksi (YYYY-MM-DD)")
    date_to: Optional[str] = Field(None, description="Tanggal akhir transaksi (YYYY-MM-DD)")
    limit: int = Field(10, ge=1, le=100, description="Jumlah produk teratas yang ditampilkan (1-100)")
    search: Optional[str] = Field(None, description="Pencarian bebas nama produk atau SKU")
    channel_name: Optional[str] = Field(None, description="Filter nama sales channel (e.g. 'WA', 'Shopee')")
    sort_by: Optional[Literal["revenue", "qty"]] = Field(None, description="Urutkan peringkat berdasarkan 'revenue' atau 'qty'")

    @field_validator("date_from")
    @classmethod
    def validate_date_from(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_from")

    @field_validator("date_to")
    @classmethod
    def validate_date_to(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_to")


class SalesOrdersFilterSchema(BaseModel):
    """Input filter schema for sales.get_sales_orders."""
    company_id: Optional[int] = Field(None, description="Company ID")
    company_ids: Optional[List[int]] = Field(None, description="Optional list of authorized company IDs")
    date_from: Optional[str] = Field(None, description="Tanggal awal transaksi (YYYY-MM-DD)")
    date_to: Optional[str] = Field(None, description="Tanggal akhir transaksi (YYYY-MM-DD)")
    status: Optional[str] = Field("confirmed", description="Filter status: 'confirmed', 'active', 'quotation', 'cancel', 'all', atau status spesifik")
    customer_name: Optional[str] = Field(None, description="Filter nama pelanggan / customer")
    customer_id: Optional[int] = Field(None, description="Filter ID kontak customer")
    channel_name: Optional[str] = Field(None, description="Filter nama sales channel (e.g. 'WA', 'Shopee', 'Tokopedia')")
    product_search: Optional[str] = Field(None, description="Filter nama produk atau kode SKU pada baris pesanan")
    search: Optional[str] = Field(None, description="Pencarian bebas nomor SO, alias, resi kurir, atau nama pelanggan")
    min_amount: Optional[float] = Field(None, description="Batas minimal total nominal order")
    max_amount: Optional[float] = Field(None, description="Batas maksimal total nominal order")
    sort_by: Optional[Literal["date_desc", "amount_desc", "amount_asc"]] = Field(
        None, description="Urutan hasil: 'date_desc' (default), 'amount_desc', atau 'amount_asc'"
    )
    limit: int = Field(20, ge=1, le=100, description="Batas jumlah baris order (1-100, default 20)")
    offset: int = Field(0, ge=0, description="Offset paginasi")

    @field_validator("date_from")
    @classmethod
    def validate_date_from(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_from")

    @field_validator("date_to")
    @classmethod
    def validate_date_to(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_to")


class ContactsFilterSchema(BaseModel):
    """Input filter schema for sales.get_contacts."""
    company_id: Optional[int] = Field(None, description="Company ID")
    company_ids: Optional[List[int]] = Field(None, description="Optional list of authorized company IDs")
    contact_id: Optional[int] = Field(None, description="Specific Contact ID")
    search: Optional[str] = Field(None, description="Pencarian nama kontak, nomor HP/WA, email, username MP, atau alamat")
    channel_name: Optional[str] = Field(None, description="Filter kontak yang bertransaksi pada channel tertentu (e.g. 'WA', 'Shopee')")
    product_search: Optional[str] = Field(None, description="Filter kontak yang membeli produk atau SKU tertentu")
    date_from: Optional[str] = Field(None, description="Tanggal awal transaksi (YYYY-MM-DD)")
    date_to: Optional[str] = Field(None, description="Tanggal akhir transaksi (YYYY-MM-DD)")
    status: Optional[str] = Field(None, description="Filter status pesanan: 'active', 'confirmed', 'quotation', 'cancel', 'all'")
    is_vendor: Optional[bool] = Field(None, description="Filter kontak vendor (True) vs pelanggan (False)")
    limit: int = Field(20, ge=1, le=100, description="Batas jumlah kontak (1-100, default 20)")
    offset: int = Field(0, ge=0, description="Offset paginasi")

    @field_validator("date_from")
    @classmethod
    def validate_date_from(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_from")

    @field_validator("date_to")
    @classmethod
    def validate_date_to(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_to")


class SalesOrderDetailFilterSchema(BaseModel):
    """Input filter schema for sales.get_sales_order_detail."""
    company_id: Optional[int] = Field(None, description="Company ID")
    order_ref: str = Field(..., description="Nomor SO (misal 'SO/2026/001'), ID order, invoice marketplace, atau nomor resi kurir")


class SalesSummaryFilterSchema(BaseModel):
    """Input filter schema for sales.get_sales_summary."""
    company_id: Optional[int] = Field(None, description="Company ID")
    date_from: Optional[str] = Field(None, description="Tanggal awal periode pelaporan (YYYY-MM-DD)")
    date_to: Optional[str] = Field(None, description="Tanggal akhir periode pelaporan (YYYY-MM-DD)")
    channel: Optional[str] = Field(None, description="Filter sales channel tertentu")
    status: Optional[str] = Field("confirmed", description="Filter status ('confirmed' default, 'all', 'quotation', 'cancel')")

    @field_validator("date_from")
    @classmethod
    def validate_date_from(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_from")

    @field_validator("date_to")
    @classmethod
    def validate_date_to(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_to")


class AccountHealthFilterSchema(BaseModel):
    """Input filter schema for marketplace.get_account_health."""
    company_id: Optional[int] = Field(None, description="Company ID")
    company_ids: Optional[List[int]] = Field(None, description="Optional additional authorized company IDs for multi-branch summary")
    marketplace: Optional[str] = Field(None, description="Optional platform filter (e.g. 'Shopee', 'TikTok Shop', 'Tokopedia', 'Lazada')")
    account_id: Optional[int] = Field(None, description="Optional single MarketplaceAccount ID")
    only_issues: bool = Field(False, description="Return only accounts with expired/expiring tokens, missing configs, inactive shops, or sync errors")
    error_window_hours: int = Field(24, ge=1, le=168, description="Lookback window in hours (1-168) for SystemLogEvent fetch errors")


class MarketplaceOrdersFilterSchema(BaseModel):
    """Input filter schema for marketplace.get_marketplace_orders."""
    company_id: Optional[int] = Field(None, description="Company ID")
    company_ids: Optional[List[int]] = Field(None, description="Optional additional authorized company IDs")
    marketplace: Optional[str] = Field(None, description="Optional platform filter (e.g. 'Shopee', 'TikTok Shop', 'Tokopedia', 'Lazada')")
    account_id: Optional[int] = Field(None, description="Optional MarketplaceAccount ID")
    mp_status: Optional[str] = Field("all", description="Status group ('processing', 'completed', 'cancelled', 'all') or exact platform status")
    sla_status: Literal["all", "overdue_shipping", "due_soon", "missing_awb", "status_mismatch"] = Field(
        "all", description="SLA/diagnostic filter: 'all', 'overdue_shipping', 'due_soon', 'missing_awb', 'status_mismatch'"
    )
    date_from: Optional[str] = Field(None, description="Optional start quotation_date (YYYY-MM-DD)")
    date_to: Optional[str] = Field(None, description="Optional end quotation_date (YYYY-MM-DD)")
    search: Optional[str] = Field(None, description="Search mp_invoice_number, SO number, AWB, no_reservasi, or buyer_name")
    limit: int = Field(20, ge=1, le=100, description="Max rows (1-100)")
    offset: int = Field(0, ge=0, description="Pagination offset")

    @field_validator("date_from")
    @classmethod
    def validate_date_from(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_from")

    @field_validator("date_to")
    @classmethod
    def validate_date_to(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_to")


class KilatOrderDiagnosticsFilterSchema(BaseModel):
    """Input filter schema for marketplace.get_kilat_order_diagnostics."""
    company_id: Optional[int] = Field(None, description="Company ID")
    company_ids: Optional[List[int]] = Field(None, description="Optional additional authorized company IDs")
    account_id: Optional[int] = Field(None, description="Optional MarketplaceAccount ID")
    reference: Optional[str] = Field(None, description="Optional reservation code, Shopee invoice, or pasted '(No. Reservasi: ...)' string")
    issue_type: Literal["all", "unupdated_invoice", "duplicate_collision", "missing_so_link"] = Field(
        "all", description="Filter: 'all', 'unupdated_invoice', 'duplicate_collision', or 'missing_so_link'"
    )
    date_from: Optional[str] = Field(None, description="Optional start quotation_date (YYYY-MM-DD)")
    date_to: Optional[str] = Field(None, description="Optional end quotation_date (YYYY-MM-DD)")
    limit: int = Field(20, ge=1, le=100, description="Max rows (1-100)")
    offset: int = Field(0, ge=0, description="Pagination offset")

    @field_validator("date_from")
    @classmethod
    def validate_date_from(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_from")

    @field_validator("date_to")
    @classmethod
    def validate_date_to(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_to")


class ProductMappingStatusFilterSchema(BaseModel):
    """Input filter schema for marketplace.get_product_mapping_status."""
    company_id: Optional[int] = Field(None, description="Company ID")
    company_ids: Optional[List[int]] = Field(None, description="Optional additional authorized company IDs")
    marketplace: Optional[str] = Field(None, description="Optional platform filter")
    account_id: Optional[int] = Field(None, description="Optional MarketplaceAccount ID")
    mapping_status: Literal[
        "all",
        "mapped",
        "unmapped",
        "unmapped_by_system",
        "unmapped_by_user",
        "exceptions",
        "multipliers_only",
        "duplicates",
    ] = Field(
        "all",
        description="Filter: 'all', 'mapped', 'unmapped', 'unmapped_by_system', 'unmapped_by_user', 'exceptions', 'multipliers_only', 'duplicates'",
    )
    search: Optional[str] = Field(None, description="Search ERP SKU, ERP product name, marketplace SKU, or listing name")
    include_deleted: bool = Field(False, description="Include soft-deleted mapping lines (is_delete=True)")
    limit: int = Field(30, ge=1, le=100, description="Max rows (1-100)")
    offset: int = Field(0, ge=0, description="Pagination offset")


class StockSyncSimulationFilterSchema(BaseModel):
    """Input filter schema for marketplace.get_stock_sync_simulation."""
    company_id: Optional[int] = Field(None, description="Company ID")
    account_id: Optional[int] = Field(None, description="Optional MarketplaceAccount ID")
    marketplace: Optional[str] = Field(None, description="Optional platform filter (e.g. 'Shopee')")
    company_ids: Optional[List[int]] = Field(None, description="Optional additional authorized company IDs")
    sku: Optional[str] = Field(None, description="Optional ERP SKU or marketplace SKU filter")
    product_id: Optional[int] = Field(None, description="Optional ProductVariant ID filter")
    only_discrepancies: bool = Field(
        False,
        description="Return only items where simulated push qty differs from last StockUpdateLog or has config warnings",
    )
    limit: int = Field(20, ge=1, le=50, description="Max items (1-50)")
    offset: int = Field(0, ge=0, description="Pagination offset")


class SyncExecutionLogsFilterSchema(BaseModel):
    """Input filter schema for marketplace.get_sync_execution_logs."""
    company_id: Optional[int] = Field(None, description="Company ID")
    log_type: Literal["all", "stock", "price", "pcls_price"] = Field(
        "all", description="Log stream: 'all', 'stock', 'price', or 'pcls_price'"
    )
    company_ids: Optional[List[int]] = Field(None, description="Optional additional authorized company IDs")
    account_id: Optional[int] = Field(None, description="Optional MarketplaceAccount ID")
    marketplace: Optional[str] = Field(None, description="Optional platform filter")
    status: Literal["FAILED", "SKIPPED", "SUCCESS", "all"] = Field(
        "FAILED", description="Status filter: 'FAILED' (default), 'SKIPPED', 'SUCCESS', or 'all'"
    )
    trigger_source: Optional[str] = Field(
        None,
        description="Optional StockUpdateLog trigger source ('CELERY_BEAT', 'MANUAL_BULK', 'MANUAL_SELECTIVE', 'MANUAL_API', 'SYSTEM')",
    )
    sku: Optional[str] = Field(None, description="Optional SKU or product name keyword")
    date_from: Optional[str] = Field(None, description="Start date (YYYY-MM-DD). Defaults to 7 days ago")
    date_to: Optional[str] = Field(None, description="End date (YYYY-MM-DD). Defaults to today")
    include_summary: bool = Field(True, description="Include aggregate success/failed/skipped summary")
    limit: int = Field(20, ge=1, le=100, description="Max rows per log stream (1-100)")
    offset: int = Field(0, ge=0, description="Pagination offset")

    @field_validator("date_from")
    @classmethod
    def validate_date_from(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_from")

    @field_validator("date_to")
    @classmethod
    def validate_date_to(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_to")


class PriceComparisonFilterSchema(BaseModel):
    """Input filter schema for marketplace.get_price_comparison."""
    company_id: Optional[int] = Field(None, description="Company ID")
    pricelist_id: Optional[int] = Field(None, description="ERP PricelistModel ID. Auto-resolved from account_id AccountOrder if omitted")
    account_id: Optional[int] = Field(None, description="MarketplaceAccount ID to compare")
    company_ids: Optional[List[int]] = Field(None, description="Optional additional authorized company IDs")
    search: Optional[str] = Field(None, description="SKU or product name search")
    filter_status: Literal["all", "mismatch", "match", "missing_mp_price", "missing_erp_price"] = Field(
        "mismatch", description="Filter: 'all', 'mismatch' (default), 'match', 'missing_mp_price', 'missing_erp_price'"
    )
    min_diff_pct: Optional[float] = Field(None, description="Optional minimum absolute variance percentage filter")
    include_wholesales: bool = Field(False, description="Include WholesaleMPModel / multi-qty PricelistLineModel tier rules")
    include_live: bool = Field(False, description="Fetch live Shopee/TikTok prices for returned page (capped at 20 SKUs, read-only)")
    limit: int = Field(20, ge=1, le=50, description="Max rows (1-50, clamped to 20 when include_live=True)")
    offset: int = Field(0, ge=0, description="Pagination offset")

    @model_validator(mode="after")
    def clamp_live_limit(self) -> "PriceComparisonFilterSchema":
        if self.include_live and self.limit > 20:
            self.limit = 20
        return self


class FeeAndEscrowAnomaliesFilterSchema(BaseModel):
    """Input filter schema for marketplace.get_fee_and_escrow_anomalies."""
    company_id: Optional[int] = Field(None, description="Company ID")
    company_ids: Optional[List[int]] = Field(None, description="Optional additional authorized company IDs")
    marketplace: Optional[str] = Field(None, description="Optional platform filter")
    account_id: Optional[int] = Field(None, description="Optional MarketplaceAccount ID")
    category: Literal["all", "negative_fee", "high_fee", "zero_income", "erp_mp_mismatch"] = Field(
        "all", description="Filter: 'all', 'negative_fee', 'high_fee', 'zero_income', or 'erp_mp_mismatch'"
    )
    min_pct: float = Field(20.0, description="Minimum fee % threshold for high_fee")
    max_pct: Optional[float] = Field(None, description="Optional maximum fee % bound")
    mp_status: Optional[str] = Field(None, description="Optional platform order status filter")
    exclude_cancelled: bool = Field(True, description="Exclude cancelled orders (which naturally have 0 expected income)")
    date_from: Optional[str] = Field(None, description="Start date (YYYY-MM-DD). Defaults to 30 days ago")
    date_to: Optional[str] = Field(None, description="End date (YYYY-MM-DD). Defaults to today")
    order_by: Literal[
        "-fee_percentage",
        "fee_percentage",
        "-fee_difference",
        "fee_difference",
        "-mp_total",
        "-date",
    ] = Field("-fee_percentage", description="Sort key")
    limit: int = Field(20, ge=1, le=100, description="Max rows (1-100)")
    offset: int = Field(0, ge=0, description="Pagination offset")

    @field_validator("date_from")
    @classmethod
    def validate_date_from(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_from")

    @field_validator("date_to")
    @classmethod
    def validate_date_to(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_to")


class DeductionAndSettlementSummaryFilterSchema(BaseModel):
    """Input filter schema for marketplace.get_deduction_and_settlement_summary."""
    company_id: Optional[int] = Field(None, description="Company ID")
    company_ids: Optional[List[int]] = Field(None, description="Optional additional authorized company IDs")
    marketplace: Optional[str] = Field(None, description="Optional platform filter")
    account_id: Optional[int] = Field(None, description="Optional MarketplaceAccount ID")
    date_from: Optional[str] = Field(None, description="Start date (YYYY-MM-DD). Defaults to start of current month")
    date_to: Optional[str] = Field(None, description="End date (YYYY-MM-DD). Defaults to today")
    date_field: Literal["received_date", "created_at", "order_date"] = Field(
        "received_date", description="Date field to filter: 'received_date' (default), 'created_at', or 'order_date'"
    )
    deduction_category: Optional[str] = Field(None, description="Optional category name filter (e.g. 'Komisi Platform', 'Biaya Layanan')")
    only_unposted_journals: bool = Field(False, description="Filter orders missing an active posted JournalEntry")
    include_order_details: bool = Field(False, description="Include paginated settlement rows alongside executive totals")
    limit: int = Field(20, ge=1, le=100, description="Max detail rows (1-100)")
    offset: int = Field(0, ge=0, description="Pagination offset")

    @field_validator("date_from")
    @classmethod
    def validate_date_from(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_from")

    @field_validator("date_to")
    @classmethod
    def validate_date_to(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_to")


class CashflowAndBalanceSummaryFilterSchema(BaseModel):
    """Input filter schema for marketplace.get_cashflow_and_balance_summary."""
    company_id: Optional[int] = Field(None, description="Company ID")
    company_ids: Optional[List[int]] = Field(None, description="Optional additional authorized company IDs")
    marketplace: Optional[str] = Field(None, description="Optional platform filter (e.g. 'Shopee', 'TikTok Shop')")
    account_id: Optional[int] = Field(None, description="Optional MarketplaceAccount ID")
    date_from: Optional[str] = Field(None, description="Start cashflow date (YYYY-MM-DD). Defaults to 30 days ago")
    date_to: Optional[str] = Field(None, description="End cashflow date (YYYY-MM-DD). Defaults to today")
    include_cashflow_logs: bool = Field(False, description="Include paginated SETTLEMENT/WITHDRAWAL/ADJUSTMENT log rows")
    include_live: bool = Field(False, description="Query live Shopee/TikTok wallet balances without writing to DB snapshots")
    limit: int = Field(30, ge=1, le=100, description="Max cashflow log rows (1-100)")
    offset: int = Field(0, ge=0, description="Pagination offset")

    @field_validator("date_from")
    @classmethod
    def validate_date_from(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_from")

    @field_validator("date_to")
    @classmethod
    def validate_date_to(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_to")


class CompetitorInsightsFilterSchema(BaseModel):
    """Input filter schema for marketplace.get_competitor_insights."""
    company_id: Optional[int] = Field(None, description="Company ID")
    company_ids: Optional[List[int]] = Field(None, description="Optional additional authorized company IDs")
    group_id: Optional[int] = Field(None, description="Optional MarketplaceCompetitorGroup ID")
    product_id: Optional[int] = Field(None, description="Optional internal ERP ProductVariant ID")
    sku: Optional[str] = Field(None, description="Optional internal ERP SKU")
    marketplace: Optional[str] = Field(None, description="Optional platform filter (e.g. 'Shopee', 'TikTok Shop')")
    date_from: Optional[str] = Field(None, description="Start inspection_date (YYYY-MM-DD). Defaults to 30 days ago")
    date_to: Optional[str] = Field(None, description="End inspection_date (YYYY-MM-DD). Defaults to today")
    focus: Literal["overview", "price_drops", "top_velocity", "stale_audits"] = Field(
        "overview", description="View focus: 'overview', 'price_drops', 'top_velocity', or 'stale_audits'"
    )
    stale_days: int = Field(7, ge=1, le=365, description="Threshold days without audit to flag active competitors as stale")
    limit: int = Field(20, ge=1, le=100, description="Max rows (1-100)")
    offset: int = Field(0, ge=0, description="Pagination offset")

    @field_validator("date_from")
    @classmethod
    def validate_date_from(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_from")

    @field_validator("date_to")
    @classmethod
    def validate_date_to(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_to")


class VendorInvoiceFilterSchema(BaseModel):
    """Input filter schema for vendor bills & invoices (TASK-4.2)."""
    vendor_name: Optional[str] = Field(None, description="Nama lengkap atau alias vendor / supplier (e.g. 'PT Sumber Makmur')")
    vendor_id: Optional[int] = Field(None, description="ID spesifik vendor jika tersedia")
    date_from: Optional[str] = Field(None, description="Tanggal awal transaksi (YYYY-MM-DD)")
    date_to: Optional[str] = Field(None, description="Tanggal akhir transaksi (YYYY-MM-DD)")
    status: Optional[str] = Field(None, description="Status tagihan: 'draft', 'posted', 'paid', 'unpaid', dll.")
    search: Optional[str] = Field(None, description="Pencarian bebas nomor invoice / referensi / catatan")
    include_lines: Optional[bool] = Field(None, description="Include line items detail for each invoice")
    company_id: Optional[int] = Field(None, description="Company ID (scoping multi-company)")
    limit: int = Field(50, ge=1, le=200, description="Batas jumlah record (1-200)")
    offset: int = Field(0, ge=0, description="Pagination offset")

    @field_validator("date_from")
    @classmethod
    def validate_date_from(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_from")

    @field_validator("date_to")
    @classmethod
    def validate_date_to(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_str(v, "date_to")

    def to_journal_filter(self) -> "JournalEntriesFilterSchema":
        """Convert to standard JournalEntriesFilterSchema payload for ERP Django Core."""
        search_term = self.search or self.vendor_name
        return JournalEntriesFilterSchema(
            company_id=self.company_id,
            vendor_name=self.vendor_name,
            vendor_id=self.vendor_id,
            status=self.status,
            date_from=self.date_from,
            date_to=self.date_to,
            search=search_term,
            include_lines=self.include_lines,
            limit=self.limit,
            offset=self.offset,
        )


# ==============================================================================
# 2. Output & Response Contract Schemas (Django Core ai_tools/core/response.py)
# ==============================================================================

class ERPErrorDetail(BaseModel):
    """Standard error detail object returned by Django Core."""
    code: str = Field(..., description="Error code identifier")
    message: str = Field(..., description="Human-readable error explanation")
    details: Dict[str, Any] = Field(default_factory=dict, description="Detailed validation/context data")


class ERPResponseEnvelope(BaseModel, Generic[T]):
    """Standard response envelope from FastPrint ERP AI Tools layer."""
    success: bool = Field(..., description="True if execution succeeded, False otherwise")
    data: Optional[T] = Field(None, description="Result payload if success is True")
    error: Optional[ERPErrorDetail] = Field(None, description="Error details if success is False")


# ------------------------------------------------------------------------------
# 2.1 Accounting Tool Data Models
# ------------------------------------------------------------------------------

class AccountBalanceTotals(BaseModel):
    opening_balance: float = 0.0
    period_debit: float = 0.0
    period_credit: float = 0.0
    ending_balance: float = 0.0


class AccountBalanceItem(BaseModel):
    account_id: int
    account_code: str
    account_name: Optional[str] = None
    opening_balance: Optional[float] = None
    period_debit: Optional[float] = None
    period_credit: Optional[float] = None
    ending_balance: Optional[float] = None
    total_debit: float = 0.0
    total_credit: float = 0.0
    net_balance: float = 0.0
    balance_position: Optional[str] = None


class AccountBalanceDataSchema(BaseModel):
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    totals: Optional[AccountBalanceTotals] = None
    accounts: List[AccountBalanceItem] = Field(default_factory=list)
    total_debit: float = 0.0
    total_credit: float = 0.0
    net_balance: float = 0.0


class GeneralLedgerTransactionItem(BaseModel):
    date: Optional[str] = None
    entry_number: Optional[str] = None
    ref: Optional[str] = None
    description: Optional[str] = None
    partner_name: Optional[str] = None
    debit: float = 0.0
    credit: float = 0.0
    running_balance: float = 0.0


class GeneralLedgerAccountItem(BaseModel):
    account_id: int
    account_code: str
    account_name: Optional[str] = None
    opening_balance: float = 0.0
    period_debit: float = 0.0
    period_credit: float = 0.0
    closing_balance: float = 0.0
    items: List[GeneralLedgerTransactionItem] = Field(default_factory=list)


class GeneralLedgerDataSchema(BaseModel):
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    accounts: List[GeneralLedgerAccountItem] = Field(default_factory=list)


class ProfitAndLossSummarySchema(BaseModel):
    total_revenue: float = 0.0
    cogs: float = 0.0
    gross_profit: float = 0.0
    operating_expenses: float = 0.0
    operating_profit: float = 0.0
    other_net: float = 0.0
    net_profit: float = 0.0
    gross_margin_percent: float = 0.0


class ProfitAndLossDataSchema(BaseModel):
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    summary: ProfitAndLossSummarySchema = Field(default_factory=ProfitAndLossSummarySchema)
    breakdown: Optional[List[Dict[str, Any]]] = None


class BalanceSheetNetBalanceCheck(BaseModel):
    is_balanced: bool = True
    difference: float = 0.0
    assets_equal_liabilities_plus_equity: bool = True


class BalanceSheetSummarySchema(BaseModel):
    total_assets: float = 0.0
    total_liabilities: float = 0.0
    total_equity: float = 0.0
    total_liabilities_and_equity: float = 0.0
    net_balance_check: Optional[BalanceSheetNetBalanceCheck] = None


class BalanceSheetDataSchema(BaseModel):
    as_of_date: Optional[str] = None
    summary: BalanceSheetSummarySchema = Field(default_factory=BalanceSheetSummarySchema)
    breakdown: Optional[List[Dict[str, Any]]] = None


class VendorBillItem(BaseModel):
    id: int
    number: str
    vendor_id: Optional[int] = None
    vendor_name: Optional[str] = None
    bill_date: Optional[str] = None
    due_date: Optional[str] = None
    total_amount: float = 0.0
    paid_amount: float = 0.0
    residual_balance: float = 0.0
    payment_status: Optional[str] = None
    due_status: Optional[str] = None


class VendorBillsDataSchema(BaseModel):
    total_count: int = 0
    limit: int = 50
    offset: int = 0
    bills: List[VendorBillItem] = Field(default_factory=list)


class BankReconciliationSummarySchema(BaseModel):
    bank_statement_balance: float = 0.0
    erp_book_balance: float = 0.0
    difference: float = 0.0
    unreconciled_statement_lines_count: int = 0
    unreconciled_gl_lines_count: int = 0


class BankReconciliationDataSchema(BaseModel):
    journal_id: Optional[int] = None
    journal_name: Optional[str] = None
    bank_account_code: Optional[str] = None
    bank_account_name: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    summary: BankReconciliationSummarySchema = Field(default_factory=BankReconciliationSummarySchema)


class JournalEntryItem(BaseModel):
    id: int
    number: str
    date: Optional[str] = None
    description: Optional[str] = None
    reference: Optional[str] = None
    status: Optional[str] = None
    journal_id: Optional[int] = None
    journal_name: Optional[str] = None
    vendor_id: Optional[int] = None
    vendor_name: Optional[str] = None
    total_debit: float = 0.0
    total_credit: float = 0.0
    total_amount: Optional[float] = 0.0
    lines: Optional[List[Dict[str, Any]]] = None


class JournalEntriesDataSchema(BaseModel):
    total_count: int
    limit: int
    offset: int
    entries: List[JournalEntryItem] = Field(default_factory=list)
    calculations: Optional[Dict[str, Any]] = None


# ------------------------------------------------------------------------------
# 2.2 Inventory Tool Data Models
# ------------------------------------------------------------------------------

class StockBreakdownItem(BaseModel):
    warehouse_id: Optional[int] = None
    warehouse_name: Optional[str] = None
    location_id: Optional[int] = None
    location_name: Optional[str] = None
    qty: float = 0.0


class StockLevelItem(BaseModel):
    product_id: Optional[int] = None
    code: Optional[str] = None
    name: Optional[str] = None
    company_id: Optional[int] = None
    company_name: Optional[str] = None
    location_id: Optional[int] = None
    location_name: Optional[str] = None
    qty_on_hand: float = 0.0
    breakdown: Optional[List[StockBreakdownItem]] = None


class StockLevelDataSchema(BaseModel):
    mode: Optional[str] = "single_product"
    product_id: Optional[int] = None
    code: Optional[str] = None
    name: Optional[str] = None
    company_id: Optional[int] = 1
    company_name: Optional[str] = None
    location_id: Optional[int] = None
    qty_on_hand: float = 0.0
    breakdown: Optional[List[StockBreakdownItem]] = None
    products: Optional[List[StockLevelItem]] = None


class ProductCategoryRef(BaseModel):
    id: Optional[int] = None
    name: Optional[str] = None


class UomRef(BaseModel):
    id: Optional[int] = None
    name: Optional[str] = None


class ProductInfoItem(BaseModel):
    id: int
    code: str
    name: str
    full_name: Optional[str] = None
    internal_reference: Optional[str] = None
    barcode: Optional[str] = None
    category: Optional[Union[ProductCategoryRef, Dict[str, Any], str]] = None
    uom: Optional[Union[UomRef, Dict[str, Any], str]] = None
    purchase_uom: Optional[Union[UomRef, Dict[str, Any], str]] = None
    last_unit_price: float = 0.0
    default_cost: Optional[float] = None
    can_be_sold: bool = True
    can_be_purchased: bool = True
    is_bundling: bool = False
    traceability: Optional[str] = None
    weight: Optional[float] = None
    volume: Optional[float] = None


class ProductInfoDataSchema(BaseModel):
    total_found: int = 0
    products: List[ProductInfoItem] = Field(default_factory=list)


class LocationItem(BaseModel):
    id: int
    name: str
    full_name: Optional[str] = None
    warehouse_id: Optional[int] = None
    warehouse_name: Optional[str] = None
    type: Optional[str] = None
    category: Optional[str] = None
    parent_id: Optional[int] = None


class LocationsDataSchema(BaseModel):
    total_found: int = 0
    locations: List[LocationItem] = Field(default_factory=list)


class StockHistoryItem(BaseModel):
    date: Optional[str] = None
    reference: Optional[str] = None
    product_id: Optional[int] = None
    code: Optional[str] = None
    product_name: Optional[str] = None
    operation_type: Optional[str] = None
    location_id: Optional[int] = None
    location_name: Optional[str] = None
    qty_done: float = 0.0
    balance: Optional[float] = None


class StockHistoryDataSchema(BaseModel):
    total_count: int = 0
    history: List[StockHistoryItem] = Field(default_factory=list)


class TransferLineItem(BaseModel):
    product_id: Optional[int] = None
    product_code: Optional[str] = None
    product_name: Optional[str] = None
    quantity: float = 0.0
    uom: Optional[str] = None


class TransferItem(BaseModel):
    id: int
    ref: Optional[str] = None
    name: Optional[str] = None
    source_document: Optional[str] = None
    status: Optional[str] = None
    operation_type: Optional[str] = None
    date: Optional[str] = None
    contact_id: Optional[int] = None
    contact_name: Optional[str] = None
    lines: Optional[List[Union[TransferLineItem, Dict[str, Any]]]] = None


class TransfersDataSchema(BaseModel):
    total_count: int = 0
    transfers: List[TransferItem] = Field(default_factory=list)


class LowStockAlertItem(BaseModel):
    product_id: Optional[int] = None
    code: Optional[str] = None
    name: Optional[str] = None
    company_id: Optional[int] = None
    company_name: Optional[str] = None
    warehouse_id: Optional[int] = None
    warehouse_name: Optional[str] = None
    location_id: Optional[int] = None
    location_name: Optional[str] = None
    qty_on_hand: float = 0.0
    threshold: float = 0.0
    min_qty: Optional[float] = None


class LowStockAlertsDataSchema(BaseModel):
    company_id: Optional[int] = None
    company_name: Optional[str] = None
    total_found: int = 0
    total_count: Optional[int] = None
    alerts: List[LowStockAlertItem] = Field(default_factory=list)


class StockAdjustmentItem(BaseModel):
    id: int
    reference: Optional[str] = None
    product_id: Optional[int] = None
    product_name: Optional[str] = None
    location_id: Optional[int] = None
    location_name: Optional[str] = None
    theoretical_qty: float = 0.0
    real_qty: float = 0.0
    difference: float = 0.0
    status: Optional[str] = None
    date: Optional[str] = None


class StockAdjustmentsDataSchema(BaseModel):
    total_count: int = 0
    adjustments: List[StockAdjustmentItem] = Field(default_factory=list)


class ScrapRecordItem(BaseModel):
    id: int
    reference: Optional[str] = None
    product_id: Optional[int] = None
    product_name: Optional[str] = None
    quantity: float = 0.0
    source_location_name: Optional[str] = None
    reason: Optional[str] = None
    date: Optional[str] = None


class ScrapRecordsDataSchema(BaseModel):
    total_count: int = 0
    scraps: List[ScrapRecordItem] = Field(default_factory=list)


class WarehouseItem(BaseModel):
    id: int
    name: str
    code: Optional[str] = None
    company_id: Optional[int] = None
    locations: Optional[List[Dict[str, Any]]] = None


class WarehousesDataSchema(BaseModel):
    total_found: int = 0
    warehouses: List[WarehouseItem] = Field(default_factory=list)


class ProductCategoryItem(BaseModel):
    id: int
    name: str
    parent_id: Optional[int] = None
    costing_method: Optional[str] = None
    standard_uom: Optional[str] = None


class ProductCategoriesDataSchema(BaseModel):
    total_found: int = 0
    categories: List[ProductCategoryItem] = Field(default_factory=list)


# ------------------------------------------------------------------------------
# Concrete Envelope Types for Documentation & Type Checking
# ------------------------------------------------------------------------------

class AccountBalanceResponseSchema(ERPResponseEnvelope[AccountBalanceDataSchema]):
    pass


class JournalEntriesResponseSchema(ERPResponseEnvelope[JournalEntriesDataSchema]):
    pass


class GeneralLedgerResponseSchema(ERPResponseEnvelope[GeneralLedgerDataSchema]):
    pass


class ProfitAndLossResponseSchema(ERPResponseEnvelope[ProfitAndLossDataSchema]):
    pass


class BalanceSheetResponseSchema(ERPResponseEnvelope[BalanceSheetDataSchema]):
    pass


class VendorBillsResponseSchema(ERPResponseEnvelope[VendorBillsDataSchema]):
    pass


class BankReconciliationResponseSchema(ERPResponseEnvelope[BankReconciliationDataSchema]):
    pass


class StockLevelResponseSchema(ERPResponseEnvelope[StockLevelDataSchema]):
    pass


class ProductInfoResponseSchema(ERPResponseEnvelope[ProductInfoDataSchema]):
    pass


class LocationsResponseSchema(ERPResponseEnvelope[LocationsDataSchema]):
    pass


class StockHistoryResponseSchema(ERPResponseEnvelope[StockHistoryDataSchema]):
    pass


class TransfersResponseSchema(ERPResponseEnvelope[TransfersDataSchema]):
    pass


class LowStockAlertsResponseSchema(ERPResponseEnvelope[LowStockAlertsDataSchema]):
    pass


class StockAdjustmentsResponseSchema(ERPResponseEnvelope[StockAdjustmentsDataSchema]):
    pass


class ScrapRecordsResponseSchema(ERPResponseEnvelope[ScrapRecordsDataSchema]):
    pass


class WarehousesResponseSchema(ERPResponseEnvelope[WarehousesDataSchema]):
    pass


class ProductCategoriesResponseSchema(ERPResponseEnvelope[ProductCategoriesDataSchema]):
    pass


# ------------------------------------------------------------------------------
# RAG Knowledge Search Schemas
# ------------------------------------------------------------------------------

class KnowledgeSearchFilterSchema(BaseModel):
    """Input filter schema for searching qualitative knowledge, SOPs, and policies."""
    query: str = Field(..., description="Kata kunci atau pertanyaan penelusuran SOP, kebijakan, atau profil master")
    category: Optional[Literal["sop", "product_spec", "policy", "vendor_terms", "general"]] = Field(
        None, description="Kategori dokumen knowledge opsional"
    )
    top_k: int = Field(3, ge=1, le=10, description="Jumlah chunk paling relevan yang ingin diambil")
    company_id: Optional[int] = Field(None, description="Company ID untuk multi-tenancy scoping")


class KnowledgeChunkItem(BaseModel):
    id: Optional[str] = None
    title: str = Field(..., description="Judul dokumen atau rujukan (e.g. 'SOP Retur Barang v1.2')")
    category: str = Field(..., description="Kategori dokumen (sop, policy, spec, dll.)")
    content: str = Field(..., description="Teks isi dokumen chunk")
    similarity_score: float = Field(0.0, description="Cosine similarity score (0.0 - 1.0)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata dokumen: pasal, halaman, tanggal, dll.")


class KnowledgeSearchDataSchema(BaseModel):
    total_found: int = 0
    chunks: List[KnowledgeChunkItem] = Field(default_factory=list)


class KnowledgeSearchResponseSchema(ERPResponseEnvelope[KnowledgeSearchDataSchema]):
    pass


# ==============================================================================
# 3. Canonical Domain Mapping Registry
# ==============================================================================

TOOL_DOMAIN_MAP: Dict[str, str] = {
    "fetch_vendor_invoices": "accounting",
    "get_account_balance": "accounting",
    "get_journal_entries": "accounting",
    "get_general_ledger": "accounting",
    "get_profit_and_loss": "accounting",
    "get_balance_sheet": "accounting",
    "get_vendor_bills": "accounting",
    "get_bank_reconciliation_summary": "accounting",
    "get_stock_level": "inventory",
    "get_product_info": "inventory",
    "get_locations": "inventory",
    "get_stock_history": "inventory",
    "get_transfers": "inventory",
    "get_low_stock_alerts": "inventory",
    "get_stock_adjustments": "inventory",
    "get_scrap_records": "inventory",
    "get_warehouses": "inventory",
    "get_product_categories": "inventory",
    "get_top_selling_products": "sales",
    "get_sales_orders": "sales",
    "get_sales_order_detail": "sales",
    "get_sales_summary": "sales",
    "get_contacts": "sales",
    "get_account_health": "marketplace",
    "get_marketplace_orders": "marketplace",
    "get_kilat_order_diagnostics": "marketplace",
    "get_product_mapping_status": "marketplace",
    "get_stock_sync_simulation": "marketplace",
    "get_sync_execution_logs": "marketplace",
    "get_price_comparison": "marketplace",
    "get_fee_and_escrow_anomalies": "marketplace",
    "get_deduction_and_settlement_summary": "marketplace",
    "get_cashflow_and_balance_summary": "marketplace",
    "get_competitor_insights": "marketplace",
}

