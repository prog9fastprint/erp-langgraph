"""
Unit and integration tests for FastPrint ERP Accounting AI Tools (7 Tools Specification).
Validates schema contracts, client methods, routing heuristics, parameter extraction,
verifier record counting, and markdown synthesizer formatting.
"""
import unittest
from datetime import date
from unittest.mock import AsyncMock
from pydantic import ValidationError

from app.schemas.tools import (
    GeneralLedgerFilterSchema,
    ProfitAndLossFilterSchema,
    BalanceSheetFilterSchema,
    VendorBillsFilterSchema,
    BankReconciliationFilterSchema,
    # Response envelopes & data schemas
    ProfitAndLossResponseSchema,
    TOOL_DOMAIN_MAP,
)
from app.graph.nodes.router import classify_query_rule_based
from app.graph.nodes.extractor import (
    _fallback_extract_general_ledger,
    _fallback_extract_profit_and_loss,
    _fallback_extract_balance_sheet,
    _fallback_extract_vendor_bills,
)
from app.graph.nodes.verifier import _calculate_records_count
from app.graph.nodes.synthesizer import _format_single_tool_presentation
from app.services.django_client import DjangoCoreClient


class TestAccountingSchemas(unittest.TestCase):
    """Test suite validating all 7 accounting Pydantic schemas and response envelopes."""

    def test_all_7_accounting_tools_in_domain_map(self):
        """All 7 accounting tools must be registered in TOOL_DOMAIN_MAP."""
        expected_tools = [
            "get_account_balance",
            "get_journal_entries",
            "get_general_ledger",
            "get_profit_and_loss",
            "get_balance_sheet",
            "get_vendor_bills",
            "get_bank_reconciliation_summary",
        ]
        for tool_name in expected_tools:
            self.assertIn(tool_name, TOOL_DOMAIN_MAP)
            self.assertEqual(TOOL_DOMAIN_MAP[tool_name], "accounting")

    def test_profit_and_loss_filter_defaults(self):
        """Profit and loss filter defaults include_breakdown to False (context optimization)."""
        schema = ProfitAndLossFilterSchema(company_id=1, date_from="2026-01-01", date_to="2026-12-31")
        self.assertFalse(schema.include_breakdown)
        self.assertEqual(schema.date_from, "2026-01-01")
        self.assertEqual(schema.date_to, "2026-12-31")

    def test_balance_sheet_filter_defaults_and_sync(self):
        """Balance sheet filter syncs date_to to as_of_date and defaults include_breakdown to False."""
        schema = BalanceSheetFilterSchema(company_id=1, date_to="2026-12-31")
        self.assertEqual(schema.as_of_date, "2026-12-31")
        self.assertFalse(schema.include_breakdown)

    def test_general_ledger_filter_defaults(self):
        """General ledger filter sets default status to 'post', limit to 50, and offset to 0."""
        schema = GeneralLedgerFilterSchema(account_code="111100", company_id=1)
        self.assertEqual(schema.status, "post")
        self.assertEqual(schema.limit, 50)
        self.assertEqual(schema.offset, 0)
        self.assertEqual(schema.account_code, "111100")

    def test_vendor_bills_filter_due_status(self):
        """Vendor bills filter supports valid due_status values and rejects invalid ones."""
        schema = VendorBillsFilterSchema(due_status="overdue", company_id=1)
        self.assertEqual(schema.due_status, "overdue")

        with self.assertRaises(ValidationError):
            VendorBillsFilterSchema(due_status="invalid_status")

    def test_bank_reconciliation_filter_validation(self):
        """Bank reconciliation filter validates date strings."""
        schema = BankReconciliationFilterSchema(journal_id=5, date_from="2026-01-01", date_to="2026-01-31")
        self.assertEqual(schema.journal_id, 5)
        self.assertEqual(schema.date_from, "2026-01-01")

        with self.assertRaises(ValidationError):
            BankReconciliationFilterSchema(date_from="not-a-date")

    def test_response_envelope_schemas(self):
        """Response envelope schemas correctly wrap their respective data models."""
        pl_data = {
            "success": True,
            "data": {
                "date_from": "2026-01-01",
                "date_to": "2026-12-31",
                "summary": {
                    "total_revenue": 500000000.0,
                    "cogs": 300000000.0,
                    "gross_profit": 200000000.0,
                    "gross_margin_percent": 40.0,
                    "net_profit": 125000000.0,
                },
            },
        }
        envelope = ProfitAndLossResponseSchema.model_validate(pl_data)
        self.assertTrue(envelope.success)
        self.assertEqual(envelope.data.summary.net_profit, 125000000.0)


class TestAccountingRouter(unittest.TestCase):
    """Test suite validating rule-based routing for accounting tools."""

    def test_route_profit_and_loss(self):
        res = classify_query_rule_based("Bagaimana laporan laba rugi perusahaan tahun 2026?")
        self.assertIsNotNone(res)
        self.assertEqual(res["detected_domain"], "accounting")
        self.assertEqual(res["detected_tool"], "get_profit_and_loss")

    def test_route_balance_sheet(self):
        res = classify_query_rule_based("Cek neraca keuangan per 31 Desember 2026")
        self.assertIsNotNone(res)
        self.assertEqual(res["detected_domain"], "accounting")
        self.assertEqual(res["detected_tool"], "get_balance_sheet")

    def test_route_general_ledger(self):
        res = classify_query_rule_based("Tampilkan buku besar akun kas kecil 111000 bulan Januari")
        self.assertIsNotNone(res)
        self.assertEqual(res["detected_domain"], "accounting")
        self.assertEqual(res["detected_tool"], "get_general_ledger")

    def test_route_vendor_bills(self):
        res = classify_query_rule_based("Ada tagihan vendor apa saja yang lewat jatuh tempo?")
        self.assertIsNotNone(res)
        self.assertEqual(res["detected_domain"], "accounting")
        self.assertEqual(res["detected_tool"], "get_vendor_bills")

    def test_route_bank_reconciliation(self):
        res = classify_query_rule_based("Cek status rekonsiliasi bank BCA bulan ini")
        self.assertIsNotNone(res)
        self.assertEqual(res["detected_domain"], "accounting")
        self.assertEqual(res["detected_tool"], "get_bank_reconciliation_summary")

    def test_route_account_balance(self):
        res = classify_query_rule_based("Berapa saldo kas operasional saat ini?")
        self.assertIsNotNone(res)
        self.assertEqual(res["detected_domain"], "accounting")
        self.assertEqual(res["detected_tool"], "get_account_balance")

    def test_route_journal_entries(self):
        res = classify_query_rule_based("Tampilkan entri jurnal transaksi bulan ini")
        self.assertIsNotNone(res)
        self.assertEqual(res["detected_domain"], "accounting")
        self.assertEqual(res["detected_tool"], "get_journal_entries")


class TestAccountingExtractor(unittest.TestCase):
    """Test suite validating heuristic parameter extraction for accounting tools."""

    def test_extract_profit_and_loss_compact_by_default(self):
        params = _fallback_extract_profit_and_loss("Laba rugi bulan lalu", ref_date=date(2026, 3, 15))
        self.assertFalse(params["include_breakdown"])
        self.assertEqual(params.get("date_from"), "2026-02-01")
        self.assertEqual(params.get("date_to"), "2026-02-28")

    def test_extract_profit_and_loss_breakdown_detected(self):
        params = _fallback_extract_profit_and_loss("Tampilkan rincian laba rugi per akun bulan ini", ref_date=date(2026, 3, 15))
        self.assertTrue(params["include_breakdown"])
        self.assertEqual(params.get("date_from"), "2026-03-01")
        self.assertEqual(params.get("date_to"), "2026-03-31")

    def test_extract_balance_sheet_as_of_date(self):
        params = _fallback_extract_balance_sheet("Neraca keuangan per 2026-12-31")
        self.assertEqual(params.get("as_of_date"), "2026-12-31")
        self.assertFalse(params["include_breakdown"])

    def test_extract_general_ledger_code_and_dates(self):
        params = _fallback_extract_general_ledger("Buku besar akun 111100 bulan ini", ref_date=date(2026, 4, 10))
        self.assertEqual(params.get("account_code"), "111100")
        self.assertEqual(params.get("date_from"), "2026-04-01")
        self.assertEqual(params.get("date_to"), "2026-04-30")

    def test_extract_vendor_bills_overdue(self):
        params = _fallback_extract_vendor_bills("Tagihan vendor PT Abadi yang lewat jatuh tempo")
        self.assertEqual(params.get("due_status"), "overdue")
        self.assertEqual(params.get("vendor_name"), "PT Abadi")


class TestAccountingVerifier(unittest.TestCase):
    """Test suite validating record counting and verification logic for accounting tools."""

    def test_verify_profit_and_loss_summary_is_valid(self):
        payload = {
            "date_from": "2026-01-01",
            "date_to": "2026-12-31",
            "summary": {
                "total_revenue": 500000000.0,
                "gross_profit": 200000000.0,
                "net_profit": 125000000.0,
            },
        }
        self.assertEqual(_calculate_records_count(payload), 1)

    def test_verify_balance_sheet_summary_is_valid(self):
        payload = {
            "as_of_date": "2026-12-31",
            "summary": {
                "total_assets": 850000000.0,
                "total_liabilities": 250000000.0,
                "total_equity": 600000000.0,
            },
        }
        self.assertEqual(_calculate_records_count(payload), 1)

    def test_verify_bank_reconciliation_summary_is_valid(self):
        payload = {
            "journal_id": 5,
            "journal_name": "Bank BCA",
            "summary": {
                "bank_statement_balance": 150000000.0,
                "erp_book_balance": 150000000.0,
                "difference": 0.0,
            },
        }
        self.assertEqual(_calculate_records_count(payload), 1)

    def test_verify_vendor_bills_list_count(self):
        payload = {
            "total_count": 2,
            "bills": [
                {"id": 1, "number": "BILL/2026/001", "total_amount": 10000000.0},
                {"id": 2, "number": "BILL/2026/002", "total_amount": 5000000.0},
            ],
        }
        self.assertEqual(_calculate_records_count(payload), 2)

    def test_verify_empty_accounting_result(self):
        empty_bills = {"total_count": 0, "bills": []}
        self.assertEqual(_calculate_records_count(empty_bills), 0)


class TestAccountingSynthesizer(unittest.TestCase):
    """Test suite validating Markdown presentation formatting for accounting tools."""

    def test_synthesize_profit_and_loss(self):
        tool_result = {
            "date_from": "2026-01-01",
            "date_to": "2026-12-31",
            "summary": {
                "total_revenue": 500000000.0,
                "cogs": 300000000.0,
                "gross_profit": 200000000.0,
                "gross_margin_percent": 40.0,
                "operating_expenses": 80000000.0,
                "operating_profit": 120000000.0,
                "other_net": 5000000.0,
                "net_profit": 125000000.0,
            },
        }
        output = _format_single_tool_presentation("get_profit_and_loss", tool_result)
        self.assertIn("Laporan Laba Rugi (Profit & Loss)", output)
        self.assertIn("Pendapatan Usaha (Revenue)", output)
        self.assertIn("Rp 500.000.000,00", output)
        self.assertIn("40.0%", output)
        self.assertIn("Laba Bersih Periode Berjalan", output)
        self.assertIn("Rp 125.000.000,00", output)

    def test_synthesize_balance_sheet(self):
        tool_result = {
            "as_of_date": "2026-12-31",
            "summary": {
                "total_assets": 850000000.0,
                "total_liabilities": 250000000.0,
                "total_equity": 600000000.0,
                "total_liabilities_and_equity": 850000000.0,
                "net_balance_check": {
                    "is_balanced": True,
                    "difference": 0.0,
                },
            },
        }
        output = _format_single_tool_presentation("get_balance_sheet", tool_result)
        self.assertIn("Laporan Neraca Keuangan (Balance Sheet)", output)
        self.assertIn("Rp 850.000.000,00", output)
        self.assertIn("SEIMBANG (BALANCED)", output)

    def test_synthesize_vendor_bills(self):
        tool_result = {
            "total_count": 1,
            "bills": [
                {
                    "id": 88,
                    "number": "BILL/2026/001",
                    "vendor_name": "PT Vendor Abadi",
                    "bill_date": "2026-01-10",
                    "due_date": "2026-01-20",
                    "total_amount": 10000000.0,
                    "paid_amount": 2500000.0,
                    "residual_balance": 7500000.0,
                    "payment_status": "partial",
                    "due_status": "overdue",
                }
            ],
        }
        output = _format_single_tool_presentation("get_vendor_bills", tool_result)
        self.assertIn("BILL/2026/001", output)
        self.assertIn("PT Vendor Abadi", output)
        self.assertIn("Rp 7.500.000,00", output)
        self.assertIn("Tagihan Lewat Jatuh Tempo (Overdue)", output)

    def test_synthesize_bank_reconciliation(self):
        tool_result = {
            "journal_id": 5,
            "journal_name": "Bank BCA",
            "bank_account_code": "111101",
            "bank_account_name": "Bank BCA Operasional",
            "date_from": "2026-01-01",
            "date_to": "2026-01-31",
            "summary": {
                "bank_statement_balance": 150000000.0,
                "erp_book_balance": 150000000.0,
                "difference": 0.0,
                "unreconciled_statement_lines_count": 0,
                "unreconciled_gl_lines_count": 0,
            },
        }
        output = _format_single_tool_presentation("get_bank_reconciliation_summary", tool_result)
        self.assertIn("Ringkasan Rekonsiliasi Bank — Bank BCA", output)
        self.assertIn("Rp 150.000.000,00", output)
        self.assertIn("Terekonsiliasi Sempurna (Fully Reconciled)", output)


class TestAccountingDjangoClient(unittest.IsolatedAsyncioTestCase):
    """Test suite validating DjangoCoreClient async execution for accounting tools."""

    async def test_fetch_accounting_tools_dispatch(self):
        mock_client = AsyncMock(spec=DjangoCoreClient)
        mock_client.execute_tool = AsyncMock(return_value={"success": True, "data": {}})

        client = DjangoCoreClient()
        client.execute_tool = mock_client.execute_tool

        await client.fetch_general_ledger(params={"account_code": "111100"}, company_id=1)
        client.execute_tool.assert_called_with(
            domain="accounting",
            tool_name="get_general_ledger",
            params={"account_code": "111100"},
            company_id=1,
            user_token=None,
        )

        await client.fetch_profit_and_loss(params={"include_breakdown": False}, company_id=1)
        client.execute_tool.assert_called_with(
            domain="accounting",
            tool_name="get_profit_and_loss",
            params={"include_breakdown": False},
            company_id=1,
            user_token=None,
        )

        await client.fetch_balance_sheet(params={"as_of_date": "2026-12-31"}, company_id=1)
        client.execute_tool.assert_called_with(
            domain="accounting",
            tool_name="get_balance_sheet",
            params={"as_of_date": "2026-12-31"},
            company_id=1,
            user_token=None,
        )

        await client.fetch_vendor_bills(params={"due_status": "overdue"}, company_id=1)
        client.execute_tool.assert_called_with(
            domain="accounting",
            tool_name="get_vendor_bills",
            params={"due_status": "overdue"},
            company_id=1,
            user_token=None,
        )

        await client.fetch_bank_reconciliation_summary(params={"journal_id": 5}, company_id=1)
        client.execute_tool.assert_called_with(
            domain="accounting",
            tool_name="get_bank_reconciliation_summary",
            params={"journal_id": 5},
            company_id=1,
            user_token=None,
        )


if __name__ == "__main__":
    unittest.main()
