"""
Unit tests for AI Tools Pydantic Schemas (FastPrint ERP Native Tools).
Validates TASK-3.1 requirements (DoD for input validation and response contracts).
"""
import unittest
from pydantic import ValidationError

from app.schemas.tools import (
    # Input schemas
    AccountBalanceFilterSchema,
    JournalEntriesFilterSchema,
    StockLevelFilterSchema,
    ProductInfoFilterSchema,
    # Envelope & Error
    ERPResponseEnvelope,
    # Output schemas
    AccountBalanceResponseSchema,
    JournalEntriesResponseSchema,
    StockLevelResponseSchema,
    ProductInfoResponseSchema,
)


class TestInputFilterSchemas(unittest.TestCase):
    """Test suite for LLM tool input parameters validation."""

    def test_account_balance_valid(self):
        """Valid account balance input should parse successfully."""
        data = {
            "company_id": 1,
            "account_code": "111100",
            "date_from": "2026-01-01",
            "date_to": "2026-12-31",
        }
        schema = AccountBalanceFilterSchema(**data)
        self.assertEqual(schema.account_code, "111100")
        self.assertEqual(schema.date_from, "2026-01-01")
        self.assertEqual(schema.date_to, "2026-12-31")

    def test_account_balance_invalid_date_format_raises_error(self):
        """Invalid date format (not YYYY-MM-DD) must raise ValidationError (DoD)."""
        with self.assertRaises(ValidationError):
            AccountBalanceFilterSchema(date_from="01/01/2026")

        with self.assertRaises(ValidationError):
            AccountBalanceFilterSchema(date_to="invalid-date")

    def test_journal_entries_valid_defaults(self):
        """Journal entries should populate default limit (50) and offset (0)."""
        schema = JournalEntriesFilterSchema()
        self.assertEqual(schema.limit, 50)
        self.assertEqual(schema.offset, 0)
        self.assertIsNone(schema.status)

    def test_journal_entries_limit_bounds_validation(self):
        """Limit < 1 or > 200 must raise ValidationError (DoD)."""
        with self.assertRaises(ValidationError):
            JournalEntriesFilterSchema(limit=0)

        with self.assertRaises(ValidationError):
            JournalEntriesFilterSchema(limit=201)

        # Valid bounds
        valid_schema = JournalEntriesFilterSchema(limit=200)
        self.assertEqual(valid_schema.limit, 200)

    def test_journal_entries_invalid_date_format(self):
        """Journal entries invalid date format must raise ValidationError."""
        with self.assertRaises(ValidationError):
            JournalEntriesFilterSchema(date_from="2026-13-45")

    def test_stock_level_valid_with_product_id(self):
        """Stock level check with product_id should succeed."""
        schema = StockLevelFilterSchema(product_id=42, company_id=1)
        self.assertEqual(schema.product_id, 42)
        self.assertIsNone(schema.code)

    def test_stock_level_valid_with_code(self):
        """Stock level check with code/sku should succeed."""
        schema = StockLevelFilterSchema(code="SKU-PAPER-A4", location_id=3)
        self.assertEqual(schema.code, "SKU-PAPER-A4")
        self.assertEqual(schema.location_id, 3)

    def test_stock_level_missing_identifiers_raises_error(self):
        """Must provide either product_id or code; otherwise raise ValidationError."""
        with self.assertRaises(ValidationError) as ctx:
            StockLevelFilterSchema(company_id=1)
        self.assertIn("Either 'product_id' or 'code'", str(ctx.exception))

    def test_product_info_valid_defaults(self):
        """Product info should populate default limit (10) and include_variants (True)."""
        schema = ProductInfoFilterSchema(search="tinta")
        self.assertEqual(schema.search, "tinta")
        self.assertEqual(schema.limit, 10)
        self.assertTrue(schema.include_variants)

    def test_product_info_limit_bounds_validation(self):
        """Limit < 1 or > 50 must raise ValidationError (DoD)."""
        with self.assertRaises(ValidationError):
            ProductInfoFilterSchema(limit=0)

        with self.assertRaises(ValidationError):
            ProductInfoFilterSchema(limit=51)


class TestOutputContractSchemas(unittest.TestCase):
    """Test suite for Django Core response envelope and data models."""

    def test_erp_error_response_envelope(self):
        """Validates error response envelope structure from ai_tools/core/response.py."""
        raw_error_payload = {
            "success": False,
            "data": None,
            "error": {
                "code": "AUTHENTICATION_REQUIRED",
                "message": "Authentication required.",
                "details": {"required_header": "Authorization"},
            },
        }
        envelope = ERPResponseEnvelope[None](**raw_error_payload)
        self.assertFalse(envelope.success)
        self.assertIsNotNone(envelope.error)
        self.assertEqual(envelope.error.code, "AUTHENTICATION_REQUIRED")
        self.assertEqual(envelope.error.details["required_header"], "Authorization")

    def test_account_balance_success_response(self):
        """Validates successful account balance response payload from Django Core."""
        raw_payload = {
            "success": True,
            "data": {
                "accounts": [
                    {
                        "account_id": 12,
                        "account_code": "111100",
                        "account_name": "Kas Kecil",
                        "total_debit": 150000000.0,
                        "total_credit": 75000000.0,
                        "net_balance": 75000000.0,
                        "balance_position": "debit",
                    }
                ],
                "total_debit": 150000000.0,
                "total_credit": 75000000.0,
                "net_balance": 75000000.0,
            },
            "error": None,
        }
        resp = AccountBalanceResponseSchema(**raw_payload)
        self.assertTrue(resp.success)
        self.assertEqual(len(resp.data.accounts), 1)
        self.assertEqual(resp.data.accounts[0].account_code, "111100")
        self.assertEqual(resp.data.net_balance, 75000000.0)

    def test_journal_entries_success_response(self):
        """Validates successful journal entries response payload from Django Core."""
        raw_payload = {
            "success": True,
            "data": {
                "total_count": 1,
                "limit": 50,
                "offset": 0,
                "entries": [
                    {
                        "id": 101,
                        "number": "JE/2026/001",
                        "date": "2026-01-15",
                        "description": "Pembayaran vendor supplier",
                        "reference": "REF-9988",
                        "status": "post",
                        "journal_id": 7,
                        "journal_name": "Vendor Bill Journal",
                        "vendor_id": 5,
                        "vendor_name": "PT Sumber Kertas",
                        "total_debit": 2500000.0,
                        "total_credit": 2500000.0,
                    }
                ],
            },
            "error": None,
        }
        resp = JournalEntriesResponseSchema(**raw_payload)
        self.assertTrue(resp.success)
        self.assertEqual(resp.data.total_count, 1)
        self.assertEqual(resp.data.entries[0].vendor_name, "PT Sumber Kertas")

    def test_stock_level_success_response(self):
        """Validates successful stock level response payload from Django Core."""
        raw_payload = {
            "success": True,
            "data": {
                "product_id": 42,
                "code": "SKU-INK-CYAN",
                "name": "Tinta Cyan 100ml",
                "company_id": 1,
                "company_name": "FastPrint Surabaya",
                "location_id": 2,
                "qty_on_hand": 150.0,
            },
            "error": None,
        }
        resp = StockLevelResponseSchema(**raw_payload)
        self.assertTrue(resp.success)
        self.assertEqual(resp.data.qty_on_hand, 150.0)
        self.assertEqual(resp.data.code, "SKU-INK-CYAN")

    def test_product_info_success_response(self):
        """Validates successful product info response payload from Django Core."""
        raw_payload = {
            "success": True,
            "data": {
                "total_found": 1,
                "products": [
                    {
                        "id": 42,
                        "code": "SKU-INK-CYAN",
                        "name": "Tinta Cyan",
                        "full_name": "Tinta Cyan FastPrint 100ml",
                        "internal_reference": "INK-C-01",
                        "barcode": "8991234567890",
                        "last_unit_price": 45000.0,
                        "can_be_sold": True,
                        "can_be_purchased": True,
                        "is_bundling": False,
                    }
                ],
            },
            "error": None,
        }
        resp = ProductInfoResponseSchema(**raw_payload)
        self.assertTrue(resp.success)
        self.assertEqual(resp.data.total_found, 1)
        self.assertEqual(resp.data.products[0].last_unit_price, 45000.0)

    def test_invalid_response_structure_raises_error(self):
        """Incompatible response types from backend must raise ValidationError."""
        raw_bad_payload = {
            "success": True,
            "data": {
                "accounts": "this should be a list, not string",
            },
        }
        with self.assertRaises(ValidationError):
            AccountBalanceResponseSchema(**raw_bad_payload)


class TestToolDomainMapRegistry(unittest.TestCase):
    """Test suite ensuring TOOL_DOMAIN_MAP is centralized and consistent across modules."""

    def test_tool_domain_map_centralized_definition(self):
        from app.schemas.tools import TOOL_DOMAIN_MAP
        from app.graph.nodes.router import TOOL_DOMAIN_MAP as ROUTER_DOMAIN_MAP
        from app.graph.nodes.extractor import TOOL_DOMAIN_MAP as EXTRACTOR_DOMAIN_MAP

        self.assertIs(ROUTER_DOMAIN_MAP, TOOL_DOMAIN_MAP)
        self.assertIs(EXTRACTOR_DOMAIN_MAP, TOOL_DOMAIN_MAP)
        self.assertEqual(TOOL_DOMAIN_MAP["fetch_vendor_invoices"], "accounting")
        self.assertEqual(TOOL_DOMAIN_MAP["get_account_balance"], "accounting")
        self.assertEqual(TOOL_DOMAIN_MAP["get_journal_entries"], "accounting")
        self.assertEqual(TOOL_DOMAIN_MAP["get_stock_level"], "inventory")
        self.assertEqual(TOOL_DOMAIN_MAP["get_product_info"], "inventory")
        self.assertEqual(TOOL_DOMAIN_MAP["get_locations"], "inventory")


if __name__ == "__main__":
    unittest.main()
