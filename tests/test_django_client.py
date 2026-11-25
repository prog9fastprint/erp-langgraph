"""
Unit tests for Asynchronous Django Core HTTP Client (TASK-3.2).
Validates user token forwarding (DoD), header propagation, structured error
handling (401, 403, 404, 500), tool execution, and helper methods.
"""
import unittest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from app.core.config import settings
from app.services.django_client import (
    DjangoCoreClient,
    AsyncERPClient,
    DjangoCoreClientError,
    DjangoAuthError,
    DjangoPermissionError,
    DjangoNotFoundError,
    DjangoServerError,
)
from app.schemas.tools import (
    AccountBalanceFilterSchema,
    JournalEntriesFilterSchema,
    StockLevelFilterSchema,
    ProductInfoFilterSchema,
)


class TestDjangoCoreClientTokenAndHeaders(unittest.IsolatedAsyncioTestCase):
    """Test suite verifying user token forwarding, fallback token, and header injection."""

    async def test_user_token_forwarded_in_authorization_header(self):
        """Definition of Done: Request HTTP forwards user_token in Authorization: Bearer {user_token}."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True, "data": {"status": "ok"}}

        mock_async_client = MagicMock(spec=httpx.AsyncClient)
        mock_async_client.request = AsyncMock(return_value=mock_response)
        mock_async_client.is_closed = False

        client = DjangoCoreClient(
            base_url="http://test-erp:8000",
            token="system-fallback-token",
            service_key="test-internal-service-key",
            client=mock_async_client,
        )

        res = await client.execute_tool(
            domain="accounting",
            tool_name="get_account_balance",
            params={"account_id": 10},
            user_token="custom-user-jwt-999",
        )

        self.assertEqual(res, {"status": "ok"})
        mock_async_client.request.assert_called_once()
        call_kwargs = mock_async_client.request.call_args.kwargs

        headers = call_kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer custom-user-jwt-999")
        self.assertEqual(headers["X-Internal-Service-Key"], "test-internal-service-key")
        self.assertEqual(headers["X-AI-Source"], "LangGraph")
        self.assertEqual(headers["Content-Type"], "application/json")

    async def test_fallback_to_system_token_when_user_token_is_none(self):
        """When user_token is None, fallback to configured client token."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True, "data": {"result": 1}}

        mock_async_client = MagicMock(spec=httpx.AsyncClient)
        mock_async_client.request = AsyncMock(return_value=mock_response)
        mock_async_client.is_closed = False

        client = DjangoCoreClient(
            base_url="http://test-erp:8000",
            token="configured-system-token",
            client=mock_async_client,
        )

        res = await client.execute_tool(
            domain="accounting",
            tool_name="get_account_balance",
            params={},
            user_token=None,
        )

        self.assertEqual(res, {"result": 1})
        headers = mock_async_client.request.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer configured-system-token")

    async def test_auto_login_when_no_token_configured(self):
        """When neither user_token nor client.token is present, trigger async login()."""
        mock_async_client = MagicMock(spec=httpx.AsyncClient)
        mock_async_client.is_closed = False

        # First call is login (POST), second is execute_tool (POST)
        login_resp = MagicMock(spec=httpx.Response)
        login_resp.status_code = 200
        login_resp.json.return_value = {"access": "newly-acquired-jwt"}

        exec_resp = MagicMock(spec=httpx.Response)
        exec_resp.status_code = 200
        exec_resp.json.return_value = {"success": True, "data": {"logged_in": True}}

        mock_async_client.post = AsyncMock(return_value=login_resp)
        mock_async_client.request = AsyncMock(return_value=exec_resp)

        client = DjangoCoreClient(
            base_url="http://test-erp:8000",
            token=None,
            username="admin",
            password="secretpassword",
            client=mock_async_client,
        )

        with patch.object(settings, "ERP_LOGIN_SECRET", "test-login-secret"):
            res = await client.execute_tool(domain="inventory", tool_name="get_stock_level")
        self.assertEqual(res, {"logged_in": True})
        self.assertEqual(client.token, "newly-acquired-jwt")
        headers = mock_async_client.request.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer newly-acquired-jwt")

    async def test_login_retry_on_transient_500_success(self):
        """When login experiences transient 500, it retries once and acquires token."""
        mock_async_client = MagicMock(spec=httpx.AsyncClient)
        mock_async_client.is_closed = False

        err_resp = MagicMock(spec=httpx.Response)
        err_resp.status_code = 500
        err_resp.text = "Internal Server Error"

        ok_resp = MagicMock(spec=httpx.Response)
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"access": "token-after-retry"}

        mock_async_client.post = AsyncMock(side_effect=[err_resp, ok_resp])

        client = DjangoCoreClient(
            base_url="http://test-erp:8000",
            username="admin",
            password="secretpassword",
            client=mock_async_client,
        )

        with patch.object(settings, "ERP_LOGIN_SECRET", "test-login-secret"):
            token = await client.login()
        self.assertEqual(token, "token-after-retry")
        self.assertEqual(mock_async_client.post.call_count, 2)

    async def test_login_html_error_sanitization(self):
        """When login receives an HTML debug page, it extracts clean title text."""
        mock_async_client = MagicMock(spec=httpx.AsyncClient)
        mock_async_client.is_closed = False

        html_resp = MagicMock(spec=httpx.Response)
        html_resp.status_code = 500
        html_resp.text = "<!DOCTYPE html><html><head><title>ConnectionError at /external_api/get_token/</title></head><body>Error details</body></html>"

        # Both attempts fail with the HTML error
        mock_async_client.post = AsyncMock(return_value=html_resp)

        client = DjangoCoreClient(
            base_url="http://test-erp:8000",
            username="admin",
            password="secretpassword",
            client=mock_async_client,
        )

        with patch.object(settings, "ERP_LOGIN_SECRET", "test-login-secret"), \
             self.assertRaises(DjangoCoreClientError) as ctx:
            await client.login()
        self.assertIn("ConnectionError at /external_api/get_token/", str(ctx.exception))
        self.assertNotIn("<!DOCTYPE html>", str(ctx.exception))


class TestStructuredErrorHandling(unittest.IsolatedAsyncioTestCase):
    """Test suite for HTTP status 401, 403, 404, 500 error mapping (DoD)."""

    def setUp(self):
        self.mock_client = MagicMock(spec=httpx.AsyncClient)
        self.mock_client.is_closed = False
        self.client = DjangoCoreClient(
            base_url="http://test-erp:8000",
            token="mock-token",
            client=self.mock_client,
        )

    async def test_401_raises_django_auth_error(self):
        """HTTP 401 must raise DjangoAuthError."""
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 401
        resp.text = "Token expired or signature invalid"
        self.mock_client.request = AsyncMock(return_value=resp)

        with self.assertRaises(DjangoAuthError) as ctx:
            await self.client.execute_tool(
                domain="accounting",
                tool_name="get_journal_entries",
                user_token="expired-token",
            )
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("Unauthorized (401)", str(ctx.exception))

    async def test_403_raises_django_permission_error(self):
        """HTTP 403 must raise DjangoPermissionError."""
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 403
        resp.text = "User does not have access to company 99"
        self.mock_client.request = AsyncMock(return_value=resp)

        with self.assertRaises(DjangoPermissionError) as ctx:
            await self.client.execute_tool(
                domain="accounting",
                tool_name="get_journal_entries",
                company_id=99,
            )
        self.assertEqual(ctx.exception.status_code, 403)
        self.assertIn("Forbidden (403)", str(ctx.exception))

    async def test_404_raises_django_not_found_error(self):
        """HTTP 404 must raise DjangoNotFoundError."""
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 404
        resp.text = "Tool or resource not found"
        self.mock_client.request = AsyncMock(return_value=resp)

        with self.assertRaises(DjangoNotFoundError) as ctx:
            await self.client.execute_tool(domain="unknown_domain", tool_name="unknown_tool")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertIn("Not Found (404)", str(ctx.exception))

    async def test_500_raises_django_server_error(self):
        """HTTP 500 must raise DjangoServerError."""
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 500
        resp.text = "Database connection timed out in Django"
        self.mock_client.request = AsyncMock(return_value=resp)

        with self.assertRaises(DjangoServerError) as ctx:
            await self.client.execute_tool(domain="inventory", tool_name="get_product_info")
        self.assertEqual(ctx.exception.status_code, 500)
        self.assertIn("Internal Server Error (500)", str(ctx.exception))

    async def test_502_bad_gateway_raises_django_server_error(self):
        """HTTP 502/503 must also raise DjangoServerError."""
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 502
        resp.text = "Bad Gateway"
        self.mock_client.request = AsyncMock(return_value=resp)

        with self.assertRaises(DjangoServerError) as ctx:
            await self.client.execute_tool(domain="inventory", tool_name="get_product_info")
        self.assertEqual(ctx.exception.status_code, 502)

    async def test_erp_tool_envelope_failure_raises_django_core_client_error(self):
        """When HTTP status is 200 but envelope has success=false, raise DjangoCoreClientError."""
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.return_value = {
            "success": False,
            "data": None,
            "error": {"code": "INVALID_PARAMS", "message": "Account ID does not exist"},
        }
        self.mock_client.request = AsyncMock(return_value=resp)

        with self.assertRaises(DjangoCoreClientError) as ctx:
            await self.client.execute_tool(domain="accounting", tool_name="get_account_balance")
        self.assertIn("Django Core Tool Error", str(ctx.exception))

    async def test_network_connection_error_raises_django_core_client_error(self):
        """Network drop or timeout raises DjangoCoreClientError."""
        self.mock_client.request = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

        with self.assertRaises(DjangoCoreClientError) as ctx:
            await self.client.execute_tool(domain="accounting", tool_name="get_account_balance")
        self.assertIn("Network error", str(ctx.exception))

    async def test_http_redirect_to_login_raises_django_auth_error(self):
        """HTTP 302 redirect to login/temp_url raises DjangoAuthError with clear message."""
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 302
        resp.is_redirect = True
        resp.headers = {"location": "/?temp_url=/ai/tools/inventory/get_stock_level/"}
        resp.text = ""
        self.mock_client.request = AsyncMock(return_value=resp)

        from app.services.django_client import DjangoAuthError
        with self.assertRaises(DjangoAuthError) as ctx:
            await self.client.execute_tool(domain="inventory", tool_name="get_stock_level")
        self.assertIn("halaman login web", str(ctx.exception))

    async def test_empty_response_body_raises_django_core_client_error(self):
        """Empty HTTP 200 response body raises informative DjangoCoreClientError."""
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.is_redirect = False
        resp.headers = {}
        resp.text = ""
        self.mock_client.request = AsyncMock(return_value=resp)

        with self.assertRaises(DjangoCoreClientError) as ctx:
            await self.client.execute_tool(domain="inventory", tool_name="get_stock_level")
        self.assertIn("empty response", str(ctx.exception))


class TestToolExecutionAndHelperMethods(unittest.IsolatedAsyncioTestCase):
    """Test suite verifying helper methods and schema parameter passing."""

    def setUp(self):
        self.mock_client = MagicMock(spec=httpx.AsyncClient)
        self.mock_client.is_closed = False
        self.client = DjangoCoreClient(
            base_url="http://test-erp:8000",
            token="mock-token",
            client=self.mock_client,
        )

    async def test_fetch_account_balance_with_pydantic_schema(self):
        """fetch_account_balance passes cleaned Pydantic schema parameters."""
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.return_value = {
            "success": True,
            "data": {"net_balance": 500000.0, "total_debit": 1000000.0, "total_credit": 500000.0},
        }
        self.mock_client.request = AsyncMock(return_value=resp)

        schema = AccountBalanceFilterSchema(
            account_code="111100",
            date_from="2026-01-01",
            date_to="2026-06-30",
        )
        res = await self.client.fetch_account_balance(params=schema, user_token="token-abc")

        self.assertEqual(res["net_balance"], 500000.0)
        call_kwargs = self.mock_client.request.call_args.kwargs
        self.assertEqual(call_kwargs["url"], "http://test-erp:8000/ai/tools/accounting/get_account_balance/")
        self.assertEqual(
            call_kwargs["json"]["parameters"],
            {"account_code": "111100", "date_from": "2026-01-01", "date_to": "2026-06-30"},
        )

    async def test_fetch_journal_entries_with_pydantic_schema(self):
        """fetch_journal_entries passes pagination and filter parameters."""
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.return_value = {
            "success": True,
            "data": {"total_count": 1, "entries": [{"id": 1, "number": "JE/001"}]},
        }
        self.mock_client.request = AsyncMock(return_value=resp)

        schema = JournalEntriesFilterSchema(status="posted", limit=25)
        res = await self.client.fetch_journal_entries(params=schema)

        self.assertEqual(res["total_count"], 1)
        call_kwargs = self.mock_client.request.call_args.kwargs
        self.assertEqual(call_kwargs["url"], "http://test-erp:8000/ai/tools/accounting/get_journal_entries/")
        self.assertEqual(call_kwargs["json"]["parameters"]["status"], "posted")
        self.assertEqual(call_kwargs["json"]["parameters"]["limit"], 25)

    async def test_fetch_stock_level_with_pydantic_schema(self):
        """fetch_stock_level correctly calls inventory.get_stock_level."""
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.return_value = {
            "success": True,
            "data": {"product_id": 101, "qty_on_hand": 150.0},
        }
        self.mock_client.request = AsyncMock(return_value=resp)

        schema = StockLevelFilterSchema(code="SKU-TEST-99")
        res = await self.client.fetch_stock_level(params=schema)

        self.assertEqual(res["qty_on_hand"], 150.0)
        call_kwargs = self.mock_client.request.call_args.kwargs
        self.assertEqual(call_kwargs["url"], "http://test-erp:8000/ai/tools/inventory/get_stock_level/")
        self.assertEqual(call_kwargs["json"]["parameters"], {"code": "SKU-TEST-99", "limit": 20, "offset": 0})

    async def test_fetch_product_info_with_pydantic_schema(self):
        """fetch_product_info correctly calls inventory.get_product_info."""
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.return_value = {
            "success": True,
            "data": {"total_found": 1, "products": [{"id": 1, "name": "Ink Cyan"}]},
        }
        self.mock_client.request = AsyncMock(return_value=resp)

        schema = ProductInfoFilterSchema(search="Cyan", limit=5)
        res = await self.client.fetch_product_info(params=schema)

        self.assertEqual(res["total_found"], 1)
        call_kwargs = self.mock_client.request.call_args.kwargs
        self.assertEqual(call_kwargs["url"], "http://test-erp:8000/ai/tools/inventory/get_product_info/")
        self.assertEqual(call_kwargs["json"]["parameters"], {"search": "Cyan", "limit": 5, "include_variants": True})

    async def test_fetch_vendor_invoices_delegation(self):
        """fetch_vendor_invoices compatibility alias delegates to fetch_journal_entries."""
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.return_value = {
            "success": True,
            "data": {"total_count": 2, "entries": [{"id": 1, "number": "INV/01"}]},
        }
        self.mock_client.request = AsyncMock(return_value=resp)

        res = await self.client.fetch_vendor_invoices(
            params={"status": "posted", "search": "FastPrint Supplier"},
            user_token="invoice-user-token",
        )

        self.assertEqual(res["total_count"], 2)
        call_kwargs = self.mock_client.request.call_args.kwargs
        self.assertEqual(call_kwargs["url"], "http://test-erp:8000/ai/tools/accounting/get_journal_entries/")
        self.assertEqual(call_kwargs["headers"]["Authorization"], "Bearer invoice-user-token")

    async def test_discover_tools(self):
        """discover_tools retrieves tool catalog via GET /ai/tools/."""
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.return_value = {
            "success": True,
            "tools": [{"name": "get_stock_level", "domain": "inventory"}],
        }
        self.mock_client.request = AsyncMock(return_value=resp)

        tools = await self.client.discover_tools()
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["name"], "get_stock_level")


class TestContextManagerAndAliases(unittest.IsolatedAsyncioTestCase):
    """Test suite for async context manager and class aliases."""

    async def test_async_context_manager(self):
        """DjangoCoreClient can be used with async with and closes client on exit."""
        mock_async_client = MagicMock(spec=httpx.AsyncClient)
        mock_async_client.is_closed = False
        mock_async_client.aclose = AsyncMock()

        async with DjangoCoreClient(
            base_url="http://test-erp:8000",
            token="test-token",
            client=mock_async_client,
        ) as client:
            self.assertIsInstance(client, DjangoCoreClient)

    def test_async_erp_client_alias(self):
        """AsyncERPClient is a direct alias to DjangoCoreClient."""
        self.assertIs(AsyncERPClient, DjangoCoreClient)


if __name__ == "__main__":
    unittest.main()
