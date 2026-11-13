"""
Asynchronous HTTP Client for FastPrint ERP Django Core (AI Tools Layer).
Provides non-blocking async execution using httpx.AsyncClient with user token forwarding,
structured error handling, and support for FastPrint ERP tools.
"""
import asyncio
import hashlib
import logging
import re
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import weakref
import httpx

from app.core.config import settings
from app.core.context import get_current_erp_datetime
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
)

logger = logging.getLogger("django_client")


# ==============================================================================
# Structured Exception Hierarchy
# ==============================================================================

class DjangoCoreClientError(Exception):
    """Base exception for all Django Core HTTP client errors."""
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        details: Any = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(status_code={self.status_code}, message='{self.message}')"


class DjangoAuthError(DjangoCoreClientError):
    """Raised when authentication fails or token is expired/invalid (HTTP 401)."""
    def __init__(self, message: str = "Authentication failed or token expired.", status_code: int = 401, details: Any = None):
        super().__init__(message=message, status_code=status_code, details=details)


class DjangoPermissionError(DjangoCoreClientError):
    """Raised when user lacks permission to access tool, company, or resource (HTTP 403)."""
    def __init__(self, message: str = "Permission denied for requested resource or company.", status_code: int = 403, details: Any = None):
        super().__init__(message=message, status_code=status_code, details=details)


class DjangoNotFoundError(DjangoCoreClientError):
    """Raised when the requested endpoint, tool, or resource does not exist (HTTP 404)."""
    def __init__(self, message: str = "Requested endpoint, tool, or resource not found.", status_code: int = 404, details: Any = None):
        super().__init__(message=message, status_code=status_code, details=details)


class DjangoServerError(DjangoCoreClientError):
    """Raised when Django Core backend encounters an internal server error (HTTP 500..599)."""
    def __init__(self, message: str = "Django Core internal server error.", status_code: int = 500, details: Any = None):
        super().__init__(message=message, status_code=status_code, details=details)


# ==============================================================================
# Asynchronous Django Core Client
# ==============================================================================

def _clean_params(params: Any) -> Dict[str, Any]:
    """Helper to convert Pydantic schema or dictionary into cleaned dict without None."""
    if params is None:
        return {}
    if hasattr(params, "model_dump"):
        raw = params.model_dump()
    elif isinstance(params, dict):
        raw = params
    else:
        raw = dict(params)
    return {k: v for k, v in raw.items() if v is not None}


_SHARED_CLIENT_LOCK = threading.Lock()
_SHARED_HTTPX_CLIENTS: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient]" = (
    weakref.WeakKeyDictionary()
)

_CACHED_SYSTEM_TOKEN: Optional[str] = None
_TOKEN_LOCK: Optional[asyncio.Lock] = None
_TOKEN_LOCK_LOOP: Optional[asyncio.AbstractEventLoop] = None


def _get_token_lock() -> asyncio.Lock:
    """Lazily initialize module-level token lock bound to the active event loop."""
    global _TOKEN_LOCK, _TOKEN_LOCK_LOOP
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if _TOKEN_LOCK is None or (running_loop is not None and _TOKEN_LOCK_LOOP is not running_loop):
        _TOKEN_LOCK = asyncio.Lock()
        _TOKEN_LOCK_LOOP = running_loop
    return _TOKEN_LOCK


async def _invalidate_cached_system_token(expected_token: Optional[str]) -> None:
    """Clear module-level cached system token only if it still matches expected_token."""
    global _CACHED_SYSTEM_TOKEN
    async with _get_token_lock():
        if _CACHED_SYSTEM_TOKEN == expected_token:
            _CACHED_SYSTEM_TOKEN = None


def _get_shared_httpx_client(timeout: Optional[float] = None) -> Tuple[httpx.AsyncClient, bool]:
    """
    Acquire or create a module-level shared httpx.AsyncClient bound to the running event loop.
    Falls back to a standalone unshared client when called without a running event loop.
    """
    effective_timeout = timeout if timeout is not None else settings.REQUEST_TIMEOUT_SEC
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        return httpx.AsyncClient(timeout=effective_timeout), False

    with _SHARED_CLIENT_LOCK:
        for loop_key in list(_SHARED_HTTPX_CLIENTS.keys()):
            if loop_key.is_closed():
                _SHARED_HTTPX_CLIENTS.pop(loop_key, None)

        shared_client = _SHARED_HTTPX_CLIENTS.get(running_loop)
        if shared_client is None or shared_client.is_closed:
            shared_client = httpx.AsyncClient(timeout=effective_timeout)
            _SHARED_HTTPX_CLIENTS[running_loop] = shared_client
        return shared_client, True


async def aclose_shared_django_client() -> None:
    """
    Close shared module-level httpx.AsyncClient pool(s) for the current event loop
    and clear the cached system JWT token. Called during FastAPI lifespan shutdown.
    """
    global _CACHED_SYSTEM_TOKEN
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    clients_to_close: List[httpx.AsyncClient] = []
    with _SHARED_CLIENT_LOCK:
        for loop_key in list(_SHARED_HTTPX_CLIENTS.keys()):
            if loop_key.is_closed():
                _SHARED_HTTPX_CLIENTS.pop(loop_key, None)
            elif running_loop is None or loop_key is running_loop:
                shared_client = _SHARED_HTTPX_CLIENTS.pop(loop_key, None)
                if shared_client is not None:
                    clients_to_close.append(shared_client)

    for client in clients_to_close:
        try:
            if not client.is_closed:
                await client.aclose()
        except Exception as exc:
            logger.warning("Error closing shared httpx.AsyncClient pool: %s", exc)

    _CACHED_SYSTEM_TOKEN = None


class DjangoCoreClient:
    """
    Asynchronous client for interacting with FastPrint ERP Django Core API via HTTPX.
    Supports user token forwarding, service keys, and structured error handling.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        default_company_id: Optional[int] = None,
        timeout: Optional[int] = None,
        service_key: Optional[str] = None,
        client: Optional[httpx.AsyncClient] = None,
        require_user_token: bool = False,
    ):
        self.base_url = (base_url or settings.ERP_BASE_URL).rstrip("/")
        self._external_client = client is not None
        explicit_token = (token or settings.ERP_API_TOKEN or "").strip() or None
        if explicit_token is None and not self._external_client and _CACHED_SYSTEM_TOKEN:
            explicit_token = _CACHED_SYSTEM_TOKEN
        self.token = explicit_token
        self.username = (username or settings.ERP_USERNAME or "").strip() or None
        self.password = (password or settings.ERP_PASSWORD or "").strip() or None
        self.default_company_id = default_company_id or settings.DEFAULT_COMPANY_ID
        self.timeout = timeout if timeout is not None else settings.REQUEST_TIMEOUT_SEC
        self.service_key = service_key or settings.ERP_API_TOKEN
        self._shared_client_ref: Optional[httpx.AsyncClient] = None
        if client is not None:
            self.client = client
            self._uses_shared_client = False
        else:
            shared_client, uses_shared = _get_shared_httpx_client(timeout=self.timeout)
            self.client = shared_client
            self._uses_shared_client = uses_shared
            if uses_shared:
                self._shared_client_ref = shared_client
        self.require_user_token = bool(require_user_token)

    async def __aenter__(self) -> "DjangoCoreClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """
        Close underlying httpx.AsyncClient if owned exclusively by this instance.
        Shared module-level pooled clients stay open until aclose_shared_django_client().
        """
        if self._external_client:
            return
        if self._uses_shared_client and self.client is self._shared_client_ref:
            return
        if self.client is not None and not self.client.is_closed:
            await self.client.aclose()

    def _generate_secret_key(self) -> str:
        """Generate time-based MD5 secret key for /external_api/get_token/."""
        now = get_current_erp_datetime().strftime("%Y-%m-%d-%H:%M")
        raw_string = f"erp_master_nexus-{now}"
        return hashlib.md5(raw_string.encode()).hexdigest()

    async def _perform_login(self) -> str:
        """Execute POST /external_api/get_token/ to acquire a fresh JWT access token."""
        if not self.username or not self.password:
            raise DjangoAuthError(
                "ERP credentials missing. Provide valid user_token, ERP_API_TOKEN, or ERP_USERNAME and ERP_PASSWORD."
            )

        url = f"{self.base_url}/external_api/get_token/"
        headers = {
            "key": self._generate_secret_key(),
            "Content-Type": "application/json",
        }
        payload = {
            "username": self.username,
            "password": self.password,
        }

        post_kwargs: Dict[str, Any] = {"json": payload, "headers": headers}
        if self._uses_shared_client and self.client is self._shared_client_ref and self.timeout != settings.REQUEST_TIMEOUT_SEC:
            post_kwargs["timeout"] = self.timeout

        last_exc: Optional[Exception] = None
        for attempt in range(2):
            try:
                resp = await self.client.post(url, **post_kwargs)
            except httpx.RequestError as exc:
                last_exc = DjangoCoreClientError(f"Network error during login to ERP: {exc}")
                if attempt == 0:
                    await asyncio.sleep(0.5)
                    continue
                raise last_exc

            if resp.status_code in (401, 403):
                raise DjangoAuthError(
                    f"Authentication failed during ERP login (status {resp.status_code}): {resp.text}",
                    status_code=resp.status_code,
                    details=resp.text,
                )
            elif resp.status_code >= 500 and attempt == 0:
                logger.warning(f"Transient HTTP {resp.status_code} during ERP login, retrying once...")
                await asyncio.sleep(0.5)
                continue
            elif resp.status_code != 200:
                error_text = resp.text or ""
                if "<html" in error_text.lower():
                    title_match = re.search(r"<title>(.*?)</title>", error_text, re.IGNORECASE | re.DOTALL)
                    clean_msg = title_match.group(1).strip() if title_match else f"HTTP {resp.status_code} Error"
                    clean_msg = " ".join(clean_msg.split())
                else:
                    clean_msg = error_text[:300]
                raise DjangoCoreClientError(
                    f"Failed to authenticate with ERP (status {resp.status_code}): {clean_msg}",
                    status_code=resp.status_code,
                    details=resp.text,
                )

            data = resp.json()
            token = data.get("access")
            if not token:
                raise DjangoAuthError(f"No access token returned from ERP login response: {data}")

            self.token = token
            logger.info("Successfully acquired new JWT token from ERP via async login.")
            return token

    async def login(self) -> str:
        """
        Authenticate against ERP external_api/get_token/ to obtain JWT access token.
        Uses a lock-guarded module cache for managed internal clients so concurrent
        requests share one login call.
        """
        global _CACHED_SYSTEM_TOKEN
        if self._external_client:
            return await self._perform_login()

        async with _get_token_lock():
            if _CACHED_SYSTEM_TOKEN:
                self.token = _CACHED_SYSTEM_TOKEN
                return _CACHED_SYSTEM_TOKEN
            token = await self._perform_login()
            _CACHED_SYSTEM_TOKEN = token
            return token

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        user_token: Optional[str] = None,
        retries: int = 1,
    ) -> Dict[str, Any]:
        """
        Internal async request helper with user token forwarding, service key,
        structured status code error mapping, and envelope unwrapping.
        """
        clean_user_token = (user_token or "").strip() or None
        if self.require_user_token and not clean_user_token:
            raise DjangoAuthError(
                "Autentikasi pengguna (user_token) diperlukan.",
                status_code=401,
            )

        active_token = clean_user_token or self.token
        if not active_token:
            active_token = await self.login()

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        req_headers = {
            "Authorization": f"Bearer {active_token}",
            "Content-Type": "application/json",
            "X-AI-Source": "LangGraph",
        }
        if self.service_key:
            req_headers["X-Internal-Service-Key"] = self.service_key
        if headers:
            req_headers.update(headers)

        try:
            req_kwargs: Dict[str, Any] = {
                "method": method,
                "url": url,
                "params": params,
                "json": json,
                "headers": req_headers,
            }
            if self._uses_shared_client and self.client is self._shared_client_ref and self.timeout != settings.REQUEST_TIMEOUT_SEC:
                req_kwargs["timeout"] = self.timeout
            resp = await self.client.request(**req_kwargs)
        except httpx.RequestError as exc:
            raise DjangoCoreClientError(f"Network error calling Django Core endpoint {endpoint}: {exc}")

        # HTTP Status Error Handling (DoD: 401, 403, 404, 500)
        if 300 <= resp.status_code < 400:
            resp_headers = getattr(resp, "headers", None)
            redirect_url = resp_headers.get("location", "") if resp_headers is not None and hasattr(resp_headers, "get") else ""
            if "temp_url=" in redirect_url or "login" in redirect_url:
                raise DjangoAuthError(
                    f"Django Core mengalihkan request API ke halaman login web (HTTP {resp.status_code} Redirect to '{redirect_url}'). "
                    f"Pastikan token JWT valid dan endpoint '/ai/' diizinkan oleh middleware autentikasi ERP.",
                    status_code=resp.status_code,
                    details={"location": redirect_url, "text": resp.text},
                )
            raise DjangoCoreClientError(
                f"Unexpected HTTP {resp.status_code} redirect from Django Core to '{redirect_url}'.",
                status_code=resp.status_code,
                details={"location": redirect_url, "text": resp.text},
            )

        if resp.status_code == 401:
            if not clean_user_token and not self.require_user_token and retries > 0:
                logger.warning("System JWT token expired (401). Attempting re-login...")
                if not self._external_client:
                    await _invalidate_cached_system_token(active_token)
                await self.login()
                return await self._request(
                    method=method,
                    endpoint=endpoint,
                    params=params,
                    json=json,
                    headers=headers,
                    user_token=user_token,
                    retries=retries - 1,
                )
            raise DjangoAuthError(
                f"Unauthorized (401): {resp.text or 'Authentication failed or token expired.'}",
                status_code=401,
                details=resp.text,
            )

        if resp.status_code == 403:
            raise DjangoPermissionError(
                f"Forbidden (403): {resp.text or 'Access forbidden. User lacks permission for this tool or company.'}",
                status_code=403,
                details=resp.text,
            )

        if resp.status_code == 404:
            raise DjangoNotFoundError(
                f"Not Found (404): {resp.text or 'Requested tool, endpoint, or resource not found on Django Core.'}",
                status_code=404,
                details=resp.text,
            )

        if 500 <= resp.status_code < 600:
            error_text = (resp.text or "").strip()
            if "<html" in error_text.lower():
                title_match = re.search(r"<title>(.*?)</title>", error_text, re.IGNORECASE | re.DOTALL)
                clean_msg = title_match.group(1).strip() if title_match else f"HTTP {resp.status_code} Error"
                clean_msg = " ".join(clean_msg.split())
            else:
                clean_msg = error_text or "Internal server error occurred on Django Core backend."
            raise DjangoServerError(
                f"Internal Server Error ({resp.status_code}): {clean_msg}",
                status_code=resp.status_code,
                details=resp.text,
            )

        if resp.status_code >= 400:
            raise DjangoCoreClientError(
                f"Django Core request failed with status {resp.status_code}: {resp.text}",
                status_code=resp.status_code,
                details=resp.text,
            )

        raw_text = (resp.text or "").strip()
        if not raw_text:
            raise DjangoCoreClientError(
                f"Django Core returned an empty response (HTTP {resp.status_code}). Endpoint: {endpoint}",
                status_code=resp.status_code,
            )

        try:
            res_data = resp.json()
        except Exception:
            preview = raw_text[:200] + ("..." if len(raw_text) > 200 else "")
            raise DjangoCoreClientError(
                f"Invalid JSON returned from Django Core (HTTP {resp.status_code}): {preview}",
                status_code=resp.status_code,
                details=raw_text,
            )

        # Standard ERP envelope validation: {"success": true, "data": ...}
        if isinstance(res_data, dict) and "success" in res_data:
            if not res_data.get("success", False):
                err_msg = res_data.get("error", "Unknown error from Django Core tools API.")
                raise DjangoCoreClientError(f"Django Core Tool Error: {err_msg}", details=res_data)
            return res_data.get("data", res_data)

        return res_data

    async def discover_tools(self, user_token: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Call async GET /ai/tools/ to fetch dynamic schema registry from ERP.
        """
        data = await self._request("GET", "/ai/tools/", user_token=user_token)
        if isinstance(data, dict) and "tools" in data:
            return data["tools"]
        elif isinstance(data, list):
            return data
        return []

    # Backward compatibility alias
    list_tools = discover_tools

    async def execute_tool(
        self,
        domain: str,
        tool_name: str,
        params: Optional[Dict[str, Any]] = None,
        company_id: Optional[int] = None,
        user_token: Optional[str] = None,
        confirm: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute tool asynchronously via POST /ai/tools/<domain>/<tool_name>/
        """
        endpoint = f"/ai/tools/{domain}/{tool_name}/"
        cleaned_params = _clean_params(params)
        payload = {
            "company_id": company_id if company_id is not None else self.default_company_id,
            "params": cleaned_params,
            "parameters": cleaned_params,
        }
        if confirm:
            payload["confirm"] = True

        return await self._request("POST", endpoint, json=payload, user_token=user_token)

    # ==========================================================================
    # Helper Methods for 4 Native FastPrint ERP Tools
    # ==========================================================================

    async def fetch_account_balance(
        self,
        params: Union[AccountBalanceFilterSchema, Dict[str, Any]],
        user_token: Optional[str] = None,
        company_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Call accounting.get_account_balance tool."""
        return await self.execute_tool(
            domain="accounting",
            tool_name="get_account_balance",
            params=_clean_params(params),
            company_id=company_id,
            user_token=user_token,
        )

    async def fetch_journal_entries(
        self,
        params: Union[JournalEntriesFilterSchema, Dict[str, Any]],
        user_token: Optional[str] = None,
        company_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Call accounting.get_journal_entries tool."""
        return await self.execute_tool(
            domain="accounting",
            tool_name="get_journal_entries",
            params=_clean_params(params),
            company_id=company_id,
            user_token=user_token,
        )

    async def fetch_general_ledger(
        self,
        params: Union[GeneralLedgerFilterSchema, Dict[str, Any]],
        user_token: Optional[str] = None,
        company_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Call accounting.get_general_ledger tool."""
        return await self.execute_tool(
            domain="accounting",
            tool_name="get_general_ledger",
            params=_clean_params(params),
            company_id=company_id,
            user_token=user_token,
        )

    async def fetch_profit_and_loss(
        self,
        params: Union[ProfitAndLossFilterSchema, Dict[str, Any]],
        user_token: Optional[str] = None,
        company_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Call accounting.get_profit_and_loss tool."""
        return await self.execute_tool(
            domain="accounting",
            tool_name="get_profit_and_loss",
            params=_clean_params(params),
            company_id=company_id,
            user_token=user_token,
        )

    async def fetch_balance_sheet(
        self,
        params: Union[BalanceSheetFilterSchema, Dict[str, Any]],
        user_token: Optional[str] = None,
        company_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Call accounting.get_balance_sheet tool."""
        return await self.execute_tool(
            domain="accounting",
            tool_name="get_balance_sheet",
            params=_clean_params(params),
            company_id=company_id,
            user_token=user_token,
        )

    async def fetch_vendor_bills(
        self,
        params: Union[VendorBillsFilterSchema, Dict[str, Any]],
        user_token: Optional[str] = None,
        company_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Call accounting.get_vendor_bills tool."""
        return await self.execute_tool(
            domain="accounting",
            tool_name="get_vendor_bills",
            params=_clean_params(params),
            company_id=company_id,
            user_token=user_token,
        )

    async def fetch_bank_reconciliation_summary(
        self,
        params: Union[BankReconciliationFilterSchema, Dict[str, Any]],
        user_token: Optional[str] = None,
        company_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Call accounting.get_bank_reconciliation_summary tool."""
        return await self.execute_tool(
            domain="accounting",
            tool_name="get_bank_reconciliation_summary",
            params=_clean_params(params),
            company_id=company_id,
            user_token=user_token,
        )

    async def fetch_stock_level(
        self,
        params: Union[StockLevelFilterSchema, Dict[str, Any]],
        user_token: Optional[str] = None,
        company_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Call inventory.get_stock_level tool."""
        return await self.execute_tool(
            domain="inventory",
            tool_name="get_stock_level",
            params=_clean_params(params),
            company_id=company_id,
            user_token=user_token,
        )

    async def fetch_product_info(
        self,
        params: Union[ProductInfoFilterSchema, Dict[str, Any]],
        user_token: Optional[str] = None,
        company_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Call inventory.get_product_info tool."""
        return await self.execute_tool(
            domain="inventory",
            tool_name="get_product_info",
            params=_clean_params(params),
            company_id=company_id,
            user_token=user_token,
        )

    async def fetch_locations(
        self,
        params: Optional[Union[LocationsFilterSchema, Dict[str, Any]]] = None,
        user_token: Optional[str] = None,
        company_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Call inventory.get_locations tool."""
        return await self.execute_tool(
            domain="inventory",
            tool_name="get_locations",
            params=_clean_params(params),
            company_id=company_id,
            user_token=user_token,
        )

    async def fetch_stock_history(
        self,
        params: Optional[Union[StockHistoryFilterSchema, Dict[str, Any]]] = None,
        user_token: Optional[str] = None,
        company_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Call inventory.get_stock_history tool (Phase 2)."""
        return await self.execute_tool(
            domain="inventory",
            tool_name="get_stock_history",
            params=_clean_params(params),
            company_id=company_id,
            user_token=user_token,
        )

    async def fetch_transfers(
        self,
        params: Optional[Union[TransfersFilterSchema, Dict[str, Any]]] = None,
        user_token: Optional[str] = None,
        company_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Call inventory.get_transfers tool (Phase 2)."""
        return await self.execute_tool(
            domain="inventory",
            tool_name="get_transfers",
            params=_clean_params(params),
            company_id=company_id,
            user_token=user_token,
        )

    async def fetch_low_stock_alerts(
        self,
        params: Optional[Union[LowStockAlertsFilterSchema, Dict[str, Any]]] = None,
        user_token: Optional[str] = None,
        company_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Call inventory.get_low_stock_alerts tool (Phase 3)."""
        return await self.execute_tool(
            domain="inventory",
            tool_name="get_low_stock_alerts",
            params=_clean_params(params),
            company_id=company_id,
            user_token=user_token,
        )

    async def fetch_stock_adjustments(
        self,
        params: Optional[Union[StockAdjustmentsFilterSchema, Dict[str, Any]]] = None,
        user_token: Optional[str] = None,
        company_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Call inventory.get_stock_adjustments tool (Phase 3)."""
        return await self.execute_tool(
            domain="inventory",
            tool_name="get_stock_adjustments",
            params=_clean_params(params),
            company_id=company_id,
            user_token=user_token,
        )

    async def fetch_scrap_records(
        self,
        params: Optional[Union[ScrapRecordsFilterSchema, Dict[str, Any]]] = None,
        user_token: Optional[str] = None,
        company_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Call inventory.get_scrap_records tool (Phase 3)."""
        return await self.execute_tool(
            domain="inventory",
            tool_name="get_scrap_records",
            params=_clean_params(params),
            company_id=company_id,
            user_token=user_token,
        )

    async def fetch_warehouses(
        self,
        params: Optional[Union[WarehousesFilterSchema, Dict[str, Any]]] = None,
        user_token: Optional[str] = None,
        company_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Call inventory.get_warehouses tool (Phase 3)."""
        return await self.execute_tool(
            domain="inventory",
            tool_name="get_warehouses",
            params=_clean_params(params),
            company_id=company_id,
            user_token=user_token,
        )

    async def fetch_product_categories(
        self,
        params: Optional[Union[ProductCategoriesFilterSchema, Dict[str, Any]]] = None,
        user_token: Optional[str] = None,
        company_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Call inventory.get_product_categories tool (Phase 3)."""
        return await self.execute_tool(
            domain="inventory",
            tool_name="get_product_categories",
            params=_clean_params(params),
            company_id=company_id,
            user_token=user_token,
        )

    async def fetch_knowledge(
        self,
        params: Any,
        user_token: Optional[str] = None,
        company_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Fetch semantically relevant knowledge chunks (SOPs, product specs, policies)
        from PGVector via Django Core endpoint POST /ai/tools/knowledge/search/.
        """
        return await self.execute_tool(
            domain="knowledge",
            tool_name="search",
            params=_clean_params(params),
            company_id=company_id,
            user_token=user_token,
        )

    async def fetch_vendor_invoices(
        self,
        params: Any,
        user_token: Optional[str] = None,
        company_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compatibility alias matching TASK-3.2 specification.
        In FastPrint ERP, vendor bills & invoices are recorded in Journal Entries.
        Delegates directly to fetch_journal_entries.
        """
        return await self.fetch_journal_entries(
            params=params,
            user_token=user_token,
            company_id=company_id,
        )


# Class alias for seamless naming compatibility
AsyncERPClient = DjangoCoreClient
