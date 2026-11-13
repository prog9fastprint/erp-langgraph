"""
HTTP REST Client for FastPrint ERP AI Tools Layer using HTTPX.
Handles authentication, token refresh, tool discovery, and tool execution.
"""
import hashlib
import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
import httpx

from app.core.config import settings
from app.core.context import get_current_erp_datetime

logger = logging.getLogger("erp_client")


class ERPClientError(Exception):
    """Base exception for ERP client errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, details: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details


class ERPClient:
    """
    Synchronous HTTP Client interacting with FastPrint ERP /ai/tools/ API via HTTPX.
    Companion to asynchronous DjangoCoreClient (used in StateGraph pipeline nodes).
    Retained for synchronous scripts, CLI tools, and unit tests.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        default_company_id: Optional[int] = None,
        timeout: Optional[int] = None,
    ):
        self.base_url = (base_url or settings.ERP_BASE_URL).rstrip("/")
        self.token = token or settings.ERP_API_TOKEN
        self.username = username or settings.ERP_USERNAME
        self.password = password or settings.ERP_PASSWORD
        self.default_company_id = default_company_id or settings.DEFAULT_COMPANY_ID
        self.timeout = timeout if timeout is not None else settings.REQUEST_TIMEOUT_SEC
        self.client = httpx.Client(timeout=self.timeout)

    def _generate_secret_key(self) -> str:
        """Generate time-based MD5 secret key for /external_api/get_token/."""
        now = get_current_erp_datetime().strftime("%Y-%m-%d-%H:%M")
        raw_string = f"erp_master_nexus-{now}"
        return hashlib.md5(raw_string.encode()).hexdigest()

    def login(self) -> str:
        """
        Authenticate against ERP external_api/get_token/ to obtain JWT access token.
        """
        if not self.username or not self.password:
            raise ERPClientError("ERP credentials missing. Provide ERP_API_TOKEN or ERP_USERNAME and ERP_PASSWORD.")

        url = f"{self.base_url}/external_api/get_token/"
        headers = {
            "key": self._generate_secret_key(),
            "Content-Type": "application/json",
        }
        payload = {
            "username": self.username,
            "password": self.password,
        }

        for attempt in range(2):
            try:
                resp = self.client.post(url, json=payload, headers=headers)
            except httpx.RequestError as exc:
                if attempt == 0:
                    time.sleep(0.5)
                    continue
                raise ERPClientError(f"Network error during login to ERP: {exc}")

            if resp.status_code >= 500 and attempt == 0:
                logger.warning(f"Transient HTTP {resp.status_code} during ERP login, retrying once...")
                time.sleep(0.5)
                continue
            elif resp.status_code != 200:
                error_text = resp.text or ""
                if "<html" in error_text.lower():
                    title_match = re.search(r"<title>(.*?)</title>", error_text, re.IGNORECASE | re.DOTALL)
                    clean_msg = title_match.group(1).strip() if title_match else f"HTTP {resp.status_code} Error"
                    clean_msg = " ".join(clean_msg.split())
                else:
                    clean_msg = error_text[:300]
                raise ERPClientError(
                    f"Failed to authenticate with ERP (status {resp.status_code}): {clean_msg}",
                    status_code=resp.status_code,
                )

            data = resp.json()
            token = data.get("access")
            if not token:
                raise ERPClientError(f"No access token returned from ERP login response: {data}")

            self.token = token
            logger.info("Successfully acquired new JWT token from ERP.")
            return token

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        retries: int = 1,
    ) -> Dict[str, Any]:
        """
        Internal request helper with automatic 401 token refresh and ERP envelope validation.
        """
        if not self.token:
            self.login()

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        req_headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "X-AI-Source": "AI_TOOL",
        }
        if headers:
            req_headers.update(headers)

        try:
            resp = self.client.request(
                method=method,
                url=url,
                params=params,
                json=json,
                headers=req_headers,
            )
        except httpx.RequestError as exc:
            raise ERPClientError(f"Network error calling ERP endpoint {endpoint}: {exc}")

        # Handle token expiration (HTTP 401)
        if resp.status_code == 401 and retries > 0:
            logger.warning("JWT token expired or invalid (401). Attempting re-login...")
            self.login()
            return self._request(
                method=method,
                endpoint=endpoint,
                params=params,
                json=json,
                headers=headers,
                retries=retries - 1,
            )

        if resp.status_code >= 400:
            raise ERPClientError(
                f"ERP request failed with status {resp.status_code}: {resp.text}",
                status_code=resp.status_code,
            )

        try:
            res_data = resp.json()
        except Exception:
            raise ERPClientError(f"Invalid JSON returned from ERP: {resp.text}")

        # Standard ERP envelope checks: {"success": true, "data": ...}
        if isinstance(res_data, dict) and "success" in res_data:
            if not res_data.get("success", False):
                err_msg = res_data.get("error", "Unknown error from ERP tools API.")
                raise ERPClientError(f"ERP Tool Error: {err_msg}", details=res_data)
            return res_data.get("data", res_data)

        return res_data

    def discover_tools(self) -> List[Dict[str, Any]]:
        """
        Call GET /ai/tools/ to fetch dynamic schema registry from ERP.
        """
        data = self._request("GET", "/ai/tools/")
        if isinstance(data, dict) and "tools" in data:
            return data["tools"]
        elif isinstance(data, list):
            return data
        return []

    # Alias for backward compatibility
    list_tools = discover_tools

    def execute_tool(
        self,
        domain: str,
        tool_name: str,
        params: Optional[Dict[str, Any]] = None,
        company_id: Optional[int] = None,
        confirm: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute tool via POST /ai/tools/<domain>/<tool_name>/
        """
        endpoint = f"/ai/tools/{domain}/{tool_name}/"
        payload = {
            "company_id": company_id if company_id is not None else self.default_company_id,
            "params": params or {},
            "parameters": params or {},
        }
        if confirm:
            payload["confirm"] = True
        return self._request("POST", endpoint, json=payload)
