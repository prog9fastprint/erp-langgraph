"""
Mekari Talenta HRIS Connector for LangGraph AI Orchestrator.
Supports Employee directory, Time Off / Leave balances, Attendance logs, and Payroll summaries
via Mekari Open API v2.
"""
from typing import Any, Dict, List, Optional
import logging
import httpx

from app.connectors.base import BaseSystemConnector
from app.core.config import settings

logger = logging.getLogger("talenta_connector")


class TalentaConnector(BaseSystemConnector):
    """
    Connector for Mekari Talenta HRIS via REST API.
    Handles Employee queries, Attendance, Time Off (Leave), and Payroll summaries.
    """

    TALENTA_TOOLS: List[str] = [
        "talenta_get_employee",
        "talenta_get_time_off_balance",
        "talenta_get_attendance",
        "talenta_get_payroll_summary",
    ]

    def __init__(
        self,
        name: str = "talenta",
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            name=name,
            connector_type="talenta_api",
            description="Mekari Talenta HRIS Open API Connector",
            config=config,
        )
        self.api_url = (api_url or settings.TALENTA_API_URL or "https://api.mekari.com/v2/talenta").rstrip("/")
        self.api_key = api_key or settings.TALENTA_API_KEY or ""
        self.client_id = client_id or settings.TALENTA_CLIENT_ID or ""
        self.client_secret = client_secret or settings.TALENTA_CLIENT_SECRET or ""
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT_SEC)
        return self._client

    def _get_headers(self, user_token: Optional[str] = None) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        token = user_token or self.api_key
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if self.client_id:
            headers["X-Client-Id"] = self.client_id
        return headers

    async def execute_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute HRIS tools against Talenta Mekari API.
        """
        client = await self._get_client()
        ctx = context or {}
        user_token = ctx.get("user_token")
        headers = self._get_headers(user_token)

        if tool_name == "talenta_get_employee":
            emp_id = params.get("employee_id")
            query = params.get("search") or params.get("name")
            endpoint = f"{self.api_url}/v2/employees/{emp_id}" if emp_id else f"{self.api_url}/v2/employees"
            req_params = {}
            if query:
                req_params["q"] = query
            if params.get("branch_id") or params.get("company_id"):
                req_params["branch_id"] = params.get("branch_id") or params.get("company_id")

            resp = await client.get(endpoint, params=req_params, headers=headers)
            return {"system": "talenta", "tool": tool_name, "data": resp.json()}

        elif tool_name == "talenta_get_time_off_balance":
            emp_id = params.get("employee_id")
            year = params.get("year")
            endpoint = f"{self.api_url}/v2/time-off/balances"
            req_params = {}
            if emp_id:
                req_params["employee_id"] = emp_id
            if year:
                req_params["year"] = year

            resp = await client.get(endpoint, params=req_params, headers=headers)
            return {"system": "talenta", "tool": tool_name, "data": resp.json()}

        elif tool_name == "talenta_get_attendance":
            endpoint = f"{self.api_url}/v2/attendance/reports"
            req_params = {
                "start_date": params.get("date_from") or params.get("start_date"),
                "end_date": params.get("date_to") or params.get("end_date"),
            }
            if params.get("employee_id"):
                req_params["employee_id"] = params["employee_id"]
            if params.get("branch_id"):
                req_params["branch_id"] = params["branch_id"]

            resp = await client.get(endpoint, params=req_params, headers=headers)
            return {"system": "talenta", "tool": tool_name, "data": resp.json()}

        elif tool_name == "talenta_get_payroll_summary":
            period = params.get("period") or params.get("month")
            endpoint = f"{self.api_url}/v2/payroll/summaries"
            req_params = {}
            if period:
                req_params["period"] = period
            if params.get("branch_id"):
                req_params["branch_id"] = params["branch_id"]

            resp = await client.get(endpoint, params=req_params, headers=headers)
            return {"system": "talenta", "tool": tool_name, "data": resp.json()}

        else:
            raise NotImplementedError(f"Tool '{tool_name}' is not supported by TalentaConnector.")

    async def discover_tools(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        return [
            {
                "tool_name": "talenta_get_employee",
                "domain": "hris",
                "description": "Cek data profil karyawan, jabatan, departemen, dan status kepegawaian di Talenta HRIS.",
            },
            {
                "tool_name": "talenta_get_time_off_balance",
                "domain": "hris",
                "description": "Cek sisa kuota cuti tahunan, cuti khusus, izin sakit, dan status pengajuan cuti di Talenta HRIS.",
            },
            {
                "tool_name": "talenta_get_attendance",
                "domain": "hris",
                "description": "Log presensi kehadiran karyawan, jam masuk/pulang, keterlambatan, dan lembur di Talenta HRIS.",
            },
            {
                "tool_name": "talenta_get_payroll_summary",
                "domain": "hris",
                "description": "Ringkasan payroll, komponen gaji, dan status slip gaji bulanan karyawan di Talenta HRIS.",
            },
        ]

    def get_supported_tools(self) -> List[str]:
        return list(self.TALENTA_TOOLS)

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
