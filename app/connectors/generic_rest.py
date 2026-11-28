"""
Generic REST API Connector for LangGraph AI Orchestrator.
Allows arbitrary external microservices or SaaS systems to be plugged in as tool providers via configuration.
"""
from typing import Any, Dict, List, Optional
import logging
import httpx

from app.connectors.base import BaseSystemConnector
from app.core.config import settings

logger = logging.getLogger("generic_rest_connector")


class GenericRestConnector(BaseSystemConnector):
    """
    Plug-and-play Generic REST Connector.
    Reads tool endpoints from configuration dict or JSON file.
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        tool_endpoints: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            name=name,
            connector_type="generic_rest",
            description=f"Generic REST Connector for {name}",
            config=config,
        )
        self.base_url = base_url.rstrip("/")
        self.tool_endpoints = tool_endpoints or {}
        self.default_headers = headers or {}
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT_SEC)
        return self._client

    async def execute_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        endpoint = self.tool_endpoints.get(tool_name)
        if not endpoint:
            # Fallback convention: /api/tools/<tool_name>
            endpoint = f"/api/tools/{tool_name}"

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        client = await self._get_client()

        req_headers = dict(self.default_headers)
        ctx = context or {}
        if ctx.get("user_token"):
            req_headers["Authorization"] = f"Bearer {ctx['user_token']}"
        if ctx.get("headers"):
            req_headers.update(ctx["headers"])

        resp = await client.post(url, json=params, headers=req_headers)
        resp.raise_for_status()
        return resp.json()

    async def discover_tools(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        return [
            {"tool_name": t, "domain": self.name, "description": f"Tool {t} on {self.name}"}
            for t in self.tool_endpoints.keys()
        ]

    def get_supported_tools(self) -> List[str]:
        return list(self.tool_endpoints.keys())

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
