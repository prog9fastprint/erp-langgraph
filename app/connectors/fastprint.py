"""
FastPrint Django Core Connector.
Adapts the legacy/existing DjangoCoreClient to the unified BaseSystemConnector interface.
"""
from typing import Any, Dict, List, Optional
import logging

from app.connectors.base import BaseSystemConnector
from app.schemas.tools import TOOL_DOMAIN_MAP
from app.services.django_client import DjangoCoreClient

logger = logging.getLogger("fastprint_connector")


class FastPrintConnector(BaseSystemConnector):
    """
    Connector for FastPrint ERP Django Core backend.
    Preserves 100% backward compatibility for all accounting, inventory, sales,
    and marketplace tools.
    """

    def __init__(
        self,
        name: str = "fastprint",
        client: Optional[DjangoCoreClient] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            name=name,
            connector_type="erp_django",
            description="FastPrint ERP Django Core Tools Connector",
            config=config,
        )
        self.client = client or DjangoCoreClient()
        self._owned_client = client is None

    async def execute_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute an ERP tool via DjangoCoreClient.
        """
        ctx = context or {}
        user_token = ctx.get("user_token")
        company_id = params.get("company_id") or ctx.get("company_id")
        confirm = bool(params.get("confirm") or ctx.get("confirm", False))

        domain = TOOL_DOMAIN_MAP.get(tool_name) or ctx.get("domain") or "accounting"

        return await self.client.execute_tool(
            domain=domain,
            tool_name=tool_name,
            params=params,
            company_id=company_id,
            user_token=user_token,
            confirm=confirm,
        )

    async def discover_tools(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch dynamically registered tools from Django Core /ai/tools/.
        """
        ctx = context or {}
        user_token = ctx.get("user_token")
        try:
            return await self.client.discover_tools(user_token=user_token)
        except Exception as exc:
            logger.warning(f"Could not discover tools from FastPrint ERP: {exc}")
            return []

    def get_supported_tools(self) -> List[str]:
        return list(TOOL_DOMAIN_MAP.keys())

    async def health_check(self) -> bool:
        try:
            tools = await self.discover_tools()
            return isinstance(tools, list)
        except Exception:
            return False

    async def aclose(self) -> None:
        if self._owned_client:
            await self.client.aclose()
