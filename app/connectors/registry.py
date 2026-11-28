"""
Central Connector Registry for LangGraph Multi-System AI Orchestration.
Dispatches tool requests to the appropriate backend connector (FastPrint, Odoo, Talenta, Generic REST).
"""
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from app.connectors.base import BaseSystemConnector
from app.connectors.fastprint import FastPrintConnector
from app.connectors.odoo import OdooConnector
from app.connectors.talenta import TalentaConnector
from app.connectors.generic_rest import GenericRestConnector
from app.core.config import settings

logger = logging.getLogger("connector_registry")


class ConnectorRegistry:
    """
    Registry for managing and resolving external system connectors.
    Acts as the single point of entry for tool execution across multiple enterprise platforms.
    """

    def __init__(self) -> None:
        self._connectors: Dict[str, BaseSystemConnector] = {}
        self._tool_connector_map: Dict[str, str] = {}
        self._domain_connector_map: Dict[str, str] = {}
        self._default_connector_name = "fastprint"
        self._initialized = False

    def initialize_default_connectors(self) -> None:
        """Initialize standard connectors based on environment configuration."""
        if self._initialized:
            return

        # 1. FastPrint Django ERP (default)
        fastprint = FastPrintConnector()
        self.register(fastprint, is_default=True)
        for t in fastprint.get_supported_tools():
            self._tool_connector_map[t] = fastprint.name
        for d in ("accounting", "inventory", "sales", "marketplace", "knowledge"):
            self._domain_connector_map[d] = fastprint.name

        # 2. Odoo ERP (if configured or loaded as plugin)
        odoo = OdooConnector()
        self.register(odoo)
        for t in odoo.get_supported_tools():
            self._tool_connector_map[t] = odoo.name
        for d in ("retail_inventory", "pos"):
            self._domain_connector_map[d] = odoo.name

        # 3. Mekari Talenta HRIS (if configured or loaded as plugin)
        talenta = TalentaConnector()
        self.register(talenta)
        for t in talenta.get_supported_tools():
            self._tool_connector_map[t] = talenta.name
        self._domain_connector_map["hris"] = talenta.name

        # 4. External config JSON (if present)
        cfg_path = settings.CONNECTORS_CONFIG_PATH
        if cfg_path and os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
                for conn_info in cfg_data.get("connectors", []):
                    c_name = conn_info.get("name")
                    c_type = conn_info.get("type")
                    if c_type == "generic_rest":
                        gr = GenericRestConnector(
                            name=c_name,
                            base_url=conn_info.get("base_url", ""),
                            tool_endpoints=conn_info.get("tool_endpoints", {}),
                            headers=conn_info.get("headers", {}),
                        )
                        self.register(gr)
                        for t in gr.get_supported_tools():
                            self._tool_connector_map[t] = gr.name
                logger.info(f"Loaded dynamic connectors from {cfg_path}")
            except Exception as exc:
                logger.warning(f"Failed to load dynamic connectors from {cfg_path}: {exc}")

        self._initialized = True

    def register(self, connector: BaseSystemConnector, is_default: bool = False) -> None:
        """Register a new connector instance."""
        self._connectors[connector.name] = connector
        for tool_name in connector.get_supported_tools():
            self._tool_connector_map[tool_name] = connector.name
        if is_default:
            self._default_connector_name = connector.name
        logger.info(f"Registered connector '{connector.name}' (type: {connector.connector_type})")

    def get_connector(self, name: str) -> Optional[BaseSystemConnector]:
        self.initialize_default_connectors()
        return self._connectors.get(name)

    def resolve_connector_for_tool(
        self,
        tool_name: str,
        domain: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> BaseSystemConnector:
        """
        Determine which connector owns the given tool.
        Priority:
        1. Explicit system/connector hint in context ('target_connector' or 'system')
        2. Exact tool_name mapping
        3. Domain-level connector mapping
        4. Default connector (FastPrint)
        """
        self.initialize_default_connectors()
        ctx = context or {}

        # 1. Context override
        target_name = ctx.get("target_connector") or ctx.get("system")
        if target_name and target_name in self._connectors:
            return self._connectors[target_name]

        # 2. Tool mapping
        if tool_name in self._tool_connector_map:
            c_name = self._tool_connector_map[tool_name]
            if c_name in self._connectors:
                return self._connectors[c_name]

        # 3. Domain mapping
        if domain and domain in self._domain_connector_map:
            c_name = self._domain_connector_map[domain]
            if c_name in self._connectors:
                return self._connectors[c_name]

        # 4. Default fallback
        return self._connectors.get(self._default_connector_name) or list(self._connectors.values())[0]

    async def execute_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
        domain: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Optional[str]]:
        """
        Execute a tool by dynamically routing to the appropriate connector.
        Returns:
            Tuple of (result_data, error_string)
        """
        connector = self.resolve_connector_for_tool(tool_name, domain=domain, context=context)
        try:
            res = await connector.execute_tool(tool_name, params=params, context=context)
            return res, None
        except Exception as exc:
            logger.warning(f"Error executing '{tool_name}' via connector '{connector.name}': {exc}")
            return None, str(exc)

    async def discover_all_tools(self) -> List[Dict[str, Any]]:
        """Collect tool schemas across all registered connectors."""
        self.initialize_default_connectors()
        all_tools: List[Dict[str, Any]] = []
        for c in self._connectors.values():
            try:
                tools = await c.discover_tools()
                for t in tools:
                    t["connector"] = c.name
                    all_tools.append(t)
            except Exception as exc:
                logger.warning(f"Failed to discover tools for connector '{c.name}': {exc}")
        return all_tools

    async def aclose_all(self) -> None:
        """Clean up all connector resources."""
        for c in self._connectors.values():
            try:
                await c.aclose()
            except Exception as exc:
                logger.warning(f"Error closing connector '{c.name}': {exc}")


_GLOBAL_CONNECTOR_REGISTRY: Optional[ConnectorRegistry] = None


def get_connector_registry() -> ConnectorRegistry:
    """Singleton getter for ConnectorRegistry."""
    global _GLOBAL_CONNECTOR_REGISTRY
    if _GLOBAL_CONNECTOR_REGISTRY is None:
        _GLOBAL_CONNECTOR_REGISTRY = ConnectorRegistry()
        _GLOBAL_CONNECTOR_REGISTRY.initialize_default_connectors()
    return _GLOBAL_CONNECTOR_REGISTRY
