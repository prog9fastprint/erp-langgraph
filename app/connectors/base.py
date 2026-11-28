"""
Base System Connector Interface for LangGraph AI Orchestrator.
Defines the standard abstract contract that every backend ERP, HRIS, or CRM connector must implement.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseSystemConnector(ABC):
    """
    Abstract Base Class for Enterprise System Connectors.
    Decouples LangGraph agent reasoning from specific backend implementations
    (e.g., FastPrint Django, Odoo XML-RPC, Mekari Talenta REST, SAP, Shopify).
    """

    def __init__(
        self,
        name: str,
        connector_type: str,
        description: str = "",
        config: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.connector_type = connector_type
        self.description = description
        self.config = config or {}

    @abstractmethod
    async def execute_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a tool against the backend system.

        Args:
            tool_name: Canonical name of the tool to invoke.
            params: Dictionary of parameters/arguments for the tool.
            context: Optional execution context (e.g. user_token, tenant_id, company_id, headers).

        Returns:
            Dict containing the execution result data.
        """
        pass

    @abstractmethod
    async def discover_tools(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Discover tools exposed by this backend system dynamically.
        Returns a list of tool definition metadata dictionaries.
        """
        pass

    @abstractmethod
    def get_supported_tools(self) -> List[str]:
        """
        Return the list of tool names natively handled by this connector.
        """
        pass

    async def health_check(self) -> bool:
        """
        Optional health probe. Default returns True.
        """
        return True

    async def aclose(self) -> None:
        """
        Clean up any open HTTP sessions, connections, or resources.
        """
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', type='{self.connector_type}')>"
