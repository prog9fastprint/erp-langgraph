"""
Connectors Package for LangGraph AI Orchestrator.
Exposes BaseSystemConnector, specific ERP/HRIS connectors, and the central ConnectorRegistry.
"""
from app.connectors.base import BaseSystemConnector
from app.connectors.fastprint import FastPrintConnector
from app.connectors.odoo import OdooConnector
from app.connectors.talenta import TalentaConnector
from app.connectors.generic_rest import GenericRestConnector
from app.connectors.registry import ConnectorRegistry, get_connector_registry

__all__ = [
    "BaseSystemConnector",
    "FastPrintConnector",
    "OdooConnector",
    "TalentaConnector",
    "GenericRestConnector",
    "ConnectorRegistry",
    "get_connector_registry",
]
