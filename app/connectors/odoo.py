"""
Odoo ERP Connector for LangGraph AI Orchestrator.
Supports Retail, Point of Sale (POS), Multi-Warehouse Stock, and Sales Orders via Odoo JSON-RPC.
"""
from typing import Any, Dict, List, Optional
import logging
import httpx

from app.connectors.base import BaseSystemConnector
from app.core.config import settings

logger = logging.getLogger("odoo_connector")


class OdooConnector(BaseSystemConnector):
    """
    Connector for Odoo ERP (v14 - v18) via standard JSON-RPC protocol.
    Enables checking stock in physical/transit warehouses, POS orders, products, and quotations.
    """

    ODOO_TOOLS: List[str] = [
        "odoo_get_stock",
        "odoo_get_products",
        "odoo_get_pos_orders",
        "odoo_get_sales_orders",
        "odoo_execute_kw",
    ]

    def __init__(
        self,
        name: str = "odoo",
        url: Optional[str] = None,
        db: Optional[str] = None,
        username: Optional[str] = None,
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            name=name,
            connector_type="odoo_rpc",
            description="Odoo ERP Retail & POS Connector",
            config=config,
        )
        self.url = (url or settings.ODOO_URL or "").rstrip("/")
        self.db = db or settings.ODOO_DB or ""
        self.username = username or settings.ODOO_USERNAME or ""
        self.api_key = api_key or settings.ODOO_API_KEY or ""
        self._uid: Optional[int] = None
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT_SEC)
        return self._client

    async def _authenticate(self) -> int:
        """Authenticate via Odoo JSON-RPC /web/session/authenticate or common.authenticate."""
        if self._uid is not None:
            return self._uid

        if not self.url or not self.db or not self.username:
            raise ValueError("Odoo connection credentials (url, db, username) are not fully configured.")

        client = await self._get_client()
        payload = {
            "jsonrpc": "2.0",
            "method": "call",
            "params": {
                "db": self.db,
                "login": self.username,
                "password": self.api_key,
            },
            "id": 1,
        }
        resp = await client.post(f"{self.url}/jsonrpc", json=payload)
        data = resp.json()
        if data.get("error"):
            raise ConnectionError(f"Odoo Auth Error: {data['error']}")
        uid = data.get("result")
        if not uid:
            raise PermissionError("Failed to authenticate with Odoo (invalid credentials).")
        self._uid = int(uid)
        return self._uid

    async def execute_kw(
        self,
        model: str,
        method: str,
        args: List[Any],
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Execute standard Odoo ORM method via JSON-RPC."""
        uid = await self._authenticate()
        client = await self._get_client()
        payload = {
            "jsonrpc": "2.0",
            "method": "call",
            "params": {
                "service": "object",
                "method": "execute_kw",
                "args": [self.db, uid, self.api_key, model, method, args, kwargs or {}],
            },
            "id": 2,
        }
        resp = await client.post(f"{self.url}/jsonrpc", json=payload)
        data = resp.json()
        if data.get("error"):
            raise RuntimeError(f"Odoo RPC Error ({model}.{method}): {data['error']}")
        return data.get("result")

    async def execute_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute Retail / Inventory / POS tools against Odoo.
        """
        if tool_name == "odoo_get_stock":
            product_query = params.get("product_name") or params.get("sku") or params.get("search")
            location_id = params.get("location_id") or params.get("warehouse_id")
            domain: List[Any] = [["quantity", ">", 0]]
            if product_query:
                domain.append(["product_id.name", "ilike", product_query])
            if location_id:
                domain.append(["location_id", "=", int(location_id)])

            quants = await self.execute_kw(
                "stock.quant",
                "search_read",
                [domain],
                {
                    "fields": ["product_id", "location_id", "quantity", "reserved_quantity"],
                    "limit": params.get("limit", 50),
                },
            )
            return {
                "system": "odoo",
                "tool": "odoo_get_stock",
                "total_records": len(quants),
                "items": quants,
            }

        elif tool_name == "odoo_get_products":
            search = params.get("search") or params.get("name")
            domain = []
            if search:
                domain.append(["name", "ilike", search])
            products = await self.execute_kw(
                "product.template",
                "search_read",
                [domain],
                {
                    "fields": ["name", "default_code", "list_price", "qty_available", "uom_id"],
                    "limit": params.get("limit", 20),
                },
            )
            return {
                "system": "odoo",
                "tool": "odoo_get_products",
                "total_records": len(products),
                "products": products,
            }

        elif tool_name == "odoo_get_pos_orders":
            domain = []
            if params.get("date_from"):
                domain.append(["date_order", ">=", params["date_from"]])
            if params.get("date_to"):
                domain.append(["date_order", "<=", params["date_to"]])
            if params.get("session_id"):
                domain.append(["session_id", "=", int(params["session_id"])])

            orders = await self.execute_kw(
                "pos.order",
                "search_read",
                [domain],
                {
                    "fields": ["name", "date_order", "partner_id", "amount_total", "state", "lines"],
                    "limit": params.get("limit", 30),
                },
            )
            return {
                "system": "odoo",
                "tool": "odoo_get_pos_orders",
                "total_orders": len(orders),
                "orders": orders,
            }

        elif tool_name == "odoo_get_sales_orders":
            domain = []
            if params.get("search"):
                domain.append(["name", "ilike", params["search"]])
            orders = await self.execute_kw(
                "sale.order",
                "search_read",
                [domain],
                {
                    "fields": ["name", "date_order", "partner_id", "amount_total", "state"],
                    "limit": params.get("limit", 30),
                },
            )
            return {
                "system": "odoo",
                "tool": "odoo_get_sales_orders",
                "total_orders": len(orders),
                "orders": orders,
            }

        elif tool_name == "odoo_execute_kw":
            model = params.get("model", "")
            method = params.get("method", "search_read")
            args = params.get("args", [])
            kwargs = params.get("kwargs", {})
            res = await self.execute_kw(model, method, args, kwargs)
            return {"system": "odoo", "result": res}

        else:
            raise NotImplementedError(f"Tool '{tool_name}' is not supported by OdooConnector.")

    async def discover_tools(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        return [
            {
                "tool_name": "odoo_get_stock",
                "domain": "retail_inventory",
                "description": "Cek kuantitas stok fisik dan available di berbagai gudang/cabang/toko ritel Odoo.",
            },
            {
                "tool_name": "odoo_get_products",
                "domain": "retail_inventory",
                "description": "Katalog master produk, SKU, barcode, dan harga jual ritel Odoo.",
            },
            {
                "tool_name": "odoo_get_pos_orders",
                "domain": "pos",
                "description": "Daftar transaksi kasir Point of Sale (POS) ritel Odoo dan status settlement.",
            },
            {
                "tool_name": "odoo_get_sales_orders",
                "domain": "sales",
                "description": "Pesanan penjualan (Sales Order) dan penawaran ritel Odoo.",
            },
        ]

    def get_supported_tools(self) -> List[str]:
        return list(self.ODOO_TOOLS)

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
