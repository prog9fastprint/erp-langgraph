"""
Unit and Integration Tests for Marketplace Domain AI Tools (11 Read-Only Tools).
Covers:
- Pydantic filter schema validation & TOOL_DOMAIN_MAP registration
- ToolRegistryService catalog & RBAC (required_permission=None)
- DjangoCoreClient async dispatch for all 11 marketplace endpoints
- Router classification, collision suppression, write guardrail, and scalar workflow routing
- Extractor parameter resolution, include_live opt-in, and status whitelist isolation
- ToolNode is_active preservation, native company_ids + HTTP 403 fallback, and price-comparison account pre-lookup
- Verifier clean-audit 0-issue semantics & strict-identifier guard
- Synthesizer deterministic Markdown presentation for all 11 tools & follow-up suggestions
"""
import unittest
from datetime import date
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from app.core.context import resolve_tool_record_count, sanitize_erp_payload
from app.graph.nodes.extractor import TOOL_SCHEMA_REGISTRY, extractor_node
from app.graph.nodes.router import router_node
from app.graph.nodes.synthesizer import format_deterministic_answer, generate_deterministic_followups
from app.graph.nodes.tool_node import tool_node
from app.graph.nodes.verifier import _calculate_records_count, verifier_node
from app.graph.state import create_initial_state
from app.graph.workflow import decide_after_router
from app.schemas.tools import (
    AccountHealthFilterSchema,
    CashflowAndBalanceSummaryFilterSchema,
    CompetitorInsightsFilterSchema,
    DeductionAndSettlementSummaryFilterSchema,
    FeeAndEscrowAnomaliesFilterSchema,
    KilatOrderDiagnosticsFilterSchema,
    MarketplaceOrdersFilterSchema,
    PriceComparisonFilterSchema,
    ProductMappingStatusFilterSchema,
    StockSyncSimulationFilterSchema,
    SyncExecutionLogsFilterSchema,
    TOOL_DOMAIN_MAP,
)
from app.services.django_client import DjangoCoreClient, DjangoPermissionError
from app.services.tool_registry_service import ToolRegistryService


MARKETPLACE_TOOLS = [
    "get_account_health",
    "get_marketplace_orders",
    "get_kilat_order_diagnostics",
    "get_product_mapping_status",
    "get_stock_sync_simulation",
    "get_sync_execution_logs",
    "get_price_comparison",
    "get_fee_and_escrow_anomalies",
    "get_deduction_and_settlement_summary",
    "get_cashflow_and_balance_summary",
    "get_competitor_insights",
]


class TestMarketplaceTools(unittest.IsolatedAsyncioTestCase):
    """Comprehensive test suite for all 11 Marketplace domain AI tools."""

    def test_schema_validation_and_registries(self):
        # 1. Valid schema instantiations & constraints
        s1 = AccountHealthFilterSchema(marketplace="Shopee", only_issues=True, error_window_hours=48)
        self.assertEqual(s1.error_window_hours, 48)
        with self.assertRaises(ValidationError):
            AccountHealthFilterSchema(error_window_hours=200)

        s2 = MarketplaceOrdersFilterSchema(sla_status="overdue_shipping", date_from="2026-11-01", date_to="2026-11-25")
        self.assertEqual(s2.sla_status, "overdue_shipping")

        s3 = KilatOrderDiagnosticsFilterSchema(reference="260929AASAD26MZABF4", issue_type="unupdated_invoice")
        self.assertEqual(s3.issue_type, "unupdated_invoice")

        s4 = ProductMappingStatusFilterSchema(mapping_status="exceptions")
        self.assertEqual(s4.mapping_status, "exceptions")

        s5 = StockSyncSimulationFilterSchema(sku="INK-100", only_discrepancies=True, limit=50)
        self.assertEqual(s5.limit, 50)

        s6 = SyncExecutionLogsFilterSchema(log_type="stock", status="FAILED", trigger_source="CELERY_BEAT")
        self.assertEqual(s6.status, "FAILED")

        # PriceComparisonFilterSchema clamps limit <= 20 when include_live=True
        s7 = PriceComparisonFilterSchema(account_id=10, include_live=True, limit=50)
        self.assertEqual(s7.limit, 20)

        s8 = FeeAndEscrowAnomaliesFilterSchema(category="high_fee", min_pct=25.0)
        self.assertEqual(s8.min_pct, 25.0)

        s9 = DeductionAndSettlementSummaryFilterSchema(only_unposted_journals=True, date_field="received_date")
        self.assertTrue(s9.only_unposted_journals)

        s10 = CashflowAndBalanceSummaryFilterSchema(include_live=True, include_cashflow_logs=True)
        self.assertTrue(s10.include_live)

        s11 = CompetitorInsightsFilterSchema(focus="price_drops", stale_days=14)
        self.assertEqual(s11.stale_days, 14)

        for t_name in MARKETPLACE_TOOLS:
            self.assertEqual(TOOL_DOMAIN_MAP.get(t_name), "marketplace")
            self.assertIn(t_name, TOOL_SCHEMA_REGISTRY)

    def test_tool_registry_catalog_and_rbac(self):
        registry = ToolRegistryService()
        for t_name in MARKETPLACE_TOOLS:
            meta = registry.get_tool_by_name(t_name)
            self.assertIsNotNone(meta, f"Missing {t_name} in ToolRegistryService")
            self.assertEqual(meta["domain"], "marketplace")
            self.assertIsNone(meta["required_permission"])
            self.assertFalse(meta["is_write"])

        # Public discovery when user_roles is None (matches chatProxyView.py)
        found = registry.search_tools_fallback(
            query="kesehatan akun token expired marketplace",
            domain="marketplace",
            user_roles=None,
        )
        self.assertIn("get_account_health", [t["tool_name"] for t in found])

    @patch.object(DjangoCoreClient, "execute_tool", new_callable=AsyncMock)
    async def test_django_client_marketplace_methods(self, mock_exec):
        mock_exec.return_value = {"summary": {}}
        client = DjangoCoreClient()

        calls = [
            (client.fetch_account_health, "get_account_health", {"only_issues": True}),
            (client.fetch_marketplace_orders, "get_marketplace_orders", {"sla_status": "overdue_shipping"}),
            (client.fetch_kilat_order_diagnostics, "get_kilat_order_diagnostics", {"issue_type": "unupdated_invoice"}),
            (client.fetch_product_mapping_status, "get_product_mapping_status", {"mapping_status": "unmapped"}),
            (client.fetch_stock_sync_simulation, "get_stock_sync_simulation", {"sku": "SKU-01"}),
            (client.fetch_sync_execution_logs, "get_sync_execution_logs", {"status": "FAILED"}),
            (client.fetch_price_comparison, "get_price_comparison", {"account_id": 5}),
            (client.fetch_fee_and_escrow_anomalies, "get_fee_and_escrow_anomalies", {"min_pct": 20.0}),
            (client.fetch_deduction_and_settlement_summary, "get_deduction_and_settlement_summary", {"only_unposted_journals": True}),
            (client.fetch_cashflow_and_balance_summary, "get_cashflow_and_balance_summary", {"include_live": False}),
            (client.fetch_competitor_insights, "get_competitor_insights", {"focus": "price_drops"}),
        ]

        for method, expected_tool, params in calls:
            await method(params=params, company_id=7)
            mock_exec.assert_called_with(
                domain="marketplace",
                tool_name=expected_tool,
                params=params,
                company_id=7,
                user_token=None,
            )

    def test_router_all_11_tools_and_guardrails(self):
        cases = [
            ("Cek kesehatan akun & token marketplace yang expired", "get_account_health"),
            ("Order marketplace yang melewati batas kirim (overdue SLA)", "get_marketplace_orders"),
            ("Cek Pesanan Kilat yang nomor invoicenya belum update", "get_kilat_order_diagnostics"),
            ("Cek produk marketplace yang belum ter-mapping (unmapped)", "get_product_mapping_status"),
            ("Simulasi stok marketplace untuk SKU 91010600", "get_stock_sync_simulation"),
            ("Kenapa update stok atau harga marketplace gagal hari ini?", "get_sync_execution_logs"),
            ("Bandingkan harga ERP vs harga Shopee yang selisih", "get_price_comparison"),
            ("Cek anomali potongan marketplace di atas 20% bulan ini", "get_fee_and_escrow_anomalies"),
            ("Settlement marketplace mana yang belum ter-posting ke jurnal akuntansi?", "get_deduction_and_settlement_summary"),
            ("Berapa saldo dompet & pending escrow Shopee saat ini?", "get_cashflow_and_balance_summary"),
            ("Kompetitor mana yang menurunkan harga minggu ini?", "get_competitor_insights"),
        ]
        for query, expected_tool in cases:
            res = router_node(create_initial_state(user_query=query))
            self.assertEqual(res.get("detected_domain"), "marketplace", msg=f"Failed domain for: {query}")
            self.assertEqual(res.get("detected_tool"), expected_tool, msg=f"Failed tool for: {query}")
            self.assertEqual(decide_after_router(res), "extractor")

        # Collision suppression check: settlement unposted journal should NOT invoke get_journal_entries
        res_settle = router_node(create_initial_state("Settlement marketplace mana yang belum ter-posting ke jurnal akuntansi?"))
        self.assertEqual(res_settle.get("detected_tools"), ["get_deduction_and_settlement_summary"])

        # Explicit COA query stays in accounting
        res_coa = router_node(create_initial_state("Berapa saldo akun 11285 bank tiktok?"))
        self.assertEqual(res_coa.get("detected_domain"), "accounting")
        self.assertEqual(res_coa.get("detected_tool"), "get_account_balance")

        # Write guardrails route directly to out_of_scope
        for w_query in ("Update stok Shopee sekarang", "Push harga ke TikTok Shop", "Gabung pesanan kilat"):
            res_w = router_node(create_initial_state(w_query))
            self.assertEqual(res_w.get("detected_domain"), "out_of_scope", msg=f"Write not guarded: {w_query}")
            self.assertIsNone(res_w.get("detected_tool"))

        # Greeting + marketplace intent
        res_hi = router_node(create_initial_state("Halo, cek kesehatan akun marketplace"))
        self.assertEqual(res_hi.get("detected_domain"), "marketplace")
        self.assertEqual(res_hi.get("detected_tool"), "get_account_health")

        # Short follow-up preserves marketplace domain
        followup_state = create_initial_state(
            user_query="di cabang Jakarta",
            messages=[
                HumanMessage(content="Cek order marketplace yang belum ada resi AWB"),
                AIMessage(content="Hasil ditemukan."),
                HumanMessage(content="di cabang Jakarta"),
            ],
        )
        res_fu = router_node(followup_state)
        self.assertEqual(res_fu.get("detected_domain"), "marketplace")
        self.assertEqual(res_fu.get("detected_tool"), "get_marketplace_orders")
        self.assertEqual(res_fu.get("company_id"), 8)

    def test_extractor_marketplace_parameters(self):
        ref_today = date(2026, 11, 25)

        # 1. Sync logs preserves status="FAILED" and "SKIPPED" (not wiped by ALLOWED_JOURNAL_STATUSES)
        res_logs = extractor_node(
            {
                "messages": [HumanMessage(content="Kenapa update stok marketplace skipped hari ini?")],
                "company_id": 7,
                "detected_domain": "marketplace",
                "detected_tool": "get_sync_execution_logs",
                "user_query": "Kenapa update stok marketplace skipped hari ini?",
            },
            ref_date=ref_today,
        )
        p_logs = res_logs.get("extracted_params") or {}
        self.assertEqual(p_logs.get("log_type"), "stock")
        self.assertEqual(p_logs.get("status"), "SKIPPED")
        self.assertEqual(p_logs.get("date_from"), "2026-11-25")

        # 2. Sales orders preserves status="quotation"
        res_so = extractor_node(
            {
                "messages": [HumanMessage(content="Daftar quotation bulan ini")],
                "company_id": 7,
                "detected_domain": "sales",
                "detected_tool": "get_sales_orders",
                "user_query": "Daftar quotation bulan ini",
            },
            ref_date=ref_today,
        )
        self.assertEqual((res_so.get("extracted_params") or {}).get("status"), "quotation")

        # 3. include_live defaults False, True only on explicit live intent
        res_pc_cache = extractor_node(
            {
                "messages": [HumanMessage(content="Bandingkan harga ERP vs Shopee")],
                "company_id": 7,
                "detected_domain": "marketplace",
                "detected_tool": "get_price_comparison",
                "user_query": "Bandingkan harga ERP vs Shopee",
            },
            ref_date=ref_today,
        )
        self.assertFalse((res_pc_cache.get("extracted_params") or {}).get("include_live"))

        res_pc_live = extractor_node(
            {
                "messages": [HumanMessage(content="Bandingkan harga live Shopee vs pricelist sekarang")],
                "company_id": 7,
                "detected_domain": "marketplace",
                "detected_tool": "get_price_comparison",
                "user_query": "Bandingkan harga live Shopee vs pricelist sekarang",
            },
            ref_date=ref_today,
        )
        self.assertTrue((res_pc_live.get("extracted_params") or {}).get("include_live"))

        # 4. Kilat reservation token extraction
        res_kilat = extractor_node(
            {
                "messages": [HumanMessage(content="Diagnosa (No. Reservasi: 260929AASAD26MZABF4)")],
                "company_id": 7,
                "detected_domain": "marketplace",
                "detected_tool": "get_kilat_order_diagnostics",
                "user_query": "Diagnosa (No. Reservasi: 260929AASAD26MZABF4)",
            },
            ref_date=ref_today,
        )
        self.assertEqual((res_kilat.get("extracted_params") or {}).get("reference"), "260929AASAD26MZABF4")

        # 5. Fee anomalies min_pct extraction
        res_fee = extractor_node(
            {
                "messages": [HumanMessage(content="Cek anomali potongan Shopee di atas 25% bulan ini")],
                "company_id": 7,
                "detected_domain": "marketplace",
                "detected_tool": "get_fee_and_escrow_anomalies",
                "user_query": "Cek anomali potongan Shopee di atas 25% bulan ini",
            },
            ref_date=ref_today,
        )
        p_fee = res_fee.get("extracted_params") or {}
        self.assertEqual(p_fee.get("min_pct"), 25.0)
        self.assertEqual(p_fee.get("marketplace"), "Shopee")
        self.assertEqual(p_fee.get("date_from"), "2026-11-01")

    async def test_tool_node_is_active_preservation_and_multi_company_403_fallback(self):
        # Default sanitize_erp_payload still strips is_active
        self.assertNotIn("is_active", sanitize_erp_payload({"id": 1, "is_active": True}))

        # 1. Single-call native company_ids on marketplace + is_active preserved for get_account_health
        mock_client = AsyncMock(spec=DjangoCoreClient)
        mock_client.fetch_account_health.return_value = {
            "company_ids": [7, 8],
            "error_window_hours": 24,
            "summary": {"total_accounts": 1, "active_accounts": 1, "healthy_accounts": 1, "accounts_with_issues": 0},
            "accounts": [
                {"account_id": 10, "account_name": "Shopee SBY", "marketplace": "Shopee", "is_active": True, "state": "Authenticated"}
            ],
        }

        state_mc = {
            "user_query": "Cek kesehatan akun marketplace cabang Surabaya dan Jakarta",
            "detected_domain": "marketplace",
            "detected_tool": "get_account_health",
            "company_id": 7,
            "company_ids": [7, 8],
            "is_multi_company": True,
            "extracted_params": {},
        }
        res_mc = await tool_node(state_mc, client=mock_client)
        self.assertEqual(res_mc["verification_status"], "VALID")
        self.assertEqual(mock_client.fetch_account_health.await_count, 1)
        called_params = mock_client.fetch_account_health.await_args.kwargs["params"]
        self.assertEqual(called_params["company_ids"], [7, 8])
        self.assertTrue(res_mc["tool_result"]["accounts"][0]["is_active"])

        # 2. HTTP 403 fallback to per-branch fan-out isolates unauthorized company
        async def _health_side_effect(params=None, user_token=None, company_id=None):
            if params and params.get("company_ids"):
                raise DjangoPermissionError("Forbidden (403): company 8 unauthorized")
            if company_id == 8:
                raise DjangoPermissionError("Forbidden (403): company 8 unauthorized")
            return {
                "company_ids": [7],
                "summary": {"total_accounts": 1, "active_accounts": 1},
                "accounts": [{"account_id": 10, "account_name": "Shopee SBY", "is_active": True}],
            }

        mock_client_403 = AsyncMock(spec=DjangoCoreClient)
        mock_client_403.fetch_account_health.side_effect = _health_side_effect
        res_403 = await tool_node(state_mc, client=mock_client_403)
        self.assertEqual(res_403["verification_status"], "VALID")
        self.assertEqual(res_403["unauthorized_companies"], [8])
        self.assertIn(7, res_403["multi_company_results"])
        self.assertIsNone(res_403["multi_company_results"][8])

    async def test_tool_node_price_comparison_account_pre_resolution(self):
        mock_client = AsyncMock(spec=DjangoCoreClient)
        mock_client.fetch_account_health.return_value = {
            "accounts": [
                {"account_id": 11, "account_name": "Shopee SBY", "marketplace": "Shopee", "company_id": 7, "is_active": True, "has_order_config": True},
                {"account_id": 12, "account_name": "Shopee JKT", "marketplace": "Shopee", "company_id": 7, "is_active": True, "has_order_config": True},
            ]
        }
        mock_client.fetch_price_comparison.side_effect = [
            {
                "company_ids": [7],
                "pricelist": {"id": 1, "name": "PL SBY"},
                "filter_status": "mismatch",
                "include_live": False,
                "summary": {"total_evaluated": 10, "match_count": 9, "mismatch_count": 1, "missing_mp_price": 0, "missing_erp_price": 0},
                "total_count": 1,
                "items": [{"erp_sku": "SKU-1", "account_name": "Shopee SBY", "comparison_status": "mismatch"}],
            },
            {
                "company_ids": [7],
                "pricelist": {"id": 2, "name": "PL JKT"},
                "filter_status": "mismatch",
                "include_live": False,
                "summary": {"total_evaluated": 5, "match_count": 4, "mismatch_count": 1, "missing_mp_price": 0, "missing_erp_price": 0},
                "total_count": 1,
                "items": [{"erp_sku": "SKU-2", "account_name": "Shopee JKT", "comparison_status": "mismatch"}],
            },
        ]

        state = {
            "user_query": "Bandingkan harga ERP vs harga Shopee",
            "detected_domain": "marketplace",
            "detected_tool": "get_price_comparison",
            "company_id": 7,
            "extracted_params": {"filter_status": "mismatch"},
        }
        res = await tool_node(state, client=mock_client)
        self.assertEqual(res["verification_status"], "VALID")
        self.assertEqual(res["tool_result"]["summary"]["mismatch_count"], 2)
        self.assertEqual(len(res["tool_result"]["items"]), 2)
        self.assertEqual(mock_client.fetch_price_comparison.await_count, 2)

        # Missing configured account fails fast with clear error code
        mock_empty = AsyncMock(spec=DjangoCoreClient)
        mock_empty.fetch_account_health.return_value = {"accounts": []}
        res_err = await tool_node(state, client=mock_empty)
        self.assertEqual(res_err["verification_status"], "ERROR")
        self.assertIn("MISSING_PRICELIST_OR_ACCOUNT", res_err["error_message"])

    def test_verifier_clean_audit_and_strict_guards(self):
        # 1. Clean audit with 0 anomalies when summary exists -> VALID (no retry)
        anom_clean = {
            "date_window": {"date_from": "2026-11-01", "date_to": "2026-11-25"},
            "filters": {"category": "all"},
            "summary": {"total_anomalies": 0},
            "total_count": 0,
            "anomalies": [],
        }
        self.assertEqual(_calculate_records_count(anom_clean), 1)
        v_clean = verifier_node({
            "detected_domain": "marketplace",
            "detected_tool": "get_fee_and_escrow_anomalies",
            "tool_result": anom_clean,
            "retry_count": 0,
        })
        self.assertEqual(v_clean["verification_status"], "VALID")

        # 2. Clean audit on sync logs (0 failed out of 120 runs) -> VALID
        logs_clean = {
            "stock_summary": {"total": 120, "success": 120, "failed": 0},
            "stock_logs_total": 0,
            "stock_logs": [],
        }
        self.assertEqual(_calculate_records_count(logs_clean), 1)
        self.assertGreaterEqual(resolve_tool_record_count(logs_clean), 1)

        # 3. True empty order search triggers RETRY
        orders_empty = {"total_count": 0, "orders": []}
        v_retry = verifier_node({
            "detected_domain": "marketplace",
            "detected_tool": "get_marketplace_orders",
            "extracted_params": {"sla_status": "overdue_shipping"},
            "tool_result": orders_empty,
            "retry_count": 0,
        })
        self.assertEqual(v_retry["verification_status"], "RETRY")

        # 4. Strict Kilat reference miss goes straight to EMPTY
        kilat_empty = {"summary": {"total_kilat_scanned": 0}, "total_count": 0, "records": []}
        v_kilat = verifier_node({
            "detected_domain": "marketplace",
            "detected_tool": "get_kilat_order_diagnostics",
            "extracted_params": {"reference": "260929AASAD26MZABF4"},
            "tool_result": kilat_empty,
            "retry_count": 0,
        })
        self.assertEqual(v_kilat["verification_status"], "EMPTY")

    def test_synthesizer_all_11_tools_and_guardrails(self):
        payloads = [
            ("get_account_health", {
                "error_window_hours": 24,
                "summary": {"total_accounts": 1, "active_accounts": 1, "healthy_accounts": 0, "accounts_with_issues": 1, "expired_tokens": 1, "expiring_soon_tokens": 0, "missing_configs_count": 0, "recent_fetch_errors_count": 1},
                "accounts": [{"account_name": "Shopee SBY", "marketplace": "Shopee", "company_name": "Surabaya", "state": "Authenticated", "token_status": "expired", "hours_until_expiry": -2.5, "shop_status": "Open", "issues": ["token_expired"]}],
            }, "Shopee SBY"),
            ("get_marketplace_orders", {
                "sla_status_filter": "overdue_shipping", "mp_status_filter": "processing", "total_count": 1,
                "orders": [{"mp_invoice_number": "260101ABC", "so_number": "SO/2026/001", "marketplace": "Shopee", "account_name": "Shopee SBY", "mp_order_status": "READY_TO_SHIP", "erp_status": "Sales Order", "awb_number": None, "max_shipping_date": "2026-11-24 12:00", "sla_flags": ["overdue_shipping", "missing_awb"], "mp_total": 150000.0}],
            }, "260101ABC"),
            ("get_kilat_order_diagnostics", {
                "summary": {"total_kilat_scanned": 1, "unupdated_invoice_count": 1, "duplicate_collision_count": 0, "missing_so_link_count": 0},
                "total_count": 1,
                "records": [{"mp_invoice_number": "RES123", "no_reservasi": "260929AASAD26MZABF4", "invoice_updated": False, "so_number": "SO/2026/009", "erp_status": "Draft", "account_name": "Shopee SBY", "issues": ["unupdated_invoice"]}],
            }, "/marketplace/tools/handling-order-kilat/"),
            ("get_product_mapping_status", {
                "mapping_status_filter": "exceptions", "total_count": 1,
                "exceptions": [{"account_name": "Shopee SBY", "marketplace": "Shopee", "erp_sku": "INK-100", "erp_name": "Tinta", "parent_sku": "P1", "child_sku": "C1"}],
            }, "ExceptionProduct"),
            ("get_stock_sync_simulation", {
                "summary": {"total_evaluated": 1, "discrepancy_or_warning_count": 1}, "total_count": 1,
                "simulations": [{"erp_sku": "INK-100", "mp_sku": "MP-100", "account_name": "Shopee SBY", "marketplace": "Shopee", "stock_setting": "Onhand", "raw_erp_stock": 20.0, "qty_multiplier": 2, "simulated_push_stock": 10, "last_stock_log": {"new_stock": 15, "status": "SUCCESS"}, "diff_vs_last_log": -5, "masked_shared_branches_count": 1, "warnings": ["bundle_multiplier_2"]}],
            }, "shared_branches_masked:1"),
            ("get_sync_execution_logs", {
                "log_type": "price", "status_filter": "FAILED", "date_window": {"date_from": "2026-11-18", "date_to": "2026-11-25"},
                "price_summary": {"total": 10, "success": 8, "failed": 2, "avg_duration_ms": 110.5},
                "price_logs_total": 1,
                "price_logs": [{"created_at": "2026-11-25 10:00:00", "account_name": "Shopee SBY", "erp_sku": "INK-100", "qty_tier": "wholesale >= 10", "old_price": 50000.0, "new_price": 45000.0, "status": "FAILED", "normalized_error": {"code": "WHOLESALE_PRICE_MISMATCH", "message": "Tier mismatch"}}],
            }, "WHOLESALE_PRICE_MISMATCH"),
            ("get_price_comparison", {
                "pricelist": {"id": 1, "name": "PL Shopee"}, "include_live": True,
                "live_fallback_errors": {"11": "Timeout"},
                "summary": {"total_evaluated": 1, "match_count": 0, "mismatch_count": 1, "missing_mp_price": 0, "missing_erp_price": 0},
                "total_count": 1,
                "items": [{"erp_sku": "INK-100", "erp_name": "Tinta", "account_name": "Shopee SBY", "expected_erp_bundle_price": 50000.0, "marketplace_price": 45000.0, "price_source": "db_cache", "difference": -5000.0, "difference_pct": 10.0, "comparison_status": "mismatch"}],
            }, "Catatan Live Fallback"),
            ("get_fee_and_escrow_anomalies", {
                "date_window": {"date_from": "2026-11-01", "date_to": "2026-11-25"},
                "summary": {"total_anomalies": 1, "negative_fee_count": 0, "high_fee_count": 1, "zero_income_count": 0, "erp_mp_mismatch_count": 0, "total_fee_difference_idr": 30000.0},
                "total_count": 1,
                "anomalies": [{"mp_invoice_number": "INVMP01", "so_number": "SO/2026/050", "account_name": "Shopee SBY", "marketplace": "Shopee", "mp_total": 100000.0, "seller_expected_income": 70000.0, "fee_difference": 30000.0, "fee_percentage": 30.0, "anomaly_types": ["high_fee"]}],
            }, "30.00%"),
            ("get_deduction_and_settlement_summary", {
                "date_field": "received_date", "date_window": {"date_from": "2026-11-01", "date_to": "2026-11-25"},
                "summary": {"total_orders": 10, "total_gross": 1000000.0, "total_deduction": 120000.0, "total_tax": 10000.0, "total_nett": 870000.0, "effective_deduction_pct": 12.0, "unposted_journal_orders_count": 2, "unposted_journal_nett_amount": 150000.0, "arithmetic_mismatch_count": 0},
                "by_deduction_category": [{"category": "Komisi Platform", "line_count": 10, "total_amount": 80000.0}],
                "by_store": [{"account_name": "Shopee SBY", "marketplace": "Shopee", "orders": 10, "amount_gross": 1000000.0, "deduction_fee": 120000.0, "tax": 10000.0, "amount_nett": 870000.0}],
            }, "Komisi Platform"),
            ("get_cashflow_and_balance_summary", {
                "date_window": {"date_from": "2026-10-26", "date_to": "2026-11-25"}, "include_live": False,
                "summary": {"total_available_balance": 2000000.0, "total_pending_balance": 500000.0, "total_balance": 2500000.0, "period_settlement_inflow": 3000000.0, "period_withdrawal_outflow": 1000000.0, "period_adjustment_net": 0.0, "stale_or_failed_snapshots": 0},
                "accounts": [{"account_name": "Shopee SG", "marketplace": "Shopee", "currency": "SGD", "available_balance": 100.0, "pending_balance": 25.0, "total_balance": 125.0, "balance_source": "db_snapshot", "snapshot_date": "2026-11-25", "warnings": []}],
            }, "SGD 125.00"),
            ("get_competitor_insights", {
                "focus": "price_drops", "stale_days": 7, "date_window": {"date_from": "2026-11-18", "date_to": "2026-11-25"},
                "summary": {"active_competitors": 2, "stale_competitors_count": 1, "competitors_with_price_drops": 1, "total_period_sold_velocity": 45.0},
                "total_count": 1,
                "competitors": [{"store_name": "Toko Saingan", "marketplace": "Shopee", "group_name": "Tinta", "erp_sku": "INK-100", "current_price": 42000.0, "net_price_delta": -3000.0, "price_drop_events": 1, "period_sold_velocity": 45.0, "audit_status": "fresh"}],
            }, "Toko Saingan"),
        ]

        for t_name, payload, expected_substr in payloads:
            out = format_deterministic_answer({
                "user_query": f"Test {t_name}",
                "detected_domain": "marketplace",
                "detected_tool": t_name,
                "verification_status": "VALID",
                "tool_result": payload,
            })
            self.assertIn(expected_substr, out, msg=f"Formatter failed for {t_name}")
            followups = generate_deterministic_followups({
                "detected_domain": "marketplace",
                "detected_tool": t_name,
                "verification_status": "VALID",
            })
            self.assertTrue(len(followups) >= 1)

        # Write out_of_scope response points to /marketplace/
        write_oos = format_deterministic_answer({
            "user_query": "Update stok Shopee sekarang",
            "detected_domain": "out_of_scope",
            "verification_status": "VALID",
        })
        self.assertIn("/marketplace/", write_oos)

        # MISSING_PRICELIST_OR_ACCOUNT guidance
        err_pl = format_deterministic_answer({
            "user_query": "Bandingkan harga ERP vs Shopee",
            "detected_domain": "marketplace",
            "detected_tool": "get_price_comparison",
            "verification_status": "ERROR",
            "error_message": "MISSING_PRICELIST_OR_ACCOUNT: Tidak ada konfigurasi",
        })
        self.assertIn("AccountOrder", err_pl)


if __name__ == "__main__":
    unittest.main()
