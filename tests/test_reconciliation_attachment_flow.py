"""
Unit tests for PDF Attachment & Bank Reconciliation Flow in LangGraph Orchestrator.
Covers schema validation, client method binding, attachment-aware routing,
parameter extraction, record verification, deterministic formatting, and
proactive inspect clarification output.
"""
import unittest

from app.core.context import resolve_tool_record_count, sanitize_erp_payload
from app.graph.nodes.extractors.accounting import (
    _fallback_extract_bank_reconciliation,
    _fallback_extract_reconcile_bank_statement,
)
from app.graph.nodes.formatters.accounting import (
    format_get_bank_reconciliation_summary,
    format_reconcile_bank_statement,
)
from app.graph.nodes.router import classify_query_rule_based
from app.graph.nodes.router_semantic import try_route_pdf_attachment
from app.graph.nodes.synthesizer import (
    _build_pdf_inspect_clarification_output,
    format_deterministic_answer,
)
from app.graph.state import create_initial_state
from app.schemas.chat import ChatAttachmentItem, ChatRequest
from app.schemas.tools import (
    BankReconciliationFilterSchema,
    ReconcileBankStatementFilterSchema,
    TOOL_DOMAIN_MAP,
)
from app.services.django_client import DjangoCoreClient


class TestReconciliationAttachmentFlow(unittest.TestCase):
    """Verify end-to-end attachment and read-only bank reconciliation handling."""

    def test_chat_request_allows_empty_query_with_attachment(self):
        req = ChatRequest(
            query="",
            attachments=[
                ChatAttachmentItem(
                    attachment_id="att-1",
                    filename="rekening_bca.pdf",
                    file_path="aiassistant_uploads/thread1/rekening_bca.pdf",
                    mime_type="application/pdf",
                    ext=".pdf",
                    size=12345,
                )
            ],
        )
        self.assertEqual(req.query, "[Berkas Dilampirkan: rekening_bca.pdf]")
        self.assertEqual(len(req.attachments), 1)

        with self.assertRaises(ValueError):
            ChatRequest(query="   ", attachments=[])

    def test_reconcile_schemas_and_client_binding(self):
        schema = ReconcileBankStatementFilterSchema(
            pdf_path="aiassistant_uploads/t1/bca.pdf",
            attachment_id="att-1",
            mode="reconcile",
            consolidate_branches=True,
            include_cross_check=True,
            account_codes=["11200HO"],
        )
        self.assertEqual(schema.mode, "reconcile")
        self.assertTrue(schema.consolidate_branches)
        self.assertTrue(schema.include_cross_check)

        summary_schema = BankReconciliationFilterSchema(consolidate_branches=True)
        self.assertTrue(summary_schema.consolidate_branches)

        self.assertEqual(TOOL_DOMAIN_MAP.get("reconcile_bank_statement"), "accounting")
        self.assertTrue(hasattr(DjangoCoreClient, "reconcile_bank_statement"))

    def test_create_initial_state_preserves_attachments(self):
        att = [{"filename": "koran.pdf", "file_path": "aiassistant_uploads/t1/koran.pdf", "ext": ".pdf"}]
        state = create_initial_state(
            user_query="",
            attachments=att,
            thread_attachments=att,
        )
        self.assertEqual(state["attachments"], att)
        self.assertEqual(state["thread_attachments"], att)

    def test_try_route_pdf_attachment_inspect_followup_and_non_pdf(self):
        att = [{"filename": "bca_apr.pdf", "file_path": "aiassistant_uploads/t1/bca_apr.pdf", "ext": ".pdf"}]
        state_current = create_initial_state(user_query="", attachments=att)

        res_inspect = try_route_pdf_attachment("", state_current)
        self.assertIsNotNone(res_inspect)
        self.assertEqual(res_inspect["detected_tool"], "reconcile_bank_statement")
        self.assertEqual(res_inspect["detected_domain"], "accounting")

        state_thread = create_initial_state(user_query="Tolong rekonsiliasi rekening koran BCA ini", thread_attachments=att)
        res_reconcile = try_route_pdf_attachment("Tolong rekonsiliasi rekening koran BCA ini", state_thread)
        self.assertIsNotNone(res_reconcile)
        self.assertEqual(res_reconcile["detected_tool"], "reconcile_bank_statement")

        state_img = create_initial_state(
            user_query="cek penjualan",
            attachments=[{"filename": "foto.png", "file_path": "aiassistant_uploads/t1/foto.png", "ext": ".png"}],
        )
        self.assertIsNone(try_route_pdf_attachment("cek penjualan", state_img))

    def test_fallback_extractor_populates_pdf_path_mode_and_flags(self):
        att = [{
            "attachment_id": "att-99",
            "filename": "mandiri.pdf",
            "file_path": "aiassistant_uploads/t2/mandiri.pdf",
            "ext": ".pdf",
        }]
        state = create_initial_state(user_query="[Berkas Dilampirkan: mandiri.pdf]", attachments=att)
        inspect_params = _fallback_extract_reconcile_bank_statement(
            "[Berkas Dilampirkan: mandiri.pdf]",
            state=state,
        )
        self.assertEqual(inspect_params["pdf_path"], "aiassistant_uploads/t2/mandiri.pdf")
        self.assertEqual(inspect_params["attachment_id"], "att-99")
        self.assertEqual(inspect_params["mode"], "inspect")
        self.assertTrue(inspect_params["consolidate_branches"])

        reconcile_params = _fallback_extract_reconcile_bank_statement(
            "audit mutasi 11200HO salah kamar semua cabang",
            state=state,
        )
        self.assertEqual(reconcile_params["mode"], "reconcile")
        self.assertTrue(reconcile_params["consolidate_branches"])
        self.assertTrue(reconcile_params["include_cross_check"])
        self.assertEqual(reconcile_params["account_codes"], ["11200HO"])

        single_branch_params = _fallback_extract_reconcile_bank_statement(
            "rekonsiliasi cabang aktif saja",
            state=state,
        )
        self.assertEqual(single_branch_params["mode"], "reconcile")
        self.assertFalse(single_branch_params["consolidate_branches"])

        chip_single_branch = _fallback_extract_reconcile_bank_statement(
            "Rekonsiliasi file mandiri.pdf khusus untuk company Main Store saja.",
            state=state,
        )
        self.assertEqual(chip_single_branch["mode"], "reconcile")
        self.assertFalse(chip_single_branch["consolidate_branches"])

        summary_cons = _fallback_extract_bank_reconciliation(
            "cek ringkasan rekonsiliasi bank BCA semua cabang bulan ini"
        )
        self.assertTrue(summary_cons.get("consolidate_branches"))

    def test_rule_based_classifier_and_thread_pdf_isolation(self):
        # Must not raise NameError when state is omitted
        res_no_state = classify_query_rule_based("Cek ringkasan rekonsiliasi bank BCA")
        self.assertIsNotNone(res_no_state)
        self.assertEqual(res_no_state["detected_tool"], "get_bank_reconciliation_summary")

        att = [{"filename": "bca_apr.pdf", "file_path": "aiassistant_uploads/t1/bca_apr.pdf", "ext": ".pdf"}]
        state_with_pdf = create_initial_state(user_query="cek rekonsiliasi bank", thread_attachments=att)
        res_with_state = classify_query_rule_based("cek rekonsiliasi bank", state=state_with_pdf)
        self.assertIsNotNone(res_with_state)
        self.assertEqual(res_with_state["detected_tool"], "reconcile_bank_statement")

        # Unrelated Turn-2 queries must NOT be hijacked by prior thread PDF
        self.assertIsNone(try_route_pdf_attachment("Cek mutasi kartu stok SKU 91010600", state_with_pdf))
        self.assertIsNone(try_route_pdf_attachment("Cek selisih harga Shopee", state_with_pdf))
        self.assertIsNone(try_route_pdf_attachment("Cek stok tinta di semua cabang", state_with_pdf))

        # Contextual follow-up after reconcile_bank_statement SHOULD route to PDF tool
        state_with_pdf["prior_detected_tool"] = "reconcile_bank_statement"
        res_followup = try_route_pdf_attachment("Cek pengeluaran yang salah kamar", state_with_pdf)
        self.assertIsNotNone(res_followup)
        self.assertEqual(res_followup["detected_tool"], "reconcile_bank_statement")

    def test_context_verifier_counts_and_sanitizes_inspect_payload(self):
        payload = {
            "mode": "inspect",
            "statement_meta": {"bank_code": "BCA", "account_number": "1234567890"},
            "checksum_gate": {"passed": True, "discrepancy": 0.0},
            "erp_accounts_used": [],
            "suggested_clarifications": [
                {"label": "Rekonsiliasi Penuh", "query": "Lakukan rekonsiliasi penuh", "icon": "fa-scale-balanced"}
            ],
        }
        cleaned = sanitize_erp_payload(payload)
        self.assertIn("erp_accounts_used", cleaned)
        self.assertIn("statement_meta", cleaned)
        self.assertEqual(resolve_tool_record_count(cleaned), 1)

    def test_formatter_and_inspect_clarification_output(self):
        inspect_payload = {
            "mode": "inspect",
            "statement_meta": {
                "bank_code": "BCA",
                "account_number": "0881234567",
                "account_holder": "PT FAST PRINT INDONESIA",
                "statement_period": "2026-04-01 s/d 2026-04-30",
                "cr_count": 20,
                "db_count": 22,
                "saldo_awal": 10000000,
                "saldo_akhir": 15000000,
            },
            "checksum_gate": {"passed": True, "discrepancy": 0},
            "account_detection": {
                "candidate_account_codes": ["11200HO", "11200MDN"],
                "is_shared_multi_branch": True,
            },
            "suggested_clarifications": [
                {
                    "label": "Rekonsiliasi Penuh (Gabung Cabang)",
                    "query": "Lakukan rekonsiliasi penuh semua cabang dari berkas ini",
                    "icon": "fa-check-double",
                }
            ],
        }

        state = create_initial_state(
            user_query="",
            attachments=[{"filename": "bca_apr.pdf", "file_path": "aiassistant_uploads/t1/bca_apr.pdf", "ext": ".pdf"}],
        )
        state["detected_domain"] = "accounting"
        state["detected_tool"] = "reconcile_bank_statement"
        state["verification_status"] = "VALID"
        state["tool_result"] = inspect_payload

        clarif_out = _build_pdf_inspect_clarification_output(state)
        self.assertIsNotNone(clarif_out)
        self.assertEqual(clarif_out["verification_status"], "CLARIFICATION")
        self.assertTrue(clarif_out["needs_clarification"])
        self.assertIn("11200HO", clarif_out["final_answer"])
        self.assertEqual(len(clarif_out["suggested_followups"]), 1)

        reconcile_payload = {
            "mode": "reconcile",
            "status": "DISCREPANCY_FOUND",
            "scope": "multi_branch_consolidated",
            "statement_meta": inspect_payload["statement_meta"],
            "checksum_gate": {"passed": True},
            "erp_accounts_used": [{"account_code": "11200HO", "company_name": "Head Office"}],
            "summary": {
                "bank_total_cr": 5000000,
                "erp_total_debit": 5000000,
                "difference_inflow": 0,
                "bank_total_db": 25000,
                "erp_total_credit": 0,
                "difference_outflow": 25000,
                "matched_count": 40,
                "unmatched_bank_inflows_count": 0,
                "unmatched_bank_outflows_count": 1,
                "unmatched_erp_count": 0,
            },
            "unmatched_bank_inflows": [],
            "unmatched_bank_outflows": [
                {
                    "date": "2026-04-30",
                    "amount": 25000,
                    "description": "BIAYA ADM BULANAN",
                    "root_cause": "BIAYA_ADMIN_BANK",
                    "diagnosis": "Biaya Administrasi Bank belum dijurnal",
                    "recommended_debit": "52001HO",
                    "recommended_credit": "11200HO",
                }
            ],
            "unmatched_erp_entries": [],
            "recommended_adjustments": [
                {
                    "no": 1,
                    "tanggal": "2026-04-30",
                    "tipe_penyesuaian": "Biaya Adm Bank",
                    "akun_debit": "52001HO",
                    "akun_kredit": "11200HO",
                    "nominal": 25000,
                }
            ],
        }
        formatted_reconcile = format_reconcile_bank_statement(
            "reconcile_bank_statement",
            reconcile_payload,
        )
        self.assertIn("BIAYA ADM BULANAN", formatted_reconcile)
        self.assertIn("25.000", formatted_reconcile)

        state["tool_result"] = reconcile_payload
        state["query_mode"] = "live_erp"
        synthesized = format_deterministic_answer(state)
        self.assertIn("Hasil Rekonsiliasi Rekening Koran BCA", synthesized)
        self.assertIn("BIAYA ADM BULANAN", synthesized)


if __name__ == "__main__":
    unittest.main()
