"""
Unit tests for Synthesizer Node (TASK-4.5).
Validates deterministic markdown formatting for vendor invoices, calculations blocks,
account balances, inventory stock, product catalog, empty results, error conditions,
negative guardrails, AIMessage message history appending, and LLM fallback.
"""
import unittest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, HumanMessage

from app.graph.state import create_initial_state
from app.graph.nodes.synthesizer import (
    synthesizer_node,
    format_deterministic_answer,
    _format_rupiah,
)


class TestRupiahFormatter(unittest.TestCase):
    """Test standard Indonesian currency notation."""

    def test_format_rupiah_standard_floats(self):
        self.assertEqual(_format_rupiah(1500000), "Rp 1.500.000,00")
        self.assertEqual(_format_rupiah(25000.5), "Rp 25.000,50")
        self.assertEqual(_format_rupiah(0), "Rp 0,00")


class TestSynthesizerFormatting(unittest.TestCase):
    """Test suite validating markdown generation across various states."""

    def test_format_vendor_invoices_with_calculations(self):
        """Definition of Done: Calculations and transaction details properly formatted."""
        state = create_initial_state(user_query="Tagihan vendor PT Sumber Makmur")
        state["detected_domain"] = "accounting"
        state["detected_tool"] = "fetch_vendor_invoices"
        state["verification_status"] = "VALID"
        state["tool_result"] = {
            "total_count": 2,
            "entries": [
                {
                    "number": "BILL/2026/001",
                    "date": "2026-11-05",
                    "vendor_name": "PT Sumber Makmur",
                    "total_debit": 5000000.0,
                    "status": "unpaid",
                },
                {
                    "number": "BILL/2026/002",
                    "date": "2026-11-12",
                    "vendor_name": "PT Sumber Makmur",
                    "total_debit": 3500000.0,
                    "status": "unpaid",
                },
            ],
            "calculations": {
                "total_unpaid": 8500000.0,
                "total_paid": 0.0,
                "grand_total": 8500000.0,
            },
        }

        output = format_deterministic_answer(state)
        self.assertIn("BILL/2026/001", output)
        self.assertIn("PT Sumber Makmur", output)
        self.assertIn("Rp 8.500.000,00", output)
        self.assertIn("Ringkasan Kalkulasi Total", output)

    def test_format_vendor_invoices_empty_entries(self):
        """Empty invoice entries renders fallback message instead of empty heading."""
        state = create_initial_state(user_query="Tagihan vendor PT Sumber Makmur")
        state["detected_domain"] = "accounting"
        state["detected_tool"] = "fetch_vendor_invoices"
        state["verification_status"] = "VALID"
        state["tool_result"] = {
            "total_count": 0,
            "entries": [],
            "calculations": {},
        }
        output = format_deterministic_answer(state)
        self.assertIn("Tidak ada transaksi tercatat pada periode ini.", output)

    def test_format_account_balance(self):
        """Account balance summary formatting."""
        state = create_initial_state(user_query="Saldo kas")
        state["detected_domain"] = "accounting"
        state["detected_tool"] = "get_account_balance"
        state["verification_status"] = "VALID"
        state["tool_result"] = {
            "accounts": [
                {
                    "account_code": "111100",
                    "account_name": "Kas Utama",
                    "total_debit": 150000000.0,
                    "total_credit": 50000000.0,
                    "net_balance": 100000000.0,
                    "balance_position": "DEBIT",
                }
            ]
        }

        output = format_deterministic_answer(state)
        self.assertIn("Kas Utama", output)
        self.assertIn("111100", output)
        self.assertIn("Rp 100.000.000,00", output)

    def test_format_stock_level(self):
        """Inventory stock level formatting."""
        state = create_initial_state(user_query="Cek stok kertas A4")
        state["detected_domain"] = "inventory"
        state["detected_tool"] = "get_stock_level"
        state["verification_status"] = "VALID"
        state["tool_result"] = {
            "code": "SKU-PAPER-A4",
            "name": "Kertas HVS A4 80gsm",
            "company_name": "FastPrint Surabaya",
            "qty_on_hand": 250.0,
        }

        output = format_deterministic_answer(state)
        self.assertIn("SKU-PAPER-A4", output)
        self.assertIn("Kertas HVS A4 80gsm", output)
        self.assertIn("250.0 unit", output)

    def test_format_product_info(self):
        """Product catalog info formatting."""
        state = create_initial_state(user_query="Detail produk printer")
        state["detected_domain"] = "inventory"
        state["detected_tool"] = "get_product_info"
        state["verification_status"] = "VALID"
        state["tool_result"] = {
            "products": [
                {
                    "code": "PRN-EPSON-L3110",
                    "name": "Epson EcoTank L3110",
                    "price": 2450000.0,
                }
            ]
        }

        output = format_deterministic_answer(state)
        self.assertIn("Epson EcoTank L3110", output)
        self.assertIn("Rp 2.450.000,00", output)

    def test_format_empty_status(self):
        """Empty status provides polite explanation and retry notes."""
        state = create_initial_state(user_query="Tagihan vendor Khayalan")
        state["detected_domain"] = "accounting"
        state["detected_tool"] = "fetch_vendor_invoices"
        state["verification_status"] = "EMPTY"
        state["extracted_params"] = {"vendor_name": "Khayalan"}

        output = format_deterministic_answer(state)
        self.assertIn("Data Tidak Ditemukan", output)
        self.assertIn("Khayalan", output)
        self.assertIn("maksimal 2x", output)

    def test_format_empty_status_multi_tool_displays_all_filters(self):
        """Empty status in multi-tool execution includes filter hints from all active tools."""
        state = create_initial_state(user_query="Cek stok Kertas A4 dan tagihan vendor PT Maju")
        state["is_multi_tool"] = True
        state["verification_status"] = "EMPTY"
        state["extracted_params"] = {"code": "KERTAS-A4"}
        state["active_tools"] = [
            {"tool": "get_stock_level", "domain": "inventory", "params": {"code": "KERTAS-A4"}},
            {"tool": "fetch_vendor_invoices", "domain": "accounting", "params": {"vendor_name": "PT Maju"}},
        ]

        output = format_deterministic_answer(state)
        self.assertIn("Data Tidak Ditemukan", output)
        self.assertIn("KERTAS-A4", output)
        self.assertIn("PT Maju", output)

    def test_format_error_status(self):
        """Error status provides polite technical troubleshooting advice."""
        state = create_initial_state(user_query="Cek jurnal")
        state["verification_status"] = "ERROR"
        state["error_message"] = "JWT token expired (HTTP 401)"

        output = format_deterministic_answer(state)
        self.assertIn("Kendala Akses Data ERP", output)
        self.assertIn("JWT token expired", output)
        self.assertIn("login ulang", output)

    def test_format_error_product_not_found(self):
        """PRODUCT_NOT_FOUND error provides polite product not found message instead of JWT message."""
        state = create_initial_state(user_query="aku mau tanya terkait stok BARANG-PALSU")
        state["detected_domain"] = "inventory"
        state["detected_tool"] = "get_stock_level"
        state["verification_status"] = "ERROR"
        state["error_message"] = 'Not Found (404): {"success": false, "data": null, "error": {"code": "PRODUCT_NOT_FOUND", "message": "Product variant not found for given criteria.", "details": {}}}'
        state["extracted_params"] = {"code": "BARANG-PALSU"}

        output = format_deterministic_answer(state)
        self.assertIn("Produk Tidak Ditemukan", output)
        self.assertIn("BARANG-PALSU", output)
        self.assertNotIn("Kendala Akses Data ERP", output)
        self.assertNotIn("token autentikasi (JWT) masih aktif", output)

    def test_format_negative_guardrail_customer_inquiry(self):
        """Customer inquiry triggers guardrail message redirecting to Sales module."""
        state = create_initial_state(user_query="Cek tagihan pelanggan Budi")
        state["detected_domain"] = "general"
        state["detected_tool"] = None

        output = format_deterministic_answer(state)
        self.assertIn("Tagihan Pelanggan (Customer)", output)
        self.assertIn("Penjualan (Sales / Invoicing)", output)


class TestSynthesizerNodeExecution(unittest.TestCase):
    """Test suite validating synthesizer_node node updates and LangGraph memory appending."""

    def test_synthesizer_appends_ai_message_to_state(self):
        state = create_initial_state(user_query="Halo")
        state["detected_domain"] = "general"
        state["detected_tool"] = None

        update = synthesizer_node(state)
        self.assertIn("final_answer", update)
        self.assertIn("messages", update)
        self.assertEqual(len(update["messages"]), 1)
        self.assertIsInstance(update["messages"][0], AIMessage)
        self.assertEqual(update["messages"][0].content, update["final_answer"])

    def test_synthesizer_uses_llm_when_provided(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Laporan tagihan vendor berhasil diringkas oleh AI."
        mock_llm.invoke.return_value = mock_response

        state = create_initial_state(user_query="Tagihan vendor PT Sumber Makmur")
        state["detected_domain"] = "accounting"
        state["detected_tool"] = "fetch_vendor_invoices"
        state["verification_status"] = "VALID"
        state["tool_result"] = {"total_count": 1, "entries": [{"number": "BILL/1"}]}

        update = synthesizer_node(state, llm=mock_llm)
        self.assertEqual(update["final_answer"], "Laporan tagihan vendor berhasil diringkas oleh AI.")

    def test_format_no_previous_query_to_retry(self):
        """Empty history retry displays specific helpful guidance instead of generic greeting."""
        state = create_initial_state(user_query="coba lagi")
        state["detected_domain"] = "general"
        state["detected_tool"] = None
        state["error_message"] = "NO_PREVIOUS_QUERY_TO_RETRY"

        output = format_deterministic_answer(state)
        self.assertIn("Tidak Ada Permintaan Sebelumnya", output)
        self.assertIn("Belum ada riwayat transaksi atau pertanyaan sebelumnya", output)

    def test_retry_callout_header_prefixed_to_answer(self):
        """Synthesizer node prepends retry callout header when retry_instruction is set."""
        state = create_initial_state(user_query="coba lagi")
        state["detected_domain"] = "inventory"
        state["detected_tool"] = "get_stock_level"
        state["verification_status"] = "VALID"
        state["retry_instruction"] = "Mengulang permintaan: Cek stok Art Paper 150g"
        state["tool_result"] = [{
            "code": "AP-150",
            "name": "Art Paper 150g",
            "qty_on_hand": 50.0,
        }]

        update = synthesizer_node(state)
        self.assertIn("> 🔄 **Mengulang permintaan: Cek stok Art Paper 150g**", update["final_answer"])
        self.assertIn("AP-150", update["final_answer"])

    def test_synthesizer_extracts_llm_followups_from_codeblock(self):
        """Synthesizer extracts JSON followups generated by LLM and strips codeblock from markdown."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = (
            "Berikut ringkasan stok produk HVS A4:\n- Total: 50 unit\n\n"
            "```json:followups\n"
            "[\n"
            '  {"label": "Riwayat Kartu Stok", "query": "Cek riwayat kartu stok HVS A4", "icon": "fa-clock-rotate-left"},\n'
            '  {"label": "Stok Cabang Lain", "query": "Cek stok HVS A4 di cabang Bandung", "icon": "fa-warehouse"}\n'
            "]\n"
            "```"
        )
        mock_llm.invoke.return_value = mock_response

        state = create_initial_state(user_query="Cek stok HVS A4")
        state["detected_domain"] = "inventory"
        state["detected_tool"] = "get_stock_level"
        state["verification_status"] = "VALID"
        state["extracted_params"] = {"code": "HVS-A4"}

        update = synthesizer_node(state, llm=mock_llm)
        self.assertNotIn("```json:followups", update["final_answer"])
        self.assertIn("Berikut ringkasan stok produk HVS A4:", update["final_answer"])
        self.assertIn("suggested_followups", update)
        self.assertEqual(len(update["suggested_followups"]), 2)
        self.assertEqual(update["suggested_followups"][0]["label"], "Riwayat Kartu Stok")
        self.assertEqual(update["suggested_followups"][0]["query"], "Cek riwayat kartu stok HVS A4")
        self.assertEqual(update["suggested_followups"][1]["label"], "Stok Cabang Lain")

    def test_synthesizer_extracts_llm_followups_from_object_block_with_header(self):
        """Synthesizer extracts raw JSON object followups with 'Rekomendasi lanjutan' header and cleans markdown."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = (
            "Berikut ringkasan stok produk FP KARET STEMPEL FLASH BULAT 42 (SKU 91010600):\n"
            "- Total Stok: 1.500 pcs\n\n"
            "Rekomendasi lanjutan (READ-ONLY)\n"
            "{\n"
            '  "followups": [\n'
            '    {\n'
            '      "label": "📦 Distribusi Stok",\n'
            '      "query": "Apakah distribusi stok SKU 91010600 di setiap gudang/rak?",\n'
            '      "icon": "📦"\n'
            '    },\n'
            '    {\n'
            '      "label": "⏰ Tagihan Overdue",\n'
            '      "query": "Apakah ada tagihan overdue untuk produk SKU 91010600?",\n'
            '      "icon": "⏰"\n'
            '    },\n'
            '    {\n'
            '      "label": "🏦 Rekonsiliasi Bank",\n'
            '      "query": "Apakah ada selisih rekonsiliasi rekening bank terkait SKU 91010600?",\n'
            '      "icon": "🏦"\n'
            '    }\n'
            '  ]\n'
            "}"
        )
        mock_llm.invoke.return_value = mock_response

        state = create_initial_state(user_query="Cek stok SKU 91010600")
        state["detected_domain"] = "inventory"
        state["detected_tool"] = "get_stock_level"
        state["verification_status"] = "VALID"
        state["extracted_params"] = {"code": "91010600"}

        update = synthesizer_node(state, llm=mock_llm)
        self.assertNotIn("Rekomendasi lanjutan", update["final_answer"])
        self.assertNotIn('"followups"', update["final_answer"])
        self.assertIn("Berikut ringkasan stok produk FP KARET STEMPEL FLASH BULAT 42 (SKU 91010600):", update["final_answer"])
        self.assertIn("suggested_followups", update)
        self.assertEqual(len(update["suggested_followups"]), 3)
        self.assertEqual(update["suggested_followups"][0]["label"], "Distribusi Stok")
        self.assertEqual(update["suggested_followups"][0]["icon"], "fa-arrow-right")
        self.assertEqual(update["suggested_followups"][1]["label"], "Tagihan Overdue")
        self.assertEqual(update["suggested_followups"][1]["icon"], "fa-arrow-right")
        self.assertEqual(update["suggested_followups"][2]["label"], "Rekonsiliasi Bank")
        self.assertEqual(update["suggested_followups"][2]["icon"], "fa-arrow-right")

    def test_deterministic_followups_inventory_with_sku(self):
        """Deterministic followups generate SKU-specific queries when product code exists."""
        state = create_initial_state(user_query="Cek stok PRN-L3110")
        state["detected_domain"] = "inventory"
        state["detected_tool"] = "get_stock_level"
        state["verification_status"] = "VALID"
        state["extracted_params"] = {"code": "PRN-L3110"}

        update = synthesizer_node(state)
        self.assertIn("suggested_followups", update)
        labels = [f["label"] for f in update["suggested_followups"]]
        self.assertTrue(any("PRN-L3110" in l for l in labels))
        queries = [f["query"] for f in update["suggested_followups"]]
        self.assertTrue(any("PRN-L3110" in q for q in queries))

    def test_deterministic_followups_accounting_with_vendor(self):
        """Deterministic followups generate vendor-specific queries when vendor name exists."""
        state = create_initial_state(user_query="Tagihan vendor PT Sumber Makmur")
        state["detected_domain"] = "accounting"
        state["detected_tool"] = "fetch_vendor_invoices"
        state["verification_status"] = "VALID"
        state["extracted_params"] = {"vendor_name": "PT Sumber Makmur"}

        update = synthesizer_node(state)
        self.assertIn("suggested_followups", update)
        queries = [f["query"] for f in update["suggested_followups"]]
        self.assertTrue(any("PT Sumber Makmur" in q for q in queries))

    def test_deterministic_followups_empty_status(self):
        """Empty verification status generates widening and scope advice followups."""
        state = create_initial_state(user_query="Cek transaksi 2026-01-01")
        state["detected_domain"] = "accounting"
        state["verification_status"] = "EMPTY"
        state["extracted_params"] = {"date_from": "2026-01-01", "date_to": "2026-01-02"}

        update = synthesizer_node(state)
        self.assertIn("suggested_followups", update)
        labels = [f["label"] for f in update["suggested_followups"]]
        self.assertIn("Perluas Tanggal", labels)


if __name__ == "__main__":
    unittest.main()
