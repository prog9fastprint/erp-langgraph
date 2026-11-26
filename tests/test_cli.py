"""
Unit tests for interactive CLI runner (cli.py).
Validates CLI loop commands: help/init banner, exit, company switching,
token setting, verbose toggle, session reset, and workflow invocation.
"""
import io
import unittest
from unittest.mock import patch

import cli


class TestCLIRunner(unittest.TestCase):
    """Test suite validating interactive CLI behavior and commands."""

    @patch("builtins.input", side_effect=["exit"])
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_cli_exit_immediately(self, mock_stdout, mock_input):
        cli.main()
        output = mock_stdout.getvalue()
        self.assertIn("FastPrint ERP — LangGraph StateGraph Assistant CLI", output)
        self.assertIn("Sampai jumpa! Sesi CLI diakhiri.", output)

    @patch("builtins.input", side_effect=["verbose", "co 2", "token test_jwt_token_12345", "reset", "quit"])
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_cli_special_commands(self, mock_stdout, mock_input):
        cli.main()
        output = mock_stdout.getvalue()
        self.assertIn("Mode Verbose: AKTIF", output)
        self.assertIn("Active Company ID diubah ke: 2", output)
        self.assertIn("User token diset", output)
        self.assertIn("Sesi percakapan di-reset", output)
        self.assertIn("Sampai jumpa!", output)

    @patch("cli.run_workflow_sync")
    @patch("builtins.input", side_effect=["Berapa tagihan vendor PT Sumber?", "q"])
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_cli_workflow_invocation(self, mock_stdout, mock_input, mock_run):
        mock_run.return_value = {
            "detected_domain": "accounting",
            "detected_tool": "fetch_vendor_invoices",
            "extracted_params": {"vendor_name": "PT Sumber"},
            "verification_status": "VALID",
            "retry_count": 0,
            "final_answer": "Ditemukan 1 tagihan sebesar Rp 5.000.000.",
        }

        cli.main()
        output = mock_stdout.getvalue()
        self.assertIn("[Route: accounting -> fetch_vendor_invoices | Status: VALID]", output)
        self.assertIn("Ditemukan 1 tagihan sebesar Rp 5.000.000.", output)
        mock_run.assert_called_once()

    @patch("cli.run_workflow_sync")
    @patch("builtins.input", side_effect=["verbose", "Cek saldo kas", "exit"])
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_cli_verbose_mode(self, mock_stdout, mock_input, mock_run):
        mock_run.return_value = {
            "detected_domain": "accounting",
            "detected_tool": "get_account_balance",
            "extracted_params": {},
            "verification_status": "VALID",
            "retry_count": 0,
            "tool_result": {"total_count": 1, "entries": []},
            "final_answer": "Saldo kas Rp 10.000.000.",
        }

        cli.main()
        output = mock_stdout.getvalue()
        self.assertIn("Mode Verbose: AKTIF", output)
        self.assertIn("--- [Rincian Eksekusi Pipeline StateGraph] ---", output)
        self.assertIn("Domain        : accounting", output)
        self.assertIn("Saldo kas Rp 10.000.000.", output)

    @patch("builtins.input", side_effect=KeyboardInterrupt)
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_cli_keyboard_interrupt(self, mock_stdout, mock_input):
        cli.main()
        output = mock_stdout.getvalue()
        self.assertIn("Sesi dihentikan.", output)


if __name__ == "__main__":
    unittest.main()
