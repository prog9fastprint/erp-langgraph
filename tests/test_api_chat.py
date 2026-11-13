"""
Unit tests for FastAPI Chat Endpoint (/api/v1/chat) and Chat Schemas (R1).
Validates ChatRequest and ChatResponse models, parameter propagation,
validation error handling (422), workflow error propagation, and 500 error handling.
"""
from enum import Enum
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app, ChatRequest, ChatResponse


class DummyVerificationEnum(str, Enum):
    """Enum simulating verification status object."""
    VALID = "VALID"
    EMPTY = "EMPTY"
    ERROR = "ERROR"


class TestChatSchemas(unittest.TestCase):
    """Test suite for ChatRequest and ChatResponse Pydantic schemas."""

    def test_chat_request_defaults(self):
        """ChatRequest sets proper default values."""
        req = ChatRequest(query="Cek saldo kas")
        self.assertEqual(req.query, "Cek saldo kas")
        self.assertEqual(req.company_id, 1)
        self.assertIsNone(req.user_token)
        self.assertIsNone(req.thread_id)

    def test_chat_request_explicit_fields(self):
        """ChatRequest accepts explicit company_id, user_token, and thread_id."""
        req = ChatRequest(
            query="Cek stok kertas A4",
            company_id=3,
            user_token="jwt_secret_token",
            thread_id="thread_session_789",
        )
        self.assertEqual(req.query, "Cek stok kertas A4")
        self.assertEqual(req.company_id, 3)
        self.assertEqual(req.user_token, "jwt_secret_token")
        self.assertEqual(req.thread_id, "thread_session_789")

    def test_chat_request_empty_query_fails_validation(self):
        """Empty query raises validation error."""
        with self.assertRaises(ValidationError):
            ChatRequest(query="")

    def test_chat_request_whitespace_query_fails_validation(self):
        """Whitespace-only query raises validation error."""
        with self.assertRaises(ValidationError):
            ChatRequest(query="    ")

    def test_chat_request_extra_fields_ignored(self):
        """Extra unknown fields are ignored without raising error."""
        req = ChatRequest(query="Halo", unknown_field="ignored")
        self.assertEqual(req.query, "Halo")
        self.assertFalse(hasattr(req, "unknown_field"))

    def test_chat_response_serialization(self):
        """ChatResponse serializes correctly to dictionary."""
        resp = ChatResponse(
            final_answer="Total stok: 50 unit",
            detected_domain="inventory",
            verification_status="VALID",
            error_message=None,
        )
        data = resp.model_dump()
        self.assertEqual(data["final_answer"], "Total stok: 50 unit")
        self.assertEqual(data["detected_domain"], "inventory")
        self.assertEqual(data["verification_status"], "VALID")
        self.assertIsNone(data["error_message"])

    def test_chat_response_defaults(self):
        """ChatResponse fields default to None."""
        resp = ChatResponse()
        self.assertIsNone(resp.final_answer)
        self.assertIsNone(resp.detected_domain)
        self.assertIsNone(resp.verification_status)
        self.assertIsNone(resp.error_message)


class TestChatEndpoint(unittest.TestCase):
    """Test suite for FastAPI POST /api/v1/chat endpoint."""

    def setUp(self):
        self.client = TestClient(app)

    def test_health_check_endpoint(self):
        """GET /health returns 200 OK and healthy status."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "healthy"})

    @patch("app.main.arun_workflow", new_callable=AsyncMock)
    def test_chat_success(self, mock_workflow):
        """POST /api/v1/chat executes workflow and returns 200 with structured ChatResponse."""
        mock_workflow.return_value = {
            "final_answer": "Saldo kas PT FastPrint per 31 Desember adalah Rp 25.000.000",
            "detected_domain": "accounting",
            "verification_status": "VALID",
            "error_message": None,
        }

        response = self.client.post(
            "/api/v1/chat",
            json={"query": "Berapa saldo kas bulan ini?"},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["final_answer"], "Saldo kas PT FastPrint per 31 Desember adalah Rp 25.000.000")
        self.assertEqual(data["detected_domain"], "accounting")
        self.assertEqual(data["verification_status"], "VALID")
        self.assertIsNone(data["error_message"])

        mock_workflow.assert_awaited_once_with(
            user_query="Berapa saldo kas bulan ini?",
            user_token=None,
            company_id=1,
            thread_id=None,
            company_id_explicit=False,
        )

    @patch("app.main.arun_workflow", new_callable=AsyncMock)
    def test_chat_returns_suggested_followups(self, mock_workflow):
        """POST /api/v1/chat includes suggested_followups in ChatResponse."""
        mock_workflow.return_value = {
            "final_answer": "Stok barang tersedia 120 pcs",
            "detected_domain": "inventory",
            "verification_status": "VALID",
            "error_message": None,
            "suggested_followups": [
                {"label": "Riwayat Mutasi", "query": "Cek riwayat kartu stok", "icon": "fa-clock"}
            ],
        }

        response = self.client.post(
            "/api/v1/chat",
            json={"query": "Cek stok SKU-8889"},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("suggested_followups", data)
        self.assertEqual(len(data["suggested_followups"]), 1)
        self.assertEqual(data["suggested_followups"][0]["label"], "Riwayat Mutasi")

    @patch("app.main.arun_workflow", new_callable=AsyncMock)
    def test_chat_parameter_forwarding(self, mock_workflow):
        """Explicit parameters (company_id, user_token, thread_id) are forwarded to arun_workflow."""
        mock_workflow.return_value = {
            "final_answer": "Stok barang tersedia 120 pcs",
            "detected_domain": "inventory",
            "verification_status": "VALID",
            "error_message": None,
        }

        payload = {
            "query": "Cek stok SKU-8889 di gudang utama",
            "company_id": 2,
            "user_token": "bearer_jwt_auth_xyz",
            "thread_id": "thread_custom_abc_456",
        }
        response = self.client.post("/api/v1/chat", json=payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["final_answer"], "Stok barang tersedia 120 pcs")
        self.assertEqual(data["detected_domain"], "inventory")

        mock_workflow.assert_awaited_once_with(
            user_query="Cek stok SKU-8889 di gudang utama",
            user_token="bearer_jwt_auth_xyz",
            company_id=2,
            thread_id="thread_custom_abc_456",
            company_id_explicit=True,
        )

    @patch("app.main.arun_workflow", new_callable=AsyncMock)
    def test_chat_history_parameter_forwarding(self, mock_workflow):
        """History parameter in ChatRequest is forwarded to arun_workflow."""
        mock_workflow.return_value = {
            "final_answer": "Stok barang tersedia",
            "detected_domain": "inventory",
            "verification_status": "VALID",
            "error_message": None,
        }
        history = [
            {"role": "user", "content": "cek stok tinta"},
            {"role": "assistant", "content": "Tinta ready 100 unit"},
        ]
        payload = {
            "query": "coba cek lagi",
            "company_id": 1,
            "thread_id": "thread_hydrated_123",
            "history": history,
        }
        response = self.client.post("/api/v1/chat", json=payload)
        self.assertEqual(response.status_code, 200)

        mock_workflow.assert_awaited_once()
        call_kwargs = mock_workflow.await_args.kwargs
        self.assertEqual(call_kwargs.get("user_query"), "coba cek lagi")
        self.assertEqual(call_kwargs.get("thread_id"), "thread_hydrated_123")
        self.assertIsNotNone(call_kwargs.get("history"))
        self.assertEqual(len(call_kwargs["history"]), 2)

    def test_chat_empty_query_validation_422(self):
        """Empty query string returns HTTP 422 Unprocessable Entity."""
        response = self.client.post("/api/v1/chat", json={"query": ""})
        self.assertEqual(response.status_code, 422)

    def test_chat_whitespace_query_validation_422(self):
        """Whitespace query string returns HTTP 422 Unprocessable Entity."""
        response = self.client.post("/api/v1/chat", json={"query": "    \t\n  "})
        self.assertEqual(response.status_code, 422)

    def test_chat_missing_query_validation_422(self):
        """Missing query payload returns HTTP 422 Unprocessable Entity."""
        response = self.client.post("/api/v1/chat", json={})
        self.assertEqual(response.status_code, 422)

    def test_chat_invalid_company_id_type_422(self):
        """Non-integer company_id returns HTTP 422 Unprocessable Entity."""
        response = self.client.post(
            "/api/v1/chat",
            json={"query": "cek stok", "company_id": "invalid_company"},
        )
        self.assertEqual(response.status_code, 422)

    @patch("app.main.arun_workflow", new_callable=AsyncMock)
    def test_chat_workflow_error_state(self, mock_workflow):
        """Workflow error state is returned cleanly with 200 status and error_message preserved."""
        mock_workflow.return_value = {
            "final_answer": "Terjadi kendala autentikasi ke server ERP.",
            "detected_domain": "accounting",
            "verification_status": "ERROR",
            "error_message": "Sesi token JWT kadaluarsa (HTTP 401)",
        }

        response = self.client.post(
            "/api/v1/chat",
            json={"query": "Tampilkan jurnal penyesuaian"},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["verification_status"], "ERROR")
        self.assertEqual(data["error_message"], "Sesi token JWT kadaluarsa (HTTP 401)")
        self.assertIn("Terjadi kendala", data["final_answer"])

    @patch("app.main.arun_workflow", new_callable=AsyncMock)
    def test_chat_workflow_enum_verification_status(self, mock_workflow):
        """Enum verification_status in state is converted to string in response."""
        mock_workflow.return_value = {
            "final_answer": "Valid response",
            "detected_domain": "inventory",
            "verification_status": DummyVerificationEnum.VALID,
            "error_message": None,
        }

        response = self.client.post(
            "/api/v1/chat",
            json={"query": "Stok barang A"},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["verification_status"], "VALID")

    @patch("app.main.arun_workflow", new_callable=AsyncMock)
    def test_chat_workflow_internal_server_error_500(self, mock_workflow):
        """Unexpected exception in arun_workflow raises HTTP 500 Internal Server Error."""
        mock_workflow.side_effect = RuntimeError("OpenRouter API connection timeout")

        response = self.client.post(
            "/api/v1/chat",
            json={"query": "Berapa piutang dagang?"},
        )

        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertIn("Chat workflow failed", data["detail"])
        self.assertIn("OpenRouter API connection timeout", data["detail"])

    @patch("app.main.arun_workflow", new_callable=AsyncMock)
    def test_chat_multi_turn_thread_propagation(self, mock_workflow):
        """Sequential queries with thread_id propagate the same thread_id across requests."""
        mock_workflow.return_value = {
            "final_answer": "Jawaban multi-turn",
            "detected_domain": "general",
            "verification_status": "VALID",
            "error_message": None,
        }

        thread_id = "conversation_session_999"

        # Turn 1
        res1 = self.client.post(
            "/api/v1/chat",
            json={"query": "Siapa nama CEO FastPrint?", "thread_id": thread_id},
        )
        self.assertEqual(res1.status_code, 200)
        mock_workflow.assert_awaited_with(
            user_query="Siapa nama CEO FastPrint?",
            user_token=None,
            company_id=1,
            thread_id=thread_id,
            company_id_explicit=False,
        )

        # Turn 2
        res2 = self.client.post(
            "/api/v1/chat",
            json={"query": "Dimana kantor cabangnya?", "thread_id": thread_id},
        )
        self.assertEqual(res2.status_code, 200)
        mock_workflow.assert_awaited_with(
            user_query="Dimana kantor cabangnya?",
            user_token=None,
            company_id=1,
            thread_id=thread_id,
            company_id_explicit=False,
        )


if __name__ == "__main__":
    unittest.main()
