"""
Unit tests for VectorService (Dedicated PGVector + Google Gemini Embedding).
Tests embedding generation, query retrieval, text chunking, and fallback mechanisms.
"""
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.vector_service import VectorService
from scripts.ingest_knowledge import split_text_into_chunks


class TestVectorService(unittest.IsolatedAsyncioTestCase):
    """Test suite for VectorService with Gemini embeddings and PGVector mock."""

    def setUp(self):
        self.service = VectorService(
            db_url="postgresql://mock_user:mock_pass@localhost:5432/mock_vdb",
            gemini_api_key="mock_gemini_key",
            embedding_model="gemini-embedding-001",
        )

    def test_text_chunking(self):
        text = "Paragraf satu berisi kalimat.\n\nParagraf dua berisi teks penjelasan.\n\nParagraf tiga penutup."
        chunks = split_text_into_chunks(text, max_chunk_words=10, overlap_words=2)
        self.assertTrue(len(chunks) >= 1)
        self.assertIn("Paragraf", chunks[0])

    @patch("app.services.vector_service.VectorService._get_genai_client")
    async def test_get_embedding_gemini(self, mock_get_client):
        # Mock Google GenAI client
        mock_client = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.values = [0.01] * 768  # 768 dimensions
        mock_response = MagicMock()
        mock_response.embeddings = [mock_embedding]

        mock_client.models.embed_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        vector = await self.service.get_embedding("SOP retur barang cetak")
        self.assertIsNotNone(vector)
        self.assertEqual(len(vector), 768)
        self.assertEqual(vector[0], 0.01)

    async def test_get_embedding_empty_text(self):
        vector = await self.service.get_embedding("   ")
        self.assertIsNone(vector)

    @patch("app.services.vector_service.VectorService.get_embedding")
    @patch("asyncpg.create_pool", new_callable=AsyncMock)
    async def test_search_knowledge_success(self, mock_create_pool, mock_get_embedding):
        # 768-dim mock vector
        mock_get_embedding.return_value = [0.05] * 768

        # Mock asyncpg connection and rows
        mock_conn = AsyncMock()
        mock_row = {
            "id": 1,
            "title": "SOP Retur Barang v1.2",
            "category": "sop",
            "content": "Pengajuan maksimal 3x24 jam.",
            "similarity_score": 0.885,
            "metadata": '{"source_file": "sop_retur.md"}',
        }
        mock_conn.fetch.return_value = [mock_row]

        # Mock asyncpg pool and acquire context manager
        mock_pool = MagicMock()
        mock_pool._closed = False
        mock_pool.close = AsyncMock()

        mock_acquire_cm = MagicMock()
        mock_acquire_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire_cm.__aexit__ = AsyncMock(return_value=None)
        mock_pool.acquire.return_value = mock_acquire_cm
        mock_create_pool.return_value = mock_pool

        results = await self.service.search_knowledge(
            query="Bagaimana SOP retur barang?",
            company_id=1,
            top_k=3,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "SOP Retur Barang v1.2")
        self.assertEqual(results[0]["similarity_score"], 0.885)
        self.assertIn("source_file", results[0]["metadata"])
        mock_create_pool.assert_awaited_once()
        mock_pool.acquire.assert_called_once()

    async def test_search_knowledge_no_db_url(self):
        service_no_db = VectorService(db_url=None, gemini_api_key="mock_key")
        with patch.object(service_no_db, "get_embedding", return_value=[0.1] * 768):
            results = await service_no_db.search_knowledge("Query")
            self.assertEqual(results, [])

    @patch("asyncpg.create_pool", new_callable=AsyncMock)
    async def test_get_pool_reuse(self, mock_create_pool):
        mock_pool = MagicMock()
        mock_pool._closed = False
        mock_create_pool.return_value = mock_pool

        pool1 = await self.service.get_pool()
        pool2 = await self.service.get_pool()

        self.assertIs(pool1, mock_pool)
        self.assertIs(pool2, mock_pool)
        mock_create_pool.assert_awaited_once()

    @patch("asyncpg.create_pool", new_callable=AsyncMock)
    async def test_close_pool(self, mock_create_pool):
        mock_pool = MagicMock()
        mock_pool._closed = False
        mock_pool.close = AsyncMock()
        mock_create_pool.return_value = mock_pool

        pool = await self.service.get_pool()
        self.assertIsNotNone(pool)
        self.assertIsNotNone(self.service._pool)

        await self.service.close_pool()
        mock_pool.close.assert_awaited_once()
        self.assertIsNone(self.service._pool)

    @patch("app.services.vector_service.VectorService.get_embedding", return_value=[0.05] * 768)
    @patch.object(VectorService, "get_pool", return_value=None)
    async def test_search_knowledge_pool_failure(self, mock_get_pool, mock_get_embedding):
        results = await self.service.search_knowledge("Query")
        self.assertEqual(results, [])

    @patch("pgvector.asyncpg.register_vector", new_callable=AsyncMock)
    @patch("asyncpg.create_pool", new_callable=AsyncMock)
    async def test_get_pool_init_registers_vector(self, mock_create_pool, mock_register_vector):
        mock_pool = MagicMock()
        mock_pool._closed = False
        mock_create_pool.return_value = mock_pool

        await self.service.get_pool()
        self.assertTrue(mock_create_pool.awaited)
        init_cb = mock_create_pool.call_args.kwargs.get("init")
        self.assertIsNotNone(init_cb)

        mock_conn = AsyncMock()
        await init_cb(mock_conn)
        mock_register_vector.assert_awaited_once_with(mock_conn)

    async def test_get_embedding_lru_cache_hits_and_skips_none(self):
        """Repeated normalized queries reuse cached embeddings without redundant API calls; errors are never cached."""
        fake_resp = MagicMock()
        fake_emb = MagicMock()
        fake_emb.values = [0.25] * 768
        fake_resp.embeddings = [fake_emb]

        fake_client = MagicMock()
        fake_client.models.embed_content = MagicMock(return_value=fake_resp)
        self.service._genai_client = fake_client

        v1 = await self.service.get_embedding("Cek   Laba Rugi")
        v2 = await self.service.get_embedding("cek laba rugi")
        self.assertEqual(v1, [0.25] * 768)
        self.assertEqual(v2, [0.25] * 768)
        self.assertEqual(fake_client.models.embed_content.call_count, 1)

        # Mutating returned copy must not corrupt cache
        v1[0] = 9.9
        v3 = await self.service.get_embedding("cek laba rugi")
        self.assertAlmostEqual(v3[0], 0.25)

        # Errors returning None must not be cached
        fake_client.models.embed_content.side_effect = RuntimeError("429")
        err_vec = await self.service.get_embedding("query gagal")
        self.assertIsNone(err_vec)
        fake_client.models.embed_content.side_effect = None
        ok_vec = await self.service.get_embedding("query gagal")
        self.assertEqual(ok_vec, [0.25] * 768)


if __name__ == "__main__":
    unittest.main()
