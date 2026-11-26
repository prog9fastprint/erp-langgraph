"""
Dedicated Vector Service & Google Gemini Embedding for FastPrint ERP.
Provides semantic embedding generation via Google GenAI SDK (gemini-embedding-001)
and vector similarity retrieval using asyncpg and pgvector on a dedicated PostgreSQL instance.
"""
import asyncio
from collections import OrderedDict
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple
import weakref

from app.core.config import settings

logger = logging.getLogger("vector_service")

# Dedicated table name for vector chunks
KNOWLEDGE_TABLE_NAME = "ai_knowledge_chunks"


class VectorService:
    """
    Dedicated PGVector service interacting with Google Gemini Embedding model.
    Supports asyncpg connection pooling and pgvector HNSW distance queries.
    """

    def __init__(
        self,
        db_url: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        self.db_url = db_url or settings.VECTOR_DB_URL
        self.gemini_api_key = gemini_api_key or settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
        self.embedding_model = embedding_model or settings.GEMINI_EMBEDDING_MODEL
        self._genai_client = None
        self._pools_by_loop: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, Any]" = (
            weakref.WeakKeyDictionary()
        )
        self._fallback_pool: Optional[Any] = None
        self._pool_locks_by_loop: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = (
            weakref.WeakKeyDictionary()
        )
        self._fallback_lock: Optional[asyncio.Lock] = None
        self._embedding_cache: "OrderedDict[Tuple[str, str], List[float]]" = OrderedDict()
        self._embedding_cache_max_size: int = 512

    @property
    def _pool(self) -> Optional[Any]:
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            if self._fallback_pool is not None:
                return self._fallback_pool
            for candidate in reversed(list(self._pools_by_loop.values())):
                if candidate is not None and not getattr(candidate, "_closed", False):
                    return candidate
            return None
        return self._pools_by_loop.get(running_loop, self._fallback_pool)

    @_pool.setter
    def _pool(self, value: Optional[Any]) -> None:
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._fallback_pool = value
            if value is None:
                self._pools_by_loop.clear()
            return
        if value is None:
            self._pools_by_loop.pop(running_loop, None)
            self._fallback_pool = None
        else:
            self._pools_by_loop[running_loop] = value

    def _get_pool_lock(self) -> asyncio.Lock:
        """Lazily initialize pool lock bound to the currently running event loop."""
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            if self._fallback_lock is None:
                self._fallback_lock = asyncio.Lock()
            return self._fallback_lock

        lock = self._pool_locks_by_loop.get(running_loop)
        if lock is None:
            lock = asyncio.Lock()
            self._pool_locks_by_loop[running_loop] = lock
        return lock

    async def get_pool(self) -> Optional[Any]:
        """
        Get or initialize the dedicated asyncpg connection pool for PGVector.
        Registers pgvector extension types on each newly established connection.
        """
        if not self.db_url:
            return None

        existing_pool = self._pool
        if existing_pool is not None and not getattr(existing_pool, "_closed", False):
            return existing_pool

        async with self._get_pool_lock():
            existing_pool = self._pool
            if existing_pool is not None and not getattr(existing_pool, "_closed", False):
                return existing_pool

            try:
                import asyncpg
                from pgvector.asyncpg import register_vector

                clean_url = self.db_url.replace("postgresql+asyncpg://", "postgresql://")

                async def _init_connection(conn):
                    await register_vector(conn)

                created_pool = await asyncpg.create_pool(
                    clean_url,
                    min_size=1,
                    max_size=10,
                    command_timeout=settings.REQUEST_TIMEOUT_SEC,
                    init=_init_connection,
                )
                self._pool = created_pool
                logger.info("Dedicated PGVector asyncpg connection pool initialized successfully.")
                return created_pool
            except Exception as exc:
                logger.error(f"Failed to create asyncpg connection pool: {exc}")
                return None

    async def close_pool(self) -> None:
        """
        Gracefully close all connections in the active asyncpg pool.
        """
        async with self._get_pool_lock():
            active_pool = self._pool
            if active_pool is not None:
                try:
                    if not getattr(active_pool, "_closed", False):
                        await active_pool.close()
                except Exception as exc:
                    logger.warning(f"Error while closing asyncpg connection pool: {exc}")
                finally:
                    self._pool = None
                logger.info("Dedicated PGVector asyncpg connection pool closed.")

    async def initialize_table(self) -> bool:
        """
        Create ai_knowledge_chunks table and HNSW index if they do not exist.
        """
        pool = await self.get_pool()
        if pool is None:
            logger.warning("PGVector connection pool unavailable. Cannot initialize knowledge table.")
            return False

        ddl = f"""
            CREATE EXTENSION IF NOT EXISTS vector;

            CREATE TABLE IF NOT EXISTS {KNOWLEDGE_TABLE_NAME} (
                id SERIAL PRIMARY KEY,
                company_id INT NOT NULL DEFAULT 1,
                title VARCHAR(255) NOT NULL,
                category VARCHAR(64) NOT NULL DEFAULT 'sop',
                content TEXT NOT NULL,
                metadata JSONB DEFAULT '{{}}'::jsonb,
                is_active BOOLEAN DEFAULT TRUE,
                embedding vector(768) NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_{KNOWLEDGE_TABLE_NAME}_embedding 
            ON {KNOWLEDGE_TABLE_NAME} USING hnsw (embedding vector_cosine_ops);

            CREATE INDEX IF NOT EXISTS idx_{KNOWLEDGE_TABLE_NAME}_company_active 
            ON {KNOWLEDGE_TABLE_NAME} (company_id, is_active);

            CREATE INDEX IF NOT EXISTS idx_{KNOWLEDGE_TABLE_NAME}_company_cat 
            ON {KNOWLEDGE_TABLE_NAME} (company_id, category);
        """
        try:
            async with pool.acquire() as conn:
                await conn.execute(ddl)
            logger.info(f"Table '{KNOWLEDGE_TABLE_NAME}' and HNSW index verified successfully.")
            return True
        except Exception as exc:
            logger.error(f"Failed to initialize table '{KNOWLEDGE_TABLE_NAME}': {exc}")
            return False

    def _get_genai_client(self):
        """Lazy load and initialize Google GenAI Client."""
        if self._genai_client is None:
            if not self.gemini_api_key:
                logger.warning("GEMINI_API_KEY is not set. Gemini embeddings will not be available.")
                return None
            try:
                from google import genai
                self._genai_client = genai.Client(api_key=self.gemini_api_key)
            except Exception as exc:
                logger.error(f"Failed to initialize Google GenAI Client: {exc}")
                return None
        return self._genai_client

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate 768-dimensional float embedding using Google Gemini API.

        Args:
            text: Input string to embed.

        Returns:
            List of 768 float numbers, or None if client/API unavailable.
        """
        if not text or not text.strip():
            return None

        normalized_text = " ".join(text.strip().lower().split())
        cache_key = (self.embedding_model, normalized_text)
        cached_vec = self._embedding_cache.get(cache_key)
        if cached_vec is not None:
            self._embedding_cache.move_to_end(cache_key)
            return list(cached_vec)

        client = self._get_genai_client()
        if client is None:
            return None

        try:
            from google.genai import types
            config = types.EmbedContentConfig(output_dimensionality=768)
            response = await asyncio.to_thread(
                client.models.embed_content,
                model=self.embedding_model,
                contents=text.strip(),
                config=config,
            )
            if response and hasattr(response, "embeddings") and response.embeddings:
                vec = list(response.embeddings[0].values)
                self._embedding_cache[cache_key] = vec
                self._embedding_cache.move_to_end(cache_key)
                while len(self._embedding_cache) > self._embedding_cache_max_size:
                    self._embedding_cache.popitem(last=False)
                return list(vec)
        except Exception as exc:
            logger.error(f"Error calling Gemini embed_content API: {exc}")
            return None

        return None

    async def search_knowledge(
        self,
        query: str,
        company_id: int = 1,
        top_k: int = 3,
        category: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity search on Dedicated PostgreSQL PGVector database.

        Args:
            query: User search query string.
            company_id: Target ERP company ID for multi-tenant scoping.
            top_k: Number of nearest neighbor chunks to return.
            category: Optional document category filter ('sop', 'product_spec', etc.).
            threshold: Minimum cosine similarity score (defaults to settings.RAG_SIMILARITY_THRESHOLD).

        Returns:
            List of matching chunk dictionaries with title, content, similarity_score, metadata.
        """
        if not query or not query.strip():
            return []

        min_score = threshold if threshold is not None else float(settings.RAG_SIMILARITY_THRESHOLD)

        # 1. Generate query embedding via Gemini
        query_vector = await self.get_embedding(query)
        if query_vector is None:
            logger.info("Gemini query embedding unavailable; returning empty knowledge context.")
            return []

        # 2. Query Dedicated PostgreSQL via connection pool
        if not self.db_url:
            logger.info("VECTOR_DB_URL not configured. Returning empty vector search results.")
            return []

        pool = await self.get_pool()
        if pool is None:
            logger.warning("Dedicated PGVector connection pool unavailable. Returning empty results.")
            return []

        try:
            # Format vector list for pgvector (asyncpg accepts list of floats)
            vector_arr = [float(v) for v in query_vector]

            base_sql = f"""
                SELECT 
                    id,
                    title,
                    category,
                    content,
                    metadata,
                    1 - (embedding <=> $1) AS similarity_score
                FROM {KNOWLEDGE_TABLE_NAME}
                WHERE company_id = $2 AND is_active = TRUE
            """
            params = [vector_arr, int(company_id)]
            param_idx = 3

            if category:
                base_sql += f" AND category = ${param_idx}"
                params.append(category)
                param_idx += 1

            base_sql += f" AND (1 - (embedding <=> $1)) >= ${param_idx}"
            params.append(min_score)
            param_idx += 1

            base_sql += f" ORDER BY embedding <=> $1 ASC LIMIT ${param_idx};"
            params.append(int(top_k))

            async with pool.acquire() as conn:
                rows = await conn.fetch(base_sql, *params)

            results = []
            for row in rows:
                meta_val = row["metadata"]
                if isinstance(meta_val, str):
                    try:
                        meta_val = json.loads(meta_val)
                    except Exception:
                        meta_val = {}

                results.append({
                    "id": str(row["id"]),
                    "title": row["title"],
                    "category": row["category"],
                    "content": row["content"],
                    "similarity_score": round(float(row["similarity_score"]), 4),
                    "metadata": meta_val or {},
                })

            return results
        except Exception as exc:
            logger.warning(f"Error querying Dedicated PGVector database: {exc}")
            return []


# Global singleton instance
vector_service = VectorService()
