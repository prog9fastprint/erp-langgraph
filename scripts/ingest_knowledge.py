"""
Ingestion script for populating Dedicated PostgreSQL (PGVector) with Google Gemini Embeddings.
Reads Markdown/text documents from knowledge folder, splits them into semantic chunks,
calls Gemini embed_content API (gemini-embedding-001, 768 dimensions), and stores them
into ai_knowledge_chunks table.
"""
import argparse
import asyncio
import glob
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_knowledge")


CREATE_TABLE_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS ai_knowledge_chunks (
    id BIGSERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    category VARCHAR(50) NOT NULL DEFAULT 'sop',
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding vector(768) NOT NULL,
    company_id INT NOT NULL DEFAULT 1,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ai_knowledge_hnsw 
ON ai_knowledge_chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_ai_knowledge_company_cat 
ON ai_knowledge_chunks (company_id, category, is_active);
"""


def split_text_into_chunks(text: str, max_chunk_words: int = 250, overlap_words: int = 40) -> List[str]:
    """Split long document text into overlapping paragraph chunks."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = []
    current_word_count = 0

    for p in paragraphs:
        words = p.split()
        if current_word_count + len(words) > max_chunk_words and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            # Keep overlap from previous paragraph if possible
            overlap = current_chunk[-1].split()[-overlap_words:] if current_chunk else []
            current_chunk = [" ".join(overlap)] if overlap else []
            current_word_count = len(overlap)

        current_chunk.append(p)
        current_word_count += len(words)

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks or [text]


async def init_vector_database(db_url: str):
    """Ensure vector extension and schema exist on dedicated PostgreSQL."""
    import asyncpg
    clean_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(clean_url)
    try:
        logger.info("Initializing dedicated PGVector table and indexes...")
        await conn.execute(CREATE_TABLE_SQL)
        logger.info("Table ai_knowledge_chunks ready.")
    finally:
        await conn.close()


async def ingest_document(
    file_path: str,
    db_url: str,
    gemini_api_key: str,
    company_id: int = 1,
    category: str = "sop",
):
    """Process a single file and store its chunks and Gemini embeddings into dedicated DB."""
    import asyncpg
    from pgvector.asyncpg import register_vector
    from google import genai

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    file_title = os.path.basename(file_path).replace(".md", "").replace("_", " ").title()
    chunks = split_text_into_chunks(content)
    logger.info(f"Ingesting '{file_title}': {len(chunks)} chunks found.")

    genai_client = genai.Client(api_key=gemini_api_key)
    clean_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(clean_url)

    try:
        await register_vector(conn)
        from google.genai import types
        embed_config = types.EmbedContentConfig(output_dimensionality=768)

        source_filename = os.path.basename(file_path)
        await conn.execute(
            """
            DELETE FROM ai_knowledge_chunks 
            WHERE company_id = $1 AND metadata->>'source_file' = $2
            """,
            int(company_id),
            source_filename,
        )

        for idx, chunk_text in enumerate(chunks, 1):
            logger.info(f"Generating Gemini embedding for chunk {idx}/{len(chunks)}...")
            res = genai_client.models.embed_content(
                model=settings.GEMINI_EMBEDDING_MODEL,
                contents=chunk_text,
                config=embed_config,
            )
            vector_values = res.embeddings[0].values
            vector_arr = [float(v) for v in vector_values]

            meta = {
                "source_file": source_filename,
                "chunk_index": idx,
                "total_chunks": len(chunks),
            }

            await conn.execute(
                """
                INSERT INTO ai_knowledge_chunks (title, category, content, metadata, embedding, company_id)
                VALUES ($1, $2, $3, $4::jsonb, $5, $6)
                """,
                f"{file_title} (Part {idx})",
                category,
                chunk_text,
                json.dumps(meta),
                vector_arr,
                int(company_id),
            )

        logger.info(f"Successfully ingested {len(chunks)} chunks for '{file_title}'.")
    finally:
        await conn.close()


async def main():
    parser = argparse.ArgumentParser(description="Ingest Markdown knowledge docs into Dedicated PGVector with Gemini.")
    parser.add_argument("--path", default="data/knowledge/*.md", help="Glob pattern for document files.")
    parser.add_argument("--db-url", default=settings.VECTOR_DB_URL, help="PostgreSQL connection string.")
    default_api_key = settings.GEMINI_API_KEY or (settings.get_gemini_api_keys()[0] if settings.get_gemini_api_keys() else None)
    parser.add_argument("--api-key", default=default_api_key, help="Google Gemini API Key.")
    parser.add_argument("--company-id", type=int, default=1, help="Target ERP Company ID.")
    parser.add_argument("--category", default="sop", help="Document category ('sop', 'product_spec', etc.).")
    args = parser.parse_args()

    if not args.db_url:
        logger.error("VECTOR_DB_URL is required. Set it in .env or pass --db-url.")
        return

    if not args.api_key:
        logger.error("GEMINI_API_KEY is required. Set it in .env or pass --api-key.")
        return

    await init_vector_database(args.db_url)
    if os.path.isdir(args.path):
        pattern = os.path.join(args.path, "**", "*.md")
        files = glob.glob(pattern, recursive=True)
    else:
        files = glob.glob(args.path)

    if not files:
        logger.warning(f"No files matched pattern: {args.path}")
        return

    for fp in files:
        await ingest_document(
            file_path=fp,
            db_url=args.db_url,
            gemini_api_key=args.api_key,
            company_id=args.company_id,
            category=args.category,
        )


if __name__ == "__main__":
    asyncio.run(main())
