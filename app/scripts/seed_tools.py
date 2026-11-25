"""
Administrative Seeding Script for FastPrint ERP Tool Catalog (TASK-5.1).
Seeds or updates the ai_tool_catalog table in PostgreSQL with pgvector embeddings.

Usage:
    python -m app.scripts.seed_tools [--update-existing] [--domain accounting] [--dry-run]
"""
import argparse
import asyncio
import logging
import sys

from app.core.config import settings
from app.services.tool_registry_service import get_tool_registry_service, CORE_TOOL_DEFINITIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("seed_tools")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Seed FastPrint ERP Tool Catalog to PostgreSQL PGVector.")
    parser.add_argument(
        "--update-existing",
        action="store_true",
        help="Update existing tools and recompute embeddings (default is incremental insert only).",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Filter tools by domain (e.g. 'accounting', 'inventory', 'sales', 'marketplace').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned seeding actions without writing to database.",
    )

    args = parser.parse_args()

    logger.info("Initializing Tool Catalog Seeding Process...")
    logger.info(f"Target Table: {settings.TOOL_CATALOG_TABLE_NAME}")
    logger.info(f"Vector DB URL configured: {bool(settings.VECTOR_DB_URL)}")
    logger.info(f"Gemini API Key configured: {bool(settings.GEMINI_API_KEY)}")
    logger.info(f"Mode: {'Dry Run' if args.dry_run else 'Live Execution'}")
    logger.info(f"Force Update: {args.update_existing}")
    if args.domain:
        logger.info(f"Domain Filter: {args.domain}")

    filtered_defs = [
        t for t in CORE_TOOL_DEFINITIONS
        if not args.domain or t["domain"] == args.domain
    ]

    logger.info(f"Found {len(filtered_defs)} tools matching domain criteria.")

    if args.dry_run:
        for t in filtered_defs:
            perm = t.get("required_permission") or "public"
            print(f" - [{t['domain']}] {t['tool_name']} (Perm: {perm}) -> {t['description'][:60]}...")
        logger.info("Dry run complete. No database changes were made.")
        return 0

    if not settings.VECTOR_DB_URL:
        logger.error("VECTOR_DB_URL is not set. Cannot connect to PostgreSQL database.")
        return 1

    registry = get_tool_registry_service()
    result = await registry.initialize_catalog(
        force_update=args.update_existing,
        domain_filter=args.domain,
    )

    logger.info(
        f"Seeding finished successfully: {result.get('created', 0)} created, "
        f"{result.get('updated', 0)} updated, {result.get('skipped', 0)} skipped."
    )
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
