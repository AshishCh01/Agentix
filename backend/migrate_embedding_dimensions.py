"""
Schema migration for the gemini-embedding-2 swap (768-dim -> 2000-dim).

gemini-embedding-2's native output is 3072 dims, but pgvector's HNSW index
has a hard cap of 2000 dims, so embeddings are requested at
output_dimensionality=2000 (see embedding_service.py / settings.py) to keep
an HNSW index usable.

Old 768-dim embeddings are NOT reinterpretable as 2000-dim vectors, and this
migration intentionally discards them rather than trying to preserve them:
it wipes `documents` (and, via ON DELETE CASCADE, `document_chunks`) before
touching the schema, which sidesteps the old data entirely -- users re-upload
their files afterward and they flow through the already-updated ingestion
pipeline as 2000-dim Gemini embeddings.

Steps performed, in order, once --confirm is passed:
  1. Report current documents / document_chunks row counts.
  2. TRUNCATE document_chunks and documents (CASCADE).
  3. Drop any existing HNSW index on document_chunks.embedding (it was built
     for 768 dimensions and can't be reused).
  4. ALTER document_chunks.embedding TYPE vector(2000).
  5. Recreate the HNSW index immediately -- safe now since the table is
     empty, so there's no need for a separate re-embed-then-index step.

NOTE: this does NOT touch files already sitting in Supabase Storage --
only the `documents` / `document_chunks` database rows are wiped. Orphaned
files in the storage bucket are left as-is; re-uploading will create new
storage objects alongside them.

Usage:
    python migrate_embedding_dimensions.py            # dry run: report only
    python migrate_embedding_dimensions.py --confirm   # actually run the migration

Take a database backup/snapshot before running with --confirm.
"""
import argparse
import asyncio

from sqlalchemy import text

from app.config.settings import settings
from app.database.connection import AsyncSessionLocal, engine

NEW_DIMENSIONS = settings.EMBEDDING_DIMENSIONS

HNSW_INDEX_NAME = "idx_document_chunks_embedding_hnsw"
HNSW_INDEX_SQL = (
    f"CREATE INDEX IF NOT EXISTS {HNSW_INDEX_NAME} ON document_chunks "
    "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);"
)


async def get_row_counts(db) -> tuple[int, int]:
    doc_count = (await db.execute(text("SELECT COUNT(*) FROM documents;"))).scalar_one()
    chunk_count = (await db.execute(text("SELECT COUNT(*) FROM document_chunks;"))).scalar_one()
    return doc_count, chunk_count


async def get_existing_hnsw_indexes(db):
    result = await db.execute(
        text("SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'document_chunks';")
    )
    return [row for row in result.fetchall() if "hnsw" in row.indexdef.lower()]


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually wipe documents/document_chunks and perform the schema "
             "change. Without this flag the script only reports current "
             "state and exits.",
    )
    args = parser.parse_args()

    async with AsyncSessionLocal() as db:
        doc_count, chunk_count = await get_row_counts(db)
        print(f"documents currently has {doc_count} row(s).")
        print(f"document_chunks currently has {chunk_count} row(s).")

        hnsw_indexes = await get_existing_hnsw_indexes(db)
        print(f"\nExisting HNSW indexes on document_chunks ({len(hnsw_indexes)}):")
        for idx in hnsw_indexes:
            print(f"  - {idx.indexname}: {idx.indexdef}")

    print("\n" + "=" * 70)
    print("WARNING: This migration will PERMANENTLY DELETE:")
    print(f"  - all {doc_count} row(s) in documents")
    print(f"  - all {chunk_count} row(s) in document_chunks (via ON DELETE CASCADE)")
    print("Files already sitting in Supabase Storage are NOT deleted by this")
    print("script -- only the database records. Users will need to re-upload")
    print("their documents afterward; re-uploads flow through the updated")
    print(f"ingestion pipeline and land as {NEW_DIMENSIONS}-dim Gemini embeddings.")
    print("\nThen it will:")
    if hnsw_indexes:
        for idx in hnsw_indexes:
            print(f"  - DROP the index '{idx.indexname}' (built for 768 dimensions)")
    print(f"  - ALTER document_chunks.embedding TYPE vector({NEW_DIMENSIONS})")
    print(f"  - Recreate the HNSW index (safe immediately -- table is now empty)")
    print("=" * 70)

    if not args.confirm:
        print("\nDry run only (no --confirm flag passed). No changes made.")
        print("Take a database backup/snapshot before re-running with --confirm.")
        await engine.dispose()
        return

    print("\n--confirm passed. Proceeding with wipe + schema migration...")

    async with engine.begin() as conn:
        print("Truncating document_chunks and documents...")
        await conn.execute(text("TRUNCATE TABLE document_chunks, documents CASCADE;"))
        print("Tables wiped.")

        if hnsw_indexes:
            for idx in hnsw_indexes:
                print(f"Dropping index {idx.indexname}...")
                await conn.execute(text(f"DROP INDEX IF EXISTS {idx.indexname};"))
            print("Index(es) dropped.")

        print(f"Altering document_chunks.embedding to vector({NEW_DIMENSIONS})...")
        await conn.execute(
            text(
                f"ALTER TABLE document_chunks ALTER COLUMN embedding "
                f"TYPE vector({NEW_DIMENSIONS}) USING NULL;"
            )
        )
        print("Column altered.")

        print("Recreating HNSW index...")
        await conn.execute(text(HNSW_INDEX_SQL))
        print("HNSW index recreated.")

    print("\nMigration complete. documents and document_chunks are empty;")
    print(f"document_chunks.embedding is now vector({NEW_DIMENSIONS}).")
    print("Re-upload documents to repopulate them with Gemini embeddings.")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
