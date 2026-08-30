"""add vector and fts indexes

Revision ID: f06fbea045e1
Revises: e671bf5ba339
Create Date: 2026-08-10 13:58:49.269374

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f06fbea045e1'
down_revision: Union[str, Sequence[str], None] = 'e671bf5ba339'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Create the GIN (full-text search) and HNSW (vector similarity) indexes on
    document_chunks that power hybrid search.

    Uses IF NOT EXISTS because a database that was bootstrapped
    out-of-band via Base.metadata.create_all() (see e671bf5ba339's docstring
    context, and ab5ddd3d013b) may already have these indexes even though
    Alembic has never run this migration against it.
    """
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_document_chunks_fts "
        "ON document_chunks USING gin (fts_tokens)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding_hnsw "
        "ON document_chunks USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )


def downgrade() -> None:
    """Drop the GIN/HNSW indexes created by upgrade()."""
    op.execute("DROP INDEX IF EXISTS idx_document_chunks_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS idx_document_chunks_fts")
