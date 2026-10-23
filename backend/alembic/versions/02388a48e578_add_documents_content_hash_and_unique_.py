"""add documents content_hash and unique scope indexes

Revision ID: 02388a48e578
Revises: 19b468402f71
Create Date: 2026-10-23 21:45:30.693134

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '02388a48e578'
down_revision: Union[str, Sequence[str], None] = '19b468402f71'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add content_hash column and scoped partial unique indexes to documents table."""
    op.add_column(
        'documents',
        sa.Column('content_hash', sa.String(length=64), nullable=True),
    )
    op.create_index(
        'ix_documents_content_hash',
        'documents',
        ['content_hash'],
        unique=False,
    )

    # Scoped partial unique indexes for deduplication
    op.create_index(
        'uq_documents_session_content_hash',
        'documents',
        ['user_id', 'session_id', 'content_hash'],
        unique=True,
        postgresql_where=sa.text('session_id IS NOT NULL AND content_hash IS NOT NULL'),
    )
    op.create_index(
        'uq_documents_kb_content_hash',
        'documents',
        ['user_id', 'knowledge_base_id', 'content_hash'],
        unique=True,
        postgresql_where=sa.text('knowledge_base_id IS NOT NULL AND content_hash IS NOT NULL'),
    )
    op.create_index(
        'uq_documents_global_content_hash',
        'documents',
        ['user_id', 'content_hash'],
        unique=True,
        postgresql_where=sa.text('session_id IS NULL AND knowledge_base_id IS NULL AND content_hash IS NOT NULL'),
    )


def downgrade() -> None:
    """Drop content_hash column and scoped partial unique indexes from documents table."""
    op.drop_index('uq_documents_global_content_hash', table_name='documents')
    op.drop_index('uq_documents_kb_content_hash', table_name='documents')
    op.drop_index('uq_documents_session_content_hash', table_name='documents')
    op.drop_index('ix_documents_content_hash', table_name='documents')
    op.drop_column('documents', 'content_hash')
