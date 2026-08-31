"""add knowledge bases

Introduces the Knowledge Base feature: a `knowledge_bases` table owned by a
user, and optional `knowledge_base_id` foreign keys on `documents` (which
document a KB owns) and `chat_sessions` (which KB a chat is scoped to for
retrieval). `documents.session_id` is relaxed to nullable so a document can
be uploaded directly into a knowledge base without belonging to any chat
session -- every other existing column/behavior is untouched.

Revision ID: 250cf577c4bd
Revises: 79afe98ca19a
Create Date: 2026-08-31 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '250cf577c4bd'
down_revision: Union[str, Sequence[str], None] = '79afe98ca19a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'knowledge_bases',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_knowledge_bases_user_id'), 'knowledge_bases', ['user_id'], unique=False
    )

    # documents.session_id: relax to nullable so KB-only uploads (no chat
    # session involved) are representable.
    op.alter_column(
        'documents', 'session_id',
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=True,
    )

    op.add_column(
        'documents', sa.Column('knowledge_base_id', postgresql.UUID(as_uuid=True), nullable=True)
    )
    op.create_index(
        op.f('ix_documents_knowledge_base_id'), 'documents', ['knowledge_base_id'], unique=False
    )
    op.create_foreign_key(
        'fk_documents_knowledge_base_id', 'documents', 'knowledge_bases',
        ['knowledge_base_id'], ['id'], ondelete='CASCADE',
    )

    op.add_column(
        'chat_sessions', sa.Column('knowledge_base_id', postgresql.UUID(as_uuid=True), nullable=True)
    )
    op.create_index(
        op.f('ix_chat_sessions_knowledge_base_id'), 'chat_sessions', ['knowledge_base_id'], unique=False
    )
    op.create_foreign_key(
        'fk_chat_sessions_knowledge_base_id', 'chat_sessions', 'knowledge_bases',
        ['knowledge_base_id'], ['id'], ondelete='SET NULL',
    )


def downgrade() -> None:
    op.drop_constraint('fk_chat_sessions_knowledge_base_id', 'chat_sessions', type_='foreignkey')
    op.drop_index(op.f('ix_chat_sessions_knowledge_base_id'), table_name='chat_sessions')
    op.drop_column('chat_sessions', 'knowledge_base_id')

    op.drop_constraint('fk_documents_knowledge_base_id', 'documents', type_='foreignkey')
    op.drop_index(op.f('ix_documents_knowledge_base_id'), table_name='documents')
    op.drop_column('documents', 'knowledge_base_id')

    op.alter_column(
        'documents', 'session_id',
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=False,
    )

    op.drop_index(op.f('ix_knowledge_bases_user_id'), table_name='knowledge_bases')
    op.drop_table('knowledge_bases')
