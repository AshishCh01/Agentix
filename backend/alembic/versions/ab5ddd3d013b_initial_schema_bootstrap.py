"""initial schema bootstrap

Creates the full base schema (users, chat_sessions, agent_logs, documents,
messages, document_chunks) plus the pgvector extension, so `alembic upgrade
head` can bootstrap a genuinely empty PostgreSQL database on its own.

This is inserted as the new root of the migration chain, ahead of
e671bf5ba339 (whose ALTER COLUMN statements previously assumed these tables
already existed, since the original dev database had been bootstrapped
out-of-band via Base.metadata.create_all() rather than through Alembic).
An already-migrated database currently stamped at e671bf5ba339 is
unaffected: Alembic only walks forward from its current revision, so this
migration is never (re-)applied there.

The GIN/HNSW indexes on document_chunks are deliberately NOT created here —
they're left to f06fbea045e1, which is the migration actually named for
that purpose and which creates them idempotently (safe whether or not they
already exist, e.g. on that same already-migrated database).

Revision ID: ab5ddd3d013b
Revises:
Create Date: 2026-08-30 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'ab5ddd3d013b'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Required for the `embedding VECTOR(2000)` column on document_chunks.
    # Pinned to SCHEMA public (rather than left to search_path) so this
    # works the same on a vanilla empty Postgres and on managed providers
    # (e.g. Supabase) that otherwise install extensions into their own
    # dedicated schema by default.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")

    op.create_table(
        'users',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('email', sa.String(), nullable=False),
        sa.Column('full_name', sa.String(), nullable=True),
        sa.Column('avatar_url', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)

    op.create_table(
        'chat_sessions',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_chat_sessions_user_id'), 'chat_sessions', ['user_id'], unique=False)

    op.create_table(
        'agent_logs',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('session_id', sa.UUID(), nullable=False),
        sa.Column('agent_name', sa.String(), nullable=False),
        sa.Column('input_data', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('output_data', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('execution_time_ms', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['session_id'], ['chat_sessions.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_agent_logs_session_id'), 'agent_logs', ['session_id'], unique=False)

    op.create_table(
        'documents',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('session_id', sa.UUID(), nullable=False),
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('file_type', sa.String(length=100), nullable=False),
        sa.Column('file_path', sa.String(length=500), nullable=False),
        sa.Column('file_size', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['session_id'], ['chat_sessions.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_documents_session_id'), 'documents', ['session_id'], unique=False)
    op.create_index(op.f('ix_documents_user_id'), 'documents', ['user_id'], unique=False)

    op.create_table(
        'messages',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('session_id', sa.UUID(), nullable=False),
        sa.Column('sender', sa.String(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('citations', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['session_id'], ['chat_sessions.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_messages_session_id'), 'messages', ['session_id'], unique=False)

    op.create_table(
        'document_chunks',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('document_id', sa.UUID(), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('page_number', sa.Integer(), nullable=True),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('embedding', pgvector.sqlalchemy.vector.VECTOR(dim=2000), nullable=False),
        sa.Column(
            'fts_tokens',
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', content)", persisted=True),
            nullable=True,
        ),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_document_chunks_document_id'), 'document_chunks', ['document_id'], unique=False)
    # NOTE: idx_document_chunks_fts and idx_document_chunks_embedding_hnsw are
    # intentionally created by the next migration (f06fbea045e1), not here.


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_document_chunks_document_id'), table_name='document_chunks')
    op.drop_table('document_chunks')

    op.drop_index(op.f('ix_messages_session_id'), table_name='messages')
    op.drop_table('messages')

    op.drop_index(op.f('ix_documents_user_id'), table_name='documents')
    op.drop_index(op.f('ix_documents_session_id'), table_name='documents')
    op.drop_table('documents')

    op.drop_index(op.f('ix_agent_logs_session_id'), table_name='agent_logs')
    op.drop_table('agent_logs')

    op.drop_index(op.f('ix_chat_sessions_user_id'), table_name='chat_sessions')
    op.drop_table('chat_sessions')

    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')

    # Deliberately not dropping the `vector` extension: it may be relied on
    # by other schemas/roles in the same database (e.g. Supabase installs it
    # at the instance level), so removing it is left to a manual, deliberate
    # operation rather than an automatic migration downgrade.
