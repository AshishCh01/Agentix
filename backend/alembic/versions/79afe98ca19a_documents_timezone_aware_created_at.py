"""documents/document_chunks created_at timezone-aware

documents.created_at and document_chunks.created_at were the only two
timestamp columns in the schema stored as naive TIMESTAMP WITHOUT TIME ZONE
(via datetime.utcnow()), unlike every other model (users, chat_sessions,
messages, agent_logs), which use TIMESTAMP WITH TIME ZONE. Comparing or
sorting across the two kinds directly raises TypeError in Python, or
silently misorders if compared as raw strings.

Converts both columns to TIMESTAMPTZ. Existing values were always populated
via datetime.utcnow(), i.e. they are already UTC wall-clock times with no
tzinfo -- so the conversion explicitly interprets them as UTC (via `AT TIME
ZONE 'UTC'`) rather than relying on the migrating session's ambient
`TimeZone` setting, which would silently produce the wrong instant if that
setting were ever anything other than UTC.

Revision ID: 79afe98ca19a
Revises: f06fbea045e1
Create Date: 2026-08-30 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '79afe98ca19a'
down_revision: Union[str, Sequence[str], None] = 'f06fbea045e1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Convert documents/document_chunks.created_at to TIMESTAMPTZ."""
    op.alter_column(
        'documents', 'created_at',
        existing_type=sa.DateTime(),
        type_=sa.DateTime(timezone=True),
        postgresql_using="created_at AT TIME ZONE 'UTC'",
        existing_nullable=False,
    )
    op.alter_column(
        'document_chunks', 'created_at',
        existing_type=sa.DateTime(),
        type_=sa.DateTime(timezone=True),
        postgresql_using="created_at AT TIME ZONE 'UTC'",
        existing_nullable=False,
    )


def downgrade() -> None:
    """Revert documents/document_chunks.created_at to naive TIMESTAMP."""
    op.alter_column(
        'document_chunks', 'created_at',
        existing_type=sa.DateTime(timezone=True),
        type_=sa.DateTime(),
        postgresql_using="created_at AT TIME ZONE 'UTC'",
        existing_nullable=False,
    )
    op.alter_column(
        'documents', 'created_at',
        existing_type=sa.DateTime(timezone=True),
        type_=sa.DateTime(),
        postgresql_using="created_at AT TIME ZONE 'UTC'",
        existing_nullable=False,
    )
