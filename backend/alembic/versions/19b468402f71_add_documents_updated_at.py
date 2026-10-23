"""add documents updated_at

Revision ID: 19b468402f71
Revises: 250cf577c4bd
Create Date: 2026-10-20 06:38:10.863955

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '19b468402f71'
down_revision: Union[str, Sequence[str], None] = '250cf577c4bd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add updated_at column to documents table and backfill from created_at."""
    op.add_column(
        'documents',
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.execute("UPDATE documents SET updated_at = created_at WHERE updated_at IS NULL")
    op.alter_column('documents', 'updated_at', nullable=False)


def downgrade() -> None:
    """Drop updated_at column from documents table."""
    op.drop_column('documents', 'updated_at')
