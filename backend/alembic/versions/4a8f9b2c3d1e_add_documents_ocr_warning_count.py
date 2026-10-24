"""add documents ocr_warning_count

Revision ID: 4a8f9b2c3d1e
Revises: 02388a48e578
Create Date: 2026-10-24 08:15:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4a8f9b2c3d1e'
down_revision: Union[str, Sequence[str], None] = '02388a48e578'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add ocr_warning_count column to documents table."""
    op.add_column(
        'documents',
        sa.Column('ocr_warning_count', sa.Integer(), nullable=False, server_default='0'),
    )


def downgrade() -> None:
    """Drop ocr_warning_count column from documents table."""
    op.drop_column('documents', 'ocr_warning_count')
