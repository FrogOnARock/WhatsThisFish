"""Change lila_annotations.id from Integer to String

Revision ID: a1f3c8e2d905
Revises: 7c714d9c92ac
Create Date: 2026-04-14 00:00:00.000000

LILA COCO annotation IDs are mixed int/str across source datasets.
Integer PK would reject string IDs; String(255) accommodates both.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1f3c8e2d905'
down_revision: Union[str, Sequence[str], None] = '7c714d9c92ac'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        'lila_annotations',
        'id',
        existing_type=sa.Integer(),
        type_=sa.String(length=255),
        existing_nullable=False,
        postgresql_using='id::text',
    )


def downgrade() -> None:
    op.alter_column(
        'lila_annotations',
        'id',
        existing_type=sa.String(length=255),
        type_=sa.Integer(),
        existing_nullable=False,
        postgresql_using='id::integer',
    )
