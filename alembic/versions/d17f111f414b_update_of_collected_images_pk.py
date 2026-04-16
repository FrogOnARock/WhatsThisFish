"""update of collected images pk

Revision ID: d17f111f414b
Revises: 718dae771660
Create Date: 2026-04-15 17:34:53.504979

Swap lila_collected_images PK from file_name to a new id column (COCO image ID).
The annotations FK (image_id -> file_name) is re-pointed to the new id column.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd17f111f414b'
down_revision: Union[str, Sequence[str], None] = '718dae771660'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Drop the FK from annotations -> collected_images (currently points at file_name)
    op.drop_constraint(
        'fk_lila_annotations_image_id_lila_collected_images',
        'lila_annotations',
        type_='foreignkey',
    )

    # 2. Drop the old PK on file_name
    op.drop_constraint('pk_lila_collected_images', 'lila_collected_images', type_='primary')

    # 3. Add the new id column (non-nullable, will become PK)
    op.add_column(
        'lila_collected_images',
        sa.Column('id', sa.String(length=255), nullable=False, server_default=''),
    )
    # Remove the server_default — it was only needed so the NOT NULL ADD COLUMN succeeds
    # on an empty table. If the table has data you'd populate id first, then set NOT NULL.
    op.alter_column('lila_collected_images', 'id', server_default=None)

    # 4. Create the new PK on id
    op.create_primary_key('pk_lila_collected_images', 'lila_collected_images', ['id'])

    # 5. Add unique constraint on file_name (was previously the PK, so already unique,
    #    but we need an explicit constraint now)
    op.create_unique_constraint(
        'uq_lila_collected_images_file_name',
        'lila_collected_images',
        ['file_name'],
    )

    # 6. Recreate the FK from annotations -> collected_images, now pointing at id
    op.create_foreign_key(
        'fk_lila_annotations_image_id_lila_collected_images',
        'lila_annotations',
        'lila_collected_images',
        ['image_id'],
        ['id'],
    )


def downgrade() -> None:
    # Reverse: FK -> drop unique -> drop PK on id -> drop id column -> restore PK on file_name -> restore FK

    # 1. Drop the FK pointing at id
    op.drop_constraint(
        'fk_lila_annotations_image_id_lila_collected_images',
        'lila_annotations',
        type_='foreignkey',
    )

    # 2. Drop unique constraint on file_name
    op.drop_constraint(
        'uq_lila_collected_images_file_name',
        'lila_collected_images',
        type_='unique',
    )

    # 3. Drop PK on id
    op.drop_constraint('pk_lila_collected_images', 'lila_collected_images', type_='primary')

    # 4. Drop the id column
    op.drop_column('lila_collected_images', 'id')

    # 5. Restore PK on file_name
    op.create_primary_key('pk_lila_collected_images', 'lila_collected_images', ['file_name'])

    # 6. Restore FK pointing at file_name
    op.create_foreign_key(
        'fk_lila_annotations_image_id_lila_collected_images',
        'lila_annotations',
        'lila_collected_images',
        ['image_id'],
        ['file_name'],
    )
