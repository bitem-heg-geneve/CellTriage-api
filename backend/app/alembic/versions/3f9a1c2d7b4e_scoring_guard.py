"""shared triage guard: text_status/error per article, job error + SIBiLS versions, readiness table

Revision ID: 3f9a1c2d7b4e
Revises: efc03549fe1a
Create Date: 2026-11-10 15:00:00.000000

Additive only (nullable columns, one new table): rows written before the guard
keep working; the API derives a legacy text_status for them.
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '3f9a1c2d7b4e'
down_revision = 'efc03549fe1a'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('article', sa.Column('text_status', sa.String(), nullable=True))
    op.add_column('article', sa.Column('error', sa.String(), nullable=True))
    op.add_column('job', sa.Column('error', sa.String(), nullable=True))
    op.add_column('job', sa.Column('sibils_version', sa.String(), nullable=True))
    op.add_column('job', sa.Column('collection_version', sa.String(), nullable=True))
    op.add_column('job', sa.Column('sibils_warning', sa.String(), nullable=True))
    op.create_table(
        'readiness',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('ok', sa.Boolean(), nullable=False),
        sa.Column('checked_at', sa.DateTime(), nullable=False),
        sa.Column('error', sa.String(), nullable=True),
        sa.Column('detail', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )


def downgrade() -> None:
    op.drop_table('readiness')
    op.drop_column('job', 'sibils_warning')
    op.drop_column('job', 'collection_version')
    op.drop_column('job', 'sibils_version')
    op.drop_column('job', 'error')
    op.drop_column('article', 'error')
    op.drop_column('article', 'text_status')
