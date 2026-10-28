"""sound events (label-only daily sound monitoring)

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-07 10:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
import homeease.backend.storage.models


revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('sound_events',
    sa.Column('id', sa.String(length=32), nullable=False),
    sa.Column('subject_id', sa.String(length=64), nullable=False),
    sa.Column('room_id', sa.String(length=64), nullable=True),
    sa.Column('source_mode', sa.String(length=16), nullable=False),
    sa.Column('category', sa.String(length=16), nullable=False),
    sa.Column('confidence', sa.Float(), nullable=False),
    sa.Column('t_start', homeease.backend.storage.models.UTCDateTime(), nullable=False),
    sa.Column('t_end', homeease.backend.storage.models.UTCDateTime(), nullable=False),
    sa.Column('t_start_mono_ms', sa.Float(), nullable=False),
    sa.Column('t_end_mono_ms', sa.Float(), nullable=False),
    sa.Column('duration_s', sa.Float(), nullable=False),
    sa.Column('rms_db', sa.Float(), nullable=False),
    sa.Column('peak_db', sa.Float(), nullable=False),
    sa.Column('n_windows', sa.Integer(), nullable=False),
    sa.Column('model_id', sa.String(length=128), nullable=False),
    sa.Column('created_at', homeease.backend.storage.models.UTCDateTime(), nullable=False),
    sa.Column('updated_at', homeease.backend.storage.models.UTCDateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('sound_events', schema=None) as batch_op:
        batch_op.create_index('ix_sound_events_subject_start', ['subject_id', 't_start'], unique=False)
        batch_op.create_index('ix_sound_events_t_end', ['t_end'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('sound_events', schema=None) as batch_op:
        batch_op.drop_index('ix_sound_events_t_end')
        batch_op.drop_index('ix_sound_events_subject_start')

    op.drop_table('sound_events')
