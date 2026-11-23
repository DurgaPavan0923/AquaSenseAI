"""Initial database schema migration

Revision ID: 001_initial
Revises: 
Create Date: 2026-11-23 20:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Users
    op.create_table(
        'users',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('username', sa.String(length=50), nullable=False, unique=True),
        sa.Column('email', sa.String(length=100), nullable=False, unique=True),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('role', sa.String(length=20), server_default='viewer'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now())
    )

    # Models
    op.create_table(
        'models',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('model_name', sa.String(length=100), nullable=False),
        sa.Column('version', sa.String(length=50), nullable=False),
        sa.Column('algorithm', sa.String(length=50), nullable=False),
        sa.Column('artifact_path', sa.String(length=255), nullable=False),
        sa.Column('metrics_json', sa.Text(), nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default=sa.false()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now())
    )

    # Samples
    op.create_table(
        'samples',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('recorded_at', sa.DateTime(), server_default=sa.func.now(), index=True),
        sa.Column('location_id', sa.String(length=50), nullable=False, index=True),
        sa.Column('sensor_id', sa.String(length=50), nullable=False),
        sa.Column('source_type', sa.String(length=20), nullable=False),
        sa.Column('ph', sa.Float(), nullable=False),
        sa.Column('turbidity', sa.Float(), nullable=False),
        sa.Column('temperature', sa.Float(), nullable=False),
        sa.Column('dissolved_oxygen', sa.Float(), nullable=False),
        sa.Column('bod', sa.Float(), nullable=False),
        sa.Column('lead', sa.Float(), nullable=False),
        sa.Column('mercury', sa.Float(), nullable=False),
        sa.Column('arsenic', sa.Float(), nullable=False),
        sa.Column('is_synthetic', sa.Boolean(), server_default=sa.false())
    )

    # Predictions
    op.create_table(
        'predictions',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('sample_id', sa.String(length=36), sa.ForeignKey('samples.id', ondelete='CASCADE'), unique=True, nullable=False),
        sa.Column('model_id', sa.String(length=36), sa.ForeignKey('models.id'), nullable=False),
        sa.Column('predicted_class', sa.Integer(), nullable=False),
        sa.Column('predicted_label', sa.String(length=20), nullable=False),
        sa.Column('confidence_score', sa.Float(), nullable=False),
        sa.Column('probabilities_json', sa.Text(), nullable=False),
        sa.Column('conformal_set_json', sa.Text(), nullable=False),
        sa.Column('is_anomaly', sa.Boolean(), server_default=sa.false()),
        sa.Column('anomaly_score', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now())
    )

    # Explanations
    op.create_table(
        'explanations',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('prediction_id', sa.String(length=36), sa.ForeignKey('predictions.id', ondelete='CASCADE'), unique=True, nullable=False),
        sa.Column('base_value', sa.Float(), nullable=False),
        sa.Column('shap_values_json', sa.Text(), nullable=False),
        sa.Column('top_positive_feature', sa.String(length=50)),
        sa.Column('top_negative_feature', sa.String(length=50)),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now())
    )

    # Alerts
    op.create_table(
        'alerts',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('sample_id', sa.String(length=36), sa.ForeignKey('samples.id', ondelete='CASCADE'), nullable=False),
        sa.Column('prediction_id', sa.String(length=36), sa.ForeignKey('predictions.id', ondelete='SET NULL'), nullable=True),
        sa.Column('severity', sa.String(length=20), nullable=False),
        sa.Column('alert_type', sa.String(length=50), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('is_acknowledged', sa.Boolean(), server_default=sa.false(), index=True),
        sa.Column('acknowledged_at', sa.DateTime(), nullable=True),
        sa.Column('acknowledged_by', sa.String(length=36), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now())
    )


def downgrade() -> None:
    op.drop_table('alerts')
    op.drop_table('explanations')
    op.drop_table('predictions')
    op.drop_table('samples')
    op.drop_table('models')
    op.drop_table('users')
