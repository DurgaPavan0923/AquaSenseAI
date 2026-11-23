"""SQLAlchemy 2.0 declarative database models."""

import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base declarative class."""
    pass


def generate_uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), default="viewer")  # "viewer" | "operator" | "admin"
    created_at = Column(DateTime, default=datetime.utcnow)

    alerts_acknowledged = relationship("Alert", back_populates="acknowledger")


class ModelRecord(Base):
    __tablename__ = "models"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    model_name = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    algorithm = Column(String(50), nullable=False)
    artifact_path = Column(String(255), nullable=False)
    metrics_json = Column(Text, nullable=False)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    predictions = relationship("Prediction", back_populates="model")


class Sample(Base):
    __tablename__ = "samples"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    recorded_at = Column(DateTime, default=datetime.utcnow, index=True)
    location_id = Column(String(50), nullable=False, default="STATION_01", index=True)
    sensor_id = Column(String(50), nullable=False, default="SENSOR_MAIN")
    source_type = Column(String(20), nullable=False, default="iot_stream")  # iot_stream | manual_lab | batch_upload
    
    # Raw feature measurements
    ph = Column(Float, nullable=False)
    turbidity = Column(Float, nullable=False)
    temperature = Column(Float, nullable=False)
    dissolved_oxygen = Column(Float, nullable=False)
    bod = Column(Float, nullable=False)
    lead = Column(Float, nullable=False)
    mercury = Column(Float, nullable=False)
    arsenic = Column(Float, nullable=False)
    is_synthetic = Column(Boolean, default=False)

    prediction = relationship("Prediction", back_populates="sample", uselist=False, cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="sample", cascade="all, delete-orphan")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    sample_id = Column(String(36), ForeignKey("samples.id", ondelete="CASCADE"), unique=True, nullable=False)
    model_id = Column(String(36), ForeignKey("models.id"), nullable=False)
    
    predicted_class = Column(Integer, nullable=False)
    predicted_label = Column(String(20), nullable=False)
    confidence_score = Column(Float, nullable=False)
    probabilities_json = Column(Text, nullable=False)
    conformal_set_json = Column(Text, nullable=False)
    is_anomaly = Column(Boolean, default=False)
    anomaly_score = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    sample = relationship("Sample", back_populates="prediction")
    model = relationship("ModelRecord", back_populates="predictions")
    explanation = relationship("Explanation", back_populates="prediction", uselist=False, cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="prediction")


class Explanation(Base):
    __tablename__ = "explanations"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    prediction_id = Column(String(36), ForeignKey("predictions.id", ondelete="CASCADE"), unique=True, nullable=False)
    base_value = Column(Float, nullable=False)
    shap_values_json = Column(Text, nullable=False)
    top_positive_feature = Column(String(50))
    top_negative_feature = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)

    prediction = relationship("Prediction", back_populates="explanation")


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    sample_id = Column(String(36), ForeignKey("samples.id", ondelete="CASCADE"), nullable=False)
    prediction_id = Column(String(36), ForeignKey("predictions.id", ondelete="SET NULL"), nullable=True)
    severity = Column(String(20), nullable=False)  # LOW | MEDIUM | HIGH | CRITICAL
    alert_type = Column(String(50), nullable=False)
    message = Column(Text, nullable=False)
    is_acknowledged = Column(Boolean, default=False, index=True)
    acknowledged_at = Column(DateTime, nullable=True)
    acknowledged_by = Column(String(36), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    sample = relationship("Sample", back_populates="alerts")
    prediction = relationship("Prediction", back_populates="alerts")
    acknowledger = relationship("User", back_populates="alerts_acknowledged")
