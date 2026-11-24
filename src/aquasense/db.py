"""
Database models and connection handling for AquaSense AI local development and testing.
Uses SQLAlchemy ORM.
"""

from collections.abc import Generator
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


class SampleRecord(Base):
    """Stores raw sensor reading telemetry."""

    __tablename__ = "samples"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sample_code = Column(String(64), nullable=True, index=True)
    ph = Column(Float, nullable=False)
    turbidity = Column(Float, nullable=False)
    temperature = Column(Float, nullable=False)
    dissolved_oxygen = Column(Float, nullable=False)
    biochemical_oxygen_demand = Column(Float, nullable=False)
    lead = Column(Float, nullable=False)
    mercury = Column(Float, nullable=False)
    arsenic = Column(Float, nullable=False)
    source = Column(String(64), default="api_request")
    created_at = Column(DateTime, default=datetime.utcnow)


class PredictionRecord(Base):
    """Stores model screening predictions and conformal uncertainty sets."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sample_id = Column(Integer, nullable=True, index=True)
    predicted_class = Column(Integer, nullable=False)
    class_label = Column(String(32), nullable=False)
    prob_good = Column(Float, nullable=False)
    prob_moderate = Column(Float, nullable=False)
    prob_polluted = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    conformal_set = Column(String(32), nullable=False)
    needs_lab_test = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class AlertRecord(Base):
    """Stores sensor audits and regulatory compliance alerts."""

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sample_id = Column(Integer, nullable=True, index=True)
    severity = Column(String(16), nullable=False)
    status_message = Column(String(255), nullable=False)
    is_anomaly = Column(Boolean, default=False)
    violations_json = Column(Text, default="[]")
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelMetadataRecord(Base):
    """Tracks deployed model versions and performance scores."""

    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(64), nullable=False)
    version = Column(String(32), nullable=False)
    macro_f1 = Column(Float, nullable=False)
    pr_auc = Column(Float, nullable=False)
    conformal_coverage = Column(Float, nullable=False)
    trained_at = Column(DateTime, default=datetime.utcnow)


def get_engine(db_url: str = "sqlite:///./aquasense.db"):
    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    return create_engine(db_url, connect_args=connect_args)


def init_db(engine=None, db_url: str = "sqlite:///./aquasense.db"):
    if engine is None:
        engine = get_engine(db_url)
    Base.metadata.create_all(bind=engine)
    return engine


def get_db_session(engine=None, db_url: str = "sqlite:///./aquasense.db") -> Generator[Session, None, None]:
    if engine is None:
        engine = get_engine(db_url)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
