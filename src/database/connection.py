"""Database connection and session lifecycle management."""

from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.common.config import settings
from src.common.logger import logger
from src.database.models import Base

# Support SQLite thread safety if using SQLite
db_url = settings.effective_database_url
connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}

engine = create_engine(
    db_url,
    connect_args=connect_args,
    echo=False,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Initializes schema and tables in the target database."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables verified and initialized", url=settings.effective_database_url.split("@")[-1])
    except Exception as e:
        logger.error("Database initialization failed", error=str(e))
        raise


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding scoped database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
