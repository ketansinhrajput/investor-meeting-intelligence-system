"""
Database connection using SQLite via SQLAlchemy.

DB file is created automatically at backend/data/call_transcript.db
No environment variables or server required for SQLite.
"""

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# DB stored alongside other backend data
DB_PATH = Path(__file__).parent / "data" / "call_transcript.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Required for SQLite with FastAPI threads
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency that yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables if they don't exist."""
    from models.user import User  # noqa: F401 — registers model with Base
    Base.metadata.create_all(bind=engine)
