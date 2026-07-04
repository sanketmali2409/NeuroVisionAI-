"""
SQLAlchemy engine, session factory, and declarative base.

Every ORM model in `backend/database/models/` imports `Base` from here so
they all register on the same metadata object, which is what
`Base.metadata.create_all()` (called at startup for this SQLite-backed
demo setup) needs to create every table in one pass.
"""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config.settings import settings

# `check_same_thread=False` is required for SQLite when the DB is accessed
# from FastAPI's threaded request handling; it is a no-op for other engines.
_connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(settings.DATABASE_URL, connect_args=_connect_args, echo=settings.DEBUG)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a DB session and guarantees closure.

    Usage:
        @router.get("/items")
        def list_items(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables that don't yet exist.

    Imports the models module first so their `Base.metadata` registrations
    actually happen before `create_all` is called — otherwise SQLAlchemy
    would have nothing to create.
    """
    from backend.database import models  # noqa: F401  (registers models on Base)

    Base.metadata.create_all(bind=engine)
