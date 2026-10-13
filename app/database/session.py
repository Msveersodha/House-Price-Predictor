"""Session factory and the FastAPI dependency that hands out sessions."""
from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy.orm import Session, sessionmaker

from app.database.connection import get_engine


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    # expire_on_commit=False lets us read the saved row after commit without
    # triggering another SELECT.
    return sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)


def get_db() -> Iterator[Session]:
    """One session per request. Rolls back on error, always closes."""
    db = get_session_factory()()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
