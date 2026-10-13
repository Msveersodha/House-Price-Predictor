"""SQL Server connection URL + SQLAlchemy engine factory.

Engines are created lazily (never at import time) so that importing the app
does not require a reachable database.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, Engine
from sqlalchemy.exc import SQLAlchemyError

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


def build_connection_url(settings: Settings, database: str | None = None) -> URL:
    """Build a ``mssql+pyodbc`` URL.

    ``URL.create`` keeps username/password as separate fields and escapes them
    when rendered, so characters such as ``@ : / ? #`` in a password are safe.
    """
    query: dict[str, str] = {
        "driver": settings.db_driver,
        "Encrypt": settings.db_encrypt,
        "TrustServerCertificate": settings.db_trust_server_certificate,
    }
    username = password = None
    if settings.db_auth_mode == "windows":
        query["Trusted_Connection"] = "yes"
    else:
        username = settings.db_user
        password = settings.db_password.get_secret_value() if settings.db_password else None

    # A named instance (host\INSTANCE) is resolved by SQL Browser; no port then.
    port = None if "\\" in settings.db_server else settings.db_port

    return URL.create(
        "mssql+pyodbc",
        username=username,
        password=password,
        host=settings.db_server,
        port=port,
        database=database or settings.db_name,
        query=query,
    )


def create_db_engine(settings: Settings, database: str | None = None, **kwargs) -> Engine:
    """Create an engine with connection pooling configured from settings."""
    return create_engine(
        build_connection_url(settings, database),
        pool_pre_ping=True,  # drop dead connections transparently
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_recycle=settings.db_pool_recycle,
        pool_timeout=settings.db_pool_timeout,
        **kwargs,
    )


@lru_cache
def get_engine() -> Engine:
    """Application engine (one per process, created on first use)."""
    return create_db_engine(get_settings())


def get_admin_engine(settings: Settings | None = None) -> Engine:
    """Engine bound to the admin database, in AUTOCOMMIT mode.

    CREATE DATABASE cannot run inside a transaction on SQL Server.
    """
    settings = settings or get_settings()
    return create_db_engine(settings, settings.db_admin_database, isolation_level="AUTOCOMMIT")


def test_connection(engine: Engine | None = None) -> bool:
    """Return True if ``SELECT 1`` works. Never raises; logs the failure."""
    try:
        with (engine or get_engine()).connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except (SQLAlchemyError, ImportError) as exc:
        logger.error("Database connectivity check failed: %s", exc.__class__.__name__)
        return False


# pytest must not collect this helper as a test.
test_connection.__test__ = False  # type: ignore[attr-defined]


if __name__ == "__main__":
    ok = test_connection()
    print("Database connection:", "OK" if ok else "FAILED (see log / README troubleshooting)")
    raise SystemExit(0 if ok else 1)
