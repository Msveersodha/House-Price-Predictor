"""Create the target database if it does not exist.

Run:  python -m app.database.create_database

Permissions: the login in .env must be a member of the ``dbcreator`` or
``sysadmin`` server role (or use Windows auth with such an account).
"""
from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.config import get_settings
from app.database.connection import get_admin_engine
from app.logging_config import configure_logging

logger = logging.getLogger(__name__)


def quote_identifier(name: str) -> str:
    """Quote a SQL Server identifier: [name] with ']' doubled."""
    return "[" + name.replace("]", "]]") + "]"


def create_database() -> bool:
    """Return True if the database was created, False if it already existed."""
    settings = get_settings()
    engine = get_admin_engine(settings)
    try:
        with engine.connect() as conn:
            exists = conn.execute(
                text("SELECT DB_ID(:name)"), {"name": settings.db_name}
            ).scalar()
            if exists is not None:
                logger.info("Database '%s' already exists.", settings.db_name)
                return False
            # Identifiers cannot be bound parameters; the name is regex-validated
            # in Settings and additionally quoted here.
            conn.execute(text(f"CREATE DATABASE {quote_identifier(settings.db_name)}"))
            logger.info("Database '%s' created.", settings.db_name)
            return True
    finally:
        engine.dispose()


if __name__ == "__main__":
    configure_logging()
    try:
        create_database()
    except SQLAlchemyError as exc:
        logger.error("Could not create database: %s", exc.__class__.__name__)
        logger.debug("details", exc_info=True)
        print("FAILED. Check server, credentials and that the login has the 'dbcreator' role.")
        raise SystemExit(1)
