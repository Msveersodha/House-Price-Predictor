"""Create the predictions table if it does not exist.

Run:  python -m app.database.create_table
"""
from __future__ import annotations

import logging

from sqlalchemy import inspect
from sqlalchemy.exc import SQLAlchemyError

from app.database.connection import get_engine
from app.database.models import Base, Prediction
from app.logging_config import configure_logging

logger = logging.getLogger(__name__)


def create_tables() -> None:
    engine = get_engine()
    Base.metadata.create_all(engine)  # checkfirst=True: existing tables are untouched
    exists = inspect(engine).has_table(Prediction.__tablename__)
    logger.info("Table '%s' %s.", Prediction.__tablename__, "is ready" if exists else "MISSING")


if __name__ == "__main__":
    configure_logging()
    try:
        create_tables()
    except SQLAlchemyError as exc:
        logger.error("Could not create tables: %s", exc.__class__.__name__)
        logger.debug("details", exc_info=True)
        print("FAILED. Run create_database first and check the .env credentials.")
        raise SystemExit(1)
