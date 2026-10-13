"""Needs a real SQL Server and a filled-in .env.

    set RUN_INTEGRATION=1        (PowerShell: $env:RUN_INTEGRATION = "1")
    pytest -m integration

Uses a separate database name so your real data is never touched.
"""
import os

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("RUN_INTEGRATION") != "1", reason="set RUN_INTEGRATION=1 to run"),
]


@pytest.fixture(scope="module")
def sql_engine():
    from app.config import Settings
    from app.database.connection import create_db_engine, get_admin_engine
    from app.database.create_database import quote_identifier
    from sqlalchemy import text

    test_db = "HousePriceDB_pytest"
    settings = Settings(db_name=test_db)
    admin = get_admin_engine(settings)
    with admin.connect() as conn:
        conn.execute(text(f"IF DB_ID('{test_db}') IS NULL CREATE DATABASE {quote_identifier(test_db)}"))
    engine = create_db_engine(settings)
    yield engine
    engine.dispose()
    with admin.connect() as conn:
        conn.execute(text(f"ALTER DATABASE {quote_identifier(test_db)} SET SINGLE_USER WITH ROLLBACK IMMEDIATE"))
        conn.execute(text(f"DROP DATABASE {quote_identifier(test_db)}"))
    admin.dispose()


def test_connection(sql_engine):
    from app.database.connection import test_connection

    assert test_connection(sql_engine) is True


def test_insert_and_read_back(sql_engine):
    from sqlalchemy.orm import Session

    from app.database.models import Base, Prediction
    from tests.conftest import VALID_PAYLOAD

    Base.metadata.create_all(sql_engine)
    with Session(sql_engine) as db:
        rec = Prediction(**VALID_PAYLOAD, predicted_value=3.5, predicted_price=350000.0)
        db.add(rec)
        db.commit()
        assert rec.id is not None
        assert db.get(Prediction, rec.id).MedInc == pytest.approx(8.3252)
