import pytest
from pydantic import ValidationError
from sqlalchemy.dialects import mssql
from sqlalchemy.schema import CreateTable

from app.config import Settings
from app.database.connection import build_connection_url
from app.database.create_database import quote_identifier
from app.database.models import Prediction


def make_settings(**kw):
    base = dict(_env_file=None, db_user="u", db_password="p")
    base.update(kw)
    return Settings(**base)


def test_special_characters_in_password_are_escaped():
    url = build_connection_url(make_settings(db_password="p@ss:w/o?r#d%"))
    rendered = url.render_as_string(hide_password=False)
    assert "p@ss:w/o?r#d%" not in rendered  # must be percent-encoded
    assert url.password == "p@ss:w/o?r#d%"  # but round-trips intact
    assert "driver=ODBC+Driver+18+for+SQL+Server" in rendered


def test_password_hidden_in_logs():
    url = build_connection_url(make_settings(db_password="topsecret"))
    assert "topsecret" not in str(url)


def test_windows_auth_url_has_no_credentials():
    s = Settings(_env_file=None, db_auth_mode="windows")
    url = build_connection_url(s)
    assert url.username is None and url.password is None
    assert url.query["Trusted_Connection"] == "yes"


def test_named_instance_omits_port():
    url = build_connection_url(make_settings(db_server=r"localhost\SQLEXPRESS"))
    assert url.port is None and url.host == r"localhost\SQLEXPRESS"


def test_default_port_used():
    assert build_connection_url(make_settings()).port == 1433


def test_admin_database_override():
    assert build_connection_url(make_settings(), "master").database == "master"


def test_sql_auth_requires_credentials(monkeypatch):
    monkeypatch.delenv("DB_USER", raising=False)
    monkeypatch.delenv("DB_PASSWORD", raising=False)
    with pytest.raises(ValidationError, match="DB_USER"):
        Settings(_env_file=None, db_auth_mode="sql")


@pytest.mark.parametrize("name", ["bad]name", "x; DROP DATABASE y", "1abc", ""])
def test_unsafe_database_name_rejected(name):
    with pytest.raises(ValidationError):
        make_settings(db_name=name)


def test_quote_identifier():
    assert quote_identifier("HousePriceDB") == "[HousePriceDB]"
    assert quote_identifier("a]b") == "[a]]b]"


def test_ddl_for_sql_server():
    ddl = str(CreateTable(Prediction.__table__).compile(dialect=mssql.dialect()))
    assert "IDENTITY" in ddl and "DATETIME2" in ddl and "CONSTRAINT" in ddl
    for col in ("MedInc", "HouseAge", "Longitude", "predicted_price", "created_at"):
        assert col in ddl
