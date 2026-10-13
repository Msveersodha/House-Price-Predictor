"""Application settings, loaded from environment variables / the .env file."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

# Database names are interpolated into CREATE DATABASE (identifiers cannot be
# bound as parameters), so only a conservative character set is accepted.
_SAFE_IDENTIFIER = r"^[A-Za-z_][A-Za-z0-9_]{0,127}$"


class Settings(BaseSettings):
    """Validated configuration. Startup fails fast if something is wrong."""

    # "model_" is a protected pydantic prefix; we want a MODEL_PATH setting.
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        protected_namespaces=("settings_",),
    )

    # --- SQL Server ---
    db_server: str = "localhost"
    db_port: int = Field(1433, ge=1, le=65535)
    db_name: str = Field("HousePriceDB", pattern=_SAFE_IDENTIFIER)
    db_admin_database: str = Field("master", pattern=_SAFE_IDENTIFIER)
    db_driver: str = "ODBC Driver 18 for SQL Server"
    db_auth_mode: Literal["sql", "windows"] = "sql"
    db_user: str | None = None
    db_password: SecretStr | None = None
    db_encrypt: Literal["yes", "no"] = "yes"
    db_trust_server_certificate: Literal["yes", "no"] = "yes"
    db_pool_size: int = Field(5, ge=1, le=100)
    db_max_overflow: int = Field(10, ge=0, le=200)
    db_pool_recycle: int = Field(1800, ge=-1)
    db_pool_timeout: int = Field(30, ge=1)

    # --- Model ---
    model_path: str = "app/models/house_price_model.pkl"
    price_currency: str = Field("USD", min_length=1, max_length=8)
    price_unit_multiplier: float = Field(100_000.0, gt=0)

    # --- API ---
    api_base_url: str = "http://127.0.0.1:8000"
    cors_origins: str = "http://localhost:8501,http://127.0.0.1:8501"
    allow_delete: bool = True
    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "text"

    @field_validator("db_server", "db_driver")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()

    @model_validator(mode="after")
    def _check_auth(self) -> "Settings":
        if self.db_auth_mode == "sql":
            missing = [
                name
                for name, value in (("DB_USER", self.db_user), ("DB_PASSWORD", self.db_password))
                if not value or (isinstance(value, SecretStr) and not value.get_secret_value())
            ]
            if missing:
                raise ValueError(
                    f"DB_AUTH_MODE=sql requires {', '.join(missing)}. "
                    "Set them in .env, or use DB_AUTH_MODE=windows."
                )
        return self

    @property
    def resolved_model_path(self) -> Path:
        path = Path(self.model_path)
        return path if path.is_absolute() else BASE_DIR / path

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings object (validated once)."""
    return Settings()
