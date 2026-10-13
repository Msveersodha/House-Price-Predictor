"""Show the ODBC drivers installed on this machine.

Run:  python -m app.database.list_drivers
"""
from __future__ import annotations

import pyodbc

INSTALL_URL = "https://learn.microsoft.com/sql/connect/odbc/download-odbc-driver-for-sql-server"


def main() -> int:
    drivers = pyodbc.drivers()
    print("Installed ODBC drivers:")
    for name in drivers:
        print(f"  - {name}")
    if not drivers:
        print("  (none found)")

    sql_server = [d for d in drivers if "SQL Server" in d]
    modern = sorted((d for d in sql_server if d.startswith("ODBC Driver")), reverse=True)
    print()
    if modern:
        print(f"Recommended: DB_DRIVER={modern[0]}")
        return 0
    if sql_server:
        print(
            "Only the legacy 'SQL Server' driver was found. It is old and lacks modern TLS.\n"
            f"Install 'ODBC Driver 18 for SQL Server': {INSTALL_URL}"
        )
        return 1
    print(f"No SQL Server ODBC driver found. Install 'ODBC Driver 18 for SQL Server':\n  {INSTALL_URL}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
