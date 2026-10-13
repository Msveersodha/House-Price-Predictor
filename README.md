# House Price Prediction API

FastAPI + scikit-learn + SQL Server + Streamlit, built around **your existing** `house_price_model.pkl`.

## 1. What your model is (verified by inspecting the file)

| | |
|---|---|
| Saved object | `sklearn.pipeline.Pipeline` (so imputing is built in) |
| Steps | `SimpleImputer(strategy="median")` -> `LGBMRegressor` (100 trees, lr 0.1, 31 leaves) |
| Saved with | scikit-learn **1.6.1** (hence the pinned version) + **LightGBM** |
| Inputs (8, in this order) | `MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude` |
| Output | a number in units of **$100,000** (California Housing convention): `3.90` means about $390,000 |

These are the standard California Housing features. The model only knows California, so the API
rejects coordinates outside it. The "$100,000" unit is inferred from that dataset, not stored in the file:
change `PRICE_UNIT_MULTIPLIER` / `PRICE_CURRENCY` in `.env` if your training target differed.

Verified here: the real `.pkl` loads at API startup, and the Berkeley sample row in section 9 returns 3.9045 (about $390,449).
**Not verified here:** a live SQL Server (none available). The SQL Server layer is tested with SQLite
plus a compile of the exact SQL Server DDL; run the integration tests (section 11) on your machine.

## 2. Architecture

```
Streamlit (frontend/app.py)  --HTTP-->  FastAPI (app/main.py)
                                          routers/prediction.py      endpoints, status codes
                                          schemas/house.py           validation (Pydantic)
                                          services/prediction_service.py   model + save in one transaction
                                          database/                  SQLAlchemy -> PyODBC -> SQL Server
```
* The model is loaded **once** at startup (lifespan). If the file is missing or its features differ
  from the schema, the API refuses to start with a clear message.
* The database is connected **lazily**; the API starts even if SQL Server is down, and
  `/health/database` reports it.
* The frontend never touches the `.pkl` or the database. Its input form is built from `GET /model/features`.
* Clients only see generic errors; details go to the server log.

## 3. Inspecting a model (any model)

```powershell
python inspect_model.py app/models/house_price_model.pkl --trust
```
Prints type, pipeline steps, expected feature names/order, input dimensions and imputer fill values.
**Pickles can execute code when loaded**: only inspect files you made or fully trust (hence `--trust`).
If feature names are not stored (`feature_names_in_` missing), you must get them and their order from the
training notebook or dataset: do not guess.

## 4. Project structure

```
app/            main.py, config.py, logging_config.py
  database/     list_drivers, connection, create_database, create_table, session, models
  schemas/      house.py       services/prediction_service.py     routers/prediction.py
  models/       house_price_model.pkl
frontend/       app.py, api_client.py
tests/          unit tests + tests/integration (real SQL Server)
inspect_model.py  .env.example  requirements.txt  requirements-dev.txt
```

## 5. Windows setup

Use Python 3.10-3.12 (64-bit).

```powershell
cd house-price-prediction-api
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env        # then edit .env
```

### 5.1 ODBC driver
```powershell
python -m app.database.list_drivers
```
If it does not list `ODBC Driver 18 for SQL Server` (or 17), install it from Microsoft:
https://learn.microsoft.com/sql/connect/odbc/download-odbc-driver-for-sql-server
(64-bit installer), reopen the terminal, and run the command again. Put the exact driver name in `DB_DRIVER`.

### 5.2 Authentication modes
**SQL Server Authentication** (`DB_AUTH_MODE=sql`, needs `DB_USER` + `DB_PASSWORD`)
1. In SSMS: server Properties -> Security -> "SQL Server and Windows Authentication mode", then restart the SQL Server service.
2. Create a login that may create databases:
   ```sql
   CREATE LOGIN house_api WITH PASSWORD = 'Use-a-strong-password-1!';
   ALTER SERVER ROLE dbcreator ADD MEMBER house_api;
   ```
   (`dbcreator` or `sysadmin` is required by `create_database`.)
3. Make sure TCP/IP is enabled (SQL Server Configuration Manager -> Network Configuration), default port 1433.

**Windows Authentication** (`DB_AUTH_MODE=windows`): leave `DB_USER`/`DB_PASSWORD` empty. The Windows account
running the app needs the same permissions. For a named instance (e.g. SQL Express) use `DB_SERVER=localhost\SQLEXPRESS`; the port is then ignored.

Special characters in passwords are fine: they are escaped automatically.

### 5.3 Model
The `.pkl` is already at `app/models/house_price_model.pkl` (or set `MODEL_PATH`).
Keep **scikit-learn==1.6.1**: a different version can load with warnings or give different results.

### 5.4 Create database, table, run
```powershell
python -m app.database.create_database
python -m app.database.create_table
uvicorn app.main:app --reload
```
Swagger: http://127.0.0.1:8000/docs. In a **second terminal** (venv activated):
```powershell
streamlit run frontend/app.py
```

## 6. Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | welcome + status |
| GET | `/health` | app + model status |
| GET | `/health/database` | database connectivity (503 if down) |
| GET | `/model/features` | feature names, limits, examples, currency |
| POST | `/predict` | validate, predict, save (201) |
| GET | `/predictions?limit=20&offset=0` | history, newest first (limit 1-100) |
| GET | `/predictions/{id}` | one prediction (404 if unknown) |
| DELETE | `/predictions/{id}` | delete (204); set `ALLOW_DELETE=false` to disable |

There is **no authentication**. Add it (and keep `ALLOW_DELETE=false`) before exposing the API beyond localhost.

## 7. Validation limits

Slightly wider than the real data, to catch nonsense but allow edge cases. NaN/Infinity, strings, missing and extra fields are rejected.
`MedInc` 0-20 (exclusive 0), `HouseAge` 0-100, `AveRooms` >0-200, `AveBedrms` >0-50, `Population` >0-50,000,
`AveOccup` >0-2,000, `Latitude` 32-43, `Longitude` -125 to -114. Edit them in `app/schemas/house.py`.

## 8. Database table

`predictions`: `id` (IDENTITY), the 8 feature columns (FLOAT), `predicted_value` (raw model output),
`predicted_price` (converted), `created_at` (DATETIME2, UTC), CHECK constraints and an index on `created_at`.
Typed columns were chosen over JSON because the feature set is fixed.

## 9. Sample request / response

Request body (the Berkeley example from the dataset):
```json
{"MedInc": 8.3252, "HouseAge": 41, "AveRooms": 6.9841, "AveBedrms": 1.0238,
 "Population": 322, "AveOccup": 2.5556, "Latitude": 37.88, "Longitude": -122.23}
```
`POST /predict` -> `201`. The price below is real output from your model; `prediction_id` and `created_at` will differ:
```json
{"prediction_id": 1, "predicted_price": 390449.35, "currency": "USD",
 "created_at": "2026-10-13T11:27:05.557562Z",
 "features": {"MedInc": 8.3252, "HouseAge": 41.0, "AveRooms": 6.9841, "AveBedrms": 1.0238,
              "Population": 322.0, "AveOccup": 2.5556, "Latitude": 37.88, "Longitude": -122.23}}
```
`Latitude: 500` -> `422` with `{"detail":[{"loc":["body","Latitude"],"msg":"Input should be less than or equal to 43",...}]}`.

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `Data source name not found` / `IM002` | Wrong `DB_DRIVER`. Run `python -m app.database.list_drivers` and copy the exact name. |
| `SSL Provider ... certificate verify failed` | Dev server with a self-signed cert: `DB_TRUST_SERVER_CERTIFICATE=yes`. |
| `Login failed for user` | Wrong credentials, or SQL authentication not enabled (5.2). |
| Timeout / `server not found` | Enable TCP/IP, check the service is running, port 1433 and firewall; named instance: `localhost\SQLEXPRESS`. |
| `CREATE DATABASE permission denied` | Login lacks `dbcreator`/`sysadmin` (5.2), or create the database manually in SSMS and skip that step. |
| API exits at startup: `Model file not found` | Check `MODEL_PATH` and that the file exists. |
| `Model needs a package ... lightgbm` | `pip install -r requirements.txt` inside the activated venv. |
| `InconsistentVersionWarning` in logs | Install scikit-learn==1.6.1. |
| Streamlit "Cannot reach the API" | Start uvicorn first; check `API_BASE_URL` / the sidebar URL. |
| `DB_AUTH_MODE=sql requires DB_USER` | Fill `.env` or use `DB_AUTH_MODE=windows`. |
| `pip install pyodbc` fails | Use 64-bit Python 3.10-3.12 (wheels available). |

## 11. Tests

```powershell
pip install -r requirements-dev.txt
pytest                       # unit tests: stub model + in-memory SQLite, no SQL Server needed
$env:RUN_INTEGRATION = "1"
pytest -m integration        # real SQL Server; uses a temporary HousePriceDB_pytest database
```
Covered: model loading (real file, missing file, corrupt, wrong features), valid/invalid/missing/extra/NaN/Infinity input,
non-finite model output, DB failure (no secrets leaked), insert, pagination, unknown ID, delete, connection-URL building
(special characters, Windows auth, named instance), SQL Server DDL, frontend client <-> API round trip.

## 12. Future improvements

API-key/JWT auth; Alembic migrations; Docker Compose with SQL Server; rate limiting; model versioning stored per prediction;
request IDs in logs; CI running the tests.
