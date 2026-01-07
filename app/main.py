import os
from datetime import datetime
from ftplib import FTP, error_perm
from io import StringIO

import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from app.settings import API_KEY, DATA_PATH, MODELS_DIR
from app.ml import train_from_csv, predict_from_row, load_latest  # 👈 añadimos load_latest

import numpy as np
import xgboost as xgb

app = FastAPI(title="API Modelo Producción (XGBoost)")

# =========================
# Seguridad
# =========================
def check_key(x_api_key: str | None):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API key inválida")


# =========================
# Config FTP (por variables de entorno)
# =========================
FTP_HOST = os.getenv("FTP_HOST", "").strip()
FTP_PORT = int(os.getenv("FTP_PORT", "21"))
FTP_USER = os.getenv("FTP_USER", "").strip()
FTP_PASS = os.getenv("FTP_PASS", "").strip()

# Ruta remota donde se guardarán los archivos
FTP_DIR = os.getenv("FTP_DIR", "/exports").strip()

# Activa/desactiva subida automática
UPLOAD_ENABLED = os.getenv("UPLOAD_ENABLED", "1").strip()  # "1" o "0"

# =========================
# Rutas locales de archivos
# =========================
LOG_PATH = os.path.join(MODELS_DIR, "predictions_log.csv")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics_latest.json")
FORECAST_PATH = os.path.join(MODELS_DIR, "forecast_26w.csv")  # 👈 nuevo

# =========================
# FTP helpers
# =========================
def _ftp_ensure_dir(ftp: FTP, remote_dir: str):
    parts = [p for p in remote_dir.split("/") if p]
    for p in parts:
        try:
            ftp.cwd(p)
        except error_perm:
            ftp.mkd(p)
            ftp.cwd(p)

def ftp_upload(local_path: str, remote_filename: str):
    if UPLOAD_ENABLED != "1":
        return
    if not (FTP_HOST and FTP_USER and FTP_PASS):
        return
    if not os.path.exists(local_path):
        return

    with FTP() as ftp:
        ftp.connect(FTP_HOST, FTP_PORT, timeout=30)
        ftp.login(FTP_USER, FTP_PASS)

        ftp.cwd("/")
        _ftp_ensure_dir(ftp, FTP_DIR)

        with open(local_path, "rb") as f:
            ftp.storbinary(f"STOR {remote_filename}", f)


# =========================
# Schemas
# =========================
class PredictRequest(BaseModel):
    row: dict


# =========================
# Endpoints básicos
# =========================
@app.get("/")
def root():
    return {"message": "API modelo tesis online. Ir a /docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


# =========================
# Entrenamiento
# =========================
@app.post("/train")
def train(x_api_key: str | None = Header(default=None)):
    check_key(x_api_key)

    metrics = train_from_csv(DATA_PATH, MODELS_DIR)

    # subir métricas por FTP
    try:
        ftp_upload(METRICS_PATH, "metrics_latest.json")
    except Exception as e:
        print("WARNING: Falló upload FTP metrics_latest.json:", repr(e))

    return {"trained": True, "metrics": metrics}


# =========================
# Predicción + log CSV + subida FTP
# =========================
@app.post("/predict")
def predict(req: PredictRequest, x_api_key: str | None = Header(default=None)):
    check_key(x_api_key)

    result = predict_from_row(MODELS_DIR, req.row)

    os.makedirs(MODELS_DIR, exist_ok=True)

    row_log = {
        "timestamp": datetime.now().isoformat(),
        **req.row,
        **result,
    }

    df_log = pd.DataFrame([row_log])

    if os.path.exists(LOG_PATH):
        df_log.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        df_log.to_csv(LOG_PATH, mode="w", header=True, index=False)

    try:
        ftp_upload(LOG_PATH, "predictions_log.csv")
    except Exception as e:
        print("WARNING: Falló upload FTP predictions_log.csv:", repr(e))

    return result


# =========================
# NUEVO: Forecast 26 semanas (racimos + cajas)
# =========================
def _build_base_row_from_csv(csv_path: str, features: list[str]):
    """
    Toma la última fila del CSV para usarla como base de features.
    """
    df = pd.read_csv(csv_path)
    last_date = None

    if "Fecha" in df.columns:
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
        df = df.sort_values("Fecha")
        ld = df["Fecha"].dropna().max()
        if pd.notna(ld):
            last_date = ld

    base = df.tail(1).to_dict(orient="records")[0] if len(df) else {}
    base.pop("RACIMOS COSECHADOS", None)
    base.pop("CAJAS PROCESADAS", None)

    row = {}
    for col in features:
        row[col] = base.get(col, np.nan)

    return row, last_date


@app.get("/forecast/26w")
def forecast_26w(x_api_key: str | None = Header(default=None), export_csv: int = 0):
    """
    Genera predicción semanal para las próximas 26 semanas:
    - pred_racimos
    - pred_cajas

    Parámetros:
      export_csv=1  -> devuelve CSV para descargar
      export_csv=0  -> devuelve JSON
    """
    check_key(x_api_key)

    mr, mc, features = load_latest(MODELS_DIR)

    base_row, last_date = _build_base_row_from_csv(DATA_PATH, features)

    # Fecha base: si no hay, hoy
    base_date = last_date if last_date is not None else pd.Timestamp.today()

    future_dates = pd.date_range(base_date + pd.Timedelta(days=7), periods=26, freq="W")

    preds = []
    row = dict(base_row)

    for fdate in future_dates:
        X = pd.DataFrame([row])

        for col in features:
            if col not in X.columns:
                X[col] = np.nan
        X = X[features]

        dmat = xgb.DMatrix(X)
        pr = float(mr.predict(dmat)[0])
        pc = float(mc.predict(dmat)[0])

        preds.append({
            "fecha": fdate.date().isoformat(),
            "semana": int(fdate.isocalendar().week),
            "pred_racimos": pr,
            "pred_cajas": pc
        })

        # Alimentación recursiva solo si existen lags de targets
        for d in (7, 14, 30, 60):
            k_r = f"RACIMOS COSECHADOS_lag_{d}"
            k_c = f"CAJAS PROCESADAS_lag_{d}"
            if k_r in row:
                row[k_r] = pr
            if k_c in row:
                row[k_c] = pc

    # Guardar CSV local (para auditoría o para subir a FTP si quieres)
    os.makedirs(MODELS_DIR, exist_ok=True)
    df_forecast = pd.DataFrame(preds)
    df_forecast.to_csv(FORECAST_PATH, index=False)

    # Subir forecast al FTP (opcional: si quieres que exista en /exports)
    try:
        ftp_upload(FORECAST_PATH, "forecast_26w.csv")
    except Exception as e:
        print("WARNING: Falló upload FTP forecast_26w.csv:", repr(e))

    if export_csv == 1:
        csv_data = df_forecast.to_csv(index=False)
        return Response(content=csv_data, media_type="text/csv")

    return {"horizon_weeks": 26, "predictions": preds}


# =========================
# Export endpoints (descarga desde la API)
# =========================
@app.get("/metrics")
def get_metrics(x_api_key: str | None = Header(default=None)):
    check_key(x_api_key)

    if not os.path.exists(METRICS_PATH):
        raise HTTPException(status_code=404, detail="No hay métricas aún. Ejecuta /train primero.")

    return FileResponse(METRICS_PATH, media_type="application/json", filename="metrics_latest.json")


@app.get("/predictions/export")
def export_predictions(x_api_key: str | None = Header(default=None)):
    check_key(x_api_key)

    if not os.path.exists(LOG_PATH):
        raise HTTPException(status_code=404, detail="No hay predicciones guardadas aún. Usa /predict primero.")

    return FileResponse(LOG_PATH, media_type="text/csv", filename="predictions_log.csv")
