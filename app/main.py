import os
from datetime import datetime
from ftplib import FTP, error_perm

import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.settings import API_KEY, DATA_PATH, MODELS_DIR
from app.ml import train_from_csv, predict_from_row

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

# Carpeta remota donde se guardarán los archivos
# Ejemplos comunes:
#   "/public_html/exports"
#   "/htdocs/exports"
#   "/exports"
FTP_DIR = os.getenv("FTP_DIR", "/exports").strip()

# Activa/desactiva subida automática
UPLOAD_ENABLED = os.getenv("UPLOAD_ENABLED", "1").strip()  # "1" o "0"

# =========================
# Rutas locales de archivos
# =========================
LOG_PATH = os.path.join(MODELS_DIR, "predictions_log.csv")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics_latest.json")


# =========================
# FTP helpers
# =========================
def _ftp_ensure_dir(ftp: FTP, remote_dir: str):
    """
    Crea (si hace falta) una ruta remota tipo /a/b/c y entra a ella.
    """
    parts = [p for p in remote_dir.split("/") if p]
    for p in parts:
        try:
            ftp.cwd(p)
        except error_perm:
            ftp.mkd(p)
            ftp.cwd(p)


def ftp_upload(local_path: str, remote_filename: str):
    """
    Sube un archivo local a FTP_DIR/remote_filename
    """
    if UPLOAD_ENABLED != "1":
        return
    if not (FTP_HOST and FTP_USER and FTP_PASS):
        # No rompe la API si no se configuró FTP en Render
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

    # Entrena y guarda artifacts (incluye metrics_latest.json)
    metrics = train_from_csv(DATA_PATH, MODELS_DIR)

    # Subir métricas al hosting por FTP
    try:
        ftp_upload(METRICS_PATH, "metrics_latest.json")
    except Exception as e:
        # No rompe el entrenamiento; se ve en logs de Render
        print("WARNING: Falló upload FTP metrics_latest.json:", repr(e))

    return {"trained": True, "metrics": metrics}


# =========================
# Predicción + log CSV + subida automática (FTP)
# =========================
@app.post("/predict")
def predict(req: PredictRequest, x_api_key: str | None = Header(default=None)):
    check_key(x_api_key)

    result = predict_from_row(MODELS_DIR, req.row)

    # Asegurar carpeta local
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Log: entrada + salida
    row_log = {
        "timestamp": datetime.now().isoformat(),
        **req.row,
        **result,
    }

    df_log = pd.DataFrame([row_log])

    # Append a CSV
    if os.path.exists(LOG_PATH):
        df_log.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        df_log.to_csv(LOG_PATH, mode="w", header=True, index=False)

    # Subir CSV al hosting por FTP
    try:
        ftp_upload(LOG_PATH, "predictions_log.csv")
    except Exception as e:
        print("WARNING: Falló upload FTP predictions_log.csv:", repr(e))

    return result


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
