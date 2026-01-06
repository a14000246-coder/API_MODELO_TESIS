from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from app.settings import API_KEY, DATA_PATH, MODELS_DIR
from app.ml import train_from_csv, predict_from_row

app = FastAPI(title="API Modelo Producción (XGBoost)")

def check_key(x_api_key: str | None):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API key inválida")

class PredictRequest(BaseModel):
    row: dict

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/train")
def train(x_api_key: str | None = Header(default=None)):
    check_key(x_api_key)
    metrics = train_from_csv(DATA_PATH, MODELS_DIR)
    return {"trained": True, "metrics": metrics}

@app.post("/predict")
def predict(req: PredictRequest, x_api_key: str | None = Header(default=None)):
    check_key(x_api_key)
    return predict_from_row(MODELS_DIR, req.row)
