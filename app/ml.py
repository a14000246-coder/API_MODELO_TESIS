# ml.py
import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

TARGETS = ["RACIMOS COSECHADOS", "CAJAS PROCESADAS"]

# =========================
# Helpers: métricas / lectura
# =========================
def load_metrics(models_dir: str) -> dict:
    path = os.path.join(models_dir, "metrics_latest.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# =========================
# Agregación mensual
# =========================
def monthly_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte datos diarios/semanales a una tabla mensual.
    - Targets: SUM (producción mensual)
    - Features numéricas: MEAN (promedio mensual)
    - Columnas no numéricas (ej. texto): se omiten por simplicidad
    """
    df = df.copy()
    if "Fecha" not in df.columns:
        raise ValueError("El CSV debe tener columna 'Fecha'")

    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df = df.dropna(subset=["Fecha"])

    # Si hay viento como categoría, pásalo a dummies ANTES de agrupar (así promedias frecuencias)
    if "Dirección del viento" in df.columns:
        df = pd.get_dummies(df, columns=["Dirección del viento"], drop_first=True)

    # Nos quedamos con columnas numéricas + Fecha
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    keep = ["Fecha"] + [c for c in num_cols if c != "Fecha"]
    df = df[keep]

    # Asegurar targets existan (si no, igual agrupa features)
    for t in TARGETS:
        if t not in df.columns:
            df[t] = np.nan

    # Definir agregaciones
    agg = {}
    for c in df.columns:
        if c == "Fecha":
            continue
        if c in TARGETS:
            agg[c] = "sum"
        else:
            agg[c] = "mean"

    df["YM"] = df["Fecha"].dt.to_period("M")
    out = df.groupby("YM", as_index=False).agg(agg)
    out["Fecha"] = out["YM"].dt.to_timestamp()  # inicio de mes
    out = out.drop(columns=["YM"])

    # Calendario
    out["MES"] = out["Fecha"].dt.month
    out["AÑO"] = out["Fecha"].dt.year
    out["MES_sin"] = np.sin(2 * np.pi * out["MES"] / 12)
    out["MES_cos"] = np.cos(2 * np.pi * out["MES"] / 12)

    return out

def make_target_features_monthly(dfm: pd.DataFrame) -> pd.DataFrame:
    """
    Crea lags/rolling mensuales para targets, para que el forecast recursivo tenga de dónde alimentarse.
    """
    dfm = dfm.sort_values("Fecha").copy()

    for t in TARGETS:
        # lags
        for k in [1, 2, 3, 6, 12]:
            dfm[f"{t}_lag_{k}m"] = dfm[t].shift(k)

        # medias móviles
        for win in [3, 6, 12]:
            dfm[f"{t}_mm_{win}m"] = dfm[t].rolling(win).mean()

    return dfm

# =========================
# Preprocess (genérico)
# =========================
def preprocess(df: pd.DataFrame):
    df = df.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df = df.dropna(subset=["Fecha"])

    # Si llegaran categorías (en mensual normalmente ya no), dummies
    if "Dirección del viento" in df.columns:
        df = pd.get_dummies(df, columns=["Dirección del viento"], drop_first=True)

    # Targets
    df = df.dropna(subset=TARGETS)

    features = [c for c in df.columns if c not in TARGETS + ["Fecha"]]
    X = df[features]
    y1 = df[TARGETS[0]]
    y2 = df[TARGETS[1]]
    return X, y1, y2, features

def train_one(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=600,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mae = float(mean_absolute_error(y_test, pred))
    r2 = float(r2_score(y_test, pred))
    return model, {"rmse": rmse, "mae": mae, "r2": r2}

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_artifacts(models_dir: str, model_r, model_c, features, metrics):
    ensure_dir(models_dir)

    booster_r = model_r.get_booster()
    booster_c = model_c.get_booster()

    booster_r.save_model(os.path.join(models_dir, "model_racimos_latest.json"))
    booster_c.save_model(os.path.join(models_dir, "model_cajas_latest.json"))

    with open(os.path.join(models_dir, "features_latest.json"), "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=2)

    with open(os.path.join(models_dir, "metrics_latest.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    booster_r.save_model(os.path.join(models_dir, f"model_racimos_{stamp}.json"))
    booster_c.save_model(os.path.join(models_dir, f"model_cajas_{stamp}.json"))

def load_latest(models_dir: str):
    mr = xgb.Booster()
    mc = xgb.Booster()

    mr.load_model(os.path.join(models_dir, "model_racimos_latest.json"))
    mc.load_model(os.path.join(models_dir, "model_cajas_latest.json"))

    with open(os.path.join(models_dir, "features_latest.json"), "r", encoding="utf-8") as f:
        features = json.load(f)

    return mr, mc, features

def train_from_csv(csv_path: str, models_dir: str, granularity: str = "raw"):
    df = pd.read_csv(csv_path)

    # Granularidad
    granularity = (granularity or "raw").lower().strip()
    if granularity in ["monthly", "mensual"]:
        dfm = monthly_aggregate(df)
        dfm = make_target_features_monthly(dfm)
        df_use = dfm.dropna(subset=TARGETS)  # asegura targets presentes
    else:
        df_use = df

    X, y_r, y_c, features = preprocess(df_use)

    model_r, m1 = train_one(X, y_r)
    model_c, m2 = train_one(X, y_c)

    metrics = {
        "racimos": m1,
        "cajas": m2,
        "rows": int(len(df_use)),
        "features_count": int(len(features)),
        "granularity": granularity
    }

    save_artifacts(models_dir, model_r, model_c, features, metrics)
    return metrics

def predict_from_row(models_dir: str, row: dict):
    mr, mc, features = load_latest(models_dir)

    X = pd.DataFrame([row])
    for col in features:
        if col not in X.columns:
            X[col] = np.nan

    X = X[features]
    dmatrix = xgb.DMatrix(X)

    pr = float(mr.predict(dmatrix)[0])
    pc = float(mc.predict(dmatrix)[0])
    return {"pred_racimos": pr, "pred_cajas": pc}


