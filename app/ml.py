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
# NUEVO: Preprocesamiento mensual (para proyección por meses)
# =========================
def monthly_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega el dataset a nivel mensual y crea lags/rolling para mejorar la variabilidad.
    - Targets (RACIMOS COSECHADOS, CAJAS PROCESADAS): SUMA mensual.
    - Otras variables numéricas: PROMEDIO mensual (ajusta a SUMA si corresponde en tu caso).
    - Dirección del viento: moda mensual (si existe).
    """
    df = df.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df = df.dropna(subset=["Fecha"]).sort_values("Fecha")

    # Mes (inicio de mes)
    df["MesFecha"] = df["Fecha"].dt.to_period("M").dt.to_timestamp()

    sum_cols = [c for c in TARGETS if c in df.columns]
    # Nota: aquí promediamos el resto de columnas numéricas. Si tienes "Precipitación" diaria en mm/día,
    # puede convenirte sumar (total mensual) en vez de promediar.
    base_exclude = set(["Fecha", "MesFecha"])
    agg_dict = {}

    for c in df.columns:
        if c in base_exclude:
            continue
        if c in sum_cols:
            agg_dict[c] = "sum"
        elif c == "Dirección del viento":
            # la tratamos aparte
            continue
        else:
            # por defecto: promedio mensual
            agg_dict[c] = "mean"

    monthly = df.groupby("MesFecha", as_index=False).agg(agg_dict)

    # Dirección del viento: moda mensual (opcional)
    if "Dirección del viento" in df.columns:
        def mode_or_nan(s):
            s = s.dropna()
            return s.mode().iloc[0] if len(s) else np.nan
        wind = df.groupby("MesFecha")["Dirección del viento"].apply(mode_or_nan).reset_index()
        monthly = monthly.merge(wind, on="MesFecha", how="left")

    monthly = monthly.rename(columns={"MesFecha": "Fecha"}).sort_values("Fecha")

    # ============================
    # LAGS MENSUALES (1,2,3,6,12)
    # ============================
    if TARGETS[0] in monthly.columns:
        for lag in [1, 2, 3, 6, 12]:
            monthly[f"{TARGETS[0]}_lag_{lag}m"] = monthly[TARGETS[0]].shift(lag)

    if TARGETS[1] in monthly.columns:
        for lag in [1, 2, 3, 6, 12]:
            monthly[f"{TARGETS[1]}_lag_{lag}m"] = monthly[TARGETS[1]].shift(lag)

    # ============================
    # ROLLING MEAN (3,6,12 meses)
    # ============================
    if TARGETS[0] in monthly.columns:
        for win in [3, 6, 12]:
            monthly[f"{TARGETS[0]}_mm_{win}m"] = monthly[TARGETS[0]].rolling(win).mean()

    if TARGETS[1] in monthly.columns:
        for win in [3, 6, 12]:
            monthly[f"{TARGETS[1]}_mm_{win}m"] = monthly[TARGETS[1]].rolling(win).mean()

    # Calendario + estacionalidad cíclica
    monthly["MES"] = monthly["Fecha"].dt.month
    monthly["AÑO"] = monthly["Fecha"].dt.year
    monthly["MES_sin"] = np.sin(2 * np.pi * monthly["MES"] / 12)
    monthly["MES_cos"] = np.cos(2 * np.pi * monthly["MES"] / 12)

    return monthly


def preprocess_monthly(df: pd.DataFrame):
    dfm = monthly_aggregate(df)

    if "Dirección del viento" in dfm.columns:
        dfm = pd.get_dummies(dfm, columns=["Dirección del viento"], drop_first=True)

    dfm = dfm.dropna(subset=TARGETS)

    features = [c for c in dfm.columns if c not in TARGETS + ["Fecha"]]
    X = dfm[features]
    y1 = dfm[TARGETS[0]]
    y2 = dfm[TARGETS[1]]
    return X, y1, y2, features

def preprocess(df: pd.DataFrame):
    df = df.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"])

    if "Dirección del viento" in df.columns:
        df = pd.get_dummies(df, columns=["Dirección del viento"], drop_first=True)

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

    # Guardar boosters (evita error _estimator_type)
    booster_r = model_r.get_booster()
    booster_c = model_c.get_booster()

    # latest
    booster_r.save_model(os.path.join(models_dir, "model_racimos_latest.json"))
    booster_c.save_model(os.path.join(models_dir, "model_cajas_latest.json"))

    with open(os.path.join(models_dir, "features_latest.json"), "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=2)

    with open(os.path.join(models_dir, "metrics_latest.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # versionado por fecha
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

def train_from_csv(csv_path: str, models_dir: str, granularity: str = "daily"):
    df = pd.read_csv(csv_path)
    if granularity == "monthly":
        X, y_r, y_c, features = preprocess_monthly(df)
    else:
        X, y_r, y_c, features = preprocess(df)

    model_r, m1 = train_one(X, y_r)
    model_c, m2 = train_one(X, y_c)

    metrics = {
        "racimos": m1,
        "cajas": m2,
        "rows": int(len(df)),
        "features_count": int(len(features)),
        "granularity": granularity,
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

