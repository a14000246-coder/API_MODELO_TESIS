import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

TARGETS = ["RACIMOS COSECHADOS", "CAJAS PROCESADAS"]

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

def train_from_csv(csv_path: str, models_dir: str):
    df = pd.read_csv(csv_path)
    X, y_r, y_c, features = preprocess(df)

    model_r, m1 = train_one(X, y_r)
    model_c, m2 = train_one(X, y_c)

    metrics = {
        "racimos": m1,
        "cajas": m2,
        "rows": int(len(df)),
        "features_count": int(len(features)),
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

