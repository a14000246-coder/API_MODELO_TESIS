import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Targets (columnas obligatorias)
TARGETS = ["RACIMOS COSECHADOS", "CAJAS PROCESADAS"]

# Columnas que suelen venir con mala codificación en CSVs
MOJIBAKE_MAP = {
    "PrecipitaciÃ³n (%)": "Precipitación (%)",
    "Precipitacion (%)": "Precipitación (%)",
}

# =========================
# Helpers
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_metrics(models_dir: str) -> dict:
    path = os.path.join(models_dir, "metrics_latest.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }

def _pick_top_features(model: xgb.XGBRegressor, feature_names: list[str], top_k: int = 45) -> list[str]:
    """
    Selecciona features por GAIN (más estable que feature_importances_).
    """
    try:
        booster = model.get_booster()
        score = booster.get_score(importance_type="gain")
        if not score:
            return feature_names[: min(top_k, len(feature_names))]

        pairs = sorted(score.items(), key=lambda x: x[1], reverse=True)
        kept = [k for k, _ in pairs if k in feature_names]
        if not kept:
            return feature_names[: min(top_k, len(feature_names))]
        return kept[: min(top_k, len(kept))]
    except Exception:
        return feature_names[: min(top_k, len(feature_names))]

def _train_with_params_ts(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    n_splits: int = 3,
    log1p: bool = True,
):
    """
    Evalúa hiperparámetros con validación temporal (TimeSeriesSplit).
    Sin early stopping para compatibilidad con entornos que no aceptan callbacks/early_stopping_rounds.
    """
    # Con pocos puntos mensuales, 3 splits suele ser más estable que 5
    n_splits = min(n_splits, max(2, len(X) - 2))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses: list[float] = []

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if log1p:
            y_tr_fit = np.log1p(y_tr)
            y_val_true = y_val
            y_val_fit = np.log1p(y_val)
        else:
            y_tr_fit = y_tr
            y_val_true = y_val
            y_val_fit = y_val

        m = xgb.XGBRegressor(**params)
        m.fit(X_tr, y_tr_fit, eval_set=[(X_val, y_val_fit)], verbose=False)

        pred = m.predict(X_val)
        if log1p:
            pred = np.expm1(pred)

        rmse = float(np.sqrt(mean_squared_error(y_val_true, pred)))
        rmses.append(rmse)

    return float(np.mean(rmses)) if rmses else float("inf")

# =========================
# Preprocesamiento
# =========================
def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.rename(columns=MOJIBAKE_MAP)
    return df

def preprocess(df: pd.DataFrame):
    """
    Preprocesamiento DAILY:
    - Respeta lags/rolling ya calculados en el CSV mejorado
    - One-hot de Dirección del viento (si existe)
    """
    df = _standardize_columns(df)
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df = df.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)

    # One-hot si existe
    if "Dirección del viento" in df.columns:
        df["Dirección del viento"] = df["Dirección del viento"].astype(str).replace({"nan": np.nan})
        df["Dirección del viento"] = df["Dirección del viento"].fillna(df["Dirección del viento"].mode().iloc[0])
        df = pd.get_dummies(df, columns=["Dirección del viento"], drop_first=True)

    # Asegurar targets
    df = df.dropna(subset=[c for c in TARGETS if c in df.columns])

    # Features: todo menos targets y Fecha
    features = [c for c in df.columns if c not in TARGETS + ["Fecha"]]
    X = df[features]
    y_r = df[TARGETS[0]]
    y_c = df[TARGETS[1]]
    return X, y_r, y_c, features

def monthly_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega a nivel mensual y RE-CREA lags/rolling mensuales.
    Importante para el CSV mejorado:
    - El CSV ya trae columnas derivadas daily (lags/rolling) basadas en targets.
      Para mensual las descartamos para evitar fuga de información y ruido.
    """
    df = _standardize_columns(df)
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df = df.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)

    # 🔥 Eliminar columnas derivadas DAILY basadas en targets (excepto los targets)
    # Ej: "RACIMOS COSECHADOS_mm_7", "RACIMOS COSECHADOS_lag_7", etc.
    for t in TARGETS:
        drop_like = [c for c in df.columns if c.startswith(f"{t}_")]
        if drop_like:
            df = df.drop(columns=drop_like, errors="ignore")

    # Quitar columnas calendario si ya venían (las recalculamos)
    for c in ["AÑO", "MES", "SEMANAS", "dia_anio", "Dia_J", "semana"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Mes (inicio de mes)
    df["MesFecha"] = df["Fecha"].dt.to_period("M").dt.to_timestamp()

    sum_cols = [c for c in TARGETS if c in df.columns]
    base_exclude = {"Fecha", "MesFecha"}

    agg_dict: dict[str, str] = {}
    for c in df.columns:
        if c in base_exclude:
            continue
        if c in sum_cols:
            agg_dict[c] = "sum"
        elif c == "Dirección del viento":
            continue
        elif c in ["Precipitación (%)"]:
            # Porcentaje -> promedio mensual
            agg_dict[c] = "mean"
        else:
            agg_dict[c] = "mean"

    monthly = df.groupby("MesFecha", as_index=False).agg(agg_dict)

    # Dirección del viento: moda mensual (si existe)
    if "Dirección del viento" in df.columns:
        def mode_or_nan(s):
            s = s.dropna()
            return s.mode().iloc[0] if len(s) else np.nan

        wind = df.groupby("MesFecha")["Dirección del viento"].apply(mode_or_nan).reset_index()
        monthly = monthly.merge(wind, on="MesFecha", how="left")

    monthly = monthly.rename(columns={"MesFecha": "Fecha"}).sort_values("Fecha").reset_index(drop=True)

    # Feature derivada lluvia por % (si existe)
    if "Precipitación (%)" in monthly.columns and "Dias_lluvia_aprox" not in monthly.columns:
        monthly["Dias_lluvia_aprox"] = (pd.to_numeric(monthly["Precipitación (%)"], errors="coerce").fillna(0) / 100.0) * 30.0

    # Lags mensuales (1,2,3,6,12)
    for t in TARGETS:
        if t in monthly.columns:
            for lag in [1, 2, 3, 6, 12]:
                monthly[f"{t}_lag_{lag}m"] = monthly[t].shift(lag)

    # Rolling mean (3,6,12)
    for t in TARGETS:
        if t in monthly.columns:
            for win in [3, 6, 12]:
                monthly[f"{t}_mm_{win}m"] = monthly[t].rolling(win).mean()

    # Calendario + estacionalidad cíclica
    monthly["MES"] = monthly["Fecha"].dt.month
    monthly["AÑO"] = monthly["Fecha"].dt.year
    monthly["MES_sin"] = np.sin(2 * np.pi * monthly["MES"] / 12)
    monthly["MES_cos"] = np.cos(2 * np.pi * monthly["MES"] / 12)

    return monthly

def preprocess_monthly(df: pd.DataFrame):
    dfm = monthly_aggregate(df)

    if "Dirección del viento" in dfm.columns:
        dfm["Dirección del viento"] = dfm["Dirección del viento"].astype(str).replace({"nan": np.nan})
        dfm["Dirección del viento"] = dfm["Dirección del viento"].fillna(dfm["Dirección del viento"].mode().iloc[0])
        dfm = pd.get_dummies(dfm, columns=["Dirección del viento"], drop_first=True)

    dfm = dfm.dropna(subset=[c for c in TARGETS if c in dfm.columns])

    features = [c for c in dfm.columns if c not in TARGETS + ["Fecha"]]
    X = dfm[features]
    y_r = dfm[TARGETS[0]]
    y_c = dfm[TARGETS[1]]
    return X, y_r, y_c, features

# =========================
# Entrenamiento
# =========================
def train_one(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    log1p: bool = True,
    top_k: int = 45,
):
    """
    - CV temporal (3 splits)
    - mini-búsqueda de hiperparámetros
    - log1p del target (default)
    - selección de features por GAIN
    """
    # Split final (último 20% como test)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    base = {
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }

    # Livianos + regularizados (sin early stopping)
    candidates = [
        {**base, "learning_rate": 0.03, "max_depth": 6, "min_child_weight": 3,
         "subsample": 0.8, "colsample_bytree": 0.8, "gamma": 0.1,
         "reg_alpha": 0.5, "reg_lambda": 2.0, "n_estimators": 900},

        {**base, "learning_rate": 0.025, "max_depth": 5, "min_child_weight": 4,
         "subsample": 0.85, "colsample_bytree": 0.85, "gamma": 0.15,
         "reg_alpha": 1.0, "reg_lambda": 3.0, "n_estimators": 1100},

        {**base, "learning_rate": 0.04, "max_depth": 7, "min_child_weight": 2,
         "subsample": 0.75, "colsample_bytree": 0.75, "gamma": 0.05,
         "reg_alpha": 0.3, "reg_lambda": 1.5, "n_estimators": 700},
    ]

    best_params = None
    best_rmse = float("inf")
    for params in candidates:
        rmse_cv = _train_with_params_ts(X_train, y_train, params, n_splits=3, log1p=log1p)
        if rmse_cv < best_rmse:
            best_rmse = rmse_cv
            best_params = params

    if best_params is None:
        best_params = candidates[0]
        best_rmse = float("inf")

    y_train_fit = np.log1p(y_train) if log1p else y_train
    y_test_true = y_test
    y_test_fit = np.log1p(y_test) if log1p else y_test

    # Entrenamiento inicial
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train_fit, eval_set=[(X_test, y_test_fit)], verbose=False)

    # Selección de features
    selected = _pick_top_features(model, list(X.columns), top_k=top_k)

    # Re-entrenamiento final
    X_train_s = X_train[selected]
    X_test_s = X_test[selected]

    model2 = xgb.XGBRegressor(**best_params)
    model2.fit(X_train_s, y_train_fit, eval_set=[(X_test_s, y_test_fit)], verbose=False)

    pred = model2.predict(X_test_s)
    if log1p:
        pred = np.expm1(pred)

    m = _metrics(y_test_true, pred)
    m["cv_rmse_mean"] = float(best_rmse)
    m["selected_features"] = int(len(selected))
    m["target_transform"] = "log1p" if log1p else "none"
    m["best_params"] = {k: best_params[k] for k in best_params if k not in ("n_jobs",)}
    return model2, m, selected

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

def train_from_csv(csv_path: str, models_dir: str, granularity: str = "daily"):
    # BOM-friendly read
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if granularity == "monthly":
        X, y_r, y_c, _ = preprocess_monthly(df)
    else:
        X, y_r, y_c, _ = preprocess(df)

    # Entrena cada target con sus propias features y guarda la UNIÓN para inferencia
    model_r, m1, feats_r = train_one(X, y_r, log1p=True, top_k=45)
    model_c, m2, feats_c = train_one(X, y_c, log1p=True, top_k=45)

    all_features = sorted(list(set(feats_r) | set(feats_c)))

    metrics = {
        "racimos": m1,
        "cajas": m2,
        "rows": int(len(df)),
        "granularity": granularity,
        "features_count": int(len(all_features)),
        "features_racimos": feats_r,
        "features_cajas": feats_c,
    }

    save_artifacts(models_dir, model_r, model_c, all_features, metrics)
    return metrics

def predict_from_row(models_dir: str, row: dict):
    mr, mc, features = load_latest(models_dir)
    meta = load_metrics(models_dir)

    X = pd.DataFrame([row])
    X = _standardize_columns(X)

    for col in features:
        if col not in X.columns:
            X[col] = np.nan

    X = X[features]
    dmatrix = xgb.DMatrix(X)

    pr = float(mr.predict(dmatrix)[0])
    pc = float(mc.predict(dmatrix)[0])

    # Si entrenamos con log1p, invertimos a escala real
    try:
        tr_r = meta.get("racimos", {}).get("target_transform")
        tr_c = meta.get("cajas", {}).get("target_transform")
        if tr_r == "log1p":
            pr = float(np.expm1(pr))
        if tr_c == "log1p":
            pc = float(np.expm1(pc))
    except Exception:
        pass

    return {"pred_racimos": pr, "pred_cajas": pc}
