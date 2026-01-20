import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

TARGETS = ["RACIMOS COSECHADOS", "CAJAS PROCESADAS"]


# =========================
# Preprocesamiento mensual (para proyección por meses)
# =========================
def monthly_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega el dataset a nivel mensual y crea lags/rolling para mejorar la variabilidad.
    - Targets (RACIMOS COSECHADOS, CAJAS PROCESADAS): SUMA mensual.
    - Otras variables: PROMEDIO mensual (ajusta a SUMA si corresponde en tu caso).
    - Dirección del viento: moda mensual (si existe).
    """
    df = df.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df = df.dropna(subset=["Fecha"]).sort_values("Fecha")

    # Mes (inicio de mes)
    df["MesFecha"] = df["Fecha"].dt.to_period("M").dt.to_timestamp()

    sum_cols = [c for c in TARGETS if c in df.columns]

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
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")

    if "Dirección del viento" in df.columns:
        df = pd.get_dummies(df, columns=["Dirección del viento"], drop_first=True)

    df = df.dropna(subset=TARGETS)

    features = [c for c in df.columns if c not in TARGETS + ["Fecha"]]
    X = df[features]
    y1 = df[TARGETS[0]]
    y2 = df[TARGETS[1]]
    return X, y1, y2, features


def _metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _pick_top_features(model: xgb.XGBRegressor, feature_names: list[str], top_k: int = 35) -> list[str]:
    """Selecciona features por importancia vía feature_importances_ (gain aproximado)."""
    importances = getattr(model, "feature_importances_", None)
    if importances is None or len(importances) != len(feature_names):
        return feature_names

    pairs = sorted(zip(feature_names, importances), key=lambda t: t[1], reverse=True)
    kept = [f for f, imp in pairs if imp > 0]
    if not kept:
        kept = feature_names
    return kept[: min(top_k, len(kept))]


def _train_with_params_ts(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    n_splits: int = 5,
    log1p: bool = True
):
    """Evalúa hiperparámetros con validación temporal (TimeSeriesSplit)."""
    # Por seguridad: con muy pocos puntos mensuales, bajar splits
    n_splits = min(n_splits, max(2, len(X) - 2))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses = []

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
        # ✅ Sin early stopping (compatibilidad Render)
        m.fit(
            X_tr,
            y_tr_fit,
            eval_set=[(X_val, y_val_fit)],
            verbose=False
        )

        pred = m.predict(X_val)
        if log1p:
            pred = np.expm1(pred)

        rmse = float(np.sqrt(mean_squared_error(y_val_true, pred)))
        rmses.append(rmse)

    return float(np.mean(rmses)) if rmses else float("inf")


def train_one(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    log1p: bool = True,
    top_k: int = 35,
    fixed_features: list[str] | None = None,
):
    """Fase 1:
    - Validación temporal (TimeSeriesSplit)
    - Búsqueda pequeña de hiperparámetros
    - Transformación log1p del target
    - Selección de features por importancia
    """

    # Si vienen features fijas (p.ej. seleccionadas con racimos), usar solo esas
    if fixed_features is not None:
        X = X[fixed_features]

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

    # ✅ Candidatos más livianos y con regularización (mejor para Render sin early stopping)
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

    # Elegir mejor set por CV temporal
    best_params = None
    best_rmse = float("inf")
    for params in candidates:
        rmse_cv = _train_with_params_ts(X_train, y_train, params, n_splits=5, log1p=log1p)
        if rmse_cv < best_rmse:
            best_rmse = rmse_cv
            best_params = params

    if best_params is None:
        best_params = candidates[0]
        best_rmse = float("inf")

    # Entrenamiento inicial con best params
    y_train_fit = np.log1p(y_train) if log1p else y_train
    y_test_true = y_test
    y_test_fit = np.log1p(y_test) if log1p else y_test

    model = xgb.XGBRegressor(**best_params)
    model.fit(
        X_train,
        y_train_fit,
        eval_set=[(X_test, y_test_fit)],
        verbose=False
    )

    # Selección de features (si no vienen fijas) y re-entrenamiento final
    selected = fixed_features if fixed_features is not None else _pick_top_features(model, list(X.columns), top_k=top_k)

    X_train_s = X_train[selected]
    X_test_s = X_test[selected]

    model2 = xgb.XGBRegressor(**best_params)
    model2.fit(
        X_train_s,
        y_train_fit,
        eval_set=[(X_test_s, y_test_fit)],
        verbose=False
    )

    pred = model2.predict(X_test_s)
    if log1p:
        pred = np.expm1(pred)

    m = _metrics(y_test_true, pred)
    m["cv_rmse_mean"] = float(best_rmse)
    m["selected_features"] = int(len(selected))
    m["target_transform"] = "log1p" if log1p else "none"
    m["best_params"] = {k: best_params[k] for k in best_params if k not in ("n_jobs",)}  # opcional
    return model2, m, selected


def load_metrics(models_dir: str) -> dict:
    """Lee metrics_latest.json si existe."""
    path = os.path.join(models_dir, "metrics_latest.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_artifacts(models_dir: str, model_r, model_c, features, metrics):
    ensure_dir(models_dir)

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
        X, y_r, y_c, _features = preprocess_monthly(df)
    else:
        X, y_r, y_c, _features = preprocess(df)

    # Entrenamos RACIMOS primero y usamos sus features seleccionadas
    model_r, m1, selected = train_one(X, y_r, log1p=True, top_k=35)
    # Entrenamos CAJAS con las mismas features (evita mismatch en inferencia)
    model_c, m2, _ = train_one(X, y_c, log1p=True, fixed_features=selected)

    metrics = {
        "racimos": m1,
        "cajas": m2,
        "rows": int(len(df)),
        "features_count": int(len(selected)),
        "granularity": granularity,
    }

    save_artifacts(models_dir, model_r, model_c, selected, metrics)
    return metrics


def predict_from_row(models_dir: str, row: dict):
    mr, mc, features = load_latest(models_dir)
    meta = load_metrics(models_dir)

    X = pd.DataFrame([row])

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
