import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# -----------------------------
# Optional dependencies
# -----------------------------
try:
    from xgboost import XGBRegressor

    HAS_XGB = True
except Exception:
    XGBRegressor = None
    HAS_XGB = False

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    HAS_STATSMODELS = True
except Exception:
    ExponentialSmoothing = None
    HAS_STATSMODELS = False

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import GRU, LSTM, Dense
    from tensorflow.keras.models import Sequential

    HAS_TF = True
except Exception:
    tf = None
    EarlyStopping = None
    LSTM = None
    GRU = None
    Dense = None
    Sequential = None
    HAS_TF = False


# ============================================================
# Core utilities
# ============================================================


def get_clean_data(df: pd.DataFrame, target_col: str = "Vazao_BBR") -> pd.DataFrame:
    """
    Remove linhas com target inválido (-1) e reseta índice.
    """
    df_clean = df[df[target_col] != -1].copy().reset_index(drop=True)
    print(
        f"  [DADOS LIMPOS] {len(df_clean)} amostras válidas (removidos {len(df) - len(df_clean)} com -1)"
    )
    return df_clean


def apply_random_mask(
    df: pd.DataFrame, missing_fraction: float, seed: int = 42
) -> pd.DataFrame:
    """
    Aplica máscara aleatória para simular missing na avaliação.
    """
    df_masked = df.copy()
    n_samples = len(df_masked)
    n_mask = max(1, int(missing_fraction * n_samples))

    rng = np.random.default_rng(seed)
    mask_indices = rng.choice(df_masked.index.to_numpy(), size=n_mask, replace=False)
    df_masked["mask_applied"] = 0
    df_masked.loc[mask_indices, "mask_applied"] = 1

    print(
        f"    Máscara aplicada: {n_mask}/{n_samples} amostras ({missing_fraction * 100:.0f}%)"
    )
    return df_masked


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, prediction_time: float = None
) -> Dict[str, Any]:
    """
    Calcula métricas de regressão de forma robusta.
    rmse fica em milhões como no seu código original.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
    if mask.sum() < 2:
        return {
            "rmse": None,
            "nrmse": None,
            "r2": None,
            "mape": None,
            "prediction_time_per_sample": None,
        }

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    try:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        nrmse = float((rmse / (np.mean(y_true) + 1e-8)) * 100)
        rmse_normalized = rmse / 1_000_000
        r2 = r2_score(y_true, y_pred)
        r2 = float(r2) if np.isfinite(r2) else None
        mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)

        time_per_sample = None
        if prediction_time is not None and len(y_true) > 0:
            time_per_sample = round((prediction_time / len(y_true)) * 1000, 4)

        return {
            "rmse": round(rmse_normalized, 2),
            "nrmse": round(nrmse, 2),
            "r2": r2,
            "mape": round(mape, 2),
            "prediction_time_per_sample": time_per_sample,
        }
    except Exception:
        return {
            "rmse": None,
            "nrmse": None,
            "r2": None,
            "mape": None,
            "prediction_time_per_sample": None,
        }


# ============================================================
# Feature engineering for tabular models
# ============================================================


def engineer_features_for_imputation(
    df: pd.DataFrame, target_col: str = "Vazao_BBR"
) -> pd.DataFrame:
    """
    Feature engineering sem leakage direto do futuro.
    """
    df = df.copy()

    if "Data" in df.columns:
        df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
        df["hour"] = df["Data"].dt.hour
        df["day_of_week"] = df["Data"].dt.dayofweek
        df["day_of_month"] = df["Data"].dt.day

    if "Atraso(ms)" in df.columns:
        df["Atraso_log"] = np.log1p(df["Atraso(ms)"].clip(lower=0))
        df["Atraso_sq"] = df["Atraso(ms)"] ** 2
    if "Hop_count" in df.columns:
        df["Hop_inv"] = 1 / (df["Hop_count"] + 1)
        df["Hop_sq"] = df["Hop_count"] ** 2

    if "Atraso(ms)" in df.columns and "Hop_count" in df.columns:
        df["Atraso_x_Hop"] = df["Atraso(ms)"] * df["Hop_count"]

    if "hour" in df.columns and "Atraso(ms)" in df.columns:
        df["Atraso_x_hour"] = df["Atraso(ms)"] * df["hour"]
    if "hour" in df.columns and "Hop_count" in df.columns:
        df["Hop_x_hour"] = df["Hop_count"] * df["hour"]
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    valid_mask = df[target_col] != -1
    target_series = df[target_col].copy()
    target_series[~valid_mask] = np.nan

    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"Vazao_lag{lag}"] = target_series.shift(lag)

    df["Vazao_diff1"] = target_series.diff(1)
    df["Vazao_diff2"] = target_series.diff(2)
    df["Vazao_pct_change"] = target_series.pct_change()

    shifted = target_series.shift(1)
    for w in [3, 6, 12]:
        df[f"Vazao_roll_mean_{w}"] = shifted.rolling(window=w, min_periods=1).mean()
        df[f"Vazao_roll_std_{w}"] = shifted.rolling(window=w, min_periods=2).std()
        df[f"Vazao_roll_median_{w}"] = shifted.rolling(window=w, min_periods=1).median()

    df["Vazao_roll_max_6"] = shifted.rolling(window=6, min_periods=1).max()
    df["Vazao_roll_min_6"] = shifted.rolling(window=6, min_periods=1).min()
    df["Vazao_expanding_mean"] = shifted.expanding(min_periods=1).mean()
    df["Vazao_expanding_std"] = shifted.expanding(min_periods=3).std()

    lag1 = target_series.shift(1)
    if "Atraso(ms)" in df.columns:
        df["Vazao_lag1_div_Atraso"] = lag1 / (df["Atraso(ms)"] + 1)
    if "Hop_count" in df.columns:
        df["Vazao_lag1_div_Hops"] = lag1 / (df["Hop_count"] + 1)
    if "Atraso(ms)" in df.columns and "Hop_count" in df.columns:
        df["Efficiency_lag1"] = lag1 / ((df["Atraso(ms)"] + 1) * (df["Hop_count"] + 1))

    df["Vazao_lag1_log"] = np.log1p(lag1.clip(lower=0))
    df["Vazao_lag1_sqrt"] = np.sqrt(lag1.clip(lower=0))

    df["Feature_Vazao_bbr_median"] = target_series.median()
    df["Feature_Vazao_bbr_mean"] = target_series.mean()

    df.loc[~valid_mask, target_col] = -1
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    for col in df.select_dtypes(include=[np.number]).columns:
        if col == target_col:
            continue
        if df[col].isna().any():
            if any(k in col for k in ["lag", "roll", "diff", "pct", "expanding"]):
                df[col] = df[col].fillna(0)
            else:
                med = df[col].median()
                df[col] = df[col].fillna(0 if pd.isna(med) else med)

    return df


# ============================================================
# Time-series baselines
# ============================================================


def _sequential_causal_fill(series: pd.Series, fill_func) -> pd.Series:
    s = series.copy().astype(float)
    for i in range(len(s)):
        if pd.isna(s.iloc[i]):
            hist = s.iloc[:i].dropna()
            if len(hist) == 0:
                continue
            s.iloc[i] = fill_func(hist, i)
    return s


def _fill_with_last_value(series: pd.Series) -> pd.Series:
    return series.ffill().bfill()


def _fill_with_rolling_mean_causal(series: pd.Series, window: int = 3) -> pd.Series:
    def fill_func(hist, i):
        return hist.iloc[-window:].mean()

    return _sequential_causal_fill(series, fill_func).ffill().bfill()


def _fill_with_rolling_median_causal(series: pd.Series, window: int = 3) -> pd.Series:
    def fill_func(hist, i):
        return hist.iloc[-window:].median()

    return _sequential_causal_fill(series, fill_func).ffill().bfill()


def _fill_with_ewma_causal(series: pd.Series, span: int = 3) -> pd.Series:
    def fill_func(hist, i):
        return hist.ewm(span=span, adjust=False).mean().iloc[-1]

    return _sequential_causal_fill(series, fill_func).ffill().bfill()


def _fill_with_linear_trend_causal(series: pd.Series, window: int = 5) -> pd.Series:
    def fill_func(hist, i):
        tail = hist.iloc[-window:]
        if len(tail) == 1:
            return tail.iloc[-1]
        y = tail.values.astype(float)
        x = np.arange(len(y))
        try:
            coef = np.polyfit(x, y, deg=1)
            return float(coef[0] * len(y) + coef[1])
        except Exception:
            return float(tail.iloc[-1])

    return _sequential_causal_fill(series, fill_func).ffill().bfill()


def _fill_with_seasonal_naive(
    series: pd.Series, seasonal_period: int = 24
) -> pd.Series:
    s = series.copy().astype(float)
    for i in range(len(s)):
        if pd.isna(s.iloc[i]):
            ref = i - seasonal_period
            if ref >= 0 and not pd.isna(s.iloc[ref]):
                s.iloc[i] = s.iloc[ref]
            else:
                hist = s.iloc[:i].dropna()
                if len(hist) > 0:
                    s.iloc[i] = hist.iloc[-1]
    return s.ffill().bfill()


def _fill_with_exp_smoothing(series: pd.Series, seasonal_period: int = 24) -> pd.Series:
    s = series.copy().astype(float)
    observed = s.dropna()
    if len(observed) < 5 or not HAS_STATSMODELS:
        return _fill_with_ewma_causal(s, span=5)

    try:
        trend = "add" if len(observed) >= 10 else None
        seasonal = "add" if len(observed) >= 2 * seasonal_period else None
        model = ExponentialSmoothing(
            observed,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_period if seasonal else None,
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True)
        filled = s.copy()
        missing_idx = np.where(filled.isna())[0]
        if len(missing_idx) == 0:
            return filled
        fcst = fit.forecast(len(missing_idx))
        for idx, val in zip(missing_idx, np.array(fcst)):
            filled.iloc[idx] = val
        return filled.ffill().bfill()
    except Exception:
        return _fill_with_ewma_causal(s, span=5)


# ============================================================
# Supervised lag models
# ============================================================


def _create_lag_supervised_dataset(
    series: pd.Series, lags: int = 12
) -> Tuple[np.ndarray, np.ndarray]:
    vals = series.astype(float).values
    X, y = [], []
    for i in range(lags, len(vals)):
        window = vals[i - lags : i]
        target = vals[i]
        if np.any(np.isnan(window)) or np.isnan(target):
            continue
        X.append(window.copy())
        y.append(target)
    if not X:
        return np.empty((0, lags)), np.empty((0,))
    return np.asarray(X), np.asarray(y)


def _sequential_model_fill(
    series: pd.Series,
    predictor_func,
    lags: int = 12,
    initial_fallback: str = "median",
) -> pd.Series:
    s = series.copy().astype(float)
    for i in range(len(s)):
        if pd.isna(s.iloc[i]):
            hist = s.iloc[:i].copy()
            if len(hist.dropna()) < max(5, lags):
                if initial_fallback == "median":
                    fill_val = (
                        float(hist.dropna().median())
                        if len(hist.dropna())
                        else float(s.dropna().median())
                    )
                else:
                    fill_val = (
                        float(hist.dropna().iloc[-1])
                        if len(hist.dropna())
                        else float(s.dropna().median())
                    )
                s.iloc[i] = fill_val
                continue

            work = hist.copy().ffill().bfill()
            X_train, y_train = _create_lag_supervised_dataset(work, lags=lags)
            if len(X_train) < 10:
                s.iloc[i] = float(work.iloc[-1])
                continue

            recent = s.iloc[i - lags : i].copy()
            recent = recent.ffill().bfill()
            if recent.isna().any():
                s.iloc[i] = float(work.iloc[-1])
                continue

            try:
                pred = predictor_func(X_train, y_train, recent.values.reshape(1, -1))
                s.iloc[i] = float(pred)
            except Exception:
                s.iloc[i] = float(work.iloc[-1])

    return s.ffill().bfill()


def _predict_with_ridge(X_train, y_train, X_pred):
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )
    model.fit(X_train, y_train)
    return model.predict(X_pred)[0]


def _predict_with_knn_lag(X_train, y_train, X_pred):
    n_neighbors = min(5, len(X_train))
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "knn",
                KNeighborsRegressor(
                    n_neighbors=max(1, n_neighbors), weights="distance"
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model.predict(X_pred)[0]


def _predict_with_rf_lag(X_train, y_train, X_pred):
    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=1,
    )
    model.fit(X_train, y_train)
    return model.predict(X_pred)[0]


def _predict_with_gb_lag(X_train, y_train, X_pred):
    model = GradientBoostingRegressor(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model.predict(X_pred)[0]


def _predict_with_xgb_lag(X_train, y_train, X_pred):
    if not HAS_XGB:
        return _predict_with_gb_lag(X_train, y_train, X_pred)
    model = XGBRegressor(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.9,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model.predict(X_pred)[0]


def _fill_with_ridge_lag(series: pd.Series, lags: int = 12) -> pd.Series:
    return _sequential_model_fill(series, _predict_with_ridge, lags=lags)


def _fill_with_knn_lag(series: pd.Series, lags: int = 12) -> pd.Series:
    return _sequential_model_fill(series, _predict_with_knn_lag, lags=lags)


def _fill_with_rf_lag(series: pd.Series, lags: int = 12) -> pd.Series:
    return _sequential_model_fill(series, _predict_with_rf_lag, lags=lags)


def _fill_with_gb_lag(series: pd.Series, lags: int = 12) -> pd.Series:
    return _sequential_model_fill(series, _predict_with_gb_lag, lags=lags)


def _fill_with_xgb_lag(series: pd.Series, lags: int = 12) -> pd.Series:
    return _sequential_model_fill(series, _predict_with_xgb_lag, lags=lags)


# ============================================================
# Neural lag models (GRU / LSTM)
# ============================================================


def _build_sequence_data(
    series: pd.Series, lags: int = 12
) -> Tuple[np.ndarray, np.ndarray]:
    X, y = _create_lag_supervised_dataset(series, lags=lags)
    if len(X) == 0:
        return np.empty((0, lags, 1)), np.empty((0,))
    return X.reshape(len(X), lags, 1), y


def _predict_with_tf_sequence(X_train_3d, y_train, X_pred_3d, model_type="gru"):
    if not HAS_TF:
        raise RuntimeError("TensorFlow não disponível")

    tf.keras.utils.set_random_seed(42)

    model = Sequential()
    if model_type == "gru":
        model.add(GRU(32, input_shape=(X_train_3d.shape[1], X_train_3d.shape[2])))
    else:
        model.add(LSTM(32, input_shape=(X_train_3d.shape[1], X_train_3d.shape[2])))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ]

    if len(X_train_3d) >= 20:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_3d, y_train, test_size=0.2, shuffle=False
        )
        model.fit(
            X_tr,
            y_tr,
            validation_data=(X_val, y_val),
            epochs=40,
            batch_size=min(16, len(X_tr)),
            verbose=0,
            callbacks=callbacks,
        )
    else:
        model.fit(
            X_train_3d,
            y_train,
            epochs=20,
            batch_size=min(8, len(X_train_3d)),
            verbose=0,
        )

    pred = model.predict(X_pred_3d, verbose=0).ravel()[0]
    return float(pred)


def _sequential_neural_fill(
    series: pd.Series, lags: int = 12, model_type: str = "gru"
) -> pd.Series:
    if not HAS_TF:
        # fallback honesto e estável
        return _fill_with_xgb_lag(series, lags=lags)

    s = series.copy().astype(float)
    for i in range(len(s)):
        if pd.isna(s.iloc[i]):
            hist = s.iloc[:i].copy().ffill().bfill()
            if len(hist.dropna()) < max(25, lags + 5):
                fallback = (
                    hist.dropna().iloc[-1]
                    if len(hist.dropna())
                    else s.dropna().median()
                )
                s.iloc[i] = float(fallback)
                continue

            X_train_3d, y_train = _build_sequence_data(hist, lags=lags)
            if len(X_train_3d) < 20:
                s.iloc[i] = float(hist.iloc[-1])
                continue

            recent = s.iloc[i - lags : i].copy().ffill().bfill()
            if recent.isna().any():
                s.iloc[i] = float(hist.iloc[-1])
                continue

            X_pred_3d = recent.values.reshape(1, lags, 1)
            try:
                pred = _predict_with_tf_sequence(
                    X_train_3d, y_train, X_pred_3d, model_type=model_type
                )
                s.iloc[i] = pred
            except Exception:
                s.iloc[i] = float(hist.iloc[-1])

    return s.ffill().bfill()


def _fill_with_gru(series: pd.Series, lags: int = 12) -> pd.Series:
    return _sequential_neural_fill(series, lags=lags, model_type="gru")


def _fill_with_lstm(series: pd.Series, lags: int = 12) -> pd.Series:
    return _sequential_neural_fill(series, lags=lags, model_type="lstm")


# ============================================================
# Evaluation of baselines / models
# ============================================================


def _safe_metric_eval(name: str, y_true, y_pred, results: Dict[str, Any]):
    try:
        results[name] = calculate_metrics(y_true, y_pred)
    except Exception:
        results[name] = {
            "rmse": None,
            "nrmse": None,
            "r2": None,
            "mape": None,
            "prediction_time_per_sample": None,
        }


def evaluate_baselines(
    df_clean: pd.DataFrame,
    missing_fraction: float,
    target_col: str = "Vazao_BBR",
    seasonal_period: int = 24,
    lag_window: int = 12,
) -> Dict:
    """
    Avalia baselines clássicas, temporais, modelos com lag e GRU/LSTM.
    """
    print(f"  [BASELINE/MODELOS] Avaliando fração {missing_fraction:.0%}")

    df_masked = apply_random_mask(df_clean, missing_fraction, seed=42)
    mask_indices = df_masked[df_masked["mask_applied"] == 1].index
    y_true = df_masked.loc[mask_indices, target_col].values

    results = {}
    df_with_nan = df_masked.copy()
    df_with_nan.loc[mask_indices, target_col] = np.nan
    target_series = df_with_nan[target_col].astype(float).copy()

    # Clássicas
    mean_value = target_series.mean()
    _safe_metric_eval(
        "Mean",
        y_true,
        target_series.fillna(mean_value).loc[mask_indices].values,
        results,
    )

    median_value = target_series.median()
    _safe_metric_eval(
        "Median",
        y_true,
        target_series.fillna(median_value).loc[mask_indices].values,
        results,
    )

    try:
        imputer = KNNImputer(n_neighbors=max(1, min(5, len(df_clean) // 2)))
        knn_filled = pd.Series(
            imputer.fit_transform(df_with_nan[[target_col]]).ravel(),
            index=df_with_nan.index,
        )
        _safe_metric_eval(
            "KNNImputer", y_true, knn_filled.loc[mask_indices].values, results
        )
    except Exception:
        results["KNNImputer"] = {
            "rmse": None,
            "nrmse": None,
            "r2": None,
            "mape": None,
            "prediction_time_per_sample": None,
        }

    # Temporais
    _safe_metric_eval(
        "ForwardFill",
        y_true,
        _fill_with_last_value(target_series).loc[mask_indices].values,
        results,
    )
    _safe_metric_eval(
        "BackwardFill",
        y_true,
        target_series.bfill().ffill().loc[mask_indices].values,
        results,
    )

    rolling_mean = target_series.rolling(window=3, min_periods=1).mean().ffill().bfill()
    _safe_metric_eval(
        "RollingMean", y_true, rolling_mean.loc[mask_indices].values, results
    )

    _safe_metric_eval(
        "RollingMeanCausal_w3",
        y_true,
        _fill_with_rolling_mean_causal(target_series, window=3)
        .loc[mask_indices]
        .values,
        results,
    )
    _safe_metric_eval(
        "RollingMeanCausal_w5",
        y_true,
        _fill_with_rolling_mean_causal(target_series, window=5)
        .loc[mask_indices]
        .values,
        results,
    )
    _safe_metric_eval(
        "RollingMedian_w3",
        y_true,
        _fill_with_rolling_median_causal(target_series, window=3)
        .loc[mask_indices]
        .values,
        results,
    )
    _safe_metric_eval(
        "RollingMedian_w5",
        y_true,
        _fill_with_rolling_median_causal(target_series, window=5)
        .loc[mask_indices]
        .values,
        results,
    )
    _safe_metric_eval(
        "EWMA_span3",
        y_true,
        _fill_with_ewma_causal(target_series, span=3).loc[mask_indices].values,
        results,
    )
    _safe_metric_eval(
        "EWMA_span5",
        y_true,
        _fill_with_ewma_causal(target_series, span=5).loc[mask_indices].values,
        results,
    )
    _safe_metric_eval(
        "LinearTrend_w3",
        y_true,
        _fill_with_linear_trend_causal(target_series, window=3)
        .loc[mask_indices]
        .values,
        results,
    )
    _safe_metric_eval(
        "LinearTrend_w5",
        y_true,
        _fill_with_linear_trend_causal(target_series, window=5)
        .loc[mask_indices]
        .values,
        results,
    )

    linear_interp = target_series.interpolate(method="linear").ffill().bfill()
    _safe_metric_eval(
        "LinearInterpolation", y_true, linear_interp.loc[mask_indices].values, results
    )

    if "Data" in df_with_nan.columns:
        try:
            df_time = df_with_nan.copy()
            df_time["Data"] = pd.to_datetime(df_time["Data"], errors="coerce")
            df_time = df_time.sort_values("Data").set_index("Data")
            time_interp = (
                df_time[target_col]
                .astype(float)
                .interpolate(method="time")
                .ffill()
                .bfill()
            )
            pred_vals = time_interp.loc[
                pd.to_datetime(df_masked.loc[mask_indices, "Data"], errors="coerce")
            ].values
            _safe_metric_eval("TimeInterpolation", y_true, pred_vals, results)
        except Exception:
            results["TimeInterpolation"] = {
                "rmse": None,
                "nrmse": None,
                "r2": None,
                "mape": None,
                "prediction_time_per_sample": None,
            }

    _safe_metric_eval(
        "SeasonalNaive",
        y_true,
        _fill_with_seasonal_naive(target_series, seasonal_period=seasonal_period)
        .loc[mask_indices]
        .values,
        results,
    )
    _safe_metric_eval(
        "ExpSmoothing",
        y_true,
        _fill_with_exp_smoothing(target_series, seasonal_period=seasonal_period)
        .loc[mask_indices]
        .values,
        results,
    )

    # Modelos com lags
    _safe_metric_eval(
        "RidgeLag",
        y_true,
        _fill_with_ridge_lag(target_series, lags=lag_window).loc[mask_indices].values,
        results,
    )
    _safe_metric_eval(
        "KNNLag",
        y_true,
        _fill_with_knn_lag(target_series, lags=lag_window).loc[mask_indices].values,
        results,
    )
    _safe_metric_eval(
        "RandomForestLag",
        y_true,
        _fill_with_rf_lag(target_series, lags=lag_window).loc[mask_indices].values,
        results,
    )
    _safe_metric_eval(
        "GradientBoostingLag",
        y_true,
        _fill_with_gb_lag(target_series, lags=lag_window).loc[mask_indices].values,
        results,
    )
    _safe_metric_eval(
        "XGBLag",
        y_true,
        _fill_with_xgb_lag(target_series, lags=lag_window).loc[mask_indices].values,
        results,
    )

    # Neurais
    _safe_metric_eval(
        "GRU",
        y_true,
        _fill_with_gru(target_series, lags=lag_window).loc[mask_indices].values,
        results,
    )
    _safe_metric_eval(
        "LSTM",
        y_true,
        _fill_with_lstm(target_series, lags=lag_window).loc[mask_indices].values,
        results,
    )

    return results


# ============================================================
# Stacking model
# ============================================================


def evaluate_stacking_with_missing(
    df_clean: pd.DataFrame, missing_fraction: float, target_col: str = "Vazao_BBR"
) -> Dict:
    print(f"[STACKING] Avaliando com missing_fraction={missing_fraction:.0%}")

    df_masked = apply_random_mask(df_clean, missing_fraction, seed=42)
    mask_indices = df_masked[df_masked["mask_applied"] == 1].index

    df_train = df_masked[df_masked["mask_applied"] == 0].copy()
    df_test = df_masked[df_masked["mask_applied"] == 1].copy()

    print(f"    Treino: {len(df_train)} amostras, Teste: {len(df_test)} amostras")

    if len(df_train) < 10 or len(df_test) < 5:
        print("Dados insuficientes")
        return {
            "rmse": None,
            "nrmse": None,
            "r2": None,
            "mape": None,
            "prediction_time_per_sample": None,
        }

    df_train_feat = engineer_features_for_imputation(df_train, target_col)
    df_test_feat = engineer_features_for_imputation(df_test, target_col)

    exclude_cols = {target_col, "Data", "mask_applied"}
    feature_cols = [
        c
        for c in df_train_feat.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_train_feat[c])
    ]

    df_train_feat = df_train_feat.dropna(subset=feature_cols + [target_col])
    df_test_feat = df_test_feat.dropna(subset=feature_cols)

    if df_train_feat.empty or df_test_feat.empty:
        print("Dados insuficientes após limpeza")
        return {
            "rmse": None,
            "nrmse": None,
            "r2": None,
            "mape": None,
            "prediction_time_per_sample": None,
        }

    X_train = df_train_feat[feature_cols].fillna(0).values
    y_train = df_train_feat[target_col].values
    X_test = df_test_feat[feature_cols].fillna(0).values
    y_true = df_clean.loc[mask_indices, target_col].values

    try:
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        base_models = [
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=120, max_depth=12, random_state=42, n_jobs=-1
                ),
            ),
            (
                "gb",
                GradientBoostingRegressor(
                    n_estimators=120, max_depth=3, learning_rate=0.05, random_state=42
                ),
            ),
            (
                "knn",
                KNeighborsRegressor(
                    n_neighbors=max(1, min(5, len(X_train) // 3)), weights="distance"
                ),
            ),
        ]
        if HAS_XGB:
            base_models.insert(
                0,
                (
                    "xgb",
                    XGBRegressor(
                        n_estimators=120,
                        max_depth=4,
                        learning_rate=0.05,
                        random_state=42,
                        verbosity=0,
                        subsample=0.8,
                    ),
                ),
            )

        stacking = StackingRegressor(
            estimators=base_models,
            final_estimator=Ridge(alpha=1.0),
            cv=max(2, min(3, len(X_train) // 10)),
            n_jobs=-1,
        )

        stacking.fit(X_train_scaled, y_train_scaled)

        start = time.time()
        y_pred_scaled = stacking.predict(X_test_scaled)
        pred_time = time.time() - start

        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        metrics = calculate_metrics(y_true, y_pred, pred_time)

        if metrics["rmse"] is not None:
            print(
                f"RMSE: {metrics['rmse']:.2f}M, R²: {metrics['r2']:.4f}"
                if metrics["r2"] is not None
                else f"RMSE: {metrics['rmse']:.2f}M"
            )
        return metrics

    except Exception as e:
        print(f"Erro no stacking: {e}")
        return {
            "rmse": None,
            "nrmse": None,
            "r2": None,
            "mape": None,
            "prediction_time_per_sample": None,
        }


# ============================================================
# File-level pipeline
# ============================================================


def evaluate_file(file_path: Path, missing_fractions: List[float]) -> Optional[Dict]:
    print(f"\n{'=' * 80}")
    print(f"[AVALIANDO] {file_path.name}")
    print(f"{'=' * 80}")

    df = pd.read_csv(file_path)
    source = file_path.stem.replace("_merged", "").replace("_largest_subseries", "")
    target_col = "Vazao_BBR"

    df_clean = get_clean_data(df, target_col)
    if len(df_clean) < 20:
        print(f"Dados insuficientes: apenas {len(df_clean)} amostras")
        return None

    results = {}
    for frac in missing_fractions:
        print(f"\n  {'─' * 60}")
        print(f"FRAÇÃO DE MISSING: {frac:.0%}")
        print(f"  {'─' * 60}")

        baseline_results = evaluate_baselines(df_clean, frac, target_col)
        stacking_results = evaluate_stacking_with_missing(df_clean, frac, target_col)

        results[str(frac)] = {
            "baseline": baseline_results,
            "stacking": {"mean": {"StackingRegressor": stacking_results}},
        }

    return {"source": source, "results": results, "n_samples": len(df_clean)}


def analyze_stacking_performance(results: Dict) -> Dict[str, Any]:
    stacking_wins = 0
    total_comparisons = 0
    stacking_rmse_list = []
    best_baseline_rmse_list = []
    prediction_times = []

    for frac, data in results.items():
        baseline_data = data.get("baseline", {})
        stacking_data = (
            data.get("stacking", {}).get("mean", {}).get("StackingRegressor", {})
        )

        if not baseline_data or not stacking_data:
            continue

        stacking_rmse = stacking_data.get("rmse")
        if stacking_rmse is None:
            continue

        pred_time = stacking_data.get("prediction_time_per_sample")
        if pred_time is not None:
            prediction_times.append(pred_time)

        baseline_rmses = [
            m["rmse"] for m in baseline_data.values() if m.get("rmse") is not None
        ]
        if not baseline_rmses:
            continue

        best_baseline_rmse = min(baseline_rmses)
        total_comparisons += 1
        stacking_rmse_list.append(stacking_rmse)
        best_baseline_rmse_list.append(best_baseline_rmse)

        if stacking_rmse < best_baseline_rmse:
            stacking_wins += 1

    if total_comparisons == 0:
        return {
            "should_impute": False,
            "win_rate": 0.0,
            "avg_improvement": 0.0,
            "total_comparisons": 0,
            "avg_prediction_time_per_sample": None,
        }

    win_rate = stacking_wins / total_comparisons
    avg_stacking_rmse = np.mean(stacking_rmse_list)
    avg_best_baseline_rmse = np.mean(best_baseline_rmse_list)
    avg_improvement = (
        (avg_best_baseline_rmse - avg_stacking_rmse) / avg_best_baseline_rmse
    ) * 100
    avg_pred_time = round(np.mean(prediction_times), 4) if prediction_times else None

    return {
        "should_impute": win_rate >= 0.5,
        "win_rate": win_rate,
        "avg_improvement": avg_improvement,
        "total_comparisons": total_comparisons,
        "stacking_wins": stacking_wins,
        "avg_stacking_rmse": avg_stacking_rmse,
        "avg_baseline_rmse": avg_best_baseline_rmse,
        "avg_prediction_time_per_sample": avg_pred_time,
    }


# ============================================================
# Imputation output generation
# ============================================================


def _save_imputed_series(
    df: pd.DataFrame,
    mask_missing: pd.Series,
    imputed_series: pd.Series,
    output_dir: Path,
    filename: str,
    target_col: str,
):
    df_out = df.copy()
    df_out["is_imputed"] = 0
    df_out.loc[mask_missing, "is_imputed"] = 1
    df_out[target_col] = imputed_series.values

    cols_to_save = [
        c
        for c in [
            "Data",
            "Atraso(ms)",
            "Hop_count",
            "Bottleneck",
            target_col,
            "is_imputed",
        ]
        if c in df_out.columns
    ]
    df_out[cols_to_save].to_csv(output_dir / filename, index=False)


def impute_with_baselines(
    df: pd.DataFrame,
    target_col: str,
    output_dir: Path,
    source: str,
    seasonal_period: int = 24,
    lag_window: int = 12,
):
    mask_missing = df[target_col] == -1
    n_missing = int(mask_missing.sum())
    if n_missing == 0:
        return

    print(f"    [BASELINES/MODELOS] Imputando {n_missing} valores...")
    base_series = df[target_col].replace(-1, np.nan).astype(float)

    methods = {
        "baseline_mean": lambda s: s.fillna(s.mean()),
        "baseline_median": lambda s: s.fillna(s.median()),
        "baseline_knn": None,  # special case
        "baseline_ffill": lambda s: s.ffill().bfill(),
        "baseline_bfill": lambda s: s.bfill().ffill(),
        "baseline_rolling_mean_w3": lambda s: _fill_with_rolling_mean_causal(s, 3),
        "baseline_rolling_mean_w5": lambda s: _fill_with_rolling_mean_causal(s, 5),
        "baseline_rolling_median_w3": lambda s: _fill_with_rolling_median_causal(s, 3),
        "baseline_rolling_median_w5": lambda s: _fill_with_rolling_median_causal(s, 5),
        "baseline_ewma_span3": lambda s: _fill_with_ewma_causal(s, 3),
        "baseline_ewma_span5": lambda s: _fill_with_ewma_causal(s, 5),
        "baseline_linear_trend_w3": lambda s: _fill_with_linear_trend_causal(s, 3),
        "baseline_linear_trend_w5": lambda s: _fill_with_linear_trend_causal(s, 5),
        "baseline_linear_interp": lambda s: (
            s.interpolate(method="linear").ffill().bfill()
        ),
        "baseline_seasonal_naive": lambda s: _fill_with_seasonal_naive(
            s, seasonal_period
        ),
        "baseline_exp_smoothing": lambda s: _fill_with_exp_smoothing(
            s, seasonal_period
        ),
        "baseline_ridge_lag": lambda s: _fill_with_ridge_lag(s, lag_window),
        "baseline_knn_lag": lambda s: _fill_with_knn_lag(s, lag_window),
        "baseline_rf_lag": lambda s: _fill_with_rf_lag(s, lag_window),
        "baseline_gb_lag": lambda s: _fill_with_gb_lag(s, lag_window),
        "baseline_xgb_lag": lambda s: _fill_with_xgb_lag(s, lag_window),
        "baseline_gru": lambda s: _fill_with_gru(s, lag_window),
        "baseline_lstm": lambda s: _fill_with_lstm(s, lag_window),
    }

    for suffix, func in methods.items():
        try:
            if suffix == "baseline_knn":
                imputer = KNNImputer(n_neighbors=max(1, min(5, len(df) // 2)))
                filled = pd.Series(
                    imputer.fit_transform(df[[target_col]].replace(-1, np.nan)).ravel(),
                    index=df.index,
                )
            else:
                filled = func(base_series.copy())
            _save_imputed_series(
                df,
                mask_missing,
                filled,
                output_dir,
                f"{source}_{suffix}.csv",
                target_col,
            )
            print(f"      {suffix}: OK")
        except Exception as e:
            print(f"      {suffix} falhou: {e}")

    if "Data" in df.columns:
        try:
            tmp = df.copy()
            tmp["Data"] = pd.to_datetime(tmp["Data"], errors="coerce")
            tmp = tmp.sort_values("Data").set_index("Data")
            filled = (
                tmp[target_col]
                .replace(-1, np.nan)
                .astype(float)
                .interpolate(method="time")
                .ffill()
                .bfill()
            )
            filled = filled.reindex(tmp.index)
            tmp[target_col] = filled.values
            tmp = tmp.reset_index()
            _save_imputed_series(
                tmp,
                tmp[target_col].isna() == False,
                tmp[target_col],
                output_dir,
                f"{source}_baseline_time_interp.csv",
                target_col,
            )
            print("      baseline_time_interp: OK")
        except Exception as e:
            print(f"      baseline_time_interp falhou: {e}")


def impute_with_stacking(
    df: pd.DataFrame, target_col: str, output_dir: Path, source: str
):
    mask_missing = df[target_col] == -1
    n_missing = int(mask_missing.sum())
    if n_missing == 0:
        return

    print(f"[STACKING] Imputando {n_missing} valores...")

    try:
        df_clean = df[df[target_col] != -1].copy()
        if len(df_clean) < 10:
            print("Dados insuficientes para treinar stacking")
            return

        df_clean_feat = engineer_features_for_imputation(df_clean, target_col)
        exclude_cols = {target_col, "Data", "mask_applied"}
        feature_cols = [
            c
            for c in df_clean_feat.columns
            if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_clean_feat[c])
        ]

        X_train = df_clean_feat[feature_cols].fillna(0).values
        y_train = df_clean_feat[target_col].values

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        base_models = [
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=150,
                    max_depth=15,
                    random_state=42,
                    n_jobs=-1,
                    min_samples_split=3,
                ),
            ),
            (
                "gb",
                GradientBoostingRegressor(
                    n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42
                ),
            ),
            (
                "knn",
                KNeighborsRegressor(
                    n_neighbors=max(1, min(7, len(X_train) // 4)),
                    weights="distance",
                    p=1,
                ),
            ),
        ]
        if HAS_XGB:
            base_models.insert(
                0,
                (
                    "xgb",
                    XGBRegressor(
                        n_estimators=150,
                        max_depth=4,
                        learning_rate=0.05,
                        random_state=42,
                        verbosity=0,
                        subsample=0.8,
                    ),
                ),
            )

        stacking = StackingRegressor(
            estimators=base_models,
            final_estimator=Ridge(alpha=10.0),
            cv=max(2, min(5, len(X_train) // 10)),
            n_jobs=-1,
            passthrough=True,
        )

        print("      Treinando ensemble...")
        stacking.fit(X_train_scaled, y_train_scaled)

        df_imputed = df.copy()
        df_imputed["is_imputed"] = 0
        missing_indices = df[mask_missing].index.tolist()
        imputed_values = []

        print(f"      Imputando {len(missing_indices)} valores...")
        for idx in missing_indices:
            df_temp = df_imputed.iloc[: idx + 1].copy()
            if df_temp.loc[idx, target_col] == -1:
                valid_values = df_temp[df_temp[target_col] != -1][target_col]
                if len(valid_values) > 0:
                    df_temp.loc[idx, target_col] = valid_values.median()
                else:
                    df_temp.loc[idx, target_col] = df_clean[target_col].median()

            df_temp_feat = engineer_features_for_imputation(df_temp, target_col)
            X_pred = df_temp_feat.iloc[-1:][feature_cols].fillna(0).values
            X_pred_scaled = scaler_X.transform(X_pred)
            y_pred_scaled = stacking.predict(X_pred_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()[0]

            df_imputed.loc[idx, target_col] = y_pred
            df_imputed.loc[idx, "is_imputed"] = 1
            imputed_values.append(y_pred)

        cols_to_save = [
            c
            for c in [
                "Data",
                "Atraso(ms)",
                "Hop_count",
                "Bottleneck",
                target_col,
                "is_imputed",
            ]
            if c in df_imputed.columns
        ]
        output_file = output_dir / f"{source}_stacking.csv"
        df_imputed[cols_to_save].to_csv(output_file, index=False)

        arr = np.array(imputed_values)
        print(f"      Stacking: {output_file}")
        if len(arr):
            print(
                f"      Média: {np.mean(arr) / 1e6:.2f}M | Desvio: {np.std(arr) / 1e6:.2f}M"
            )
            print(
                f"      Min: {np.min(arr) / 1e6:.2f}M | Max: {np.max(arr) / 1e6:.2f}M"
            )
            print(f"      Valores únicos: {len(np.unique(arr))}/{n_missing}")

    except Exception as e:
        print(f"Stacking falhou: {e}")


def intelligent_imputation(file_path: Path, results: Dict, output_dir: Path):
    source = results["source"]

    print(f"\n{'=' * 80}")
    print(f"[IMPUTAÇÃO AUTOMÁTICA] {source}")
    print(f"{'=' * 80}")

    df = pd.read_csv(file_path)
    target_col = "Vazao_BBR"
    n_missing = int((df[target_col] == -1).sum())
    if n_missing == 0:
        print("Nenhum valor faltante para imputar")
        return

    print(f"{n_missing} valores faltantes encontrados")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("DECISÃO: IMPUTANDO COM TODOS OS MÉTODOS (baselines + stacking)")
    print(f"\n  {'─' * 60}")
    impute_with_baselines(df, target_col, output_dir, source)
    impute_with_stacking(df, target_col, output_dir, source)
    print(f"  {'─' * 60}")
    print(f"Imputação concluída para {source}")


def main():
    data_path = Path("../../datasets/originals")
    results_path = Path("../../results-more-baselines")
    imputed_path = Path("../../datasets/imputed-series-more-baselines")

    results_path.mkdir(exist_ok=True, parents=True)
    imputed_path.mkdir(exist_ok=True, parents=True)

    csv_files = list(data_path.glob("*_merged.csv"))
    missing_fractions = [0.2, 0.3, 0.4, 0.5]

    print(f"\n{'=' * 80}")
    print("PIPELINE COMPLETO: AVALIAÇÃO + IMPUTAÇÃO AUTOMÁTICA")
    print(f"{'=' * 80}")
    print(f"Arquivos encontrados: {len(csv_files)}")
    print(f"Frações de missing: {missing_fractions}")
    print(f"Pasta de resultados: {results_path}")
    print(f"Pasta de dados imputados: {imputed_path}")
    print(f"{'=' * 80}\n")

    all_results = {}
    summary = {"total_files": len(csv_files), "processed": 0, "failed": 0}

    for i, file_path in enumerate(csv_files, 1):
        print(f"\n{'#' * 80}")
        print(f"[{i}/{len(csv_files)}] Processando: {file_path.name}")
        print(f"{'#' * 80}")

        try:
            print(f"\n{'=' * 80}")
            print("FASE 1: AVALIAÇÃO")
            print(f"{'=' * 80}")

            result = evaluate_file(file_path, missing_fractions)
            if result is None:
                print("Arquivo ignorado (dados insuficientes)")
                summary["failed"] += 1
                continue

            source = result["source"]
            all_results[source] = result["results"]

            evaluation_file = results_path / "metrics_summary.json"
            with open(evaluation_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=4)

            print(f"Avaliação salva em: {evaluation_file}")

            print(f"\n{'=' * 80}")
            print("FASE 2: IMPUTAÇÃO")
            print(f"{'=' * 80}")

            intelligent_imputation(file_path, result, imputed_path)

            summary["processed"] += 1
            print(f"Concluído: {source}")

        except Exception as e:
            print(f"Erro processando {file_path.name}: {e}")
            summary["failed"] += 1
            continue

    print(f"\n{'=' * 80}")
    print("RESUMO FINAL")
    print(f"{'=' * 80}")
    print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    main()
