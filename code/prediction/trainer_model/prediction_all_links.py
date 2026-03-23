import glob
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rnn_trainer import RNNForecaster
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler, StandardScaler


class PredictionAllLinks:
    """
    Prediction on all links comparing stacking vs baseline methods.
    ----------------------------------------------
    This class implements a framework for training and evaluating
    prediction models (Random Forest, LSTM, GRU) on multiple time series datasets.

    """

    def __init__(
        self,
        data_dir: str,
        lags: int = 6,
        cv_splits: int = 5,
        min_split_size: int = 30,
        baseline_keep: str = "bfill,ffill,knn,mean,median",
        require_all_baselines: bool = False,
        device: str = "auto",
        window: int = 24,
        same_window: bool = False,
        epochs: int = 25,
        patience: int = 3,
        log_level: str = "INFO",
        log_dir: str = "logs",
        plots: bool = False,
        max_links: int = None,
    ):

        self.RANDOM_STATE = 42
        self.data_dir = data_dir
        self.lags = lags
        self.cv_splits = cv_splits
        self.min_split_size = min_split_size
        self.baseline_keep = baseline_keep
        self.require_all_baselines = require_all_baselines
        self.device = device
        self.window = window
        self.same_window = same_window
        self.epochs = epochs
        self.patience = patience
        self.log_level = log_level
        self.log_dir = log_dir
        self.plots = plots
        self.max_links = max_links

    def setup_logger(self, log_dir: str, level: str):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(log_dir) / f"simple_prediction_{ts}.log"
        level_value = getattr(logging, level.upper(), logging.INFO)
        fmt = "%(asctime)s | %(levelname)7s | %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        logging.basicConfig(
            level=level_value,
            format=fmt,
            datefmt=datefmt,
            handlers=[
                logging.FileHandler(log_path, encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )
        return logging.getLogger(__name__), str(log_path)

    def safe_load_csv(self, fp: str) -> pd.DataFrame:
        try:
            return pd.read_csv(fp)
        except Exception:
            return pd.read_csv(fp, sep=";")

    def choose_device(self, logger):
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(
            f"CUDA available (torch.cuda.is_available()): {torch.cuda.is_available()}"
        )
        logger.info(
            f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}"
        )
        if torch.cuda.is_available():
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"  Device {i}: {torch.cuda.get_device_name(i)}")

        if self.device == "cpu":
            logger.info("Using CPU (forced).")
            return torch.device("cpu")

        if self.device in ("auto", "cuda"):
            if torch.cuda.is_available():
                dev = torch.device("cuda")
                logger.info(f"CUDA available: {torch.cuda.get_device_name(dev)}")
                try:
                    torch.backends.cudnn.benchmark = True
                    torch.set_float32_matmul_precision("medium")
                    logger.info("CUDA optimization flags set successfully")
                except Exception as e:
                    logger.warning(f"Could not set CUDA optimizations: {e}")
                return dev
            else:
                if self.device == "cuda":
                    raise RuntimeError(
                        "CUDA requested but not available. Use device='auto' or device='cpu'."
                    )
                logger.warning("CUDA not available; falling back to CPU.")
                return torch.device("cpu")

        logger.info("CUDA not available; using CPU.")
        return torch.device("cpu")

    def compute_time_splits(
        self, n_samples: int, desired: int, min_split_size: int
    ) -> int:
        if n_samples < min_split_size * 2:
            return 2
        return min(desired, max(2, n_samples // min_split_size))

    def build_supervised(self, df: pd.DataFrame, lags=6, target_col="Atraso(ms)"):
        if "Data" in df.columns:
            df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
            df = df.sort_values("Data")
            df.set_index("Data", inplace=True)
        for lag in range(1, lags + 1):
            df[f"lag_{lag}"] = df[target_col].shift(lag)
        df[f"rolling_mean_{lags}"] = df[target_col].rolling(lags).mean().shift(1)
        for c in ["Hop_count", "Bottleneck", "Vazao_BBR", "mask_applied"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna()
        y = df[target_col].astype(float)
        X = df.drop(columns=[target_col])
        return X, y

    def compute_metrics(self, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        eps = 1e-8
        mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0
        r2 = r2_score(y_true, y_pred)
        return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}

    def build_sequences(self, series: pd.Series, window: int):
        data = series.values.astype(float)
        X_list, y_list = [], []
        for i in range(window, len(data)):
            X_list.append(data[i - window : i])
            y_list.append(data[i])
        return np.array(X_list), np.array(y_list)

    def train_rnn_cv(
        self,
        series: pd.Series,
        window=24,
        model_type="LSTM",
        epochs=25,
        folds=5,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        lr=1e-3,
        device="cpu",
        patience=3,
    ):
        data_X, data_y = self.build_sequences(series, window)
        if len(data_X) == 0:
            return {
                "rmse": np.nan,
                "mae": np.nan,
                "mape": np.nan,
                "r2": np.nan,
                "params": {},
            }
        scaler_X = StandardScaler()
        scaler_y = RobustScaler()
        data_X_scaled = scaler_X.fit_transform(data_X)
        data_y_scaled = scaler_y.fit_transform(data_y.reshape(-1, 1)).ravel()
        tscv = TimeSeriesSplit(n_splits=folds)
        metrics_accum = []
        for train_idx, test_idx in tscv.split(data_X_scaled):
            X_tr = (
                torch.tensor(data_X_scaled[train_idx], dtype=torch.float32)
                .unsqueeze(-1)
                .to(device)
            )
            y_tr = torch.tensor(data_y_scaled[train_idx], dtype=torch.float32).to(
                device
            )
            X_te = (
                torch.tensor(data_X_scaled[test_idx], dtype=torch.float32)
                .unsqueeze(-1)
                .to(device)
            )
            y_te = torch.tensor(data_y_scaled[test_idx], dtype=torch.float32).to(device)
            model = RNNForecaster(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                rnn_type=model_type,
                dropout=dropout,
            ).to(device)
            optim = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.MSELoss()
            best_loss = np.inf
            patience_counter = 0
            for _ in range(epochs):
                model.train()
                optim.zero_grad()
                pred = model(X_tr)
                loss = loss_fn(pred, y_tr)
                loss.backward()
                optim.step()
                if loss.item() < best_loss - 1e-6:
                    best_loss = loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    break
            model.eval()
            with torch.no_grad():
                pred_scaled = model(X_te).cpu().numpy()
            pred_inv = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
            y_te_inv = scaler_y.inverse_transform(
                y_te.cpu().numpy().reshape(-1, 1)
            ).ravel()
            metrics_accum.append(self.compute_metrics(y_te_inv, pred_inv))
        if not metrics_accum:
            agg = {"rmse": np.nan, "mae": np.nan, "mape": np.nan, "r2": np.nan}
        else:
            agg = {
                k: float(np.mean([m[k] for m in metrics_accum]))
                for k in metrics_accum[0]
            }
        agg["params"] = {
            "window": window,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "lr": lr,
            "epochs": epochs,
        }
        return agg

    def train_rf_cv(self, X: pd.DataFrame, y: pd.Series, cv_splits=5):
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=self.RANDOM_STATE,
            n_jobs=-1,
        )
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        metrics_accum = []
        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
            metrics_accum.append(self.compute_metrics(y_te.values, pred))
        if not metrics_accum:
            agg = {"rmse": np.nan, "mae": np.nan, "mape": np.nan, "r2": np.nan}
        else:
            agg = {
                k: float(np.mean([m[k] for m in metrics_accum]))
                for k in metrics_accum[0]
            }
        agg["params"] = {"n_estimators": 300, "max_depth": None, "max_features": "sqrt"}
        return agg

    def group_files_by_link(self, files):
        grouped = {}
        for fp in files:
            name = os.path.basename(fp)
            if "_stacking.csv" in name:
                link = name.replace("_stacking.csv", "")
                grouped.setdefault(link, {"stacking": None, "baselines": []})
                grouped[link]["stacking"] = fp
            elif "_baseline_" in name:
                parts = name.split("_baseline_")
                if len(parts) >= 2:
                    link = parts[0]
                    method = parts[1].replace(".csv", "")
                    grouped.setdefault(link, {"stacking": None, "baselines": []})
                    grouped[link]["baselines"].append((method, fp))
        return grouped

    def evaluate_df_models(self, df: pd.DataFrame, device, n_splits_rf, n_splits_rnn):
        X, y = self.build_supervised(df, lags=self.lags)
        rf_metrics = self.train_rf_cv(X, y, cv_splits=n_splits_rf)
        series = (
            df.sort_values("Data")["Atraso(ms)"]
            if "Data" in df.columns
            else df["Atraso(ms)"]
        )
        window = self.lags if self.same_window else max(self.window, self.lags + 2)
        dropout_dyn = 0.1 if len(series) > 300 else 0.3
        lstm_metrics = self.train_rnn_cv(
            series,
            window=window,
            model_type="LSTM",
            epochs=self.epochs,
            folds=n_splits_rnn,
            hidden_size=64,
            num_layers=2,
            dropout=dropout_dyn,
            lr=1e-3,
            device=device,
            patience=self.patience,
        )
        gru_metrics = self.train_rnn_cv(
            series,
            window=window,
            model_type="GRU",
            epochs=self.epochs,
            folds=n_splits_rnn,
            hidden_size=64,
            num_layers=2,
            dropout=dropout_dyn,
            lr=1e-3,
            device=device,
            patience=self.patience,
        )
        return {
            "rf": rf_metrics,
            "lstm": lstm_metrics,
            "gru": gru_metrics,
            "window": window,
        }

    def run(self):
        logger, log_path = self.setup_logger(self.log_dir, self.log_level)
        logger.info("Starting simple script")
        baseline_keep_set = {
            m.strip().lower() for m in self.baseline_keep.split(",") if m.strip()
        }
        logger.info(f"Allowed baseline methods: {sorted(baseline_keep_set)}")

        data_dir = Path(self.data_dir)
        files = sorted(glob.glob(str(data_dir / "*.csv")))
        if not files:
            logger.error("No CSV files found.")
            return
        grouped = self.group_files_by_link(files)
        logger.info(f"Links detected: {len(grouped)}")
        device = self.choose_device(logger)
        logger.info(f"Using device: {device}")

        detail_rows = []
        summary_rows = []

        # Limit processing if max_links is specified
        if self.max_links is not None:
            links_to_process = list(sorted(grouped.items()))[: self.max_links]
            logger.info(
                f"Processing limited to {len(links_to_process)} links out of {len(grouped)} total links"
            )
        else:
            links_to_process = list(sorted(grouped.items()))
            logger.info(f"Processing all {len(links_to_process)} links")

        for link, data in links_to_process:
            stacking_fp = data.get("stacking")
            baseline_files_all = data.get("baselines", [])
            baseline_files = [
                (m, fp)
                for (m, fp) in baseline_files_all
                if m.lower() in baseline_keep_set
            ]
            if self.require_all_baselines:
                present = {m.lower() for (m, _) in baseline_files}
                missing = baseline_keep_set - present
                if missing:
                    logger.warning(
                        f"[{link}] Ignored - missing required methods: {sorted(missing)}"
                    )
                    continue
            if not stacking_fp:
                logger.warning(f"[{link}] Ignored - no stacking file.")
                continue
            if not baseline_files:
                logger.warning(f"[{link}] Ignored - no allowed baseline.")
                continue

            logger.info(
                f"[{link}] Evaluating {len(baseline_files)} baselines + stacking"
            )

            best_baseline_rmse = float("inf")
            best_baseline_method = None
            best_baseline_model = None
            best_baseline_metrics_full = None

            for method, fp in baseline_files:
                try:
                    dfb = self.safe_load_csv(fp)
                    n_splits_rf = self.compute_time_splits(
                        len(dfb), self.cv_splits, self.min_split_size
                    )
                    n_splits_rnn = self.compute_time_splits(
                        len(dfb), self.cv_splits, self.min_split_size
                    )
                    res = self.evaluate_df_models(
                        dfb, device, n_splits_rf, n_splits_rnn
                    )
                    for model_name, metrics_obj in [
                        ("RandomForest", res["rf"]),
                        ("LSTM", res["lstm"]),
                        ("GRU", res["gru"]),
                    ]:
                        detail_rows.append(
                            {
                                "link": link,
                                "file": os.path.basename(fp),
                                "method": method,
                                "variant": "baseline",
                                "model": model_name,
                                "rmse": metrics_obj["rmse"],
                                "mae": metrics_obj["mae"],
                                "mape": metrics_obj["mape"],
                                "r2": metrics_obj["r2"],
                                "params": metrics_obj.get("params"),
                                "window": metrics_obj.get("params", {}).get("window"),
                                "splits_rf": n_splits_rf,
                                "splits_rnn": n_splits_rnn,
                            }
                        )
                        if metrics_obj["rmse"] < best_baseline_rmse:
                            best_baseline_rmse = metrics_obj["rmse"]
                            best_baseline_method = method
                            best_baseline_model = model_name
                            best_baseline_metrics_full = metrics_obj
                    logger.info(
                        f"[{link}] Baseline {method} RMSE RF={res['rf']['rmse']:.4f} LSTM={res['lstm']['rmse']:.4f} GRU={res['gru']['rmse']:.4f}"
                    )
                except Exception as e:
                    logger.exception(f"[{link}] Baseline {method} error: {e}")

            try:
                dfs = self.safe_load_csv(stacking_fp)
                n_splits_rf_s = self.compute_time_splits(
                    len(dfs), self.cv_splits, self.min_split_size
                )
                n_splits_rnn_s = self.compute_time_splits(
                    len(dfs), self.cv_splits, self.min_split_size
                )
                res_stack = self.evaluate_df_models(
                    dfs, device, n_splits_rf_s, n_splits_rnn_s
                )
                stacking_candidates = [
                    ("RandomForest", res_stack["rf"]),
                    ("LSTM", res_stack["lstm"]),
                    ("GRU", res_stack["gru"]),
                ]
                best_stacking_model, best_stacking_metrics = min(
                    stacking_candidates, key=lambda x: x[1]["rmse"]
                )
                best_stacking_rmse = best_stacking_metrics["rmse"]
                for model_name, metrics_obj in stacking_candidates:
                    detail_rows.append(
                        {
                            "link": link,
                            "file": os.path.basename(stacking_fp),
                            "method": "stacking_source",
                            "variant": "stacking",
                            "model": model_name,
                            "rmse": metrics_obj["rmse"],
                            "mae": metrics_obj["mae"],
                            "mape": metrics_obj["mape"],
                            "r2": metrics_obj["r2"],
                            "params": metrics_obj.get("params"),
                            "window": metrics_obj.get("params", {}).get("window"),
                            "splits_rf": n_splits_rf_s,
                            "splits_rnn": n_splits_rnn_s,
                        }
                    )
                logger.info(
                    f"[{link}] Stacking RF={res_stack['rf']['rmse']:.4f} LSTM={res_stack['lstm']['rmse']:.4f} GRU={res_stack['gru']['rmse']:.4f} -> best={best_stacking_model}:{best_stacking_rmse:.4f}"
                )
            except Exception as e:
                logger.exception(f"[{link}] Stacking error: {e}")
                best_stacking_model = None
                best_stacking_rmse = np.nan
                best_stacking_metrics = None

            improvement_pct = (
                (best_baseline_rmse - best_stacking_rmse) / best_baseline_rmse * 100.0
                if (best_baseline_rmse and not np.isnan(best_stacking_rmse))
                else np.nan
            )

            summary_rows.append(
                {
                    "link": link,
                    "baseline_methods_considered": ",".join(
                        sorted({m for (m, _) in baseline_files})
                    ),
                    "best_baseline_method": best_baseline_method,
                    "best_baseline_model": best_baseline_model,
                    "best_baseline_rmse": best_baseline_rmse,
                    "best_baseline_mae": best_baseline_metrics_full["mae"]
                    if best_baseline_metrics_full
                    else np.nan,
                    "best_baseline_mape": best_baseline_metrics_full["mape"]
                    if best_baseline_metrics_full
                    else np.nan,
                    "best_baseline_r2": best_baseline_metrics_full["r2"]
                    if best_baseline_metrics_full
                    else np.nan,
                    "best_stacking_model": best_stacking_model,
                    "best_stacking_rmse": best_stacking_rmse,
                    "best_stacking_mae": best_stacking_metrics["mae"]
                    if best_stacking_metrics
                    else np.nan,
                    "best_stacking_mape": best_stacking_metrics["mape"]
                    if best_stacking_metrics
                    else np.nan,
                    "best_stacking_r2": best_stacking_metrics["r2"]
                    if best_stacking_metrics
                    else np.nan,
                    "improvement_pct": improvement_pct,
                }
            )
            logger.info(
                f"[{link}] Best baseline {best_baseline_method}/{best_baseline_model} RMSE={best_baseline_rmse:.4f} | Stacking {best_stacking_model} RMSE={best_stacking_rmse:.4f} | improvement={improvement_pct:.2f}%"
            )

        df_detail = pd.DataFrame(detail_rows)
        df_compare = pd.DataFrame(summary_rows)
        df_detail.to_csv("simple_models_detail.csv", index=False)
        logger.info("Details saved: simple_models_detail.csv")
        if not df_compare.empty:
            df_compare.sort_values("improvement_pct", ascending=False, inplace=True)
            df_compare.to_csv("simple_comparison_summary.csv", index=False)
            logger.info("Summary saved: simple_comparison_summary.csv")
            logger.info(
                f"Stacking better in {(df_compare['improvement_pct'] > 0).sum()} of {len(df_compare)} links."
            )
        else:
            logger.warning("Empty summary: no link processed.")

        logger.info(f"Log: {log_path}")
        logger.info("End.")
