"""
SmartRail - Delay Predictor Module
Generates synthetic railway delay data, trains an XGBoost/GradientBoosting regressor,
and predicts arrival and delay for simulated trains.
Compares performance against a naive average delay baseline.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import os
import joblib

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CACHE_PATH = os.path.join(MODEL_DIR, "delay_model.joblib")

RANDOM_SEED = 42

def generate_synthetic_dataset(n_samples: int = 3000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generates ~3000 synthetic rows:
    - scheduled_hour (0-23)
    - day_of_week (0-6)
    - train_type ('Rajdhani', 'Express', 'Local')
    - upstream_delay (minutes, 0-60)
    - fog_flag (0 or 1)
    - festival_flag (0 or 1)
    -> actual_delay (minutes)
    """
    rng = np.random.RandomState(seed)

    scheduled_hours = rng.randint(0, 24, size=n_samples)
    days_of_week = rng.randint(0, 7, size=n_samples)
    
    # Train types: Rajdhani=0, Express=1, Local=2
    train_types = rng.choice(["Rajdhani", "Express", "Local"], size=n_samples, p=[0.25, 0.45, 0.30])
    
    # Upstream delay: exponential distribution capped at 60
    upstream_delays = np.clip(rng.exponential(scale=10.0, size=n_samples), 0, 75).round(1)

    # Weather & Event flags
    # Fog is more likely in early morning (hours 4-8) or late night (hours 22-24)
    fog_probs = [0.40 if (h < 9 or h > 21) else 0.05 for h in scheduled_hours]
    fog_flags = (rng.rand(n_samples) < fog_probs).astype(int)

    # Festival flags (more common on weekends or specific peaks)
    festival_probs = [0.25 if d in (5, 6) else 0.10 for d in days_of_week]
    festival_flags = (rng.rand(n_samples) < festival_probs).astype(int)

    # Ground truth formula for actual delay (with realistic interactions and random noise)
    # Rajdhani has priority clear track -> smaller propagation
    # Local has many stops -> high variance
    type_factors = {"Rajdhani": 0.65, "Express": 0.90, "Local": 1.15}
    base_propagation = np.array([type_factors[t] for t in train_types])

    # Congestion during morning peak (8-11) and evening peak (17-21)
    peak_hours = np.array([1 if (8 <= h <= 11 or 17 <= h <= 21) else 0 for h in scheduled_hours])

    noise = rng.normal(loc=0.0, scale=3.5, size=n_samples)

    actual_delays = (
        upstream_delays * base_propagation
        + fog_flags * rng.uniform(12.0, 28.0, size=n_samples)
        + festival_flags * rng.uniform(6.0, 16.0, size=n_samples)
        + peak_hours * rng.uniform(4.0, 10.0, size=n_samples)
        + noise
    )
    # Delays can't be negative (earliness capped at 0 or max 2 mins early)
    actual_delays = np.maximum(actual_delays, 0.0).round(1)

    df = pd.DataFrame({
        "scheduled_hour": scheduled_hours,
        "day_of_week": days_of_week,
        "train_type": train_types,
        "upstream_delay": upstream_delays,
        "fog_flag": fog_flags,
        "festival_flag": festival_flags,
        "actual_delay": actual_delays
    })
    return df

class DelayPredictor:
    def __init__(self):
        self.model = None
        self.model_type = "XGBoost" if XGB_AVAILABLE else "GradientBoosting"
        self.baseline_mean = 0.0
        self.metrics = {}
        self.is_trained = False

    def _preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Encodes categorical features and separates target if present."""
        df_encoded = df.copy()
        type_mapping = {"Rajdhani": 0, "Express": 1, "Local": 2}
        df_encoded["train_type_code"] = df_encoded["train_type"].map(type_mapping).fillna(1)
        
        feature_cols = ["scheduled_hour", "day_of_week", "train_type_code", "upstream_delay", "fog_flag", "festival_flag"]
        X = df_encoded[feature_cols]
        y = df_encoded["actual_delay"] if "actual_delay" in df_encoded else None
        return X, y

    def train_and_evaluate(self, n_samples: int = 3000) -> Dict[str, Any]:
        """Trains the regressor and evaluates against naive mean baseline."""
        df = generate_synthetic_dataset(n_samples=n_samples, seed=RANDOM_SEED)
        X, y = self._preprocess(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=RANDOM_SEED
        )

        self.baseline_mean = float(y_train.mean())
        y_baseline_pred = np.full_like(y_test, fill_value=self.baseline_mean)

        baseline_mae = mean_absolute_error(y_test, y_baseline_pred)
        baseline_rmse = root_mean_squared_error(y_test, y_baseline_pred)

        fitted_successfully = False

        if XGB_AVAILABLE:
            try:
                # Ensure contiguous float32 numpy arrays for cross-platform C-API compatibility
                X_train_arr = np.ascontiguousarray(X_train.values, dtype=np.float32)
                y_train_arr = np.ascontiguousarray(y_train.values, dtype=np.float32)
                X_test_arr = np.ascontiguousarray(X_test.values, dtype=np.float32)

                xgb_model = xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.08,
                    max_depth=4,
                    random_state=RANDOM_SEED,
                    n_jobs=1
                )
                xgb_model.fit(X_train_arr, y_train_arr)

                # Test prediction immediately inside try block to verify booster is intact
                test_preds = xgb_model.predict(X_test_arr)
                if test_preds is not None and len(test_preds) == len(X_test):
                    self.model = xgb_model
                    self.model_type = "XGBoost Regressor"
                    y_model_pred = np.maximum(test_preds, 0.0)
                    fitted_successfully = True
            except Exception as e:
                print(f"[SmartRail] XGBoost execution note ({e}), engaging Sklearn GradientBoosting fallback")
                fitted_successfully = False

        if not fitted_successfully:
            self.model = GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.08, max_depth=4, random_state=RANDOM_SEED
            )
            self.model.fit(X_train, y_train)
            self.model_type = "Sklearn GradientBoosting"
            y_model_pred = np.maximum(self.model.predict(X_test), 0.0)

        model_mae = mean_absolute_error(y_test, y_model_pred)
        model_rmse = root_mean_squared_error(y_test, y_model_pred)

        mae_improvement = ((baseline_mae - model_mae) / baseline_mae) * 100
        rmse_improvement = ((baseline_rmse - model_rmse) / baseline_rmse) * 100

        self.metrics = {
            "model_type": self.model_type,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "baseline_mean_delay": round(self.baseline_mean, 2),
            "baseline_mae": round(baseline_mae, 2),
            "baseline_rmse": round(baseline_rmse, 2),
            "model_mae": round(model_mae, 2),
            "model_rmse": round(model_rmse, 2),
            "mae_improvement_pct": round(mae_improvement, 1),
            "rmse_improvement_pct": round(rmse_improvement, 1),
        }
        self.is_trained = True
        return self.metrics

    def predict_delay(
        self,
        scheduled_hour: int,
        day_of_week: int,
        train_type: str,
        upstream_delay: float,
        fog_flag: int = 0,
        festival_flag: int = 0
    ) -> float:
        """Predicts actual delay (in minutes) for a single train observation."""
        if not self.is_trained or self.model is None:
            self.train_and_evaluate()

        type_mapping = {"Rajdhani": 0, "Express": 1, "Local": 2}
        type_code = type_mapping.get(train_type, 1)

        input_data = np.array([[
            float(scheduled_hour),
            float(day_of_week),
            float(type_code),
            float(upstream_delay),
            float(fog_flag),
            float(festival_flag)
        ]], dtype=np.float32)

        try:
            if "XGBoost" in self.model_type:
                pred = self.model.predict(input_data)[0]
            else:
                input_df = pd.DataFrame([{
                    "scheduled_hour": int(scheduled_hour),
                    "day_of_week": int(day_of_week),
                    "train_type_code": type_code,
                    "upstream_delay": float(upstream_delay),
                    "fog_flag": int(fog_flag),
                    "festival_flag": int(festival_flag)
                }])
                pred = self.model.predict(input_df)[0]
            return float(np.clip(pred, 0.0, 180.0).round(1))
        except Exception:
            fallback_delay = upstream_delay * 0.8 + (15.0 if fog_flag else 0.0) + (10.0 if festival_flag else 0.0)
            return float(np.clip(fallback_delay, 0.0, 180.0).round(1))

_GLOBAL_PREDICTOR = None

def get_delay_predictor() -> DelayPredictor:
    """Singleton getter for the trained DelayPredictor."""
    global _GLOBAL_PREDICTOR
    if _GLOBAL_PREDICTOR is None:
        _GLOBAL_PREDICTOR = DelayPredictor()
        _GLOBAL_PREDICTOR.train_and_evaluate()
    return _GLOBAL_PREDICTOR
