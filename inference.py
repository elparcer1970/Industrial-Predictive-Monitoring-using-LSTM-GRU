"""
Standalone inference — use this outside of the web app.

Usage:
    from inference import SensorPredictor
    p = SensorPredictor(".")
    result = p.predict(your_dataframe)
"""
import numpy as np
import pandas as pd
import joblib, json
import tensorflow as tf


class SensorPredictor:
    def __init__(self, model_dir="."):
        with open(f"{model_dir}/model_config.json") as f:
            self.cfg = json.load(f)

        best = self.cfg["best_model"].lower()
        self.model = tf.keras.models.load_model(
            f"{model_dir}/{best}_sensor_predictor.keras"
        )
        self.rul_scaler = joblib.load(f"{model_dir}/rul_scaler.pkl")

        self.W         = self.cfg["window_size"]
        self.SENSORS   = self.cfg["sensor_cols"]
        self.FEATURES  = self.cfg["all_features"]
        self.THRESHOLD = self.cfg["anomaly_threshold"]
        self.MAX_RUL   = self.cfg["max_rul"]

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy().reset_index(drop=True)
        avail = [c for c in self.SENSORS if c in df.columns]

        for col in avail:
            m = df[col].mean(); s = df[col].std() + 1e-8
            df[col] = (df[col] - m) / s

        for col in avail:
            s = df[col]
            df[f"{col}_rmean"] = s.rolling(10, min_periods=1).mean()
            df[f"{col}_rstd"]  = s.rolling(10, min_periods=1).std().fillna(0)
            df[f"{col}_trend"] = s.diff(10).fillna(0)

        df["sensor_mad"] = df[avail].apply(
            lambda r: np.mean(np.abs(r.values)), axis=1
        )

        out = pd.DataFrame(
            np.zeros((len(df), len(self.FEATURES))),
            columns=self.FEATURES
        )
        for col in df.columns:
            if col in out.columns:
                out[col] = df[col].values

        return out.values.astype(np.float32)

    def predict(self, df: pd.DataFrame) -> dict:
        """
        df : DataFrame with sensor columns, at least WINDOW rows.

        Returns dict:
          sensor_forecast : {sensor_name: predicted_value}
          rul_cycles      : float — cycles remaining
          health_pct      : float — 0-100%
          anomaly_score   : float — reconstruction error
          is_anomaly      : bool
          status          : "NORMAL" / "WARNING" / "ANOMALY"
        """
        if len(df) < self.W:
            raise ValueError(f"Need >= {self.W} rows, got {len(df)}")

        feat  = self._build_features(df)
        X     = feat[-self.W:].reshape(1, self.W, -1)
        pred  = self.model.predict(X, verbose=0)

        sens_pred  = pred[0][0]
        rul_norm   = float(pred[1][0][0])
        rul_cycles = float(
            self.rul_scaler.inverse_transform([[rul_norm]])[0][0]
        )
        health_pct = min(100.0, rul_cycles / self.MAX_RUL * 100)

        avail  = [c for c in self.SENSORS if c in df.columns]
        raw_s  = df[avail].values.astype(np.float32)
        n_sh   = min(len(sens_pred), raw_s.shape[1])
        recon  = float(np.linalg.norm(
            sens_pred[:n_sh] - raw_s[-1, :n_sh]
        ))

        if recon > self.THRESHOLD:
            status = "ANOMALY"
        elif recon > self.THRESHOLD * 0.7 or health_pct < 30:
            status = "WARNING"
        else:
            status = "NORMAL"

        return {
            "sensor_forecast": dict(zip(self.SENSORS, sens_pred.tolist())),
            "rul_cycles":      rul_cycles,
            "health_pct":      health_pct,
            "anomaly_score":   recon,
            "is_anomaly":      recon > self.THRESHOLD,
            "status":          status,
            "threshold":       self.THRESHOLD,
        }


if __name__ == "__main__":
    import sys
    predictor = SensorPredictor(".")
    print(f"✅ Model loaded — {predictor.cfg['best_model']}")
    print(f"   Sensors  : {len(predictor.SENSORS)}")
    print(f"   Window   : {predictor.W}")
    print(f"   Threshold: {predictor.THRESHOLD:.4f}")

    if len(sys.argv) > 1:
        df  = pd.read_csv(sys.argv[1])
        res = predictor.predict(df)
        print(f"\nResult for {sys.argv[1]}:")
        for k, v in res.items():
            if k != "sensor_forecast":
                print(f"  {k}: {v}")
