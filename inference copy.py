
import numpy as np, pandas as pd, joblib, json
import tensorflow as tf

class SensorPredictor:
    def __init__(self, model_dir="."):
        with open(f"{model_dir}/model_config.json") as f:
            self.cfg = json.load(f)
        self.model = tf.keras.models.load_model(
            f"{model_dir}/lstm_sensor_predictor.keras")
        self.prep  = joblib.load(f"{model_dir}/sensor_preprocessor.pkl")
        self.rul_s = joblib.load(f"{model_dir}/rul_scaler.pkl")
        self.W     = self.cfg["window_size"]
        self.T     = self.cfg["anomaly_threshold"]

    def predict(self, window_df):
        clean = self.prep.transform(window_df)
        X = clean[self.cfg["all_features"]].values
        X = X.reshape(1, self.W, -1).astype(np.float32)
        s, r = self.model.predict(X, verbose=0)
        rul_cyc = float(self.rul_s.inverse_transform([[r[0][0]]])[0][0])
        err     = float(np.linalg.norm(s[0] - X[0, -1, :len(s[0])]))
        return {
            "sensor_forecast": dict(zip(self.cfg["sensor_cols"], s[0].tolist())),
            "rul_cycles": rul_cyc,
            "anomaly_score": err,
            "is_anomaly": err > self.T,
        }

if __name__ == "__main__":
    p = SensorPredictor(".")
    print("✅ Model loaded. Best model:", p.cfg["best_model"])
