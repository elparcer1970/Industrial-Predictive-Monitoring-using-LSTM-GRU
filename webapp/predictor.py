"""
webapp/predictor.py
───────────────────
All ML inference logic: model loading, feature engineering,
prediction, chart generation, and sample data generation.
"""

import io
import os
import json
import base64

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf

# Suppress TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── Project root (one level above this file) ──────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Load model config ─────────────────────────────────────────────────────────
with open(os.path.join(BASE_DIR, "model_config.json")) as f:
    CFG = json.load(f)

WINDOW    = CFG["window_size"]
SENSORS   = CFG["sensor_cols"]
FEATURES  = CFG["all_features"]
THRESHOLD = CFG["anomaly_threshold"]
MAX_RUL   = CFG["max_rul"]
BEST      = CFG["best_model"]
AUTHOR    = CFG.get("author", "DINRAJ K DINESH")

# ── Colour palette ─────────────────────────────────────────────────────────────
PAL   = ["#00d4ff", "#ff6b6b", "#51cf66", "#ffd43b",
         "#cc5de8", "#ff922b", "#74c0fc", "#f783ac"]
BG    = "#0d0d1a"
PANEL = "#141428"


# ── Custom loss (must be registered before model.load) ────────────────────────
@tf.keras.utils.register_keras_serializable()
def rul_asymmetric_loss(y_true, y_pred):
    """Asymmetric RUL loss — over-estimating remaining life is penalised more."""
    err = y_true - y_pred
    return tf.reduce_mean(
        tf.where(err >= 0,
                 0.5 * tf.square(err),
                 1.5 * tf.square(err))
    )


# ── Load model & scaler ───────────────────────────────────────────────────────
model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, f"{BEST.lower()}_sensor_predictor.keras"),
    custom_objects={"rul_asymmetric_loss": rul_asymmetric_loss},
)
rul_scaler = joblib.load(os.path.join(BASE_DIR, "rul_scaler.pkl"))

print(f"✅ {BEST} model loaded  |  Window={WINDOW}  |  Sensors={len(SENSORS)}")


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering (mirrors Phase 3 in training notebook)
# ─────────────────────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> np.ndarray:
    df    = df.copy().reset_index(drop=True)
    avail = [c for c in SENSORS if c in df.columns]
    if not avail:
        raise ValueError(f"No matching sensor columns. Expected: {SENSORS}")

    for col in avail:
        m = df[col].mean()
        s = df[col].std() + 1e-8
        df[col] = (df[col] - m) / s

    for col in avail:
        s = df[col]
        df[f"{col}_rmean"] = s.rolling(10, min_periods=1).mean()
        df[f"{col}_rstd"]  = s.rolling(10, min_periods=1).std().fillna(0)
        df[f"{col}_trend"] = s.diff(10).fillna(0)

    df["sensor_mad"] = df[avail].apply(
        lambda r: np.mean(np.abs(r.values)), axis=1
    )

    for i in range(min(3, len(avail))):
        for j in range(i + 1, min(i + 3, len(avail))):
            key = f"r_{avail[i]}_{avail[j]}"
            df[key] = df[avail[i]] / (df[avail[j]].abs() + 1e-6)

    out = pd.DataFrame(np.zeros((len(df), len(FEATURES))), columns=FEATURES)
    for col in df.columns:
        if col in out.columns:
            out[col] = df[col].values

    return out.values.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Core prediction
# ─────────────────────────────────────────────────────────────────────────────
def run_prediction(feat_matrix: np.ndarray, raw_sensors: np.ndarray):
    X    = feat_matrix[-WINDOW:].reshape(1, WINDOW, -1)
    pred = model.predict(X, verbose=0)

    sens_pred  = pred[0][0]
    rul_norm   = float(pred[1][0][0])
    rul_cycles = float(rul_scaler.inverse_transform([[rul_norm]])[0][0])
    health_pct = min(100.0, max(0.0, rul_cycles / MAX_RUL * 100))

    n_shared  = min(len(sens_pred), raw_sensors.shape[1])
    recon_err = float(np.linalg.norm(
        sens_pred[:n_shared] - raw_sensors[-1, :n_shared]
    ))

    return sens_pred, rul_cycles, health_pct, recon_err


def get_status(recon_err: float, health_pct: float) -> str:
    if recon_err > THRESHOLD:
        return "ANOMALY"
    elif recon_err > THRESHOLD * 0.7 or health_pct < 30:
        return "WARNING"
    return "NORMAL"


# ─────────────────────────────────────────────────────────────────────────────
# Plot → base64 PNG
# ─────────────────────────────────────────────────────────────────────────────
def build_chart_b64(df_raw, sens_pred, rul_cycles, health_pct,
                    recon_err, status) -> str:
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.40)

    icons = {"NORMAL": "🟢 NORMAL", "WARNING": "🟡 WARNING",
             "ANOMALY": "🔴 ANOMALY DETECTED"}
    fig.suptitle(
        f"Sensor Prediction Dashboard  ·  {BEST} Model  ·  "
        f"Status: {icons.get(status, status)}",
        fontsize=13, color="white", y=1.01,
    )

    avail  = [c for c in SENSORS if c in df_raw.columns]
    n_show = min(5, len(avail))

    # 1) Sensor traces
    ax1  = fig.add_subplot(gs[0, :2])
    tail = min(WINDOW, len(df_raw))
    for i in range(n_show):
        ax1.plot(df_raw[avail[i]].values[-tail:], color=PAL[i], lw=2, label=avail[i])
    ax1.set_title(f"Last {tail} Sensor Readings (Input Window)", fontsize=11)
    ax1.set_xlabel("Timestep")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc="upper left", ncol=2)
    ax1.set_facecolor(PANEL)

    # 2) Actual vs Predicted
    ax2 = fig.add_subplot(gs[0, 2])
    if avail:
        col      = avail[0]
        actual_v = df_raw[col].values[-min(50, len(df_raw)):]
        pred_f   = float(sens_pred[0])
        ax2.plot(actual_v, color=PAL[0], lw=2, label="Actual")
        ax2.scatter(len(actual_v), pred_f, color="#ff6b6b", s=120,
                    zorder=5, label=f"Forecast → {pred_f:.3f}")
        ax2.set_title(f"{col}\nActual vs Forecast", fontsize=10)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor(PANEL)

    # 3) Sensor forecast bar
    ax3 = fig.add_subplot(gs[1, 0])
    n_s = min(len(sens_pred), len(SENSORS))
    ax3.bar(range(n_s), sens_pred[:n_s],
            color=[PAL[i % len(PAL)] for i in range(n_s)],
            alpha=0.85, edgecolor=BG, linewidth=0.8)
    ax3.set_xticks(range(n_s))
    ax3.set_xticklabels(SENSORS[:n_s], rotation=45, fontsize=7, ha="right")
    ax3.set_title("Predicted Sensor Values\n(Next Timestep)", fontsize=10)
    ax3.axhline(0, color="white", lw=0.8, alpha=0.4)
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.set_facecolor(PANEL)

    # 4) Health gauge
    ax4 = fig.add_subplot(gs[1, 1])
    bar_color = ("#51cf66" if health_pct > 60 else
                 "#ffd43b" if health_pct > 30 else "#ff6b6b")
    ax4.barh(["Health"], [health_pct], color=bar_color, height=0.45, alpha=0.90)
    ax4.barh(["Health"], [100 - health_pct],
             left=health_pct, color="#2a2a4a", height=0.45)
    ax4.set_xlim(0, 100)
    ax4.axvline(50, color="white",   ls="--", alpha=0.35)
    ax4.axvline(25, color="#ff6b6b", ls=":",  alpha=0.35)
    ax4.set_title(
        f"Engine Health: {health_pct:.1f}%\nRUL: {rul_cycles:.0f} / {MAX_RUL} cycles",
        fontsize=10,
    )
    ax4.set_xlabel("Health %")
    ax4.grid(True, alpha=0.3, axis="x")
    ax4.set_facecolor(PANEL)

    # 5) Anomaly score gauge
    ax5 = fig.add_subplot(gs[1, 2])
    score_pct   = min(100, recon_err / (THRESHOLD * 1.5) * 100)
    score_color = ("#ff6b6b" if recon_err > THRESHOLD else
                   "#ffd43b" if recon_err > THRESHOLD * 0.7 else "#51cf66")
    ax5.barh(["Anomaly\nScore"], [score_pct], color=score_color, height=0.4, alpha=0.9)
    ax5.barh(["Anomaly\nScore"], [100 - score_pct],
             left=score_pct, color="#2a2a4a", height=0.4)
    thresh_pct = min(100, THRESHOLD / (THRESHOLD * 1.5) * 100)
    ax5.axvline(thresh_pct, color="white", ls="--", lw=2, label="Threshold")
    ax5.set_xlim(0, 100)
    ax5.set_title(
        f"Anomaly Score: {recon_err:.4f}\nThreshold: {THRESHOLD:.4f}",
        fontsize=10,
    )
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3, axis="x")
    ax5.set_facecolor(PANEL)

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Sample data generator (NASA CMAPSS physical baselines)
# ─────────────────────────────────────────────────────────────────────────────
_S_MEAN = np.array([518.67, 642.68, 1590.09, 1408.85,
                     14.62,  21.61,  553.36, 2388.02,
                   9063.19,   1.30,   47.54,  521.72,
                   2388.08, 8141.50])


def generate_sample_df(scenario: str, n: int = 60) -> pd.DataFrame:
    rng    = np.random.default_rng(42)
    s_mean = _S_MEAN[:len(SENSORS)]
    s_std  = s_mean * 0.015
    data   = {}

    for i, col in enumerate(SENSORS):
        base = s_mean[i] if i < len(s_mean) else 1.0
        std  = s_std[i]  if i < len(s_std)  else 0.01

        if scenario == "healthy":
            data[col] = base + rng.normal(0, std, n)
        elif scenario == "degrading":
            deg = np.linspace(0, 0.4, n)
            data[col] = base * (1 + deg * 0.05) + rng.normal(0, std, n)
        elif scenario == "near_failure":
            deg = np.linspace(0.4, 0.9, n)
            data[col] = base * (1 + deg * 0.12) + rng.normal(0, std * 2, n)
        else:
            data[col] = base + rng.normal(0, std, n)

    return pd.DataFrame(data)
