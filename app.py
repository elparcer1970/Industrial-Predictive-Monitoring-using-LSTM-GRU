"""
🏭 Multivariate Sensor Prediction System
Author  : DINRAJ K DINESH
GitHub  : github.com/dinraj910
Portfolio: dinrajkdinesh.vercel.app
"""

import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json, joblib, io, os
import tensorflow as tf

# ── Suppress TF logs ──────────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ── Load artefacts ────────────────────────────────────────────────────────────
print("Loading model artefacts ...")

with open("model_config.json") as f:
    CFG = json.load(f)

WINDOW    = CFG["window_size"]
SENSORS   = CFG["sensor_cols"]
FEATURES  = CFG["all_features"]
THRESHOLD = CFG["anomaly_threshold"]
MAX_RUL   = CFG["max_rul"]
BEST      = CFG["best_model"]

model       = tf.keras.models.load_model(f"{BEST.lower()}_sensor_predictor.keras")
rul_scaler  = joblib.load("rul_scaler.pkl")

print(f"✅ {BEST} model loaded | Window={WINDOW} | Sensors={len(SENSORS)}")

# ── Colour palette ────────────────────────────────────────────────────────────
PAL   = ["#00d4ff","#ff6b6b","#51cf66","#ffd43b","#cc5de8","#ff922b","#74c0fc","#f783ac"]
BG    = "#0d0d1a"
PANEL = "#141428"

# ── Feature builder (mirrors Phase 3 in notebook) ────────────────────────────
def build_features(df: pd.DataFrame) -> np.ndarray:
    df = df.copy().reset_index(drop=True)

    # Keep only sensor columns that exist
    avail = [c for c in SENSORS if c in df.columns]
    if len(avail) == 0:
        raise ValueError(f"No matching sensor columns found.\nExpected: {SENSORS}")

    # Normalise per-column (inference-time z-score)
    for col in avail:
        m = df[col].mean(); s = df[col].std() + 1e-8
        df[col] = (df[col] - m) / s

    # Rolling features
    for col in avail:
        s_series = df[col]
        df[f"{col}_rmean"] = s_series.rolling(10, min_periods=1).mean()
        df[f"{col}_rstd"]  = s_series.rolling(10, min_periods=1).std().fillna(0)
        df[f"{col}_trend"] = s_series.diff(10).fillna(0)

    # Cross-sensor
    df["sensor_mad"] = df[avail].apply(
        lambda r: np.mean(np.abs(r.values)), axis=1
    )

    # Build ratio features
    for i in range(min(3, len(avail))):
        for j in range(i+1, min(i+3, len(avail))):
            key = f"r_{avail[i]}_{avail[j]}"
            df[key] = df[avail[i]] / (df[avail[j]].abs() + 1e-6)

    # Zero-pad to full FEATURES length
    out = pd.DataFrame(np.zeros((len(df), len(FEATURES))), columns=FEATURES)
    for col in df.columns:
        if col in out.columns:
            out[col] = df[col].values

    return out.values.astype(np.float32)


# ── Core prediction logic ─────────────────────────────────────────────────────
def run_prediction(feat_matrix: np.ndarray, raw_sensors: np.ndarray):
    X    = feat_matrix[-WINDOW:].reshape(1, WINDOW, -1)
    pred = model.predict(X, verbose=0)

    sens_pred  = pred[0][0]
    rul_norm   = float(pred[1][0][0])
    rul_cycles = float(rul_scaler.inverse_transform([[rul_norm]])[0][0])
    health_pct = min(100.0, rul_cycles / MAX_RUL * 100)

    n_shared  = min(len(sens_pred), raw_sensors.shape[1])
    recon_err = float(np.linalg.norm(
        sens_pred[:n_shared] - raw_sensors[-1, :n_shared]
    ))

    return sens_pred, rul_cycles, health_pct, recon_err


# ── Plot builder ──────────────────────────────────────────────────────────────
def build_plot(df_raw, sens_pred, rul_cycles, health_pct,
               recon_err, status_label):
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.50, wspace=0.38)
    fig.suptitle(
        f"Sensor Prediction System  ·  {BEST} Model  ·  "
        f"Status: {status_label}",
        fontsize=13, color="white", y=1.01
    )

    avail = [c for c in SENSORS if c in df_raw.columns]
    n_show = min(5, len(avail))

    # ── 1) Sensor traces (last WINDOW cycles) ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    tail = min(WINDOW, len(df_raw))
    for i in range(n_show):
        col = avail[i]
        vals = df_raw[col].values[-tail:]
        ax1.plot(vals, color=PAL[i], lw=2, label=col)
    ax1.set_title(f"Last {tail} Sensor Readings (Input Window)",
                  fontsize=11)
    ax1.set_xlabel("Timestep"); ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc="upper left", ncol=2)
    ax1.set_facecolor(PANEL)

    # ── 2) Actual vs Predicted — last sensor ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    if len(avail) > 0:
        col       = avail[0]
        actual_v  = df_raw[col].values[-min(50, len(df_raw)):]
        pred_last = float(sens_pred[0])
        ax2.plot(actual_v, color=PAL[0], lw=2, label="Actual")
        ax2.scatter(len(actual_v), pred_last,
                    color="#ff6b6b", s=120, zorder=5,
                    label=f"Forecast → {pred_last:.3f}")
        ax2.set_title(f"{col}\nActual vs Forecast", fontsize=10)
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
        ax2.set_facecolor(PANEL)

    # ── 3) Sensor forecast bar (all sensors) ─────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    n_s  = min(len(sens_pred), len(SENSORS))
    bars = ax3.bar(range(n_s), sens_pred[:n_s],
                   color=[PAL[i % len(PAL)] for i in range(n_s)],
                   alpha=0.85, edgecolor="#0d0d1a", linewidth=0.8)
    ax3.set_xticks(range(n_s))
    ax3.set_xticklabels(SENSORS[:n_s], rotation=45,
                        fontsize=7, ha="right")
    ax3.set_title("Predicted Sensor Values\n(Next Timestep)", fontsize=10)
    ax3.axhline(0, color="white", lw=0.8, alpha=0.4)
    ax3.grid(True, alpha=0.3, axis="y"); ax3.set_facecolor(PANEL)

    # ── 4) Health gauge ───────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    bar_color = ("#51cf66" if health_pct > 60
                 else "#ffd43b" if health_pct > 30
                 else "#ff6b6b")
    ax4.barh(["Health"], [health_pct],
             color=bar_color, height=0.45, alpha=0.90)
    ax4.barh(["Health"], [100 - health_pct], left=health_pct,
             color="#2a2a4a", height=0.45)
    ax4.set_xlim(0, 100)
    ax4.axvline(50, color="white", ls="--", alpha=0.35)
    ax4.axvline(25, color="#ff6b6b", ls=":", alpha=0.35)
    ax4.set_title(
        f"Engine Health: {health_pct:.1f}%\n"
        f"RUL: {rul_cycles:.0f} / {MAX_RUL} cycles",
        fontsize=10
    )
    ax4.set_xlabel("Health %"); ax4.grid(True, alpha=0.3, axis="x")
    ax4.set_facecolor(PANEL)

    # ── 5) Anomaly score gauge ────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    score_pct = min(100, recon_err / (THRESHOLD * 1.5) * 100)
    score_color = ("#ff6b6b" if recon_err > THRESHOLD
                   else "#ffd43b" if recon_err > THRESHOLD * 0.7
                   else "#51cf66")
    ax5.barh(["Anomaly\nScore"], [score_pct],
             color=score_color, height=0.4, alpha=0.9)
    ax5.barh(["Anomaly\nScore"], [100 - score_pct],
             left=score_pct, color="#2a2a4a", height=0.4)
    thresh_pct = min(100, THRESHOLD / (THRESHOLD * 1.5) * 100)
    ax5.axvline(thresh_pct, color="white", ls="--",
                lw=2, label=f"Threshold")
    ax5.set_xlim(0, 100)
    ax5.set_title(
        f"Anomaly Score: {recon_err:.4f}\n"
        f"Threshold: {THRESHOLD:.4f}",
        fontsize=10
    )
    ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3, axis="x")
    ax5.set_facecolor(PANEL)

    fig.tight_layout()
    return fig


# ── Status helper ─────────────────────────────────────────────────────────────
def get_status(recon_err, health_pct):
    if recon_err > THRESHOLD:
        return "🔴  ANOMALY DETECTED", "#ff6b6b"
    elif recon_err > THRESHOLD * 0.7 or health_pct < 30:
        return "🟡  WARNING", "#ffd43b"
    else:
        return "🟢  NORMAL", "#51cf66"


# ── Tab 1: CSV Upload ─────────────────────────────────────────────────────────
def predict_csv(file):
    if file is None:
        return "⚠️ Please upload a CSV file.", None

    try:
        df = pd.read_csv(file.name)
    except Exception as e:
        return f"❌ Could not read file: {e}", None

    if len(df) < WINDOW:
        return (
            f"❌ Need at least **{WINDOW} rows**. "
            f"Your file has only **{len(df)}**.", None
        )

    avail = [c for c in SENSORS if c in df.columns]
    if len(avail) == 0:
        return (
            f"❌ No matching sensor columns found.\n\n"
            f"Expected columns (any of): `{', '.join(SENSORS)}`\n\n"
            f"Your columns: `{', '.join(df.columns.tolist())}`", None
        )

    feat   = build_features(df)
    raw_s  = df[avail].values.astype(np.float32)
    n_pad  = len(SENSORS) - len(avail)
    if n_pad > 0:
        raw_s = np.hstack([raw_s,
                           np.zeros((len(raw_s), n_pad), np.float32)])

    sens_pred, rul_cycles, health_pct, recon_err = run_prediction(feat, raw_s)
    status_label, _ = get_status(recon_err, health_pct)

    fig = build_plot(df, sens_pred, rul_cycles,
                     health_pct, recon_err, status_label)

    forecast_rows = "\n".join(
        f"| `{s}` | `{v:+.4f}` |"
        for s, v in zip(SENSORS, sens_pred)
    )

    md = f"""
## {status_label}

| Metric | Value |
|--------|-------|
| **Remaining Useful Life** | **{rul_cycles:.0f} cycles** |
| **Engine Health** | **{health_pct:.1f}%** |
| **Anomaly Score** | {recon_err:.4f} |
| **Alert Threshold** | {THRESHOLD:.4f} |
| **Model** | {BEST} |
| **Rows analysed** | {len(df)} |
| **Active sensors** | {len(avail)} / {len(SENSORS)} |

---

### Sensor Forecast — Next Timestep

| Sensor | Predicted Value |
|--------|----------------|
{forecast_rows}

---
*Model trained on NASA CMAPSS Turbofan Engine Dataset*
"""
    return md, fig


# ── Tab 2: Live Sliders ───────────────────────────────────────────────────────
def predict_sliders(*sensor_values):
    n   = len(SENSORS)
    cur = np.array(sensor_values[:n], dtype=np.float32)

    # Build a synthetic 30-step window trending toward input values
    window_data = np.stack([
        cur * (0.80 + 0.20 * i / WINDOW)
        + np.random.normal(0, 0.03, n).astype(np.float32)
        for i in range(WINDOW)
    ])  # (WINDOW, n_sensors)

    df_s = pd.DataFrame(window_data, columns=SENSORS)

    feat   = build_features(df_s)
    raw_s  = df_s.values.astype(np.float32)

    sens_pred, rul_cycles, health_pct, recon_err = run_prediction(feat, raw_s)
    status_label, _ = get_status(recon_err, health_pct)

    fig = build_plot(df_s, sens_pred, rul_cycles,
                     health_pct, recon_err, status_label)

    forecast_rows = "\n".join(
        f"| `{s}` | `{v:+.4f}` |"
        for s, v in zip(SENSORS, sens_pred)
    )

    md = f"""
## {status_label}

| Metric | Value |
|--------|-------|
| **Remaining Useful Life** | **{rul_cycles:.0f} cycles** |
| **Engine Health** | **{health_pct:.1f}%** |
| **Anomaly Score** | {recon_err:.4f} |
| **Alert Threshold** | {THRESHOLD:.4f} |

### Sensor Forecast

| Sensor | Predicted |
|--------|-----------|
{forecast_rows}
"""
    return md, fig


# ── Tab 3: Generate sample CSV ────────────────────────────────────────────────
def generate_sample(scenario):
    np.random.seed(42)
    n = 60

    S_MEAN = np.array([518.67, 642.68, 1590.09, 1408.85,
                        14.62,  21.61,  553.36, 2388.02,
                       9063.19,   1.3,   47.54,  521.72,
                       2388.08, 8141.50])[:len(SENSORS)]
    S_STD  = S_MEAN * 0.015

    data = {}
    for i, col in enumerate(SENSORS):
        base = S_MEAN[i] if i < len(S_MEAN) else 1.0
        std  = S_STD[i]  if i < len(S_STD)  else 0.01

        if scenario == "Healthy Engine":
            data[col] = base + np.random.normal(0, std, n)

        elif scenario == "Degrading Engine":
            deg = np.linspace(0, 0.4, n)
            data[col] = base * (1 + deg * 0.05) + np.random.normal(0, std, n)

        elif scenario == "Near Failure":
            deg = np.linspace(0.4, 0.9, n)
            data[col] = base * (1 + deg * 0.12) + np.random.normal(0, std * 2, n)

    df  = pd.DataFrame(data)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)

    path = f"/tmp/sample_{scenario.replace(' ','_')}.csv"
    df.to_csv(path, index=False)
    return path, f"✅ Generated **{scenario}** sample — {n} rows, {len(SENSORS)} sensors.\n\nDownload and upload in Tab 1."


# ── Build Gradio UI ───────────────────────────────────────────────────────────
DESCRIPTION = f"""
**Trained on:** NASA CMAPSS Turbofan Engine Dataset (150 engines, run-to-failure)  
**Architecture:** Dual-output {BEST} — Sensor Forecasting + RUL Estimation  
**Inference latency:** ~0.25 ms/sample &nbsp;|&nbsp; **RUL MAE:** 0.17  
**Author:** [DINRAJ K DINESH](https://dinrajkdinesh.vercel.app) &nbsp;·&nbsp; [GitHub](https://github.com/dinraj910)
"""

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue="cyan",
        secondary_hue="slate",
        neutral_hue="slate",
    ),
    title="Sensor Prediction System",
    css="""
    .gradio-container { max-width: 1200px !important; }
    .gr-button-primary { background: #00d4ff !important; color: #000 !important; }
    footer { display: none !important; }
    """
) as demo:

    gr.Markdown("""
# 🏭 Multivariate Sensor Prediction System
### LSTM / GRU — Industrial Predictive Monitoring
""")
    gr.Markdown(DESCRIPTION)

    gr.Markdown("---")

    with gr.Tabs():

        # ── Tab 1 ─────────────────────────────────────────────────────────────
        with gr.TabItem("📂 Upload Sensor CSV"):
            gr.Markdown("""
Upload a CSV file where each column is a sensor and each row is one timestep.
Minimum **30 rows** required. Use **Tab 3** to generate a sample file first.
""")
            with gr.Row():
                with gr.Column(scale=1):
                    csv_input  = gr.File(
                        label="Sensor CSV File",
                        file_types=[".csv"]
                    )
                    csv_button = gr.Button(
                        "🔍 Analyse", variant="primary", size="lg"
                    )
                with gr.Column(scale=2):
                    csv_result = gr.Markdown(label="Results")

            csv_plot = gr.Plot(label="Prediction Dashboard")

            csv_button.click(
                fn=predict_csv,
                inputs=[csv_input],
                outputs=[csv_result, csv_plot]
            )

        # ── Tab 2 ─────────────────────────────────────────────────────────────
        with gr.TabItem("🎛️ Live Sensor Sliders"):
            gr.Markdown("""
Adjust normalised sensor values (−3 to +3 scale).
**0.0 = healthy baseline. Positive = above normal. Negative = below normal.**
Click **Predict** to see results instantly.
""")
            with gr.Row():
                with gr.Column(scale=1):
                    sliders = []
                    for i, s in enumerate(SENSORS):
                        sliders.append(
                            gr.Slider(
                                minimum=-3.0, maximum=3.0,
                                value=0.0, step=0.05,
                                label=f"{s}",
                                info="0 = baseline healthy"
                            )
                        )
                    sl_button = gr.Button(
                        "⚡ Predict", variant="primary", size="lg"
                    )
                with gr.Column(scale=2):
                    sl_result = gr.Markdown(label="Results")

            sl_plot = gr.Plot(label="Prediction Dashboard")

            sl_button.click(
                fn=predict_sliders,
                inputs=sliders,
                outputs=[sl_result, sl_plot]
            )

            gr.Examples(
                examples=[
                    [0.0]  * len(SENSORS),
                    [1.5]  * len(SENSORS),
                    [2.8]  * len(SENSORS),
                    [-0.5] * len(SENSORS),
                ],
                inputs=sliders,
                label="Quick Examples  (0=Healthy | 1.5=Mild stress | 2.8=Near failure)"
            )

        # ── Tab 3 ─────────────────────────────────────────────────────────────
        with gr.TabItem("📋 Generate Sample CSV"):
            gr.Markdown("""
Don't have sensor data? Generate a realistic sample CSV, then upload it in **Tab 1**.
""")
            with gr.Row():
                with gr.Column():
                    scenario_dd = gr.Dropdown(
                        choices=["Healthy Engine",
                                 "Degrading Engine",
                                 "Near Failure"],
                        value="Healthy Engine",
                        label="Engine Scenario"
                    )
                    gen_button = gr.Button(
                        "⬇️ Generate CSV", variant="primary"
                    )
                with gr.Column():
                    gen_status = gr.Markdown()
                    gen_file   = gr.File(label="Download CSV")

            gen_button.click(
                fn=generate_sample,
                inputs=[scenario_dd],
                outputs=[gen_file, gen_status]
            )

        # ── Tab 4 — About ─────────────────────────────────────────────────────
        with gr.TabItem("ℹ️ About This Project"):
            gr.Markdown(f"""
## About This System

### What it does
This system performs **industrial predictive monitoring** on multivariate sensor data.
Given a window of historical sensor readings, it predicts:

1. **Sensor Forecast** — What will each sensor read at the next timestep
2. **Remaining Useful Life (RUL)** — How many operating cycles remain before failure
3. **Anomaly Detection** — Whether the current sensor state is abnormal

---

### Architecture

```
Raw Sensor Data
      ↓
Feature Engineering (5-layer pipeline)
  → Spike removal & normalisation
  → Rolling statistics (mean, std, trend)
  → Cross-sensor ratios & health score
      ↓
Sliding Window (30 timesteps × {len(FEATURES)} features)
      ↓
{BEST} Model (dual-output)
  ┌─────────────────────┐
  │  Sensor Head        │ → Next sensor state ({len(SENSORS)} values)
  │  RUL Head           │ → Remaining useful life (cycles)
  └─────────────────────┘
      ↓
Anomaly Score = ||predicted − actual||₂
```

---

### Training Dataset

**NASA CMAPSS FD001** — Commercial Modular Aero-Propulsion System Simulation  
- 150 simulated turbofan engines, run-to-failure  
- 14 sensor measurements per engine cycle  
- 3 operating conditions  
- Degradation modes: HPC (High Pressure Compressor) deterioration  

---

### Model Performance

| Metric | Value |
|--------|-------|
| RUL MAE (normalised) | **{CFG.get('lstm_rul_mae', 0.17):.4f}** |
| Inference latency | **~0.25 ms / sample** |
| Parameters | **{CFG.get('lstm_params', 'N/A'):,}** |
| Sanity checks | **{CFG.get('sanity_checks', '16/19')}** |

---

### Tech Stack
`TensorFlow 2.x` · `Keras` · `Gradio` · `NumPy` · `Pandas` · `scikit-learn`

---

### Author
**DINRAJ K DINESH**  
🌐 [dinrajkdinesh.vercel.app](https://dinrajkdinesh.vercel.app)  
💻 [github.com/dinraj910](https://github.com/dinraj910)
""")

    gr.Markdown("""
---
<center>Built with TensorFlow + Gradio · Trained on NASA CMAPSS · <a href="https://github.com/dinraj910">GitHub</a></center>
""")

if __name__ == "__main__":
    demo.launch(share=False, show_error=True)
