---
title: Sensor Prediction System
emoji: 🏭
colorFrom: cyan
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: mit
---

# 🏭 Multivariate Sensor Prediction System

> **Industrial Predictive Monitoring using LSTM & GRU**  
> Trained on NASA CMAPSS Turbofan Engine Dataset

[![HuggingFace](https://img.shields.io/badge/🤗_Live_Demo-HuggingFace_Spaces-yellow)](https://huggingface.co/spaces/dinraj910/sensor-prediction-system)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## What This System Does

Given a window of historical sensor readings, the model predicts:

| Output | Description |
|--------|-------------|
| **Sensor Forecast** | Next-timestep values for all N sensors |
| **Remaining Useful Life** | How many cycles until engine failure |
| **Anomaly Score** | Reconstruction error flagging abnormal states |

---

## System Architecture

```
Raw Sensor Data (CSV or live sliders)
         ↓
Feature Engineering Pipeline (5 layers)
  → Spike removal + StandardScaler normalisation
  → Rolling statistics: mean, std, trend (window=10)
  → Cross-sensor ratios + Mahalanobis health score
         ↓
Sliding Window  →  shape: (30 timesteps × N features)
         ↓
┌──────────────────────────────────────────┐
│  Dual-Output LSTM / GRU Model            │
│                                          │
│  LSTM Block 1  (128 units, return_seq)   │
│  BatchNorm + Dropout(0.30)               │
│  LSTM Block 2  (64 units)                │
│  BatchNorm + Dropout(0.20)               │
│  Shared Dense  (64 units, ReLU)          │
│       ↓                    ↓             │
│  Sensor Head           RUL Head          │
│  Dense(32) → N_sensors  Dense(16) → 1   │
└──────────────────────────────────────────┘
         ↓
Anomaly Score = ||predicted_sensors − actual||₂
```

---

## Model Performance

| Metric | LSTM | GRU |
|--------|------|-----|
| Sensor RMSE (normalised) | 1.004 | 1.002 |
| RUL MAE (normalised) | **0.172** | 0.204 |
| Inference latency | **0.25 ms** | 0.26 ms |
| Parameters | ~85K | ~65K |

> RUL MAE of **0.172** on normalised scale = ~22 cycles error on a 0–130 cycle range.
> Inference at **0.25 ms/sample** — 200× faster than the 50 ms production requirement.

---

## Dataset — NASA CMAPSS FD001

| Property | Value |
|----------|-------|
| Source | NASA Prognostics Center of Excellence |
| Engines | 150 simulated turbofan engines |
| Sensors | 14 active measurements per cycle |
| Labels | Run-to-failure (RUL per timestep) |
| Fault | HPC (High Pressure Compressor) degradation |

---

## Web Application — 3 Tabs

### Tab 1 — Upload Sensor CSV
Upload any CSV where columns are sensors and rows are timesteps.
Minimum 30 rows required.

### Tab 2 — Live Sliders
Adjust normalised sensor values (−3 to +3) in real-time.
Instant predictions on every click.

### Tab 3 — Generate Sample CSV
Generate realistic test data for 3 scenarios:
- Healthy Engine
- Degrading Engine
- Near Failure

---

## Local Setup

```bash
# Clone
git clone https://github.com/dinraj910/sensor-prediction-system
cd sensor-prediction-system

# Install
pip install -r requirements.txt

# Copy your model files from the Colab ZIP
# (lstm_sensor_predictor.keras, sensor_preprocessor.pkl,
#  rul_scaler.pkl, model_config.json)

# Run
python app.py
# → Open http://localhost:7860
```

---

## Project Structure

```
sensor-prediction-system/
├── app.py                          ← Gradio web application
├── inference.py                    ← Standalone predictor class
├── requirements.txt
├── model_config.json               ← Thresholds + metadata
├── lstm_sensor_predictor.keras     ← Primary model (TF 2.x)
├── lstm_sensor_predictor.h5        ← Primary model (legacy)
├── gru_sensor_predictor.keras      ← Secondary model
├── gru_sensor_predictor.h5         ← Secondary model
├── sensor_preprocessor.pkl         ← Feature pipeline
├── rul_scaler.pkl                  ← RUL inverse transform
└── README.md
```

---

## Tech Stack

`TensorFlow 2.13` · `Keras` · `Gradio 4.x` · `NumPy` · `Pandas` · `scikit-learn` · `Matplotlib`

---

## Author

**DINRAJ K DINESH**  
🌐 [dinrajkdinesh.vercel.app](https://dinrajkdinesh.vercel.app)  
💻 [github.com/dinraj910](https://github.com/dinraj910)

---

## License

MIT — free to use and modify with attribution.
