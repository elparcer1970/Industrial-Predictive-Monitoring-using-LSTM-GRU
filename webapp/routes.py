"""
webapp/routes.py
────────────────
All Flask route handlers.  Registered via register_routes(app) so that
url_for() endpoint names stay unprefixed (no Blueprint prefix required).
"""

import io

import numpy as np
import pandas as pd
from flask import render_template, request, jsonify, send_file, abort

from .predictor import (
    CFG, WINDOW, SENSORS, THRESHOLD, MAX_RUL, BEST, AUTHOR,
    build_features, run_prediction, get_status,
    build_chart_b64, generate_sample_df,
)


def register_routes(app):
    """Attach all routes to the Flask app instance."""

    # ── Pages ─────────────────────────────────────────────────────────────────

    @app.route("/")
    def index():
        stats = {
            "best_model":   BEST,
            "lstm_rmse":    round(CFG.get("lstm_sensor_rmse", 1.004), 4),
            "gru_rmse":     round(CFG.get("gru_sensor_rmse",  1.002), 4),
            "lstm_rul_mae": round(CFG.get("lstm_rul_mae",     0.172), 4),
            "gru_rul_mae":  round(CFG.get("gru_rul_mae",      0.204), 4),
            "lstm_params":  CFG.get("lstm_params", 85000),
            "gru_params":   CFG.get("gru_params",  65000),
            "n_sensors":    len(SENSORS),
            "window":       WINDOW,
            "threshold":    round(THRESHOLD, 4),
            "max_rul":      MAX_RUL,
            "author":       AUTHOR,
        }
        return render_template("index.html", stats=stats)

    @app.route("/predict")
    def predict_page():
        return render_template("predict.html", sensors=SENSORS, window=WINDOW)

    @app.route("/about")
    def about_page():
        info = {
            "best_model":   BEST,
            "n_sensors":    len(SENSORS),
            "n_features":   CFG.get("n_features", 63),
            "window":       WINDOW,
            "lstm_rmse":    round(CFG.get("lstm_sensor_rmse", 1.004), 4),
            "gru_rmse":     round(CFG.get("gru_sensor_rmse",  1.002), 4),
            "lstm_rul_mae": round(CFG.get("lstm_rul_mae",     0.172), 4),
            "gru_rul_mae":  round(CFG.get("gru_rul_mae",      0.204), 4),
            "lstm_params":  f"{CFG.get('lstm_params', 85000):,}",
            "gru_params":   f"{CFG.get('gru_params',  65000):,}",
            "threshold":    round(THRESHOLD, 4),
            "max_rul":      MAX_RUL,
            "author":       AUTHOR,
            "sensors":      SENSORS,
            "date":         CFG.get("date",        "2026-03-08"),
            "tf_version":   CFG.get("tensorflow",  "2.19.0"),
        }
        return render_template("about.html", info=info)

    # ── API: CSV prediction ───────────────────────────────────────────────────

    @app.route("/api/predict/csv", methods=["POST"])
    def api_predict_csv():
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded."}), 400

        f = request.files["file"]
        if not f.filename.lower().endswith(".csv"):
            return jsonify({"error": "Only CSV files are accepted."}), 400

        try:
            df = pd.read_csv(f)
        except Exception as e:
            return jsonify({"error": f"Cannot parse CSV: {e}"}), 400

        if len(df) < WINDOW:
            return jsonify({
                "error": f"Need at least {WINDOW} rows; got {len(df)}."
            }), 400

        avail = [c for c in SENSORS if c in df.columns]
        if not avail:
            return jsonify({
                "error":        f"No sensor columns found. Expected: {SENSORS}",
                "your_columns": df.columns.tolist(),
            }), 400

        try:
            feat  = build_features(df)
            raw_s = df[avail].values.astype(np.float32)
            n_pad = len(SENSORS) - len(avail)
            if n_pad > 0:
                raw_s = np.hstack(
                    [raw_s, np.zeros((len(raw_s), n_pad), np.float32)]
                )

            sens_pred, rul_cycles, health_pct, recon_err = run_prediction(feat, raw_s)
            status = get_status(recon_err, health_pct)
            chart  = build_chart_b64(df, sens_pred, rul_cycles,
                                     health_pct, recon_err, status)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        return jsonify({
            "status":          status,
            "rul_cycles":      round(rul_cycles, 1),
            "health_pct":      round(health_pct, 2),
            "anomaly_score":   round(recon_err, 4),
            "threshold":       round(THRESHOLD, 4),
            "is_anomaly":      recon_err > THRESHOLD,
            "model":           BEST,
            "rows_analysed":   len(df),
            "active_sensors":  len(avail),
            "total_sensors":   len(SENSORS),
            "sensor_forecast": dict(
                zip(SENSORS, [round(float(v), 4) for v in sens_pred])
            ),
            "chart_b64":       chart,
        })

    # ── API: Live slider prediction ───────────────────────────────────────────

    @app.route("/api/predict/live", methods=["POST"])
    def api_predict_live():
        body = request.get_json(silent=True)
        if not body or "sensor_values" not in body:
            return jsonify({"error": "JSON body with 'sensor_values' required."}), 400

        vals = body["sensor_values"]
        if len(vals) != len(SENSORS):
            return jsonify({
                "error": f"Expected {len(SENSORS)} values, got {len(vals)}."
            }), 400

        try:
            cur = np.array(vals, dtype=np.float32)
        except (ValueError, TypeError):
            return jsonify({"error": "sensor_values must be a list of numbers."}), 400

        rng = np.random.default_rng()
        window_data = np.stack([
            cur * (0.80 + 0.20 * i / WINDOW)
            + rng.normal(0, 0.03, len(SENSORS)).astype(np.float32)
            for i in range(WINDOW)
        ])
        df_s = pd.DataFrame(window_data, columns=SENSORS)

        try:
            feat      = build_features(df_s)
            raw_s     = df_s.values.astype(np.float32)
            sens_pred, rul_cycles, health_pct, recon_err = run_prediction(feat, raw_s)
            status    = get_status(recon_err, health_pct)
            chart     = build_chart_b64(df_s, sens_pred, rul_cycles,
                                        health_pct, recon_err, status)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        return jsonify({
            "status":          status,
            "rul_cycles":      round(rul_cycles, 1),
            "health_pct":      round(health_pct, 2),
            "anomaly_score":   round(recon_err, 4),
            "threshold":       round(THRESHOLD, 4),
            "is_anomaly":      recon_err > THRESHOLD,
            "model":           BEST,
            "sensor_forecast": dict(
                zip(SENSORS, [round(float(v), 4) for v in sens_pred])
            ),
            "chart_b64":       chart,
        })

    # ── API: Download sample CSV ──────────────────────────────────────────────

    @app.route("/api/sample/<scenario>")
    def api_sample(scenario):
        if scenario not in {"healthy", "degrading", "near_failure"}:
            abort(404)

        df  = generate_sample_df(scenario)
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)

        return send_file(buf, mimetype="text/csv",
                         as_attachment=True,
                         download_name=f"sample_{scenario}.csv")

    # ── API: Health check ─────────────────────────────────────────────────────

    @app.route("/api/health")
    def api_health():
        return jsonify({
            "status":  "ok",
            "model":   BEST,
            "sensors": len(SENSORS),
            "window":  WINDOW,
        })
