"""
run.py
──────
Entry point for the Multivariate Sensor Prediction System Flask app.

Usage:
    python run.py
"""

from webapp import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
