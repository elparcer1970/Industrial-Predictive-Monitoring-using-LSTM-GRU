"""
webapp/__init__.py
──────────────────
Flask application factory.

Flask(__name__) sets root_path to the webapp/ directory, so:
  template_folder='templates'  →  webapp/templates/
  static_folder='static'       →  webapp/static/
Both are resolved automatically — no absolute paths needed.
"""

from flask import Flask


def create_app():
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload cap

    from .routes import register_routes
    register_routes(app)

    return app
