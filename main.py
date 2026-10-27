from flask import Flask, jsonify, render_template
from werkzeug.exceptions import RequestEntityTooLarge

from config import FLASK_DEBUG, MAX_CONTENT_LENGTH
from routes.upload import upload_bp


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    app.config["DEBUG"] = FLASK_DEBUG
    app.register_blueprint(upload_bp)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.errorhandler(RequestEntityTooLarge)
    def handle_request_entity_too_large(_error):
        return jsonify({"error": "Uploaded file is too large."}), 413

    return app


if __name__ == "__main__":
    create_app().run(debug=FLASK_DEBUG)
