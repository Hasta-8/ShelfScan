from flask import Flask

from .routes import register_routes

app = Flask(__name__)
app.config.from_object("app.config")

register_routes(app)
