from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_migrate import Migrate
from config import Config
from app.testing import OptimizedHybridRecommender

db = SQLAlchemy()
login = LoginManager()
migrate = Migrate()

recommender = OptimizedHybridRecommender(
    cache_dir=Config.CACHE_DIR
)

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    db.init_app(app)
    login.init_app(app)
    migrate.init_app(app, db)

    from app.routes import bp
    app.register_blueprint(bp)

    # --- Force recommender training/loading on app startup ---
    with app.app_context():
        print("Initializing recommender...")
        try:
            recommender.train(
                books_path=Config.BOOKS_PATH,
                ratings_path=Config.RATINGS_PATH,
                sample_frac=0.1,
                force_rebuild=False
            )
            print("Recommender initialized successfully.")
        except Exception as e:
            print(f"Error during recommender initialization: {e}")

    return app

from app import models