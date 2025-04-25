from app import db
from flask_login import UserMixin
import json

class User(UserMixin, db.Model):
    user_id = db.Column(db.String(50), primary_key=True)
    profile_name = db.Column(db.String(100))
    password_hash = db.Column(db.String(128))

    def get_id(self):
        return self.user_id

class Book(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), unique=True)
    description = db.Column(db.Text)
    authors = db.Column(db.Text)
    image = db.Column(db.String(400))
    preview_link = db.Column(db.String(400))
    publisher = db.Column(db.String(100))
    published_date = db.Column(db.String(20))
    info_link = db.Column(db.String(300))
    categories = db.Column(db.Text)
    ratings_count = db.Column(db.Integer)

    @property
    def authors_list(self):
        return json.loads(self.authors) if self.authors else []
    
    @property
    def categories_list(self):
        return json.loads(self.categories) if self.categories else []

class Rating(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    book_id = db.Column(db.Integer, db.ForeignKey('book.id'))
    user_id = db.Column(db.String(50), db.ForeignKey('user.user_id'))
    review_score = db.Column(db.Float)
    review_text = db.Column(db.Text)
    user = db.relationship('User', backref='ratings')

class UserBookView(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(50), db.ForeignKey('user.user_id'))
    book_id = db.Column(db.Integer, db.ForeignKey('book.id'))
    timestamp = db.Column(db.DateTime, default=db.func.now())