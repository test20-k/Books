from flask import render_template, request, redirect, url_for, flash, jsonify, Blueprint
from flask_login import login_user, logout_user, current_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from app import db, login
from app.models import User, Book, Rating,UserBookView
from app.forms import RegistrationForm, LoginForm, RatingForm, SearchForm
from app.testing import OptimizedHybridRecommender
from sqlalchemy import or_
import os
from config import Config

bp = Blueprint('main', __name__)


from app import recommender

@login.user_loader
def load_user(user_id):
    return User.query.get(user_id)

@bp.route('/')
def index():
    user_id = current_user.get_id() if current_user.is_authenticated else None
    books = []

    if user_id:
        try:
            recommendations = recommender.hybrid_recommend(user_id=user_id, top_n=8)
            if recommendations:
                books = Book.query.filter(Book.title.in_(recommendations)).all()
        except Exception as e:
            print(f"Home recommendations error: {str(e)}")

    if not books:  # fallback to popular books
        books = Book.query.order_by(Book.ratings_count.desc()).limit(8).all()

    return render_template('index.html', popular_books=books)



@bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(
            user_id=form.username.data,
            password_hash=generate_password_hash(form.password.data)
        )
        db.session.add(user)
        db.session.commit()
        flash('Registration successful. Please login.')
        return redirect(url_for('main.login'))
    return render_template('register.html', form=form)

@bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.get(form.username.data)
        if user and user.password_hash==form.password.data:
            login_user(user)
            return redirect(url_for('main.index'))
        flash('Invalid username or password')
    return render_template('login.html', form=form)


@bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('main.index'))


@bp.route('/books')
def list_books():
    page = request.args.get('page', 1, type=int)
    search_query = request.args.get('q', '')
    query = Book.query
    if search_query:
        query = query.filter(Book.title.ilike(f'%{search_query}%'))
    books = query.paginate(page=page, per_page=10)
    return render_template('books.html', books=books, search_query=search_query)

@bp.route('/book/<int:book_id>', methods=['GET', 'POST'])
def book_detail(book_id):
    book = Book.query.get_or_404(book_id)
    form = RatingForm()
    existing_rating = None
    similar_books = []
    user_id = current_user.get_id() if current_user.is_authenticated else None

    if current_user.is_authenticated:
        viewed_book = UserBookView.query.filter_by(
            user_id=current_user.user_id,
            book_id=book_id
        ).first()
        if not viewed_book:
            new_view = UserBookView(user_id=current_user.user_id, book_id=book_id)
            db.session.add(new_view)
            db.session.commit()


    ratings = Rating.query.filter_by(book_id=book_id).all()
    if ratings:
        avg_rating = sum(rating.review_score for rating in ratings) / len(ratings)
    else:
        avg_rating = 0

    # Handle rating submission (remains the same)
    if current_user.is_authenticated and form.validate_on_submit():
        try:
            if existing_rating:
                existing_rating.review_score = form.rating.data
                existing_rating.review_text = form.review.data
            else:
                new_rating = Rating(
                    user_id=current_user.get_id(),
                    book_id=book_id,
                    review_score=form.rating.data,
                    review_text=form.review.data
                )
                db.session.add(new_rating)
            db.session.commit()
            flash('Your review has been submitted!', 'success')
            return redirect(url_for('main.book_detail', book_id=book_id))
        except Exception as e:
            db.session.rollback()
            flash('Error saving your review. Please try again.', 'error')

 
    try:
        recommendations = recommender.recommend_based_on_book(book.title, top_n=5)
        if recommendations:
            similar_books_query = Book.query.filter(Book.title.in_(recommendations)).limit(5)
            similar_books = similar_books_query.all()
            for similar_book in similar_books:
                similar_book_ratings = Rating.query.filter_by(book_id=similar_book.id).all()
                if similar_book_ratings:
                    similar_book.avg_rating = sum(r.review_score for r in similar_book_ratings) / len(similar_book_ratings)
                else:
                    similar_book.avg_rating = 0
        else:
            similar_books = Book.query.order_by(Book.ratings_count.desc()).limit(5).all()
            for similar_book in similar_books:
                similar_book.avg_rating = 0
    except Exception as e:
        print(f"Recommendation error (book detail): {str(e)}")
        flash('Could not load similar book recommendations.', 'error')
        similar_books = Book.query.order_by(Book.ratings_count.desc()).limit(5).all()
        for similar_book in similar_books:
            similar_book.avg_rating = 0

    # Paginate reviews (remains the same)
    reviews_page = request.args.get('page', 1, type=int)
    reviews = Rating.query.filter_by(book_id=book_id)\
        .paginate(page=reviews_page, per_page=5)

    return render_template('book_detail.html',
                           book=book,
                           form=form,
                           existing_rating=existing_rating,
                           reviews=reviews,
                           similar_books=similar_books,
                           avg_rating=avg_rating)



@bp.route('/recommendations')
@login_required
def recommendations():
    user_id = current_user.get_id()
    books = []

    try:
        recommendations = recommender.hybrid_recommend(user_id=user_id, top_n=10)
        if recommendations:
            books = Book.query.filter(Book.title.in_(recommendations)).all()
    except Exception as e:
        print(f"User recommendations error: {str(e)}")

    if not books: 
        flash('Error generating recommendations. Showing popular books instead.', 'warning')
        books = Book.query.order_by(Book.ratings_count.desc()).limit(10).all()

    return render_template('recommendations.html', books=books)

@bp.route('/autocomplete')
def autocomplete():
    search = request.args.get('q')
    results = Book.query.filter(Book.title.ilike(f'%{search}%')).limit(10).all()
    return jsonify([book.title for book in results])

@bp.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@bp.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500



@bp.route('/reading_history')
@login_required
def reading_history():
    page = request.args.get('page', 1, type=int)
    reading_history = UserBookView.query.filter_by(user_id=current_user.user_id)\
        .order_by(UserBookView.timestamp.desc())\
        .paginate(page=page, per_page=10)  
    books = [Book.query.get(item.book_id) for item in reading_history.items]

    return render_template('reading_history.html', reading_history=reading_history, books=books)

@bp.route('/rated_books')
@login_required
def rated_books():
    page = request.args.get('page', 1, type=int)
    rated_books = Rating.query.filter_by(user_id=current_user.user_id)\
        .order_by(Rating.id.desc())\
        .paginate(page=page, per_page=10) 
    rated_book_details = []
    for rating in rated_books.items:
        book = Book.query.get(rating.book_id)
        if book:
            rated_book_details.append({'book': book, 'rating': rating})

    return render_template('rated_books.html', rated_books=rated_books, rated_book_details=rated_book_details)
