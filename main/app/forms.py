from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, TextAreaField, FloatField, SubmitField
from wtforms.validators import DataRequired, Length, NumberRange, ValidationError
from app.models import User

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[
        DataRequired(), 
        Length(min=4, max=50)
    ])
    password = PasswordField('Password', validators=[
        DataRequired(), 
        Length(min=6)
    ])
    submit = SubmitField('Register')

    def validate_username(self, username):
        user = User.query.filter_by(user_id=username.data).first()
        if user is not None:
            raise ValidationError('Please use a different username.')

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class RatingForm(FlaskForm):
    rating = FloatField('Rating', validators=[
        DataRequired(),
        NumberRange(min=0, max=5, message='Rating must be between 0 and 5')
    ])
    review = TextAreaField('Review')
    submit = SubmitField('Submit Review')

class SearchForm(FlaskForm):
    query = StringField('Search Books')
    submit = SubmitField('Search')