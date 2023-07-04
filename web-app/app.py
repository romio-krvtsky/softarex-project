import json
import pandas as pd
from flask_migrate import Migrate
from flask import Flask, render_template, request, redirect, url_for, flash, current_app
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import logging
from logging.handlers import RotatingFileHandler
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.secret_key = 'd436fda43b13884c260745681a796405'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:softarex@localhost/softarex_project'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Конфигурация логгера
log_file = 'app.log'
log_format = '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
if not os.path.exists('logs'):
    os.makedirs('logs')
handler = RotatingFileHandler('logs/' + log_file, maxBytes=10240, backupCount=10)
handler.setFormatter(logging.Formatter(log_format))
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)


def analyze_data(data):
    text = data

    current_directory = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(current_directory, 'model.h5')
    data_path = os.path.join(current_directory, 'balanced_data.csv')

    model = load_model(model_path)
    train_data = pd.read_csv(data_path)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_data["comment_text"])

    text_tokens = tokenizer.texts_to_sequences([text])

    max_sequence_length = 100

    text_tokens_padded = pad_sequences(text_tokens, maxlen=max_sequence_length, padding="post", truncating="post")

    prediction = model.predict(text_tokens_padded)
    prediction = (prediction > 0.5).astype(int)

    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    result = 'neutral'

    if np.any(prediction):
        result = ', '.join([label for prediction, label in zip(prediction[0], labels) if prediction == 1])

    print(result)

    return result


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    computation_count = db.Column(db.Integer, default=0)
    computations = db.relationship('Computation', backref='user', lazy=True)

    def get_id(self):
        return str(self.id)

    def __repr__(self):
        return f'<User {self.username}>'


class Computation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    text = db.Column(db.Text, nullable=False)
    result = db.Column(db.String(255), nullable=False)

    def __repr__(self):
        return f'<Computation {self.id}>'


login_manager = LoginManager(app)


@login_manager.user_loader
def load_user(user_id):
    with current_app.app_context():
        return db.session.get(User, int(user_id))


@app.route('/', methods=["GET", "POST"])
def home():
    if current_user.is_authenticated:
        username = current_user.username
        if request.method == "POST":
            data = request.form.get("text")
            prediction = analyze_data(data)

            user = User.query.get(current_user.id)
            computation = Computation(user_id=user.id, text=data, result=prediction)
            user.computation_count += 1
            db.session.add(computation)
            db.session.commit()

            result = {
                'data': data,
                'prediction': prediction
            }

            result_json = json.dumps(result)
            return render_template("results.html", data=data, prediction=prediction, result=result_json)
        return render_template('index.html', username=username)
    else:
        return render_template('home.html')


@app.route('/results', methods=["GET", "POST"])
def results():
    return render_template('results.html')


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        new_username = request.form.get('username')
        new_password = request.form.get('password')

        if new_username:
            current_user.username = new_username
        if new_password:
            current_user.password = new_password
        db.session.commit()

        return redirect(url_for('profile'))

    user = User.query.get(current_user.id)
    computations = Computation.query.filter_by(user_id=user.id).all()

    return render_template('profile.html', user=current_user, computations=computations)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirmed_password = request.form['confirmedPassword']

        if password != confirmed_password:
            error_message = "Passwords do not match."
            return render_template('register.html', error_message=error_message)

        user = User.query.filter_by(username=username).first()
        if user:
            flash('Username already exists.')
            return redirect(url_for('register'))

        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            login_user(user)
            app.logger.info(f'User {user.username} logged in.')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.')
            return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    app.logger.info(f'User {current_user.username} logged out.')
    logout_user()
    return redirect(url_for('home'))


if __name__ == '__main__':
    # db.create_all()
    app.run(debug=True)
