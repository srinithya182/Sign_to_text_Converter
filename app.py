import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import keras
import tensorflow as tf
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from pro.env
load_dotenv(dotenv_path='pro.env')  # Make sure to load from your pro.env file

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')  # Loaded from pro.env
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SQLALCHEMY_DATABASE_URI')  # Loaded from pro.env
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER')  # Loaded from pro.env
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure upload directory exists
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # or any folder name you want
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize database
db = SQLAlchemy(app)

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Load the model
def load_model_with_custom_objects(filepath):
    try:
        return tf.keras.models.load_model(filepath)
    except TypeError as e:
        if "batch_shape" in str(e):
            # Load the model file as an h5 file first
            with keras.utils.custom_object_scope({
                'InputLayer': lambda config: keras.layers.InputLayer(
                    shape=config.get('batch_shape')[1:] if config.get('batch_shape') else None,
                    dtype=config.get('dtype'),
                    sparse=config.get('sparse', False),
                    ragged=config.get('ragged', False),
                    name=config.get('name')
                )
            }):
                return tf.keras.models.load_model(filepath)
        else:
            raise e

model = tf.keras.models.load_model('/Users/nithyareddy/sign-language-recognition/models/isl_rnn_model.keras')

# Class labels (ensure these match your model's classes)
class_labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
    'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4', 
    '5', '6', '7', '8', '9'  # Adjust based on your dataset
]

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    date_joined = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='user', lazy=True)
    settings = db.relationship('UserSettings', backref='user', uselist=False)
    usage_statistics = db.relationship('UsageStatistics', backref='user', uselist=False)

# Prediction history model
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    predicted_class = db.Column(db.String(10), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# Translations model
class Translation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_path = db.Column(db.String(100), nullable=False)
    prediction = db.Column(db.String(10), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    type = db.Column(db.String(10), nullable=False)  # 'image', 'video', or 'webcam'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Favorites model
class Favorite(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    translation_id = db.Column(db.Integer, db.ForeignKey('translation.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref='favorites')
    translation = db.relationship('Translation', backref='favorites')

# Shared Translations model
class SharedTranslation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    translation_id = db.Column(db.Integer, db.ForeignKey('translation.id'), nullable=False)
    shared_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    shared_with = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)  # NULL if public
    public_token = db.Column(db.String(100), nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    expires_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# User Settings model
class UserSettings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    theme = db.Column(db.String(20), default='light')
    notifications_enabled = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)

# Usage Statistics model
class UsageStatistics(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    translations_count = db.Column(db.Integer, default=0)
    last_translation_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)

# Helper function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# User loader callback for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        else:
            flash('Login failed. Please check your username and password.')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match!')
            return redirect(url_for('signup'))
        
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists!')
            return redirect(url_for('signup'))
            
        existing_email = User.query.filter_by(email=email).first()
        if existing_email:
            flash('Email already registered!')
            return redirect(url_for('signup'))
        
        new_user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created successfully! Please log in.')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).limit(5).all()
    return render_template('dashboard.html', predictions=predictions)

@app.route('/history')
@login_required
def view_history():
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).all()
    return render_template('history.html', predictions=predictions)


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        img = Image.open(file_path).resize((64, 64))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        y_pred_probs = model.predict(img_array)
        y_pred = np.argmax(y_pred_probs, axis=1)
        predicted_class = class_labels[y_pred[0]]
        confidence = float(y_pred_probs[0][y_pred[0]])

        prediction = Prediction(
            filename=filename,
            predicted_class=predicted_class,
            confidence=confidence,
            user_id=current_user.id
        )
        db.session.add(prediction)
        db.session.commit()

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence,
            'file_path': f'/static/uploads/{filename}'
        })

    return jsonify({'error': 'File type not allowed'})


@app.route('/predict_image', methods=['POST'])
@login_required
def predict_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        # Preprocess and predict
        img = Image.open(save_path).convert('RGB')
        img = img.resize((64, 64))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape((1, 64, 64, 3))
        predictions = model.predict(img_array)
        confidence = np.max(predictions)
        predicted_class = class_labels[np.argmax(predictions)]

        # Save to DB
        new_pred = Prediction(
            filename=filename,
            predicted_class=predicted_class,
            confidence=float(confidence),
            user_id=current_user.id
        )
        db.session.add(new_pred)
        db.session.commit()

        return jsonify({
            'prediction': predicted_class,
            'confidence': round(confidence * 100, 2),
            'image_url': url_for('static', filename='uploads/' + filename)
        })

    return jsonify({'error': 'Invalid file type'}), 400


# Ensure the tables are created when the app starts
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)
