from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import sqlite3
import os
import numpy as np
import cv2
from datetime import datetime
import tensorflow as tf
from PIL import Image
import io
import base64
import pydicom
import json

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Custom objects for model loading
def dice_coefficient(y_true, y_pred, smooth=1):
    """Dice coefficient for segmentation"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Dice loss for segmentation"""
    return 1 - dice_coefficient(y_true, y_pred)

def combined_segmentation_loss(y_true, y_pred):
    """Combined loss for segmentation"""
    return 0.5 * tf.keras.losses.categorical_crossentropy(y_true, y_pred) + 0.5 * dice_loss(y_true, y_pred)

# Load the trained model
MODEL_PATH = 'prostate_multi_output_model.h5'  # Update with your model path
try:
    custom_objects = {
        'dice_coefficient': dice_coefficient,
        'combined_segmentation_loss': combined_segmentation_loss,
        'dice_loss': dice_loss
    }
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    print("Multi-output model loaded successfully!")
    print(f"Model inputs: {model.input_shape}")
    print(f"Model outputs: {[output.shape for output in model.outputs]}")
except Exception as e:
    model = None
    print(f"Warning: Model not found. Error: {e}")

# Database setup - Updated to include new prediction fields
def init_db():
    conn = sqlite3.connect('medical_app.db')
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Updated predictions table for multi-output model
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            filename TEXT NOT NULL,
            cancer_prediction TEXT NOT NULL,
            cancer_confidence REAL NOT NULL,
            severity_prediction TEXT NOT NULL,
            severity_confidence REAL NOT NULL,
            segmentation_available BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Add new columns to existing table if they don't exist
    try:
        cursor.execute('ALTER TABLE predictions ADD COLUMN severity_prediction TEXT')
        cursor.execute('ALTER TABLE predictions ADD COLUMN severity_confidence REAL')
        cursor.execute('ALTER TABLE predictions ADD COLUMN segmentation_available BOOLEAN DEFAULT TRUE')
        cursor.execute('ALTER TABLE predictions RENAME COLUMN prediction TO cancer_prediction')
        cursor.execute('ALTER TABLE predictions RENAME COLUMN confidence TO cancer_confidence')
    except sqlite3.OperationalError:
        # Columns already exist or other error - that's fine
        pass
    
    conn.commit()
    conn.close()

# Initialize database
init_db()

def get_db_connection():
    conn = sqlite3.connect('medical_app.db')
    conn.row_factory = sqlite3.Row
    return conn

def preprocess_image(image_path):
    """Preprocess image for multi-output model prediction - handles both DICOM and regular images"""
    try:
        file_extension = os.path.splitext(image_path)[1].lower()
        
        if file_extension == '.dcm':
            # Handle DICOM files
            print(f"Processing DICOM file: {image_path}")
            ds = pydicom.dcmread(image_path)
            
            if 'PixelData' not in ds:
                print("No pixel data in DICOM file")
                return None
            
            # Extract pixel array
            img = ds.pixel_array
            print(f"Original DICOM shape: {img.shape}")
            
            # Handle different DICOM formats
            if len(img.shape) == 3:
                # Multi-slice volume - take middle slice
                middle_slice = img.shape[0] // 2
                img = img[middle_slice]
                print(f"Taking middle slice {middle_slice}, new shape: {img.shape}")
            
            # Normalize to 0-255 range
            img_min = img.min()
            img_max = img.max()
            if img_max > img_min:
                img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                img = np.zeros_like(img, dtype=np.uint8)
            
            print(f"After normalization - min: {img.min()}, max: {img.max()}")
            
        else:
            # Handle regular image files (JPG, PNG, etc.)
            print(f"Processing regular image file: {image_path}")
            img = cv2.imread(image_path)
            if img is None:
                # Try with PIL if OpenCV fails
                pil_img = Image.open(image_path)
                img = np.array(pil_img)
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            if img is None:
                print("Failed to load image with both OpenCV and PIL")
                return None
        
        # Ensure we have a 2D grayscale image
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        print(f"Grayscale image shape: {img.shape}")
        
        # Resize to model input size (128, 128)
        img = cv2.resize(img, (128, 128))
        print(f"Resized image shape: {img.shape}")
        
        # Keep as grayscale (single channel) - this matches your training preprocessing
        # Normalize to [0, 1] - this matches your training preprocessing
        img = img.astype('float32') / 255.0
        
        # Add channel dimension for grayscale
        img = np.expand_dims(img, axis=-1)
        print(f"Grayscale with channel image shape: {img.shape}")
        
        # Add batch dimension
        img_batch = np.expand_dims(img, axis=0)
        print(f"Final batch shape: {img_batch.shape}")
        
        return img_batch
        
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def predict_image(image_path):
    """Make prediction on uploaded image using multi-output model"""
    if model is None:
        return {
            'cancer_prediction': "Model not available",
            'cancer_confidence': 0.0,
            'severity_prediction': "Model not available", 
            'severity_confidence': 0.0,
            'error': True
        }
    
    print(f"Starting prediction for: {image_path}")
    
    processed_img = preprocess_image(image_path)
    if processed_img is None:
        return {
            'cancer_prediction': "Error processing image",
            'cancer_confidence': 0.0,
            'severity_prediction': "Error processing image",
            'severity_confidence': 0.0,
            'error': True
        }
    
    try:
        print("Making prediction...")
        # Model returns [segmentation, cancer_detection, severity_classification]
        predictions = model.predict(processed_img, verbose=0)
        segmentation_pred, cancer_pred, severity_pred = predictions
        
        print(f"Raw cancer prediction: {cancer_pred}")
        print(f"Raw severity prediction: {severity_pred}")
        
        # Cancer detection
        cancer_class = np.argmax(cancer_pred[0])
        cancer_confidence = float(np.max(cancer_pred[0]))
        cancer_names = ['No Cancer', 'Cancer Detected']
        cancer_result = cancer_names[cancer_class]
        
        # Severity classification
        severity_class = np.argmax(severity_pred[0])
        severity_confidence = float(np.max(severity_pred[0]))
        severity_names = ['None', 'Low Risk', 'Medium Risk', 'High Risk']
        severity_result = severity_names[severity_class]
        
        print(f"Cancer: {cancer_result} (confidence: {cancer_confidence:.3f})")
        print(f"Severity: {severity_result} (confidence: {severity_confidence:.3f})")
        
        # Convert segmentation to image for visualization (optional)
        seg_class = np.argmax(segmentation_pred[0], axis=-1)
        
        return {
            'cancer_prediction': cancer_result,
            'cancer_confidence': cancer_confidence,
            'severity_prediction': severity_result,
            'severity_confidence': severity_confidence,
            'segmentation': seg_class,  # For potential future use
            'error': False
        }
        
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'cancer_prediction': "Prediction error",
            'cancer_confidence': 0.0,
            'severity_prediction': "Prediction error",
            'severity_confidence': 0.0,
            'error': True
        }

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        user = conn.execute(
            'SELECT * FROM users WHERE username = ?', (username,)
        ).fetchone()
        conn.close()
        
        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password!', 'error')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        conn = get_db_connection()
        
        # Check if user already exists
        existing_user = conn.execute(
            'SELECT id FROM users WHERE username = ? OR email = ?', 
            (username, email)
        ).fetchone()
        
        if existing_user:
            flash('Username or email already exists!', 'error')
        else:
            # Create new user
            password_hash = generate_password_hash(password)
            conn.execute(
                'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                (username, email, password_hash)
            )
            conn.commit()
            flash('Account created successfully! Please login.', 'success')
            conn.close()
            return redirect(url_for('login'))
        
        conn.close()
    
    return render_template('signup.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get user's recent predictions
    conn = get_db_connection()
    predictions = conn.execute('''
        SELECT * FROM predictions 
        WHERE user_id = ? 
        ORDER BY created_at DESC 
        LIMIT 10
    ''', (session['user_id'],)).fetchall()
    conn.close()
    
    return render_template('dashboard.html', predictions=predictions)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected!', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected!', 'error')
            return redirect(request.url)
        
        if file:
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Keep original extension for DICOM files
            original_ext = os.path.splitext(filename)[1]
            filename_without_ext = os.path.splitext(filename)[0]
            filename = f"{timestamp}_{filename_without_ext}{original_ext}"
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            print(f"File saved to: {filepath}")
            print(f"File exists: {os.path.exists(filepath)}")
            print(f"File size: {os.path.getsize(filepath)} bytes")
            
            # Make prediction using multi-output model
            result = predict_image(filepath)
            
            # Format the current date and time
            analysis_date = datetime.now().strftime('%B %d, %Y at %I:%M %p')
            
            if not result['error']:
                # Save to database with new schema
                conn = get_db_connection()
                conn.execute('''
                    INSERT INTO predictions 
                    (user_id, filename, cancer_prediction, cancer_confidence, 
                     severity_prediction, severity_confidence, segmentation_available) 
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    session['user_id'], 
                    filename, 
                    result['cancer_prediction'], 
                    result['cancer_confidence'],
                    result['severity_prediction'],
                    result['severity_confidence'],
                    True
                ))
                conn.commit()
                conn.close()
                
                return render_template('result.html', 
                                     filename=filename, 
                                     cancer_prediction=result['cancer_prediction'],
                                     cancer_confidence=result['cancer_confidence'],
                                     severity_prediction=result['severity_prediction'],
                                     severity_confidence=result['severity_confidence'],
                                     analysis_date=analysis_date)
            else:
                flash('Error processing image. Please try again.', 'error')
                return redirect(request.url)
    
    return render_template('predict.html')

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    predictions = conn.execute('''
        SELECT * FROM predictions 
        WHERE user_id = ? 
        ORDER BY created_at DESC
    ''', (session['user_id'],)).fetchall()
    conn.close()
    
    return render_template('history.html', predictions=predictions)

@app.route('/download_results')
def download_results():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    predictions = conn.execute('''
        SELECT * FROM predictions 
        WHERE user_id = ? 
        ORDER BY created_at DESC
    ''', (session['user_id'],)).fetchall()
    conn.close()
    
    # Create CSV content with new fields
    csv_content = "Date,Filename,Cancer_Prediction,Cancer_Confidence,Severity_Prediction,Severity_Confidence\n"
    for pred in predictions:
        # Handle both old and new database schema
        cancer_pred = getattr(pred, 'cancer_prediction', getattr(pred, 'prediction', 'N/A'))
        cancer_conf = getattr(pred, 'cancer_confidence', getattr(pred, 'confidence', 0.0))
        severity_pred = getattr(pred, 'severity_prediction', 'N/A')
        severity_conf = getattr(pred, 'severity_confidence', 0.0)
        
        csv_content += f"{pred['created_at']},{pred['filename']},{cancer_pred},{cancer_conf:.4f},{severity_pred},{severity_conf:.4f}\n"
    
    # Create file-like object
    output = io.StringIO()
    output.write(csv_content)
    output.seek(0)
    
    # Convert to bytes
    csv_bytes = io.BytesIO()
    csv_bytes.write(output.getvalue().encode('utf-8'))
    csv_bytes.seek(0)
    
    return send_file(
        csv_bytes,
        as_attachment=True,
        download_name=f'prostate_predictions_{session["username"]}.csv',
        mimetype='text/csv'
    )

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)