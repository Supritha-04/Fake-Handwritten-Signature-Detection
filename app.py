from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from skimage.transform import resize
import keras
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Setup ---
UPLOAD_FOLDER = 'uploaded_signatures'
MODEL_PATH = 'siamese_signature_model.keras'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ----- Define and register custom function -----
@keras.saving.register_keras_serializable()
def abs_diff(tensors):
    x, y = tensors
    return tf.abs(x - y)

# ----- Load the model using custom_objects -----
model = load_model(MODEL_PATH, custom_objects={'abs_diff': abs_diff})

# ----- Preprocess and verification logic -----
def preprocess_signature(img_path, target_shape=(100, 100)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    img = resize(img, target_shape, mode='constant', anti_aliasing=True)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

def verify_signature(test_path, known_paths, model, threshold=0.5):
    test_img = preprocess_signature(test_path)
    best_sim = -1  # use max similarity
    for idx, known_path in enumerate(known_paths):
        known_img = preprocess_signature(known_path)
        sim = model.predict([test_img, known_img])[0][0]  # sigmoid output [0,1]
        print(f"Comparing: test={test_path} known={known_path} similarity={sim:.4f}")
        if sim > best_sim:
            best_sim = sim
    print(f"Best similarity: {best_sim:.4f}")
    return "Genuine" if best_sim >= threshold else "Forged"

@app.route('/', methods=['GET', 'POST'])
def index():
    users = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('index.html', users=users)

@app.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    files = request.files.getlist('genuine_signatures')
    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], username)
    os.makedirs(user_folder, exist_ok=True)
    for file in files:
        file.save(os.path.join(user_folder, file.filename))
    return redirect(url_for('index'))

@app.route('/verify', methods=['POST'])
def verify():
    username = request.form['username']
    file = request.files['test_signature']
    test_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(test_path)
    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], username)
    known_paths = [os.path.join(user_folder, f) for f in os.listdir(user_folder)]
    result = verify_signature(test_path, known_paths, model, threshold=0.5)  # tune threshold empirically
    if os.path.exists(test_path):
        os.remove(test_path)
    else:
        print(f"File not found for deletion: {test_path}")
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
