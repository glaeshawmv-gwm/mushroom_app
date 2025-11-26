# predict.py
# -------------------------------
# Mushroom ML: Training + Prediction API
# -------------------------------

import os
import numpy as np
import cv2
import joblib
from flask import Flask, request, jsonify
from tensorflow.keras.applications.efficientnet import preprocess_input
from train import train_pipeline  # Make sure train_pipeline is imported

# -------------------------------
# Flask App
# -------------------------------
app = Flask(__name__)

# -------------------------------
# Training Endpoint
# -------------------------------
@app.route("/train", methods=["POST"])
def train_endpoint():
    try:
        train_pipeline()
        return jsonify({"status": "Training completed successfully!"})
    except Exception as e:
        return jsonify({"status": "Error", "message": str(e)}), 500

# -------------------------------
# Prediction Endpoint
# -------------------------------
# Load models (ensure you have already trained and created the pkls)
FEATURE_PATH = "models/feature_extractor.pkl"
RF_PATH = "models/cal_rf.pkl"
MAHAL_PATH = "models/mahalanobis.pkl"

if os.path.exists(FEATURE_PATH) and os.path.exists(RF_PATH) and os.path.exists(MAHAL_PATH):
    feature_extractor = joblib.load(FEATURE_PATH)
    cal_rf = joblib.load(RF_PATH)
    mahal_data = joblib.load(MAHAL_PATH)

    mean_vec = mahal_data["mean"]
    inv_cov = mahal_data["inv_cov"]
    THRESHOLD = mahal_data["threshold"]

    CLASSES = [
        "contamination_bacterialblotch",
        "contamination_cobweb",
        "contamination_greenmold",
        "healthy_bag",
        "healthy_mushroom"
    ]
else:
    feature_extractor = None
    cal_rf = None
    mean_vec = None
    inv_cov = None
    THRESHOLD = None
    CLASSES = []

def extract_features(img):
    """Resize and extract features using EfficientNet."""
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return feature_extractor.predict(img)[0]

@app.post("/predict")
def predict():
    if feature_extractor is None:
        return jsonify({"error": "Models not found. Please train first."}), 400

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    # Feature extraction
    feat = extract_features(img)

    # Mahalanobis distance for unknown class
    dist = np.sqrt(np.dot(np.dot((feat - mean_vec), inv_cov), (feat - mean_vec).T))
    if dist > THRESHOLD:
        return jsonify({"prediction": "not_mushroom", "confidence": 1.0})

    # Random Forest classifier
    probs = cal_rf.predict_proba([feat])[0]
    idx = np.argmax(probs)

    return jsonify({"prediction": CLASSES[idx], "confidence": float(probs[idx])})

# -------------------------------
# Basic home route
# -------------------------------
@app.get("/")
def home():
    return "Mushroom ML API is running!"

# -------------------------------
# Note: No app.run() needed for Railway
# Railway will run the container with:
# gunicorn -b 0.0.0.0:8080 predict:app
# -------------------------------
