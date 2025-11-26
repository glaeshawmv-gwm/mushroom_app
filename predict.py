import os
import numpy as np
import cv2
import joblib
from flask import Flask, request, jsonify
from tensorflow.keras.applications.efficientnet import preprocess_input
from train import train_pipeline  # Make sure train_pipeline exists

app = Flask(__name__)

# -------------------------------
# Training endpoint
@app.route("/train", methods=["POST"])
def train_endpoint():
    try:
        train_pipeline()
        return jsonify({"status": "Training completed successfully!"})
    except Exception as e:
        return jsonify({"status": "Error", "message": str(e)}), 500

# -------------------------------
# Prediction endpoint
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

    feat = extract_features(img)

    dist = np.sqrt(np.dot(np.dot((feat - mean_vec), inv_cov), (feat - mean_vec).T))
    if dist > THRESHOLD:
        return jsonify({"prediction": "not_mushroom", "confidence": 1.0})

    probs = cal_rf.predict_proba([feat])[0]
    idx = np.argmax(probs)

    return jsonify({"prediction": CLASSES[idx], "confidence": float(probs[idx])})

@app.get("/")
def home():
    return "Mushroom ML API is running!"

# -------------------------------
# Start gunicorn programmatically
if __name__ == "__main__":
    from gunicorn.app.base import BaseApplication

    PORT = int(os.environ.get("PORT", 8080))  # Railway sets this automatically

    class StandaloneApplication(BaseApplication):
        def __init__(self, app, options=None):
            self.options = options or {}
            self.application = app
            super().__init__()

        def load_config(self):
            config = {key: value for key, value in self.options.items()
                      if key in self.cfg.settings and value is not None}
            for key, value in config.items():
                self.cfg.set(key.lower(), value)

        def load(self):
            return self.application

    options = {
        "bind": f"0.0.0.0:{PORT}",
        "workers": 1,
    }

    StandaloneApplication(app, options).run()
