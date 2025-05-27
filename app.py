import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Set up logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

# Global variables for the model
model = None
label_encoder = None
feature_names = None

def load_artifacts():
    """Load the trained model and encoder from disk."""
    global model, label_encoder, feature_names
    try:
        logging.info("Loading model and encoder...")
        model = joblib.load("disease_predictor_model.pkl")
        label_encoder = joblib.load("label_encoder.pkl")

        if hasattr(model, "get_booster"):
            feature_names = model.get_booster().feature_names
        elif hasattr(model, "feature_names_in_"):
            feature_names = model.feature_names_in_
        else:
            raise AttributeError("Model lacks feature name information.")

        logging.info(f"Model and encoder loaded successfully with {len(feature_names)} features.")
    except Exception as e:
        logging.critical(f"Error loading model or encoder: {e}")
        raise RuntimeError("Failed to load model or encoder.") from e

@app.route("/")
def home():
    """Render the main page."""
    # Convert feature_names to list if it's a numpy array or pandas index
    features_list = feature_names.tolist() if hasattr(feature_names, 'tolist') else list(feature_names) if feature_names else []
    return render_template('index.html', 
                         features=features_list,
                         model_loaded=bool(model))

@app.route("/health", methods=["GET"])
def health_check():
    """Health check to confirm the API and model are working."""
    return jsonify({
        "status": "healthy" if model else "unhealthy",
        "model_loaded": bool(model),
        "features_loaded": len(feature_names) if feature_names else 0,
        "disease_classes": len(label_encoder.classes_) if label_encoder else 0,
        "timestamp": datetime.now().isoformat(),
        "version": "1.0"
    }), 200 if model else 503

@app.route("/predict", methods=["POST"])
def predict():
    """Predict diseases based on user symptoms."""
    try:
        if not request.is_json:
            logging.warning("Request is not JSON")
            return jsonify({"error": "Request must be JSON"}), 400

        input_data = request.get_json()
        if not input_data or not isinstance(input_data, dict):
            logging.warning("Invalid JSON format")
            return jsonify({"error": "Invalid input format"}), 400

        features = pd.DataFrame(0, index=[0], columns=feature_names)
        for symptom, val in input_data.items():
            if symptom in features.columns:
                try:
                    features[symptom] = 1 if bool(val) else 0
                except Exception:
                    logging.warning(f"Invalid value for symptom: {symptom} = {val}")
            else:
                logging.warning(f"Ignoring unknown symptom: {symptom}")

        valid_count = int(features.sum().sum())
        if valid_count == 0:
            return jsonify({
                "error": "No valid symptoms detected",
                "available_features": feature_names.tolist()
            }), 400

        probabilities = model.predict_proba(features)[0]
        top_indices = np.argsort(probabilities)[-3:][::-1]

        results = []
        for idx in top_indices:
            prob = float(probabilities[idx])
            disease = label_encoder.classes_[idx]
            confidence = "high" if prob > 0.7 else "medium" if prob > 0.4 else "low"

            results.append({
                "disease": disease,
                "probability": round(prob, 4),
                "confidence": confidence
            })

        return jsonify({
            "predictions": results,
            "timestamp": datetime.now().isoformat(),
            "symptoms_used": valid_count,
            "total_symptoms_available": len(feature_names)
        })

    except Exception as e:
        logging.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed",
            "details": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": ["/health (GET)", "/predict (POST)"],
        "timestamp": datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({
        "error": "Internal server error",
        "timestamp": datetime.now().isoformat()
    }), 500

if __name__ == "__main__":
    try:
        load_artifacts()
        logging.info("Starting Flask app on http://localhost:5000 ...")
        app.run(host="0.0.0.0", port=5000, debug=True)
    except Exception as e:
        logging.critical(f"App failed to start: {e}")
        print(f"Fatal error during startup: {e}")