# Flask API for image classification
import os
import json
import numpy as np
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

# ========================
# CONFIGURATION
# ========================
class Config:
    # Get absolute path of the backend directory
    BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
    # Project root is one level up
    PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
    
    MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "crop_disease_model.pth")
    LABELS_PATH = os.path.join(PROJECT_ROOT, "models", "class_labels.json")
    UPLOAD_FOLDER = os.path.join(BACKEND_DIR, "uploads")
    
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    IMG_SIZE = 224
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()

# ========================
# FLASK APP SETUP
# ========================
app = Flask(__name__)
CORS(app)
os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)

# ========================
# DISEASE INFO DATABASE
# ========================
DISEASE_TREATMENTS = {
    # (Same dictionary as before - shortened for brevity in this rewrite, 
    # ensuring all keys match PlantVillage classes)
    "Bacterial_spot": {
        "severity": "Moderate",
        "treatment": "Apply copper-based bactericides. Remove infected leaves.",
        "prevention": "Use disease-free seeds, practice crop rotation."
    },
    "Early_blight": {
        "severity": "Moderate", 
        "treatment": "Apply fungicides containing chlorothalonil. Remove debris.",
        "prevention": "Mulch around plants, water at base."
    },
    "Late_blight": {
        "severity": "Severe",
        "treatment": "Apply fungicides immediately. Destroy infected plants.",
        "prevention": "Plant resistant varieties, ensure drainage."
    },
    "Leaf_Mold": {
        "severity": "Moderate",
        "treatment": "Improve air circulation. Apply fungicides if severe.",
        "prevention": "Space plants properly, ensure ventilation."
    },
    "Septoria_leaf_spot": {
        "severity": "Moderate",
        "treatment": "Remove infected leaves, apply organic fungicides.",
        "prevention": "Rotate crops, avoid overhead watering."
    },
    "Spider_mites": {
        "severity": "Moderate",
        "treatment": "Spray with insecticidal soap or neem oil.",
        "prevention": "Avoid water stress, encourage predators."
    },
    "Target_Spot": {
        "severity": "Moderate",
        "treatment": "Apply fungicides, remove infected leaves.",
        "prevention": "Practice crop rotation, mulch soil."
    },
    "mosaic_virus": {
        "severity": "Severe",
        "treatment": "No cure. Remove infected plants immediately.",
        "prevention": "Control aphids, use virus-free seeds."
    },
    "YellowLeaf_Curl_Virus": {
        "severity": "Severe",
        "treatment": "Remove infected plants. Control whiteflies.",
        "prevention": "Use resistant varieties, install insect screens."
    },
    "healthy": {
        "severity": "None",
        "treatment": "No treatment needed.",
        "prevention": "Maintain good cultural practices."
    }
}

# ========================
# MODEL LOADING
# ========================
model = None
class_labels = {}

def load_model():
    global model, class_labels
    
    # Load labels
    try:
        with open(config.LABELS_PATH, 'r') as f:
            class_labels = json.load(f)
            # Ensure keys are integers
            class_labels = {int(k): v for k, v in class_labels.items()}
            print(f"✅ Labels loaded: {len(class_labels)} classes")
    except Exception as e:
        print(f"❌ Error loading labels: {e}")
        return

    # Load model architecture
    try:
        print("🏗️  Building model architecture...")
        model = models.mobilenet_v2(weights=None)
        
        # Recreate the classifier head same as training
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, len(class_labels))
        )
        
        # Load weights
        print(f"🔄 Loading weights from {config.MODEL_PATH}...")
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
        model.to(config.DEVICE)
        model.eval()
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None

# Initialize model
load_model()

# ========================
# PREPROCESSING
# ========================
prediction_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

def get_treatment_info(disease_name):
    for key, val in DISEASE_TREATMENTS.items():
        if key.lower() in disease_name.lower():
            return val
    return {
        "severity": "Unknown",
        "treatment": "Consult an expert.",
        "prevention": "Practice good hygiene."
    }

def parse_disease_name(class_name):
    # Handle the anomaly class "PlantVillage" created by nested folders
    if class_name.lower() == "plantvillage":
        return "Unknown", "Unidentified (Dataset issue)", False

    # Standardize delimiters
    clean_name = class_name.replace('___', ' ').replace('__', ' ').replace('_', ' ')
    parts = clean_name.split()
    
    # Extract crop (usually the first word)
    crop = parts[0]
    
    # Extract disease (rest of the words)
    if len(parts) > 1:
        disease = ' '.join(parts[1:])
    else:
        disease = "Healthy" if "healthy" in class_name.lower() else "Unknown"
        
    is_healthy = 'healthy' in class_name.lower()
    return crop, disease, is_healthy

# ========================
# API ENDPOINTS
# ========================

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "framework": "PyTorch"
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 500
    
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file"}), 400
        
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Invalid file"}), 400

    try:
        # Process image directly from memory
        img = Image.open(file.stream).convert('RGB')
        img_tensor = prediction_transform(img).unsqueeze(0).to(config.DEVICE)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
        # Get top 3
        top_probs, top_idxs = torch.topk(probs, 3)
        top_probs = top_probs.cpu().numpy()[0]
        top_idxs = top_idxs.cpu().numpy()[0]
        
        predictions = []
        for i in range(len(top_idxs)):
            class_name = class_labels[top_idxs[i]]
            confidence = float(top_probs[i]) * 100
            crop, disease, is_healthy = parse_disease_name(class_name)
            
            predictions.append({
                "class_name": class_name,
                "crop": crop,
                "disease": disease,
                "confidence": round(confidence, 2),
                "is_healthy": is_healthy
            })
            
        primary = predictions[0]
        treatment = get_treatment_info(primary['disease'])
        
        return jsonify({
            "success": True,
            "prediction": {
                **primary,
                **treatment
            },
            "top_3_predictions": predictions
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
