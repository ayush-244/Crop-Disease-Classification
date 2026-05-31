# Flask API for image classification
import os
import json
import numpy as np
import io
from flask import Flask, request, jsonify, send_from_directory
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
    FRONTEND_FOLDER = os.path.join(PROJECT_ROOT, "frontend")
    
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    IMG_SIZE = 224
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()

# ========================
# FLASK APP SETUP
# ========================
app = Flask(__name__, static_folder=config.FRONTEND_FOLDER, static_url_path='')
CORS(app)
os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)

# ========================
# DISEASE INFO DATABASE
# ========================
DISEASE_TREATMENTS = {
    "bacterial spot": {
        "severity": "Moderate",
        "treatment": "Use copper-based bactericides such as Kocide. Remove and destroy infected leaves immediately to prevent spreading.",
        "prevention": "Rotate crops every 2-3 years. Avoid overhead watering to keep foliage dry. Use disease-free seeds."
    },
    "early blight": {
        "severity": "Moderate", 
        "treatment": "Apply fungicides containing chlorothalonil or mancozeb. Prune lower leaves to improve air circulation.",
        "prevention": "Maintain consistent moisture and provide adequate spacing between plants. Mulch to prevent soil splash."
    },
    "late blight": {
        "severity": "Severe",
        "treatment": "Apply targeted fungicides like Ridomil Gold. If infection is severe, destroy the entire plant to save others.",
        "prevention": "Avoid planting during very damp/cool seasons. Ensure excellent soil drainage and use resistant varieties."
    },
    "leaf mold": {
        "severity": "Moderate",
        "treatment": "Reduce humidity in greenhouses. Apply fungicides like Daconil if the infection is spreading rapidly.",
        "prevention": "Provide adequate spacing for ventilation. Prune plants to increase air flow through the canopy."
    },
    "septoria leaf spot": {
        "severity": "Moderate",
        "treatment": "Remove infected leaves. Apply organic fungicides or those containing copper or sulfur.",
        "prevention": "Avoid working with plants when they are wet. Practice strict crop rotation and weed control."
    },
    "spider mites": {
        "severity": "Moderate",
        "treatment": "Apply Neem oil or insecticidal soap. Blast the undersides of leaves with a strong stream of water.",
        "prevention": "Keep plants well-hydrated, as spider mites thrive in hot, dry conditions. Encourage natural predators like ladybugs."
    },
    "target spot": {
        "severity": "Moderate",
        "treatment": "Apply fungicides such as chlorothalonil. Remove lower diseased leaves to reduce fungal spore load.",
        "prevention": "Ensure good soil health and drainage. Avoid wetting the leaves during irrigation."
    },
    "mosaic virus": {
        "severity": "Severe",
        "treatment": "No chemical cure exists. Immediately remove and burn infected plants to prevent the virus from spreading.",
        "prevention": "Control aphids and whiteflies which spread the virus. Use virus-resistant seeds and disinfect tools."
    },
    "yellowleaf curl virus": {
        "severity": "Severe",
        "treatment": "No cure. Remove infected plants. Focus heavily on controlling the whitefly population that transmits the virus.",
        "prevention": "Use silver plastic mulches to repel whiteflies. Plant resistant varieties and use protective covers."
    },
    "healthy": {
        "severity": "None",
        "treatment": "Continue your current care routine! Your plant looks strong and vibrant.",
        "prevention": "Maintain regular watering and fertilization. Keep an eye out for early signs of pests."
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
try:
    load_model()
except Exception as e:
    print(f"⚠️  Failed to load model on startup: {e}")
    print("App will still start but predictions won't be available until model loads.")

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
    clean_query = disease_name.lower().replace('_', ' ')
    for key, val in DISEASE_TREATMENTS.items():
        if key in clean_query:
            return val
    
    return {
        "severity": "N/A",
        "treatment": "Diagnosis unclear. Please consult with a local agricultural specialist or try a clearer photo.",
        "prevention": "Maintain general plant hygiene and monitor the area for spreading symptoms."
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

@app.route('/', methods=['GET'])
def serve_index():
    """Serve the frontend index.html"""
    return send_from_directory(config.FRONTEND_FOLDER, 'index.html')

@app.route('/<path:path>', methods=['GET'])
def serve_static(path):
    """Serve static files (CSS, JS, images)"""
    # Don't intercept API routes
    if path.startswith('api/'):
        return jsonify({"error": "Not Found"}), 404
    
    try:
        return send_from_directory(config.FRONTEND_FOLDER, path)
    except Exception as e:
        # If static file not found, serve index.html for SPA routing
        try:
            return send_from_directory(config.FRONTEND_FOLDER, 'index.html')
        except:
            return jsonify({"error": f"Not Found: {path}"}), 404

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "framework": "PyTorch",
        "num_classes": len(class_labels) if class_labels else 0
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
    port = int(os.environ.get('PORT', 5000))
    print(f"\n{'='*50}")
    print(f"🚀 Starting Crop Disease Classification API")
    print(f"   Port: {port}")
    print(f"   Host: 0.0.0.0")
    print(f"   Model loaded: {model is not None}")
    print(f"   Model path: {config.MODEL_PATH}")
    print(f"{'='*50}\n")
    app.run(host='0.0.0.0', port=port, debug=False)
