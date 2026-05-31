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
    
    import traceback
    
    # Log initialization
    print("\n" + "="*60)
    print("🔍 MODEL LOADING STARTED")
    print("="*60)
    print(f"[INIT] Model path: {config.MODEL_PATH}")
    print(f"[INIT] Labels path: {config.LABELS_PATH}")
    print(f"[INIT] Device: {config.DEVICE}")
    print(f"[INIT] Model path exists: {os.path.exists(config.MODEL_PATH)}")
    print(f"[INIT] Labels path exists: {os.path.exists(config.LABELS_PATH)}")
    
    # Load labels
    try:
        print("[LABELS] Loading class labels...")
        if not os.path.exists(config.LABELS_PATH):
            raise FileNotFoundError(f"Labels file not found: {config.LABELS_PATH}")
        
        with open(config.LABELS_PATH, 'r') as f:
            class_labels = json.load(f)
            # Ensure keys are integers
            class_labels = {int(k): v for k, v in class_labels.items()}
            print(f"✅ [LABELS] Loaded {len(class_labels)} classes: {list(class_labels.values())[:3]}...")
    except FileNotFoundError as e:
        print(f"❌ [LABELS] FileNotFoundError: {e}")
        print(f"❌ [LABELS] Traceback: {traceback.format_exc()}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ [LABELS] JSONDecodeError: {e}")
        print(f"❌ [LABELS] Traceback: {traceback.format_exc()}")
        return False
    except Exception as e:
        print(f"❌ [LABELS] Unexpected error: {e}")
        print(f"❌ [LABELS] Traceback: {traceback.format_exc()}")
        return False

    # Load model architecture
    try:
        print("[MODEL] Building MobileNetV2 architecture...")
        model = models.mobilenet_v2(weights=None)
        print("[MODEL] MobileNetV2 architecture created")
        
        # Recreate the classifier head same as training
        print("[MODEL] Rebuilding classifier head...")
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, len(class_labels))
        )
        print(f"[MODEL] Classifier head configured for {len(class_labels)} classes")
        
        # Load weights
        print(f"[WEIGHTS] Loading model weights from: {config.MODEL_PATH}")
        if not os.path.exists(config.MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {config.MODEL_PATH}")
        
        model_file_size = os.path.getsize(config.MODEL_PATH) / (1024*1024)
        print(f"[WEIGHTS] Model file size: {model_file_size:.2f} MB")
        
        print(f"[WEIGHTS] Loading with map_location={config.DEVICE}...")
        state_dict = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
        print(f"[WEIGHTS] State dict keys: {len(state_dict)} parameters")
        
        model.load_state_dict(state_dict)
        print("[WEIGHTS] State dict loaded successfully")
        
        model.to(config.DEVICE)
        print(f"[WEIGHTS] Model moved to device: {config.DEVICE}")
        
        model.eval()
        print("[WEIGHTS] Model set to evaluation mode")
        print("✅ [MODEL] Model loaded successfully!")
        print("="*60 + "\n")
        return True
        
    except FileNotFoundError as e:
        print(f"❌ [MODEL] FileNotFoundError: {e}")
        print(f"❌ [MODEL] Traceback: {traceback.format_exc()}")
        model = None
        return False
    except RuntimeError as e:
        print(f"❌ [MODEL] RuntimeError: {e}")
        print(f"❌ [MODEL] Traceback: {traceback.format_exc()}")
        model = None
        return False
    except Exception as e:
        print(f"❌ [MODEL] Unexpected error: {type(e).__name__}: {e}")
        print(f"❌ [MODEL] Traceback: {traceback.format_exc()}")
        model = None
        return False

# Initialize model
print("\n🔄 INITIALIZING APPLICATION...")
try:
    if load_model():
        print("✅ APPLICATION READY - Model loaded successfully")
    else:
        print("⚠️  APPLICATION STARTED - Model failed to load (predictions unavailable)")
except Exception as e:
    print(f"⚠️  APPLICATION STARTED - Model loading crashed: {e}")
    print(f"   Traceback: {traceback.format_exc()}")
    import traceback

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
# ERROR HANDLERS
# ========================

@app.errorhandler(404)
def not_found(error):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(error):
    print(f"[ERROR] Server error: {error}")
    return jsonify({"success": False, "error": "Internal server error"}), 500

@app.errorhandler(Exception)
def handle_exception(error):
    print(f"[ERROR] Unhandled exception: {error}")
    return jsonify({"success": False, "error": str(error)}), 500

@app.before_request
def before_request():
    print(f"[{request.method}] {request.path}")

# ========================
# API ENDPOINTS
# ========================

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "framework": "PyTorch",
        "num_classes": len(class_labels) if class_labels else 0
    })

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    import traceback
    import sys
    
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return '', 204
    
    print("\n" + "="*60)
    print("[PREDICT] NEW PREDICTION REQUEST")
    print("="*60)
    
    try:
        # ==================== REQUEST VALIDATION ====================
        print("[PREDICT:1] Checking if model is loaded...")
        if model is None:
            print("[ERROR:1] Model is None!")
            return jsonify({
                "success": False, 
                "error": "Model not loaded",
                "debug": "Backend model is None"
            }), 500
        print("[PREDICT:1] ✅ Model loaded")
        
        print("[PREDICT:2] Checking request files...")
        files_in_request = list(request.files.keys())
        print(f"[PREDICT:2] Files in request: {files_in_request}")
        
        if 'file' not in request.files:
            print(f"[ERROR:2] No 'file' key in request. Keys: {files_in_request}")
            return jsonify({
                "success": False,
                "error": "No file provided",
                "debug": f"Expected 'file' key, got: {files_in_request}"
            }), 400
        print("[PREDICT:2] ✅ File key found")
        
        # ==================== FILE VALIDATION ====================
        print("[PREDICT:3] Getting file from request...")
        file = request.files['file']
        print(f"[PREDICT:3] Filename: {file.filename}")
        
        if file.filename == '':
            print("[ERROR:3] Filename is empty")
            return jsonify({
                "success": False,
                "error": "Invalid file: filename is empty"
            }), 400
        print("[PREDICT:3] ✅ Filename valid")
        
        print("[PREDICT:4] Validating file extension...")
        if not allowed_file(file.filename):
            ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'none'
            print(f"[ERROR:4] Invalid file extension: {ext}. Allowed: {config.ALLOWED_EXTENSIONS}")
            return jsonify({
                "success": False,
                "error": f"Invalid file format: {ext}. Use PNG or JPG.",
                "allowed": list(config.ALLOWED_EXTENSIONS)
            }), 400
        print(f"[PREDICT:4] ✅ File extension valid: {file.filename.rsplit('.', 1)[1]}")
        
        # ==================== IMAGE LOADING ====================
        print("[PREDICT:5] Opening image from file stream...")
        try:
            # Reset file stream to beginning
            file.stream.seek(0)
            print("[PREDICT:5] File stream reset to beginning")
            
            img = Image.open(file.stream).convert('RGB')
            print(f"[PREDICT:5] ✅ Image opened: {img.size} pixels, mode: RGB")
        except FileNotFoundError as e:
            print(f"[ERROR:5] FileNotFoundError: {e}")
            return jsonify({
                "success": False,
                "error": "File not found",
                "debug": str(e)
            }), 400
        except IOError as e:
            print(f"[ERROR:5] IOError (corrupted image?): {e}")
            return jsonify({
                "success": False,
                "error": "Failed to read image file (possibly corrupted)",
                "debug": str(e)
            }), 400
        except Exception as e:
            print(f"[ERROR:5] Unexpected error opening image: {type(e).__name__}: {e}")
            print(f"[ERROR:5] Traceback: {traceback.format_exc()}")
            return jsonify({
                "success": False,
                "error": "Failed to open image",
                "debug": f"{type(e).__name__}: {e}"
            }), 400
        
        # ==================== IMAGE PREPROCESSING ====================
        print("[PREDICT:6] Starting image preprocessing...")
        try:
            print("[PREDICT:6] Applying transforms (resize, crop, normalize)...")
            img_tensor = prediction_transform(img).unsqueeze(0)
            print(f"[PREDICT:6] ✅ Image tensor shape after transforms: {img_tensor.shape}")
            print(f"[PREDICT:6] Image tensor dtype: {img_tensor.dtype}")
            print(f"[PREDICT:6] Image tensor device: {img_tensor.device}")
        except RuntimeError as e:
            print(f"[ERROR:6] RuntimeError during preprocessing: {e}")
            return jsonify({
                "success": False,
                "error": "Image preprocessing failed",
                "debug": f"RuntimeError: {e}"
            }), 400
        except Exception as e:
            print(f"[ERROR:6] Unexpected error preprocessing: {type(e).__name__}: {e}")
            print(f"[ERROR:6] Traceback: {traceback.format_exc()}")
            return jsonify({
                "success": False,
                "error": "Image preprocessing failed",
                "debug": f"{type(e).__name__}: {e}"
            }), 400
        
        # ==================== MOVING TENSOR TO DEVICE ====================
        print(f"[PREDICT:7] Moving tensor to device: {config.DEVICE}...")
        try:
            img_tensor = img_tensor.to(config.DEVICE)
            print(f"[PREDICT:7] ✅ Tensor moved to {config.DEVICE}")
            print(f"[PREDICT:7] Tensor memory: {img_tensor.element_size() * img_tensor.nelement() / 1024:.2f} KB")
        except RuntimeError as e:
            print(f"[ERROR:7] RuntimeError moving to device: {e}")
            return jsonify({
                "success": False,
                "error": "Failed to move tensor to device",
                "debug": f"RuntimeError: {e}"
            }), 500
        except Exception as e:
            print(f"[ERROR:7] Unexpected error: {type(e).__name__}: {e}")
            return jsonify({
                "success": False,
                "error": "Device error",
                "debug": f"{type(e).__name__}: {e}"
            }), 500
        
        # ==================== MODEL INFERENCE ====================
        print("[PREDICT:8] Starting model inference...")
        print(f"[PREDICT:8] Model device: {next(model.parameters()).device}")
        print(f"[PREDICT:8] Model training: {model.training}")
        
        try:
            print("[PREDICT:8] Running forward pass with torch.no_grad()...")
            with torch.no_grad():
                outputs = model(img_tensor)
                print(f"[PREDICT:8] ✅ Forward pass completed")
                print(f"[PREDICT:8] Model output shape: {outputs.shape}")
                print(f"[PREDICT:8] Model output dtype: {outputs.dtype}")
                
                print("[PREDICT:8] Computing softmax probabilities...")
                probs = torch.nn.functional.softmax(outputs, dim=1)
                print(f"[PREDICT:8] ✅ Softmax computed")
                print(f"[PREDICT:8] Probabilities shape: {probs.shape}")
        except RuntimeError as e:
            print(f"[ERROR:8] RuntimeError during inference: {e}")
            print(f"[ERROR:8] Traceback: {traceback.format_exc()}")
            return jsonify({
                "success": False,
                "error": "Model inference failed",
                "debug": f"RuntimeError: {e}"
            }), 500
        except MemoryError as e:
            print(f"[ERROR:8] MemoryError during inference: {e}")
            print(f"[ERROR:8] Traceback: {traceback.format_exc()}")
            return jsonify({
                "success": False,
                "error": "Out of memory during prediction",
                "debug": f"MemoryError: {e}"
            }), 500
        except Exception as e:
            print(f"[ERROR:8] Unexpected error during inference: {type(e).__name__}: {e}")
            print(f"[ERROR:8] Traceback: {traceback.format_exc()}")
            return jsonify({
                "success": False,
                "error": "Model inference failed",
                "debug": f"{type(e).__name__}: {e}"
            }), 500
        
        # ==================== POST-PROCESSING ====================
        print("[PREDICT:9] Getting top 3 predictions...")
        try:
            top_probs, top_idxs = torch.topk(probs, 3)
            top_probs = top_probs.cpu().numpy()[0]
            top_idxs = top_idxs.cpu().numpy()[0]
            print(f"[PREDICT:9] ✅ Top 3 predictions extracted")
            print(f"[PREDICT:9] Top probabilities: {top_probs}")
            print(f"[PREDICT:9] Top indices: {top_idxs}")
        except Exception as e:
            print(f"[ERROR:9] Error extracting top-k: {type(e).__name__}: {e}")
            return jsonify({
                "success": False,
                "error": "Failed to extract predictions",
                "debug": f"{type(e).__name__}: {e}"
            }), 500
        
        # ==================== BUILDING RESPONSE ====================
        print("[PREDICT:10] Building prediction response...")
        try:
            predictions = []
            for i in range(len(top_idxs)):
                idx = int(top_idxs[i])
                print(f"[PREDICT:10] Processing prediction {i+1}: index={idx}")
                
                if idx not in class_labels:
                    print(f"[ERROR:10] Index {idx} not in class_labels. Available: {list(class_labels.keys())}")
                    return jsonify({
                        "success": False,
                        "error": f"Model output index {idx} not found in labels",
                        "debug": f"Available indices: {list(class_labels.keys())}"
                    }), 500
                
                class_name = class_labels[idx]
                confidence = float(top_probs[i]) * 100
                crop, disease, is_healthy = parse_disease_name(class_name)
                
                predictions.append({
                    "class_name": class_name,
                    "crop": crop,
                    "disease": disease,
                    "confidence": round(confidence, 2),
                    "is_healthy": is_healthy
                })
                print(f"[PREDICT:10] ✅ Pred {i+1}: {class_name} ({confidence:.2f}%)")
            
            print("[PREDICT:10] ✅ All predictions processed")
        except Exception as e:
            print(f"[ERROR:10] Error building predictions: {type(e).__name__}: {e}")
            print(f"[ERROR:10] Traceback: {traceback.format_exc()}")
            return jsonify({
                "success": False,
                "error": "Failed to process predictions",
                "debug": f"{type(e).__name__}: {e}"
            }), 500
        
        # ==================== TREATMENT INFO ====================
        print("[PREDICT:11] Getting treatment information...")
        try:
            primary = predictions[0]
            treatment = get_treatment_info(primary['disease'])
            print(f"[PREDICT:11] ✅ Treatment info retrieved")
        except Exception as e:
            print(f"[ERROR:11] Error getting treatment: {type(e).__name__}: {e}")
            return jsonify({
                "success": False,
                "error": "Failed to retrieve treatment information",
                "debug": f"{type(e).__name__}: {e}"
            }), 500
        
        # ==================== FINAL RESPONSE ====================
        print("[PREDICT:12] Building final response...")
        try:
            response = {
                "success": True,
                "prediction": {
                    **primary,
                    **treatment
                },
                "top_3_predictions": predictions
            }
            print("[PREDICT:12] ✅ Response built successfully")
            print("[PREDICT] ✅ PREDICTION COMPLETED SUCCESSFULLY")
            print("="*60 + "\n")
            return jsonify(response)
        except Exception as e:
            print(f"[ERROR:12] Error building response: {type(e).__name__}: {e}")
            return jsonify({
                "success": False,
                "error": "Failed to build response",
                "debug": f"{type(e).__name__}: {e}"
            }), 500
            
    except Exception as e:
        print(f"\n[CRITICAL] UNHANDLED EXCEPTION: {type(e).__name__}")
        print(f"[CRITICAL] Message: {e}")
        print(f"[CRITICAL] Traceback:\n{traceback.format_exc()}")
        print("="*60 + "\n")
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "debug": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()
        }), 500

# ========================
# FRONTEND ROUTES (LOWEST PRIORITY)
# ========================

@app.route('/', methods=['GET'])
def serve_index():
    """Serve the frontend index.html"""
    return send_from_directory(config.FRONTEND_FOLDER, 'index.html')

@app.route('/<path:path>', methods=['GET'])
def serve_static(path):
    """Serve static files (CSS, JS, images)"""
    try:
        return send_from_directory(config.FRONTEND_FOLDER, path)
    except Exception as e:
        # If static file not found, serve index.html for SPA routing
        try:
            return send_from_directory(config.FRONTEND_FOLDER, 'index.html')
        except:
            return jsonify({"error": f"Not Found: {path}"}), 404

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
