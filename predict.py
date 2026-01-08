# Test script for single image prediction
import sys
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Config
MODEL_PATH = "models/crop_disease_model.pth"
LABELS_PATH = "models/class_labels.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_labels():
    print("Loading model...")
    # Load labels
    with open(LABELS_PATH, 'r') as f:
        labels = json.load(f)
        labels = {int(k): v for k, v in labels.items()}
        
    # Build model (Matching architecture)
    model = models.mobilenet_v2(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, len(labels))
    )
    
    # Load weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    return model, labels

def predict(image_path, model, labels):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        img = Image.open(image_path).convert('RGB')
        img_t = transform(img).unsqueeze(0).to(DEVICE)
        
        print(f"\nAnalyzing: {image_path}")
        print("-" * 50)
        
        with torch.no_grad():
            out = model(img_t)
            probs = torch.nn.functional.softmax(out, dim=1)
            
        top_probs, top_idxs = torch.topk(probs, 3)
        
        for i in range(len(top_idxs[0])):
            idx = top_idxs[0][i].item()
            prob = top_probs[0][i].item() * 100
            name = labels[idx]
            
            print(f"{i+1}. {name}: {prob:.2f}%")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
        
    model, labels = load_model_and_labels()
    predict(sys.argv[1], model, labels)
