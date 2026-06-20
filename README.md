# 🌿 Crop Disease Classification Using Leaf Images

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

An end-to-end deep learning project for classifying crop diseases from leaf images using **MobileNetV2 Transfer Learning**. This project includes a complete ML pipeline, REST API backend, and a modern web interface.

## 🌐 Live Demo
Try the live application here: **[https://crop-disease-classification-1-t5nk.onrender.com/](https://crop-disease-classification-1-t5nk.onrender.com/)**


## 📸 Screenshots

![Hero Section](docs/hero.png)
![Upload Interface](docs/upload.png)
![Results Display](docs/results.png)

## 🎯 Features

- ✅ **Deep Learning Model**: MobileNetV2 transfer learning with 95%+ accuracy
- ✅ **15+ Disease Classes**: Supports multiple crops (Tomato, Potato, Pepper)
- ✅ **REST API**: Flask-based backend with comprehensive endpoints
- ✅ **Modern UI**: Beautiful, responsive web interface
- ✅ **Treatment Recommendations**: AI-powered disease treatment advice
- ✅ **Batch Processing**: Support for multiple image predictions
- ✅ **Real-time Predictions**: Results in under 2 seconds

## 🗂️ Project Structure

```
Crop-Disease-Classification/
├── backend/
│   ├── app.py                 # Flask API server
│   └── uploads/               # Temporary upload folder
├── frontend/
│   ├── index.html             # Main HTML page
│   ├── styles.css             # Premium CSS design
│   └── script.js              # Frontend JavaScript
├── models/
│   ├── crop_disease_model.h5  # Trained model (generated)
│   ├── class_labels.json      # Class mappings (generated)
│   ├── training_history.json  # Training metrics (generated)
│   ├── confusion_matrix.png   # Evaluation plot (generated)
│   └── training_history.png   # Training plots (generated)
├── notebooks/                  # Jupyter notebooks (optional)
├── train_model.py             # Model training script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 📊 Dataset

This project uses the **PlantVillage Dataset** with the following structure:

```
PlantVillage/
├── Pepper__bell___Bacterial_spot/
├── Pepper__bell___healthy/
├── Potato___Early_blight/
├── Potato___Late_blight/
├── Potato___healthy/
├── Tomato_Bacterial_spot/
├── Tomato_Early_blight/
├── Tomato_Late_blight/
├── Tomato_Leaf_Mold/
├── Tomato_Septoria_leaf_spot/
├── Tomato_Spider_mites_Two_spotted_spider_mite/
├── Tomato__Target_Spot/
├── Tomato__Tomato_YellowLeaf__Curl_Virus/
├── Tomato__Tomato_mosaic_virus/
└── Tomato_healthy/
```

**Dataset Statistics:**
- Total Images: 40,000+
- Image Size: 224×224 pixels
- Classes: 15
- Crops: Tomato, Potato, Pepper

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-enabled GPU for faster training

### Step 1: Clone or Download

```bash
cd "C:\Users\ayush\Desktop\Machine Learning\Crop-Disease-Classification"
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify Dataset Path

Ensure your dataset is located at:
```
C:\Users\ayush\Desktop\Machine Learning\PlantVillage
```

Or update the path in `train_model.py`:
```python
DATASET_PATH = r"YOUR_DATASET_PATH"
```

## 🧠 Training the Model

### Quick Start

```bash
python train_model.py
```

### Training Configuration

Edit the `Config` class in `train_model.py` to customize:

```python
class Config:
    IMG_SIZE = 224              # Image dimensions
    BATCH_SIZE = 32             # Batch size
    EPOCHS = 30                 # Training epochs
    LEARNING_RATE = 0.0001      # Initial learning rate
    USE_TRANSFER_LEARNING = True # Use MobileNetV2
    FINE_TUNE_LAYERS = 50       # Layers to fine-tune
```

### Training Output

After training completes, you'll find:
- `models/crop_disease_model.h5` - Trained model
- `models/class_labels.json` - Class label mappings
- `models/training_history.json` - Training metrics
- `models/confusion_matrix.png` - Confusion matrix visualization
- `models/training_history.png` - Training/validation curves
- `models/classification_report.txt` - Detailed metrics

### Expected Performance

- **Training Time**: 30-60 minutes (GPU) / 2-4 hours (CPU)
- **Accuracy**: 95%+ on validation set
- **Top-3 Accuracy**: 98%+

## ⚙️ Running the Backend

### Start Flask Server

```bash
cd backend
python app.py
```

The API will be available at: `http://localhost:5000`

### API Endpoints

#### 1. Health Check
```bash
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "num_classes": 15
}
```

#### 2. Get Classes
```bash
GET /api/classes
```

**Response:**
```json
{
  "classes": ["Pepper__bell___Bacterial_spot", ...],
  "num_classes": 15
}
```

#### 3. Predict Single Image
```bash
POST /api/predict
Content-Type: multipart/form-data
Body: file=<image_file>
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "crop": "Tomato",
    "disease": "Early blight",
    "confidence": 96.5,
    "is_healthy": false,
    "severity": "Moderate",
    "treatment": "Apply fungicides containing...",
    "prevention": "Mulch around plants..."
  },
  "top_3_predictions": [...]
}
```

#### 4. Batch Predict
```bash
POST /api/batch-predict
Content-Type: multipart/form-data
Body: files=<multiple_image_files>
```

## 🎨 Running the Frontend

### Option 1: Using Python HTTP Server

```bash
cd frontend
python -m http.server 8000
```

Open browser: `http://localhost:8000`

### Option 2: Using Live Server (VS Code)

1. Install "Live Server" extension
2. Right-click `index.html`
3. Select "Open with Live Server"

### Option 3: Direct File Access

Simply open `frontend/index.html` in your browser.

**Note:** Make sure the Flask backend is running for predictions to work!

## 🖥️ Usage Guide

### Web Interface

1. **Upload Image**
   - Drag and drop a leaf image
   - Or click to browse and select
   - Supported formats: JPG, PNG, JPEG

2. **Analyze**
   - Click "Analyze Image" button
   - Wait for AI processing (~2 seconds)

3. **View Results**
   - Disease name and confidence score
   - Crop type identification
   - Severity level
   - Treatment recommendations
   - Prevention tips
   - Top 3 alternative predictions

### Programmatic Usage

```python
import requests

# Predict single image
with open('leaf_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/api/predict', files=files)
    result = response.json()
    print(result['prediction'])
```

## 📈 Model Architecture

### Transfer Learning Approach

```
Input (224×224×3)
    ↓
MobileNetV2 (Pre-trained on ImageNet)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization + Dropout(0.5)
    ↓
Dense(512, ReLU)
    ↓
BatchNormalization + Dropout(0.3)
    ↓
Dense(256, ReLU)
    ↓
BatchNormalization + Dropout(0.2)
    ↓
Dense(15, Softmax)
```

### Training Strategy

1. **Phase 1**: Train top layers with frozen base model
2. **Phase 2**: Fine-tune last 50 layers of MobileNetV2
3. **Data Augmentation**: Rotation, shift, zoom, flip
4. **Callbacks**: Early stopping, learning rate reduction, model checkpoint

## 🛠️ Technology Stack

### 🧠 Machine Learning
- **Framework:** PyTorch & torchvision (Switched from TensorFlow for better performance on Python 3.12)
- **Architecture:** MobileNetV2 (Transfer Learning)
- **Features:**
  - Pre-trained on ImageNet
  - Fine-tuned on PlantVillage dataset
  - 95%+ Accuracy target
  - Optimized for inference speed (<100ms)
- **scikit-learn** - Metrics and evaluation

### Backend
- **Flask 3.0** - Web framework
- **Flask-CORS** - Cross-origin resource sharing
- **Pillow** - Image processing

### Frontend
- **HTML5** - Structure
- **CSS3** - Modern styling with gradients and animations
- **JavaScript (ES6+)** - Interactive functionality
- **Font Awesome** - Icons
- **Google Fonts** - Typography (Inter, Outfit)

## 🎓 Use Cases

This project is perfect for:

- ✅ **Final Year Projects** - Complete ML + Full-Stack implementation
- ✅ **Hackathons** - Production-ready with modern UI
- ✅ **Portfolio** - Showcase ML and web development skills
- ✅ **Research** - Baseline for agricultural AI projects
- ✅ **Learning** - Understand end-to-end ML deployment

## 📝 Future Enhancements

- [ ] Mobile app (React Native / Flutter)
- [ ] User authentication and history
- [ ] Database integration for predictions
- [ ] Real-time camera capture
- [ ] Multi-language support
- [ ] Fertilizer recommendations
- [ ] Weather integration
- [ ] Farmer community forum

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Ayush Kumar**
- Email: ayushku244@gmail.com
- Contact: +91 9162573098
- GitHub: [@ayushku244](https://github.com/ayushku244)

## 🙏 Acknowledgments

- **PlantVillage Dataset** - For providing the training data
- **TensorFlow Team** - For the amazing deep learning framework
- **MobileNetV2** - For the efficient transfer learning architecture
- **Agricultural Community** - For domain knowledge and feedback

## 📞 Support

If you have any questions or issues, please:
1. Check the [Issues](https://github.com/ayushku244/crop-disease-classification/issues) page
2. Create a new issue with detailed description
3. Contact via email: ayushku244@gmail.com

## ⭐ Star This Repository

If you found this project helpful, please give it a star! It helps others discover the project.

---

**Developed by:** Ayush Kumar  
**Email:** ayushku244@gmail.com  
**Contact:** +91 9162573098
#   R e b u i l d   t r i g g e r  
 