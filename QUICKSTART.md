# 🚀 Quick Start Guide

This guide will help you get the Crop Disease Classification project up and running in minutes!

## 📋 Prerequisites Checklist

- [ ] Python 3.8+ installed
- [ ] pip package manager
- [ ] PlantVillage dataset downloaded
- [ ] 10GB+ free disk space
- [ ] (Optional) CUDA-enabled GPU

## ⚡ Quick Setup (5 Minutes)

### Step 1: Install Dependencies

Open PowerShell/Command Prompt in the project directory:

```bash
pip install -r requirements.txt
```

**Expected time:** 2-3 minutes

### Step 2: Verify Dataset

Make sure your dataset is at:
```
C:\Users\ayush\Desktop\Machine Learning\PlantVillage
```

If it's in a different location, update `train_model.py` line 31:
```python
DATASET_PATH = r"YOUR_PATH_HERE"
```

### Step 3: Train the Model

```bash
python train_model.py
```

**Expected time:** 
- With GPU: 30-60 minutes
- With CPU: 2-4 hours

**What happens:**
- ✅ Loads and preprocesses 40,000+ images
- ✅ Trains MobileNetV2 model
- ✅ Fine-tunes for optimal accuracy
- ✅ Saves model to `models/crop_disease_model.h5`
- ✅ Generates evaluation metrics and plots

**Training Output:**
```
🌿 CROP DISEASE CLASSIFICATION - MODEL TRAINING
================================================================================
📅 Started at: 2026-01-07 18:30:00
🖼️  Image size: 224x224
📦 Batch size: 32
🔄 Epochs: 30
📚 Transfer Learning: True
================================================================================
✅ GPU available: 1 device(s)
🔄 Creating data generators...
✅ Found 28000 training images
✅ Found 6000 validation images
✅ Number of classes: 15
...
```

### Step 4: Start the Backend

Open a **new terminal** and run:

```bash
cd backend
python app.py
```

**Expected output:**
```
🌿 CROP DISEASE CLASSIFICATION - FLASK API SERVER
================================================================================
📡 Starting server...
🔗 API will be available at: http://localhost:5000
📊 Model: ../models/crop_disease_model.h5
🏷️  Labels: ../models/class_labels.json
================================================================================
✅ Model loaded from ../models/crop_disease_model.h5
✅ Labels loaded: 15 classes
 * Running on http://0.0.0.0:5000
```

**Keep this terminal open!**

### Step 5: Start the Frontend

Open **another new terminal** and run:

```bash
cd frontend
python -m http.server 8000
```

**Expected output:**
```
Serving HTTP on :: port 8000 (http://[::]:8000/) ...
```

### Step 6: Open in Browser

Open your browser and navigate to:
```
http://localhost:8000
```

**You should see the beautiful CropCare AI interface!**

## 🎯 Testing the Application

### Test 1: Upload an Image

1. Click the upload area or drag an image
2. Select a leaf image from your dataset
3. Click "Analyze Image"
4. Wait 1-2 seconds
5. View the results!

### Test 2: Try Different Images

Test with different disease types:
- Healthy leaves
- Bacterial spot
- Early blight
- Late blight
- Leaf mold

### Test 3: Check Confidence Scores

- Healthy images should have 95%+ confidence
- Diseased images should correctly identify the disease
- Check the top 3 predictions for alternatives

## 🔧 Troubleshooting

### Problem: "Model not found"

**Solution:**
```bash
# Make sure you've trained the model first
python train_model.py
```

### Problem: "API connection failed"

**Solution:**
```bash
# Make sure Flask backend is running
cd backend
python app.py
```

### Problem: "CORS error in browser"

**Solution:**
- Flask-CORS should be installed
- Check if backend is running on port 5000
- Try accessing: http://localhost:5000/api/health

### Problem: "Out of memory during training"

**Solution:**
Edit `train_model.py` and reduce batch size:
```python
BATCH_SIZE = 16  # or even 8
```

### Problem: "Slow training on CPU"

**Solution:**
- Reduce epochs: `EPOCHS = 10`
- Reduce image size: `IMG_SIZE = 128`
- Or wait patiently (2-4 hours is normal)

### Problem: "Import errors"

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

## 📊 Expected Results

After successful training, you should see:

### Accuracy Metrics
- **Training Accuracy:** 96-98%
- **Validation Accuracy:** 95-97%
- **Top-3 Accuracy:** 98%+

### Generated Files
```
models/
├── crop_disease_model.h5          # 50-100 MB
├── class_labels.json               # < 1 KB
├── training_history.json           # < 10 KB
├── confusion_matrix.png            # Visualization
├── training_history.png            # Plots
└── classification_report.txt       # Metrics
```

## 🎓 Next Steps

### 1. Test with Your Own Images
```bash
python predict.py path/to/your/image.jpg
```

### 2. Customize the Model
- Adjust hyperparameters in `train_model.py`
- Try different architectures
- Add more data augmentation

### 3. Enhance the Frontend
- Add more features
- Customize the design
- Add user authentication

### 4. Deploy to Production
- Use Gunicorn for Flask
- Deploy to Heroku/AWS/Azure
- Add HTTPS
- Set up CI/CD

## 📞 Getting Help

If you encounter issues:

1. **Check the logs** - Look for error messages in the terminal
2. **Read the README** - Comprehensive documentation available
3. **Check requirements** - Ensure all dependencies are installed
4. **Verify paths** - Make sure dataset and model paths are correct

## 🎉 Success Checklist

- [ ] Dependencies installed
- [ ] Model trained successfully
- [ ] Backend running on port 5000
- [ ] Frontend accessible on port 8000
- [ ] Can upload and predict images
- [ ] Results display correctly
- [ ] Treatment recommendations shown

## 💡 Pro Tips

1. **Use GPU** - Training is 10x faster with GPU
2. **Save checkpoints** - Model saves best weights automatically
3. **Monitor training** - Watch the accuracy curves
4. **Test thoroughly** - Try various disease types
5. **Document changes** - Keep track of experiments

---

**Ready to start? Run the first command and let's go! 🚀**

```bash
pip install -r requirements.txt
```
