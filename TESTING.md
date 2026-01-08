# 🧪 Testing Guide - Crop Disease Classification

This guide helps you test all components of the project to ensure everything works correctly.

## 📋 Pre-Testing Checklist

- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset available at correct path
- [ ] Model trained successfully
- [ ] Backend server can start
- [ ] Frontend server can start

## 🧠 Testing the ML Model

### Test 1: Training Script

```bash
python train_model.py
```

**Expected Output:**
```
🌿 CROP DISEASE CLASSIFICATION - MODEL TRAINING
================================================================================
✅ GPU available: 1 device(s)
🔄 Creating data generators...
✅ Found 28000 training images
✅ Found 6000 validation images
✅ Number of classes: 15
...
✅ TRAINING COMPLETED SUCCESSFULLY!
🎯 Final Accuracy: 95.XX%
```

**Success Criteria:**
- ✅ No errors during execution
- ✅ Validation accuracy > 90%
- ✅ Model file created: `models/crop_disease_model.h5`
- ✅ Labels file created: `models/class_labels.json`
- ✅ Plots generated in `models/` directory

### Test 2: Quick Prediction Script

```bash
# Test with a sample image from your dataset
python predict.py "PlantVillage/Tomato_healthy/image001.jpg"
```

**Expected Output:**
```
Analyzing: PlantVillage/Tomato_healthy/image001.jpg
--------------------------------------------------

🎯 Top 3 Predictions:
==================================================

1. Tomato_healthy
   Crop: Tomato
   Disease: healthy
   Confidence: 98.45%
   ✅ HEALTHY

2. Tomato_Bacterial_spot
   Crop: Tomato
   Disease: Bacterial spot
   Confidence: 1.23%
   ⚠️ DISEASED

3. Tomato_Early_blight
   Crop: Tomato
   Disease: Early blight
   Confidence: 0.18%
   ⚠️ DISEASED
```

**Success Criteria:**
- ✅ Correct disease identified
- ✅ High confidence (>90%) for correct class
- ✅ Top-3 predictions shown

## ⚙️ Testing the Backend API

### Test 1: Start Backend Server

```bash
cd backend
python app.py
```

**Expected Output:**
```
🌿 CROP DISEASE CLASSIFICATION - FLASK API SERVER
================================================================================
📡 Starting server...
🔗 API will be available at: http://localhost:5000
✅ Model loaded from ../models/crop_disease_model.h5
✅ Labels loaded: 15 classes
 * Running on http://0.0.0.0:5000
```

**Success Criteria:**
- ✅ Server starts without errors
- ✅ Model loads successfully
- ✅ Port 5000 is accessible

### Test 2: Health Check Endpoint

**Using Browser:**
Navigate to: `http://localhost:5000/api/health`

**Using curl:**
```bash
curl http://localhost:5000/api/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "num_classes": 15
}
```

**Success Criteria:**
- ✅ Returns 200 OK status
- ✅ `model_loaded` is `true`
- ✅ `num_classes` is 15

### Test 3: Get Classes Endpoint

**Using Browser:**
Navigate to: `http://localhost:5000/api/classes`

**Using curl:**
```bash
curl http://localhost:5000/api/classes
```

**Expected Response:**
```json
{
  "classes": [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    ...
  ],
  "num_classes": 15
}
```

**Success Criteria:**
- ✅ Returns all 15 classes
- ✅ Class names are correct

### Test 4: Prediction Endpoint

**Using curl:**
```bash
curl -X POST -F "file=@test_image.jpg" http://localhost:5000/api/predict
```

**Using Python:**
```python
import requests

with open('test_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/api/predict', files=files)
    print(response.json())
```

**Expected Response:**
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

**Success Criteria:**
- ✅ Returns 200 OK status
- ✅ `success` is `true`
- ✅ Prediction contains all required fields
- ✅ Confidence is reasonable (>50%)

### Test 5: Error Handling

**Test with invalid file type:**
```bash
curl -X POST -F "file=@document.pdf" http://localhost:5000/api/predict
```

**Expected Response:**
```json
{
  "success": false,
  "error": "Invalid file type. Allowed types: jpg, png, jpeg"
}
```

**Test with no file:**
```bash
curl -X POST http://localhost:5000/api/predict
```

**Expected Response:**
```json
{
  "success": false,
  "error": "No file provided"
}
```

**Success Criteria:**
- ✅ Returns appropriate error messages
- ✅ Returns 400 status code for bad requests

## 🎨 Testing the Frontend

### Test 1: Start Frontend Server

```bash
cd frontend
python -m http.server 8000
```

**Expected Output:**
```
Serving HTTP on :: port 8000 (http://[::]:8000/) ...
```

**Success Criteria:**
- ✅ Server starts without errors
- ✅ Port 8000 is accessible

### Test 2: Load Homepage

**Open browser:** `http://localhost:8000`

**Visual Checks:**
- ✅ Page loads without errors
- ✅ Header with logo and navigation visible
- ✅ Hero section with title and stats
- ✅ Upload area visible
- ✅ Features section visible
- ✅ Footer visible
- ✅ No console errors in browser DevTools

### Test 3: Upload Functionality

**Steps:**
1. Click on upload area
2. Select an image file
3. Verify image preview appears
4. Check that "Analyze Image" button is enabled

**Success Criteria:**
- ✅ File dialog opens
- ✅ Image preview displays correctly
- ✅ Remove button appears
- ✅ Predict button becomes enabled

### Test 4: Drag and Drop

**Steps:**
1. Drag an image file over upload area
2. Drop the file
3. Verify image preview appears

**Success Criteria:**
- ✅ Drag over effect shows
- ✅ Image loads after drop
- ✅ Preview displays correctly

### Test 5: Prediction Flow

**Steps:**
1. Upload a leaf image
2. Click "Analyze Image"
3. Wait for results
4. Verify results display

**Success Criteria:**
- ✅ Loading spinner appears
- ✅ Results appear after ~2 seconds
- ✅ Crop name is correct
- ✅ Disease name is shown
- ✅ Confidence bar animates
- ✅ Severity badge shows
- ✅ Treatment text appears
- ✅ Prevention text appears
- ✅ Top 3 predictions listed

### Test 6: Reset Functionality

**Steps:**
1. Upload and predict an image
2. Click "Reset" button
3. Verify everything clears

**Success Criteria:**
- ✅ Image preview clears
- ✅ Results section hides
- ✅ Upload area returns to initial state
- ✅ Predict button is disabled

### Test 7: Responsive Design

**Test on different screen sizes:**
- Desktop (1920×1080)
- Tablet (768×1024)
- Mobile (375×667)

**Success Criteria:**
- ✅ Layout adapts correctly
- ✅ All elements are readable
- ✅ Navigation works on mobile
- ✅ Upload area is usable
- ✅ Results display properly

### Test 8: Browser Compatibility

**Test on:**
- Chrome
- Firefox
- Edge
- Safari (if available)

**Success Criteria:**
- ✅ Works on all browsers
- ✅ No visual glitches
- ✅ All features functional

## 🔗 Integration Testing

### Test 1: Full End-to-End Flow

**Steps:**
1. Start backend server
2. Start frontend server
3. Open frontend in browser
4. Upload a healthy leaf image
5. Click analyze
6. Verify correct prediction

**Success Criteria:**
- ✅ Backend receives request
- ✅ Frontend displays results
- ✅ Prediction is accurate
- ✅ All data fields populated

### Test 2: Multiple Predictions

**Steps:**
1. Predict image 1 (healthy)
2. Reset
3. Predict image 2 (diseased)
4. Reset
5. Predict image 3 (different disease)

**Success Criteria:**
- ✅ Each prediction is independent
- ✅ Results are different for different images
- ✅ No errors occur
- ✅ Performance remains consistent

### Test 3: Error Scenarios

**Test 1: Backend Down**
1. Stop backend server
2. Try to predict from frontend
3. Verify error message appears

**Test 2: Invalid Image**
1. Upload a non-leaf image (e.g., cat photo)
2. Predict
3. Verify system still works (may give low confidence)

**Test 3: Large File**
1. Try uploading a very large image (>10MB)
2. Verify error message appears

**Success Criteria:**
- ✅ Appropriate error messages shown
- ✅ System doesn't crash
- ✅ User can recover and try again

## 📊 Performance Testing

### Test 1: Prediction Speed

**Measure:**
- Time from clicking "Analyze" to results appearing

**Success Criteria:**
- ✅ < 2 seconds with GPU
- ✅ < 5 seconds with CPU

### Test 2: Model Loading Time

**Measure:**
- Time for backend to start and load model

**Success Criteria:**
- ✅ < 10 seconds

### Test 3: Frontend Loading Time

**Measure:**
- Time for page to fully load

**Success Criteria:**
- ✅ < 3 seconds on good connection

## 🐛 Common Issues & Solutions

### Issue: "Model not found"
**Solution:** Run `python train_model.py` first

### Issue: "Port already in use"
**Solution:** 
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Change port in app.py if needed
```

### Issue: "CORS error"
**Solution:** Ensure Flask-CORS is installed and configured

### Issue: "Low prediction accuracy"
**Solution:** 
- Train for more epochs
- Check if using correct dataset
- Verify image quality

### Issue: "Out of memory"
**Solution:** Reduce batch size in `train_model.py`

## ✅ Final Checklist

After all tests pass, verify:

- [ ] Model trains successfully
- [ ] Prediction accuracy > 90%
- [ ] Backend API responds correctly
- [ ] Frontend loads without errors
- [ ] Upload functionality works
- [ ] Predictions are accurate
- [ ] Error handling works
- [ ] Responsive design works
- [ ] All documentation is clear

## 🎉 Success!

If all tests pass, congratulations! Your Crop Disease Classification system is working perfectly and ready for:

- ✅ Demonstration
- ✅ Portfolio showcase
- ✅ Final year project submission
- ✅ Hackathon presentation
- ✅ Further development

---

**Happy Testing! 🚀**
