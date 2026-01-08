/**
 * Frontend logic for image upload and API interaction
 */

// ========================
// CONFIGURATION
// ========================
const CONFIG = {
    API_BASE_URL: 'http://localhost:5000/api',
    MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
    ALLOWED_TYPES: ['image/jpeg', 'image/jpg', 'image/png'],
};

// ========================
// DOM ELEMENTS
// ========================
const elements = {
    uploadArea: document.getElementById('uploadArea'),
    fileInput: document.getElementById('fileInput'),
    uploadPlaceholder: document.getElementById('uploadPlaceholder'),
    imagePreview: document.getElementById('imagePreview'),
    previewImg: document.getElementById('previewImg'),
    removeImage: document.getElementById('removeImage'),
    predictBtn: document.getElementById('predictBtn'),
    resetBtn: document.getElementById('resetBtn'),
    loadingState: document.getElementById('loadingState'),
    resultsSection: document.getElementById('resultsSection'),
    toast: document.getElementById('toast'),
    toastMessage: document.getElementById('toastMessage'),

    // Result elements
    healthBadge: document.getElementById('healthBadge'),
    cropName: document.getElementById('cropName'),
    diseaseName: document.getElementById('diseaseName'),
    confidenceFill: document.getElementById('confidenceFill'),
    confidenceText: document.getElementById('confidenceText'),
    severityBadge: document.getElementById('severityBadge'),
    treatmentText: document.getElementById('treatmentText'),
    preventionText: document.getElementById('preventionText'),
    predictionsList: document.getElementById('predictionsList'),
};

// ========================
// STATE MANAGEMENT
// ========================
let currentFile = null;

// ========================
// EVENT LISTENERS
// ========================
function initializeEventListeners() {
    // Upload area click
    elements.uploadArea.addEventListener('click', () => {
        if (!currentFile) {
            elements.fileInput.click();
        }
    });

    // File input change
    elements.fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    elements.uploadArea.addEventListener('dragover', handleDragOver);
    elements.uploadArea.addEventListener('dragleave', handleDragLeave);
    elements.uploadArea.addEventListener('drop', handleDrop);

    // Remove image
    elements.removeImage.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });

    // Predict button
    elements.predictBtn.addEventListener('click', handlePredict);

    // Reset button
    elements.resetBtn.addEventListener('click', resetAll);

    // Smooth scroll for navigation
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({ behavior: 'smooth' });

                // Update active link
                document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
                link.classList.add('active');
            }
        });
    });
}

// ========================
// FILE HANDLING
// ========================
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        validateAndPreviewFile(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    elements.uploadArea.classList.add('dragover');
}

function handleDragLeave(event) {
    event.preventDefault();
    elements.uploadArea.classList.remove('dragover');
}

function handleDrop(event) {
    event.preventDefault();
    elements.uploadArea.classList.remove('dragover');

    const file = event.dataTransfer.files[0];
    if (file) {
        validateAndPreviewFile(file);
    }
}

function validateAndPreviewFile(file) {
    // Validate file type
    if (!CONFIG.ALLOWED_TYPES.includes(file.type)) {
        showToast('Please upload a valid image file (JPG, PNG)', 'error');
        return;
    }

    // Validate file size
    if (file.size > CONFIG.MAX_FILE_SIZE) {
        showToast('File size must be less than 10MB', 'error');
        return;
    }

    // Store file and preview
    currentFile = file;
    previewImage(file);

    // Enable predict button
    elements.predictBtn.disabled = false;

    // Hide results if showing
    elements.resultsSection.style.display = 'none';
}

function previewImage(file) {
    const reader = new FileReader();

    reader.onload = (e) => {
        elements.previewImg.src = e.target.result;
        elements.uploadPlaceholder.style.display = 'none';
        elements.imagePreview.style.display = 'flex';
    };

    reader.readAsDataURL(file);
}

function resetUpload() {
    currentFile = null;
    elements.fileInput.value = '';
    elements.previewImg.src = '';
    elements.uploadPlaceholder.style.display = 'flex';
    elements.imagePreview.style.display = 'none';
    elements.predictBtn.disabled = true;
}

function resetAll() {
    resetUpload();
    elements.resultsSection.style.display = 'none';
    elements.loadingState.style.display = 'none';
}

// ========================
// PREDICTION
// ========================
async function handlePredict() {
    if (!currentFile) {
        showToast('Please select an image first', 'error');
        return;
    }

    // Show loading state
    elements.loadingState.style.display = 'block';
    elements.resultsSection.style.display = 'none';
    elements.predictBtn.disabled = true;

    try {
        // Create form data
        const formData = new FormData();
        formData.append('file', currentFile);

        // Make API request
        const response = await fetch(`${CONFIG.API_BASE_URL}/predict`, {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (data.success) {
            displayResults(data);
            showToast('Analysis complete!', 'success');
        } else {
            throw new Error(data.error || 'Prediction failed');
        }

    } catch (error) {
        console.error('Prediction error:', error);
        showToast(error.message || 'Failed to analyze image. Please try again.', 'error');
        elements.predictBtn.disabled = false;
    } finally {
        elements.loadingState.style.display = 'none';
    }
}

// ========================
// RESULTS DISPLAY
// ========================
function displayResults(data) {
    const prediction = data.prediction;
    const top3 = data.top_3_predictions;

    // Update health badge
    const isHealthy = prediction.is_healthy;
    elements.healthBadge.className = `health-badge ${isHealthy ? 'healthy' : 'diseased'}`;
    elements.healthBadge.innerHTML = `
        <i class="fas fa-${isHealthy ? 'check-circle' : 'exclamation-circle'}"></i>
        <span>${isHealthy ? 'Healthy' : 'Diseased'}</span>
    `;

    // Update crop and disease
    elements.cropName.textContent = prediction.crop;
    elements.diseaseName.textContent = prediction.disease;

    // Update confidence
    const confidence = prediction.confidence;
    elements.confidenceFill.style.width = `${confidence}%`;
    elements.confidenceText.textContent = `${confidence}%`;

    // Update severity
    const severity = prediction.severity.toLowerCase();
    elements.severityBadge.textContent = prediction.severity;
    elements.severityBadge.className = `detail-value severity-badge ${severity}`;

    // Update treatment and prevention
    elements.treatmentText.textContent = prediction.treatment;
    elements.preventionText.textContent = prediction.prevention;

    // Update top 3 predictions
    displayTop3Predictions(top3);

    // Show results with animation
    elements.resultsSection.style.display = 'block';
    elements.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    // Re-enable predict button
    elements.predictBtn.disabled = false;
}

function displayTop3Predictions(predictions) {
    elements.predictionsList.innerHTML = '';

    predictions.forEach((pred, index) => {
        const item = document.createElement('div');
        item.className = 'prediction-item';
        item.style.animationDelay = `${index * 0.1}s`;

        item.innerHTML = `
            <div class="prediction-info">
                <div class="prediction-disease">${pred.disease}</div>
                <div class="prediction-crop">${pred.crop}</div>
            </div>
            <div class="prediction-confidence">${pred.confidence}%</div>
        `;

        elements.predictionsList.appendChild(item);
    });
}

// ========================
// TOAST NOTIFICATIONS
// ========================
function showToast(message, type = 'success') {
    elements.toastMessage.textContent = message;

    // Update toast style based on type
    if (type === 'error') {
        elements.toast.style.background = 'hsl(0, 84%, 60%)';
    } else {
        elements.toast.style.background = 'hsl(142, 76%, 45%)';
    }

    // Show toast
    elements.toast.classList.add('show');

    // Hide after 3 seconds
    setTimeout(() => {
        elements.toast.classList.remove('show');
    }, 3000);
}

// ========================
// API HEALTH CHECK
// ========================
async function checkAPIHealth() {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/health`);
        const data = await response.json();

        if (data.status === 'healthy' && data.model_loaded) {
            console.log('✅ API is healthy and model is loaded');
            console.log(`📊 Number of classes: ${data.num_classes}`);
        } else {
            console.warn('⚠️ API is running but model is not loaded');
            showToast('Model not loaded. Please train the model first.', 'error');
        }
    } catch (error) {
        console.error('❌ API health check failed:', error);
        console.warn('⚠️ Make sure the Flask backend is running on http://localhost:5000');
    }
}

// ========================
// SCROLL ANIMATIONS
// ========================
function initializeScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -100px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe feature cards
    document.querySelectorAll('.feature-card').forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(30px)';
        card.style.transition = `all 0.6s ease-out ${index * 0.1}s`;
        observer.observe(card);
    });
}

// ========================
// ACTIVE NAV LINK ON SCROLL
// ========================
function initializeScrollSpy() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-link');

    window.addEventListener('scroll', () => {
        let current = '';

        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;

            if (window.pageYOffset >= sectionTop - 200) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('active');
            }
        });
    });
}

// ========================
// INITIALIZATION
// ========================
function initialize() {
    console.log('🌿 Crop Disease Classifier - Frontend Initialized');

    // Initialize event listeners
    initializeEventListeners();

    // Initialize scroll animations
    initializeScrollAnimations();

    // Initialize scroll spy
    initializeScrollSpy();

    // Check API health
    checkAPIHealth();

    console.log('✅ All systems ready');
}

// Run initialization when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialize);
} else {
    initialize();
}

// ========================
// UTILITY FUNCTIONS
// ========================

/**
 * Format file size to human readable format
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

/**
 * Debounce function for performance
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Export for potential module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        handlePredict,
        resetAll,
        showToast,
    };
}
