import io
import json
import os
from typing import List

# Disable GPU
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import keras

# Configuration
MODEL_PATH = "mushroom_model.keras"
NAMES_PATH = "mushroom_names.json"
IMAGE_SIZE = (128, 128)
MAX_TOP_N = 10

app = FastAPI(
    title="Mushroom Classifier",
    version="1.0",
    description="CPU inference backend for mushroom classification",
)

# Global variables for model and class names
model = None
class_names = None


def load_class_names(names_file: str) -> List[str]:
    """Load mushroom class names from JSON file."""
    with open(names_file, "r") as f:
        data = json.load(f)
    return data["mushroom_classes"]


def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """Convert uploaded image to preprocessed numpy array."""
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img = img.resize(IMAGE_SIZE)
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise ValueError(f"Failed to process image: {str(e)}")


@app.on_event("startup")
def startup_event():
    """Load model and class names on startup."""
    global model, class_names

    if not os.path.isfile(MODEL_PATH):
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    if not os.path.isfile(NAMES_PATH):
        raise RuntimeError(f"Names file not found: {NAMES_PATH}")

    print(f"Loading class names from {NAMES_PATH}...")
    class_names = load_class_names(NAMES_PATH)
    print(f"Loaded {len(class_names)} mushroom classes")

    print(f"Loading model from {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully")


@app.get("/", response_class=HTMLResponse)
async def web_interface():
    """Serve HTML web interface for image upload and prediction."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mushroom Classifier</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            .container {
                background: white;
                border-radius: 12px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                padding: 40px;
                max-width: 600px;
                width: 100%;
            }
            h1 {
                color: #333;
                margin-bottom: 10px;
                text-align: center;
            }
            .subtitle {
                color: #666;
                text-align: center;
                margin-bottom: 30px;
                font-size: 14px;
            }
            .form-group {
                margin-bottom: 25px;
            }
            label {
                display: block;
                margin-bottom: 8px;
                color: #333;
                font-weight: 500;
                font-size: 14px;
            }
            input[type="file"],
            input[type="number"],
            button {
                width: 100%;
                padding: 12px;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                font-size: 14px;
                font-family: inherit;
            }
            input[type="file"] {
                padding: 8px;
                cursor: pointer;
            }
            input[type="number"] {
                margin-bottom: 10px;
            }
            button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                cursor: pointer;
                font-weight: 600;
                transition: transform 0.2s, box-shadow 0.2s;
                padding: 14px;
                font-size: 16px;
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
            }
            button:active {
                transform: translateY(0);
            }
            .loading {
                display: none;
                text-align: center;
                color: #667eea;
                font-weight: 500;
            }
            .loading.show {
                display: block;
            }
            .spinner {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid #e0e0e0;
                border-top: 3px solid #667eea;
                border-radius: 50%;
                animation: spin 0.8s linear infinite;
                margin-right: 10px;
                vertical-align: middle;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            #results {
                margin-top: 30px;
                display: none;
            }
            #results.show {
                display: block;
            }
            .result-item {
                background: #f5f5f5;
                padding: 15px;
                margin-bottom: 12px;
                border-radius: 6px;
                border-left: 4px solid #667eea;
            }
            .result-name {
                font-weight: 600;
                color: #333;
                margin-bottom: 5px;
                text-transform: capitalize;
            }
            .result-confidence {
                color: #666;
                font-size: 13px;
            }
            .confidence-bar {
                background: #e0e0e0;
                height: 6px;
                border-radius: 3px;
                margin-top: 8px;
                overflow: hidden;
            }
            .confidence-fill {
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                height: 100%;
                border-radius: 3px;
                transition: width 0.3s ease;
            }
            .error {
                background: #fee;
                border-left: 4px solid #f44336;
                color: #c62828;
                padding: 15px;
                border-radius: 6px;
                margin-top: 20px;
                display: none;
            }
            .error.show {
                display: block;
            }
            .preview-container {
                margin-top: 15px;
                text-align: center;
            }
            #imagePreview {
                max-width: 200px;
                max-height: 200px;
                border-radius: 6px;
                display: none;
            }
            #imagePreview.show {
                display: inline-block;
            }
            .api-docs-link {
                text-align: center;
                margin-top: 20px;
                padding-top: 20px;
                border-top: 1px solid #e0e0e0;
            }
            .api-docs-link a {
                color: #667eea;
                text-decoration: none;
                font-size: 13px;
            }
            .api-docs-link a:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🍄 Mushroom Classifier</h1>
            <p class="subtitle">Upload an image to identify mushroom species</p>

            <form id="predictionForm">
                <div class="form-group">
                    <label for="image">Select Image (JPG/PNG)</label>
                    <input type="file" id="image" name="image" accept="image/jpeg,image/png" required>
                    <div class="preview-container">
                        <img id="imagePreview" alt="Preview">
                    </div>
                </div>

                <div class="form-group">
                    <label for="topN">Top Predictions</label>
                    <input type="number" id="topN" name="n" value="3" min="1" max="10">
                </div>

                <button type="submit">Predict Mushroom</button>
                <div class="loading" id="loading">
                    <span class="spinner"></span>Processing...
                </div>
            </form>

            <div id="error" class="error"></div>
            <div id="results"></div>

            <div class="api-docs-link">
                <a href="/docs" target="_blank">📚 API Documentation (Swagger)</a> •
                <a href="/redoc" target="_blank">📖 ReDoc</a>
            </div>
        </div>

        <script>
            const form = document.getElementById('predictionForm');
            const imageInput = document.getElementById('image');
            const imagePreview = document.getElementById('imagePreview');
            const loading = document.getElementById('loading');
            const errorDiv = document.getElementById('error');
            const resultsDiv = document.getElementById('results');

            // Show image preview
            imageInput.addEventListener('change', (e) => {
                const file = e.target.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = (event) => {
                        imagePreview.src = event.target.result;
                        imagePreview.classList.add('show');
                    };
                    reader.readAsDataURL(file);
                }
            });

            // Handle form submission
            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const image = imageInput.files[0];
                if (!image) {
                    showError('Please select an image');
                    return;
                }

                const n = document.getElementById('topN').value;

                const formData = new FormData();
                formData.append('image', image);
                formData.append('n', n);

                loading.classList.add('show');
                errorDiv.classList.remove('show');
                resultsDiv.classList.remove('show');

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.error || errorData.detail || 'Prediction failed');
                    }

                    const data = await response.json();
                    displayResults(data.top_n);
                } catch (error) {
                    showError(error.message);
                } finally {
                    loading.classList.remove('show');
                }
            });

            function displayResults(predictions) {
                let html = '<h2 style="margin-bottom: 20px; color: #333;">Top Predictions</h2>';
                predictions.forEach((pred, index) => {
                    const percentage = (pred.confidence * 100).toFixed(1);
                    html += `
                        <div class="result-item">
                            <div class="result-name">${index + 1}. ${pred.name}</div>
                            <div class="result-confidence">Confidence: ${percentage}%</div>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: ${percentage}%"></div>
                            </div>
                        </div>
                    `;
                });
                resultsDiv.innerHTML = html;
                resultsDiv.classList.add('show');
            }

            function showError(message) {
                errorDiv.innerHTML = `<strong>Error:</strong> ${message}`;
                errorDiv.classList.add('show');
            }
        </script>
    </body>
    </html>
    """
    return html_content


@app.post("/predict")
async def predict(image: UploadFile = File(...), n: int = Form(3)):
    """
    Predict mushroom species from an uploaded image.

    Parameters:
    - image: Image file (JPG or PNG)
    - n: Number of top predictions (1-10, default: 3)

    Returns:
    - top_n: List of top predictions with names and confidence scores
    """
    # Validate n parameter
    if n <= 0 or n > MAX_TOP_N:
        raise HTTPException(
            status_code=400, detail=f"n must be between 1 and {MAX_TOP_N}"
        )

    # Read image file
    file_bytes = await image.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    # Validate file format
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400, detail="Only JPG and PNG images are supported"
        )

    try:
        # Preprocess image
        img_array = preprocess_image(file_bytes)

        # Run inference
        predictions = model.predict(img_array, verbose=0)
        probs = predictions[0]

        # Get top-n predictions
        top_n_count = min(n, len(class_names))
        top_indices = np.argsort(probs)[-top_n_count:][::-1]

        results = [
            {"name": class_names[i], "confidence": float(probs[i])} for i in top_indices
        ]

        return {"top_n": results}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": "loaded" if model is not None else "not loaded",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
