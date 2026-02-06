import io
import json
import os
from typing import List

# Disable GPU
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
import keras

# Configuration
MODEL_PATH = "mushroom_model.keras"
NAMES_PATH = "mushroom_names.json"
IMAGE_SIZE = (128, 128)
MAX_TOP_N = 20

app = FastAPI(
    title="Mushroom Classifier",
    version="1.0",
    description="CPU inference backend for mushroom classification",
    root_path="/mushrooms",
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


@app.get("/")
async def web_documentation():
    """Redirect to local FastAPI docs."""
    return RedirectResponse("/docs")


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
