import io
import os
from typing import List

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
import keras

DATA_DIR = "data"
MODEL_PATH = "mushroom_model.keras"
IMAGE_SIZE = (128, 128)

app = FastAPI(title="Mushroom Classifier", version="1.0")


def load_class_names(data_dir: str) -> List[str]:
    return sorted(
        [
            d
            for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ]
    )


def preprocess_image(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.on_event("startup")
def startup_event():
    global model, class_names

    if not os.path.isdir(DATA_DIR):
        raise RuntimeError(f"Data directory not found: {DATA_DIR}")
    if not os.path.isfile(MODEL_PATH):
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    class_names = load_class_names(DATA_DIR)
    model = keras.models.load_model(MODEL_PATH)


@app.post("/predict")
async def predict(image: UploadFile = File(...), n: int = Form(3)):
    if n <= 0:
        return JSONResponse(status_code=400, content={"error": "n must be > 0"})

    file_bytes = await image.read()
    if not file_bytes:
        return JSONResponse(status_code=400, content={"error": "empty file"})

    img_array = preprocess_image(file_bytes)
    predictions = model.predict(img_array)
    probs = predictions[0]

    top_n = min(n, len(class_names))
    top_indices = np.argsort(probs)[-top_n:][::-1]

    results = [
        {"name": class_names[i], "confidence": float(probs[i])}
        for i in top_indices
    ]

    return {"top_n": results}
