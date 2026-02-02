# Mushroom Classifier Backend API

A FastAPI-based CPU inference backend for mushroom classification using a trained Keras model.

## Overview

This backend serves a trained deep learning model (`mushroom_model.keras`) that classifies mushroom images and returns the top-n possible mushroom species with confidence scores.

## Features

- **REST API** with FastAPI
- **CPU-only inference** (no GPU required)
- **Image upload support** (JPG/PNG)
- **Top-N predictions** (configurable)
- **HTML web interface** for easy testing
- **JSON API** for programmatic access

## Requirements

- Python 3.8+
- TensorFlow/Keras
- FastAPI
- Uvicorn
- Pillow (PIL)
- NumPy

## Installation

1. Install dependencies:
```bash
pip install fastapi uvicorn keras tensorflow pillow numpy
```

2. Ensure model file exists:
```bash
ls mushroom_model.keras
```

3. Ensure data directory with class names exists:
```bash
ls data/
```

## Running the Server

Start the API server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

- `--host 0.0.0.0`: Listen on all network interfaces
- `--port 8000`: Use port 8000 (change as needed)

The server will start and load the model on startup.

## API Endpoints

### 1. GET `/` - Web Interface

**Access in browser:**
```
http://localhost:8000/
```

**Response:** HTML form for image upload and prediction count

---

### 2. POST `/predict` - Prediction Endpoint

**URL:** `http://localhost:8000/predict`

**Method:** POST (multipart/form-data)

**Parameters:**
- `image` (file, required): JPG or PNG image file
- `n` (integer, optional): Number of top predictions (default: 3, max: 10)

**Response (JSON):**
```json
{
  "top_n": [
    {"name": "fleecy_milkcap", "confidence": 0.85},
    {"name": "common_inkcap", "confidence": 0.10},
    {"name": "butter_cap", "confidence": 0.05}
  ]
}
```

**Response (HTML):**
When accessed from a browser form, returns formatted HTML with results.

---

## Usage Examples

### Browser Access

1. Open `http://localhost:8000/` in your browser
2. Upload a mushroom image (JPG/PNG)
3. Enter top-N predictions (default: 3)
4. Click "Predict"
5. View results

### cURL Command

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "image=@data/fleecy_milkcap/6.png" \
  -F "n=3"
```

### Python (requests library)

```python
import requests

with open("data/fleecy_milkcap/6.png", "rb") as f:
    files = {"image": f}
    data = {"n": 3}
    response = requests.post("http://localhost:8000/predict", files=files, data=data)
    print(response.json())
```

### JavaScript (fetch API)

```javascript
const formData = new FormData();
formData.append("image", document.getElementById("imageInput").files[0]);
formData.append("n", 3);

fetch("http://localhost:8000/predict", {
  method: "POST",
  body: formData
})
.then(res => res.json())
.then(data => console.log(data.top_n));
```

## Configuration

Edit these variables in `main.py`:

```python
DATA_DIR = "data"                    # Directory with class folders
MODEL_PATH = "mushroom_model.keras"  # Path to model file
IMAGE_SIZE = (128, 128)              # Input image size
```

## Model Details

- **Framework:** Keras/TensorFlow
- **Input:** 128×128 RGB images
- **Output:** Probability distribution over mushroom classes
- **Classes:** 303+ mushroom species (from data directory)

## Error Handling

### Missing Image
```json
{"detail": [{"type": "missing", "loc": ["body", "image"], "msg": "Field required"}]}
```

### Invalid n parameter
```json
{"error": "n must be > 0"}
```

### Empty File
```json
{"error": "empty file"}
```

## Performance

- **Inference time:** ~200-500ms per image (CPU, model-dependent)
- **Batch size:** Single image per request (configurable)
- **Memory:** ~500MB-1GB (model + TensorFlow runtime)

## Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
COPY mushroom_model.keras .
COPY data/ data/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Server

Use Gunicorn + Uvicorn:
```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
```

## Troubleshooting

**Model not found:**
```
RuntimeError: Model file not found: mushroom_model.keras
```
→ Ensure `mushroom_model.keras` exists in the working directory

**Data directory not found:**
```
RuntimeError: Data directory not found: data
```
→ Ensure `data/` directory exists with subdirectories for each mushroom class

**Out of memory:**
→ Reduce batch size or use a lighter model

**Slow inference:**
→ This is normal on CPU. Consider using a GPU server for faster predictions.

## API Documentation

FastAPI auto-generates interactive API docs:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

## License

Project repository: [mushroom-categorizor-model](https://github.com/WSM-simon/mushroom-categorizor-model)
