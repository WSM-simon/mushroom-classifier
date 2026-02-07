# Mushroom Classifier

An AI-powered mushroom species identification application combining a FastAPI backend with a Next.js frontend.

## Features

- 🍄 Upload mushroom images (JPG/PNG)
- 🤖 AI-powered species classification
- 📊 Top-N predictions with confidence scores
- 🎨 Modern, responsive UI with Tailwind CSS
- ⚡ Fast inference with TensorFlow/Keras

## Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **TensorFlow/Keras** - Deep learning model
- **Uvicorn** - ASGI server

### Frontend
- **Next.js 15** - React framework
- **TypeScript** - Type-safe JavaScript
- **Tailwind CSS** - Utility-first CSS framework
- **Radix UI** - Accessible component primitives

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm or pnpm

### Installation

1. **Install Node.js dependencies:**
```bash
npm install
# or
pnpm install
```

2. **Python dependencies:**
Python dependencies will be automatically installed in a virtual environment when you run `npm run dev` or `npm run fastapi-dev`.

### Running the Application

**Development mode (runs both backend and frontend):**
```bash
npm run dev
# or
pnpm dev
```

This will start:
- FastAPI backend on http://localhost:8000
- Next.js frontend on http://localhost:3000

**Run backend only:**
```bash
npm run fastapi-dev
# or
uvicorn backend:app --reload --port 8000
```

**Run frontend only:**
```bash
npm run next-dev
# or
next dev
```

### Production Build

```bash
npm run build
npm run start
```

## Project Structure

```
mushroom-classifier/
├── app/                    # Next.js app directory
│   ├── page.tsx           # Main page component
│   ├── layout.tsx         # Root layout
│   └── globals.css        # Global styles
├── components/            # React components
│   └── ui/               # UI components (Button, Card, etc.)
├── lib/                  # Utility functions
├── public/               # Static assets
├── backend.py            # FastAPI backend
├── mushroom_model.keras  # Trained model
├── mushroom_names.json   # Class names
├── package.json          # Node.js dependencies
├── next.config.js        # Next.js configuration
├── tailwind.config.js    # Tailwind CSS config
└── tsconfig.json         # TypeScript config
```

## API Endpoints

### POST `/api/predict`
Classify a mushroom image.

**Parameters:**
- `image` (file): JPG or PNG image file
- `n` (int): Number of top predictions (1-20, default: 3)

**Response:**
```json
{
  "top_n": [
    {"name": "fleecy_milkcap", "confidence": 0.85},
    {"name": "common_inkcap", "confidence": 0.10}
  ]
}
```

### GET `/api/health`
Health check endpoint.

### GET `/api/docs`
FastAPI auto-generated Swagger documentation.

## Configuration

### Backend (backend.py)
- `MODEL_PATH`: Path to Keras model file
- `NAMES_PATH`: Path to class names JSON file
- `IMAGE_SIZE`: Input image dimensions (128×128)
- `MAX_TOP_N`: Maximum number of predictions

### Frontend (next.config.js)
- API rewrites for development/production
- Proxy configuration for FastAPI

## Development

The application uses concurrently to run both servers in development mode. The Next.js config includes rewrites that proxy API requests to the FastAPI backend:

- Development: `http://localhost:3000/api/*` → `http://localhost:8000/*`
- Production: API routes handled by serverless functions

## Deployment

### Vercel (Recommended)

1. Push code to GitHub
2. Import project to Vercel
3. Configure build settings:
   - Build Command: `npm run build`
   - Output Directory: `.next`
4. Add Python runtime support via `api/` directory

### Docker

```bash
docker build -t mushroom-classifier .
docker run -p 3000:3000 -p 8000:8000 mushroom-classifier
```

## Safety Warning

⚠️ **This is an AI model and may not be 100% accurate. Never consume mushrooms based solely on this identification. Always consult with a mycology expert before consuming wild mushrooms.**

## License

This project is part of the [mushroom-categorizor-model](https://github.com/WSM-simon/mushroom-categorizor-model) repository.
