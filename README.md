# SignBridge Backend

FastAPI backend for SignBridge. It handles authentication, user profiles, dictionary entries, feedback, translation history, file uploads, and AI-based sign recognition.

## Tech Stack

- FastAPI
- MongoDB / PyMongo
- TensorFlow, Keras, OpenCV, MediaPipe
- Boto3 for R2/S3-compatible storage
- Python dotenv and Passlib

## Prerequisites

- Python 3.10+ recommended
- A running MongoDB database
- Cloudflare R2 or another S3-compatible bucket for profile images
- Trained model files in `model_dev/saved_models/`

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in this folder with the required values.

## Environment Variables

At minimum, the backend expects:

- `MONGODB_URI`
- `FRONTEND_URL`
- `FRONTEND_VERCEL_URL`
- R2 / S3 credentials used by `app/services/r2_storage.py`

## Run

Start the API from the backend folder:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Or run the module directly:

```bash
python -m app.main
```

## Main Features

- User registration and login
- Profile image upload and storage
- Dictionary CRUD
- Feedback submission and moderation
- Translation history
- Real-time sign recognition endpoints for alphabet, number, and word models

## Project Structure

- `app/main.py` - FastAPI app entry point
- `app/controllers/` - Request handling and business logic
- `app/routes/` - API routes
- `app/models/` - Pydantic models
- `app/core/database.py` - MongoDB connection helpers
- `app/services/r2_storage.py` - Storage helpers
- `model_dev/saved_models/` - Trained AI model artifacts

## Notes

- The backend sets TensorFlow / MediaPipe environment flags in `app/main.py` before importing AI modules.
- Keep the saved model files in place, otherwise recognition endpoints will fail during startup.
