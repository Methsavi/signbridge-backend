import os
import sys

# --- MANDATORY ENVIRONMENT FIXES (Must be the absolute first lines) ---
# This forces the Protobuf library to use the pure-python implementation
# which is the only version compatible with MediaPipe on Windows.
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import uvicorn
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

# Local imports must happen AFTER the environment variables are set
from app.core.database import connect_to_mongodb, close_mongodb_connection
from app.routes.user_routes import router as user_router
from app.routes.recognition_routes import router as recognition_router
from app.controllers.recognition_controller import load_ai_models
from app.routes.feature_routes import router as feature_router
from app.routes.feedback_routes import router as feedback_router
from app.services.r2_storage import ensure_profile_image_directory

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    print("🚀 Starting SignBridge Backend...")
    connect_to_mongodb()
    ensure_profile_image_directory()

    # Initialize MediaPipe and TensorFlow
    load_ai_models()

    yield
    # --- SHUTDOWN ---
    print("🛑 Shutting down SignBridge Backend...")
    close_mongodb_connection()


app = FastAPI(title="SignBridge AI", lifespan=lifespan)

# CORS Config
# Note: allow_origins=["*"] cannot be used with allow_credentials=True (browsers reject it).
# Explicitly list all allowed origins instead.
FRONTEND_VERCEL_URL = os.getenv("FRONTEND_VERCEL_URL", "")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

allowed_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5174",
]
if FRONTEND_URL and FRONTEND_URL not in allowed_origins:
    allowed_origins.append(FRONTEND_URL)
if FRONTEND_VERCEL_URL and FRONTEND_VERCEL_URL not in allowed_origins:
    allowed_origins.append(FRONTEND_VERCEL_URL)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routes
app.include_router(user_router)
app.include_router(recognition_router)
app.include_router(feature_router)
app.include_router(feedback_router)


@app.get("/")
def root():
    return {"message": "SignBridge API is running!"}


if __name__ == "__main__":
    # reload=False ensures the environment state remains stable on Windows
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)