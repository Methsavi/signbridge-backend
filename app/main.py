import os
import uvicorn
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.core.database import connect_to_mongodb, close_mongodb_connection
from app.routes.user_routes import router as user_router
# Import the new AI components
from app.routes.recognition_routes import router as recognition_router
from app.controllers.recognition_controller import load_ai_model
from app.routes.feature_routes import router as feature_router

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    connect_to_mongodb()
    load_ai_model() # <--- Load the AI Brain here!
    yield
    # --- SHUTDOWN ---
    close_mongodb_connection()

app = FastAPI(title="SignBridge API", lifespan=lifespan)

# CORS Config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routes
app.include_router(user_router)
app.include_router(recognition_router) # <--- Add the WebSocket route
app.include_router(feature_router)

@app.get("/")
def root():
    return {"message": "SignBridge API is running!"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)