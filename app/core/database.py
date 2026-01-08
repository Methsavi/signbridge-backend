import os
from pathlib import Path
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv

# --- LOAD ENV ---
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

MONGODB_URI = os.getenv("MONGODB_URI")

client = None
db = None


def connect_to_mongodb():
    global client, db
    try:
        if not MONGODB_URI:
            print("❌ Error: MONGODB_URI is missing.")
            return None

        print(f"🔄 Connecting to Database: {MONGODB_URI}")

        # Simple connection for Localhost
        client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=2000
        )

        # Test connection
        client.admin.command('ping')

        # Get DB name from URI or default
        db_name = MONGODB_URI.split("/")[-1] or "signbridge_db"
        db = client[db_name]

        print("✅ Connected to Local MongoDB successfully")
        return db
    except ConnectionFailure as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        print("💡 HINT: Is MongoDB Community Server running?")
        print("   Open Task Manager -> Services -> Search for 'MongoDB'")
        raise


def get_database():
    global db
    if db is None:
        connect_to_mongodb()
    return db


def close_mongodb_connection():
    global client
    if client:
        client.close()
        print("✅ MongoDB connection closed")