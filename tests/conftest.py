"""
conftest.py - Test configuration for SignBridge backend.

Mocks all heavy ML and database packages at sys.modules level so
the app can be imported in CI without TensorFlow/MediaPipe/MongoDB installed.
"""

import os
import sys
import types
from unittest.mock import MagicMock, patch
import pytest

# 1. Environment variables
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["MONGO_URI"] = "mongodb://localhost:27017"
os.environ["DB_NAME"] = "signbridge_test"


# 2. Helper: fake package with __path__ so Python treats it as a package
class _FakePackage(types.ModuleType):
    def __getattr__(self, name):
        mock = MagicMock()
        setattr(self, name, mock)
        return mock


def _fake_pkg(name):
    mod = _FakePackage(name)
    mod.__path__ = []
    mod.__package__ = name
    mod.__spec__ = MagicMock()
    return mod


# 3. Mock all heavy packages BEFORE any app code is imported
for _pkg in ["cv2", "tensorflow", "keras", "mediapipe", "joblib", "scipy",
             "sklearn", "pandas", "matplotlib", "gtts", "deep_translator",
             "pymongo", "bson", "boto3", "botocore"]:
    sys.modules[_pkg] = _fake_pkg(_pkg)

for _sub in ["tensorflow.python", "tensorflow.keras", "tensorflow.keras.layers",
             "tensorflow.keras.models", "tensorflow.keras.callbacks",
             "tensorflow.keras.optimizers", "keras.src", "keras.src.saving",
             "keras.src.saving.serialization_lib", "keras.saving",
             "keras.saving.serialization_lib", "mediapipe.python",
             "mediapipe.python.solutions", "mediapipe.python.solutions.holistic",
             "mediapipe.python.solutions.drawing_utils", "scipy.interpolate",
             "sklearn.preprocessing", "matplotlib.pyplot",
             "pymongo.errors", "botocore.config", "botocore.exceptions"]:
    sys.modules[_sub] = MagicMock()

sys.modules["bson"].ObjectId = lambda x=None: str(x) if x else "mock_object_id"

# 4. Pre-import app modules now (mocks are in place), then stub startup functions
from app.controllers import recognition_controller as _rc
_rc.load_ai_models = MagicMock()

from app.core import database as _db
_db.connect_to_mongodb = MagicMock()
_db.close_mongodb_connection = MagicMock()

from app.services import r2_storage as _r2
_r2.ensure_profile_image_directory = MagicMock()


# 5. Shared TestClient fixture for all tests
@pytest.fixture(scope="session")
def client():
    """FastAPI TestClient - startup functions already replaced above."""
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as test_client:
        yield test_client
