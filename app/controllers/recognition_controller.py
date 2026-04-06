import os

# --- MANDATORY ENVIRONMENT FIXES (Must be at the absolute top) ---
# This forces the Protobuf library to use a stable implementation compatible with MediaPipe
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import tensorflow as tf
import joblib
import base64
import warnings

# Suppress technical warnings for a cleaner console
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "model_dev", "saved_models", "signbridge_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "model_dev", "saved_models", "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "model_dev", "saved_models", "encoder.pkl")

# --- GLOBAL VARIABLES ---
model = None
scaler = None
encoder = None
hands = None  # Initialized inside load_ai_model to prevent import-time crashes


def load_ai_model():
    """
    Loads the Deep Learning model and pre-processing tools.
    Must be called when the FastAPI server starts.
    """
    global model, scaler, encoder, hands

    try:
        print("🧠 Loading SignBridge AI Engine...")

        # 1. Load the Neural Network (.h5 from Colab)
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)

        # 2. Load the processing tools (scikit-learn objects)
        scaler = joblib.load(SCALER_PATH)
        encoder = joblib.load(ENCODER_PATH)

        # 3. DELAYED IMPORT: Import MediaPipe here to ensure environment variables are ready
        import mediapipe as mp
        mp_hands = mp.solutions.hands

        # Initialize MediaPipe with static mode for individual frame processing
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )

        print(f"✅ Deep Learning model and MediaPipe loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading AI components: {e}")


def process_frame(base64_image: str):
    """
    Decodes the frame, extracts landmarks for prediction,
    and returns both the result and visual landmarks for React drawing.
    """
    global model, scaler, encoder, hands

    # Critical Check: Ensure the engine actually loaded
    if hands is None or model is None or scaler is None or encoder is None:
        return {"error": "AI Engine (MediaPipe/TensorFlow) not initialized on server"}

    try:
        # 1. Decode the Base64 image from React
        if ',' in base64_image:
            base64_image = base64_image.split(',')[1]

        image_bytes = base64.b64decode(base64_image)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"error": "Invalid image payload"}

        # 2. Extract Landmarks via MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                raw_coords = []
                visual_landmarks = []  # For drawing skeleton on the React frontend

                # Extract coordinates
                for lm in hand_landmarks.landmark:
                    # Prediction data (raw x, y for the AI model)
                    raw_coords.append(lm.x)
                    raw_coords.append(lm.y)
                    # Visual data (structured x, y objects for the React Canvas)
                    visual_landmarks.append({"x": lm.x, "y": lm.y})

                # 3. Pre-process and Predict
                # Convert landmarks to a 2D array (1 sample, 42 features)
                input_data = np.array(raw_coords).reshape(1, -1)

                # Safety check: Ensure we have exactly 42 features (21 points * x,y)
                if input_data.shape[1] != 42:
                    return {"error": "Landmark extraction inconsistent (Missing points)"}

                # Scale features to match training data
                input_scaled = scaler.transform(input_data)

                # Run inference through the Neural Network
                prediction_probs = model.predict(input_scaled, verbose=0)
                class_index = np.argmax(prediction_probs)

                # Decode index to actual character (A, B, 1, 2, etc.)
                predicted_char = encoder.inverse_transform([class_index])[0]
                confidence = float(np.max(prediction_probs))

                return {
                    "sign": str(predicted_char),
                    "confidence": round(confidence * 100, 2),
                    "landmarks": visual_landmarks  # Coordinates for drawing on React
                }

        # Return empty state if no hand is detected in frame
        return {"sign": "...", "confidence": 0, "landmarks": None}

    except Exception as e:
        print(f"Prediction logic error: {e}")
        return {"error": str(e)}