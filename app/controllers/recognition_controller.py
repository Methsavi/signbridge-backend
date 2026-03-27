import pickle
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd  # <--- NEW: Import pandas
import base64
import os
import warnings

# Suppress the Google Protobuf warnings to clean up the console
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "model_dev", "saved_models", "sign_model.p")

# --- GLOBAL VARIABLES ---
model = None
mp_hands = mp.solutions.hands
hands = None
feature_columns = []  # To store the names [x0, y0, x1, y1...]


def load_ai_model():
    """Loads the trained model and initializes MediaPipe"""
    global model, hands, feature_columns

    # 1. Load Scikit-Learn Model
    try:
        with open(MODEL_PATH, 'rb') as f:
            model_dict = pickle.load(f)
            model = model_dict['model']
        print(f"✅ AI Model loaded from: {MODEL_PATH}")

        # 2. Re-create the column names exactly as they were during training
        # The pattern was x0, y0, x1, y1 ... x20, y20
        feature_columns = []
        for i in range(21):
            feature_columns.extend([f"x{i}", f"y{i}"])

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None

    # 3. Initialize MediaPipe
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )


def process_frame(base64_image: str):

    global model, hands, feature_columns

    if model is None:
        return {"error": "Model not loaded"}

    try:
        # 1. Decode Base64
        if ',' in base64_image:
            base64_image = base64_image.split(',')[1]

        image_bytes = base64.b64decode(base64_image)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"error": "Invalid image"}

        # 2. Extract Landmarks
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x)
                    data_aux.append(lm.y)

                # --- FIX: CREATE DATAFRAME WITH NAMES ---
                # We convert the raw list into a DataFrame with column names
                # This stops the "UserWarning: X does not have valid feature names"
                df = pd.DataFrame([data_aux], columns=feature_columns)

                # 3. Predict
                prediction = model.predict(df)
                predicted_character = prediction[0]

                # Get confidence
                probs = model.predict_proba(df)
                confidence = float(np.max(probs))

                return {
                    "sign": predicted_character,
                    "confidence": round(confidence * 100, 2)
                }

        return {"sign": "...", "confidence": 0}

    except Exception as e:
        # Only print real errors, not connection closed messages
        return {"error": str(e)}