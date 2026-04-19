import os

# --- ENV FIXES ---
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import tensorflow as tf
import joblib
import base64
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Alphabet
ALPHABET_MODEL = os.path.join(BASE_DIR, "model_dev", "saved_models", "alphabet", "best_alphabet_model.keras")
ALPHABET_SCALER = os.path.join(BASE_DIR, "model_dev", "saved_models", "alphabet", "alphabet_scaler.pkl")
ALPHABET_ENCODER = os.path.join(BASE_DIR, "model_dev", "saved_models", "alphabet", "alphabet_encoder.pkl")

# Number
NUMBER_MODEL = os.path.join(BASE_DIR, "model_dev", "saved_models", "number", "best_number_model.keras")
NUMBER_SCALER = os.path.join(BASE_DIR, "model_dev", "saved_models", "number", "number_scaler.pkl")
NUMBER_ENCODER = os.path.join(BASE_DIR, "model_dev", "saved_models", "number", "number_label_encoder.pkl")
NUMBER_IMPUTER = os.path.join(BASE_DIR, "model_dev", "saved_models", "number", "number_imputer.pkl")

# Word
WORD_MODEL = os.path.join(BASE_DIR, "model_dev", "saved_models", "word", "best_wlasl10_bilstm.keras")
WORD_ENCODER = os.path.join(BASE_DIR, "model_dev", "saved_models", "word", "wlasl10_label_encoder.pkl")

# --- GLOBALS ---
alphabet_model = None
alphabet_scaler = None
alphabet_encoder = None

number_model = None
number_scaler = None
number_encoder = None
number_imputer = None

word_model = None
word_encoder = None

hands = None
pose = None

frame_buffer = []
SEQUENCE_LENGTH = 32
POSE_LANDMARKS_TO_USE = 25


def load_ai_models():
    global alphabet_model, alphabet_scaler, alphabet_encoder
    global number_model, number_scaler, number_encoder, number_imputer
    global word_model, word_encoder, hands, pose

    print("BASE_DIR:", BASE_DIR)

    print("Alphabet model:", ALPHABET_MODEL, os.path.exists(ALPHABET_MODEL))
    print("Alphabet scaler:", ALPHABET_SCALER, os.path.exists(ALPHABET_SCALER))
    print("Alphabet encoder:", ALPHABET_ENCODER, os.path.exists(ALPHABET_ENCODER))

    print("Number model:", NUMBER_MODEL, os.path.exists(NUMBER_MODEL))
    print("Number scaler:", NUMBER_SCALER, os.path.exists(NUMBER_SCALER))
    print("Number encoder:", NUMBER_ENCODER, os.path.exists(NUMBER_ENCODER))
    print("Number imputer:", NUMBER_IMPUTER, os.path.exists(NUMBER_IMPUTER))

    print("Word model:", WORD_MODEL, os.path.exists(WORD_MODEL))
    print("Word encoder:", WORD_ENCODER, os.path.exists(WORD_ENCODER))

    print("🚀 Loading ALL AI models...")

    alphabet_model = tf.keras.models.load_model(ALPHABET_MODEL)
    alphabet_scaler = joblib.load(ALPHABET_SCALER)
    alphabet_encoder = joblib.load(ALPHABET_ENCODER)

    number_model = tf.keras.models.load_model(NUMBER_MODEL)
    number_scaler = joblib.load(NUMBER_SCALER)
    number_encoder = joblib.load(NUMBER_ENCODER)
    number_imputer = joblib.load(NUMBER_IMPUTER)

    word_model = tf.keras.models.load_model(WORD_MODEL)
    word_encoder = joblib.load(WORD_ENCODER)

    import mediapipe as mp

    hands_module = mp.solutions.hands
    pose_module = mp.solutions.pose

    hands = hands_module.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    )

    pose = pose_module.Pose(
        static_image_mode=True,
        min_detection_confidence=0.5
    )

    print("✅ All models loaded successfully!")


def decode_base64_image(base64_image: str):
    if ',' in base64_image:
        base64_image = base64_image.split(',')[1]

    image_bytes = base64.b64decode(base64_image)
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return frame


# =========================
# ALPHABET / NUMBER HELPERS
# =========================
def extract_hand_xyz_features_and_visual(base64_image):
    frame = decode_base64_image(base64_image)
    if frame is None:
        return None, None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if not results.multi_hand_landmarks:
        return None, None

    raw = []
    visual = []

    for lm in results.multi_hand_landmarks[0].landmark:
        raw.extend([lm.x, lm.y, lm.z])
        visual.append({"x": lm.x, "y": lm.y})

    return np.array(raw, dtype=np.float32), visual


# =========================
# WORD HELPERS
# =========================
def extract_word_frame_features(base64_image):
    frame = decode_base64_image(base64_image)
    if frame is None:
        return None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_result = hands.process(frame_rgb)
    pose_result = pose.process(frame_rgb)

    left_hand = np.zeros((21, 2), dtype=np.float32)
    right_hand = np.zeros((21, 2), dtype=np.float32)

    if hand_result.multi_hand_landmarks and hand_result.multi_handedness:
        for hand_landmarks, handedness in zip(hand_result.multi_hand_landmarks, hand_result.multi_handedness):
            side = handedness.classification[0].label.lower()
            coords = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark], dtype=np.float32)

            if side == "left":
                left_hand = coords
            elif side == "right":
                right_hand = coords

    pose_arr = np.zeros((POSE_LANDMARKS_TO_USE, 2), dtype=np.float32)
    if pose_result.pose_landmarks:
        pose_coords = np.array(
            [[lm.x, lm.y] for lm in pose_result.pose_landmarks.landmark[:POSE_LANDMARKS_TO_USE]],
            dtype=np.float32
        )
        pose_arr[:len(pose_coords)] = pose_coords

    features = np.concatenate([
        left_hand.flatten(),   # 42
        right_hand.flatten(),  # 42
        pose_arr.flatten()     # 50
    ])  # total = 134

    return features.astype(np.float32)


def normalize_sequence(sequence):
    sequence = np.array(sequence, dtype=np.float32)
    mean = sequence.mean(axis=0, keepdims=True)
    std = sequence.std(axis=0, keepdims=True) + 1e-6
    return (sequence - mean) / std


# =========================
# PREDICTORS
# =========================
def predict_alphabet(base64_image):
    features, visual = extract_hand_xyz_features_and_visual(base64_image)

    if features is None:
        return {"sign": "...", "confidence": 0, "landmarks": None}

    features = features.reshape(1, -1)
    features = alphabet_scaler.transform(features)

    probs = alphabet_model.predict(features, verbose=0)
    idx = np.argmax(probs)

    return {
        "sign": alphabet_encoder.inverse_transform([idx])[0],
        "confidence": float(np.max(probs)),
        "landmarks": visual
    }


def predict_number(base64_image):
    features, visual = extract_hand_xyz_features_and_visual(base64_image)

    if features is None:
        return {"sign": "...", "confidence": 0, "landmarks": None}

    # Model was trained with 64 features, but live extraction gives 63.
    # Append one missing feature as NaN so the saved imputer fills it.
    features_64 = np.append(features, np.nan).reshape(1, -1)

    features_64 = number_imputer.transform(features_64)
    features_64 = number_scaler.transform(features_64)

    probs = number_model.predict(features_64, verbose=0)
    idx = np.argmax(probs)

    return {
        "sign": str(number_encoder.inverse_transform([idx])[0]),
        "confidence": float(np.max(probs)),
        "landmarks": visual
    }


def predict_word(base64_image):
    global frame_buffer

    features = extract_word_frame_features(base64_image)

    if features is None:
        frame_buffer.clear()
        return {"sign": "...", "confidence": 0, "ready": False}

    frame_buffer.append(features)

    if len(frame_buffer) > SEQUENCE_LENGTH:
        frame_buffer = frame_buffer[-SEQUENCE_LENGTH:]

    if len(frame_buffer) < SEQUENCE_LENGTH:
        return {"sign": "...", "confidence": 0, "ready": False}

    seq = normalize_sequence(frame_buffer)
    seq = seq.reshape(1, SEQUENCE_LENGTH, 134)

    probs = word_model.predict(seq, verbose=0)
    idx = np.argmax(probs)

    return {
        "sign": word_encoder.inverse_transform([idx])[0],
        "confidence": float(np.max(probs)),
        "ready": True
    }