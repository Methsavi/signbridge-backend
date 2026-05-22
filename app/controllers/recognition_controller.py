import os
import time

# --- ENV FIXES ---
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import base64
import warnings
from collections import deque

# --- OPTIONAL HEAVY ML IMPORTS ---
# Wrapped in try/except so the app can start on servers (Azure) that don't
# have GPU drivers or display libraries. All recognition routes already return
# "recognition_unavailable" when these are None, so non-ML endpoints are
# completely unaffected.
try:
    import cv2
    import numpy as np
    import tensorflow as tf
    import joblib
    from scipy.interpolate import interp1d

    # Monkeypatch Keras deserialization to remove unsupported
    # 'quantization_config' entries which cause errors when a model was saved
    # with quantization metadata.
    try:
        import keras.src.saving.serialization_lib as _serialization_lib
    except Exception:
        import keras.saving.serialization_lib as _serialization_lib

    _original_deserialize = getattr(_serialization_lib, 'deserialize_keras_object', None)

    def _strip_quantization(obj):
        if isinstance(obj, dict):
            return {k: _strip_quantization(v) for k, v in obj.items() if k != 'quantization_config'}
        if isinstance(obj, list):
            return [_strip_quantization(v) for v in obj]
        return obj

    def _deserialize_wrapper(config, *args, **kwargs):
        try:
            cleaned = _strip_quantization(config)
        except Exception:
            cleaned = config
        return _original_deserialize(cleaned, *args, **kwargs)

    if _original_deserialize is not None:
        _serialization_lib.deserialize_keras_object = _deserialize_wrapper

    from tensorflow.keras.layers import Dense as _KerasDense
    class DenseCompat(_KerasDense):
        def __init__(self, *args, quantization_config=None, **kwargs):
            super().__init__(*args, **kwargs)

    _CUSTOM_OBJECTS = {
        'Dense': DenseCompat,
        'keras.layers.Dense': DenseCompat,
        'tensorflow.keras.layers.Dense': DenseCompat,
    }

    _ML_AVAILABLE = True
    warnings.filterwarnings("ignore", category=UserWarning)

except Exception as _ml_import_error:
    print(f"[recognition_controller] ML libraries unavailable: {_ml_import_error}")
    print("[recognition_controller] Recognition features disabled — all other endpoints unaffected.")
    cv2 = None
    np = None
    tf = None
    joblib = None
    interp1d = None
    _CUSTOM_OBJECTS = {}
    _ML_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Alphabet
ALPHABET_MODEL   = os.path.join(BASE_DIR, "model_dev", "saved_models", "alphabet", "best_alphabet_model.keras")
ALPHABET_SCALER  = os.path.join(BASE_DIR, "model_dev", "saved_models", "alphabet", "alphabet_scaler.pkl")
ALPHABET_ENCODER = os.path.join(BASE_DIR, "model_dev", "saved_models", "alphabet", "alphabet_encoder.pkl")

# Number
NUMBER_MODEL   = os.path.join(BASE_DIR, "model_dev", "saved_models", "number", "best_number_model.keras")
NUMBER_SCALER  = os.path.join(BASE_DIR, "model_dev", "saved_models", "number", "number_scaler.pkl")
NUMBER_ENCODER = os.path.join(BASE_DIR, "model_dev", "saved_models", "number", "number_encoder.pkl")

# Word (new Transformer model)
WORD_MODEL   = os.path.join(BASE_DIR, "model_dev", "saved_models", "word", "best_word_model.keras")
WORD_ENCODER = os.path.join(BASE_DIR, "model_dev", "saved_models", "word", "word_encoder.pkl")
WORD_SCALER  = os.path.join(BASE_DIR, "model_dev", "saved_models", "word", "word_scaler.pkl")

# ─────────────────────────────────────────────────────────────────────
# THRESHOLDS & TIMING
# ─────────────────────────────────────────────────────────────────────
ALPHABET_THRESHOLD = 0.75
NUMBER_THRESHOLD   = 0.80
WORD_THRESHOLD     = 0.40   # Lower threshold — show top3 even when less confident

HOLD_DURATION  = 0.6   # seconds user must hold a sign before commit
COOLDOWN_AFTER = 3.0   # seconds before same sign can commit again

# Word model config — must match training
WORD_SEQ_LEN   = 30
WORD_FEAT_DIM  = 126   # 21 landmarks × 2 hands × 3 (x,y,z)
WORD_MIN_FRAMES = 2

# ─────────────────────────────────────────────────────────────────────
# GLOBALS
# ─────────────────────────────────────────────────────────────────────
alphabet_model   = None
alphabet_scaler  = None
alphabet_encoder = None

number_model   = None
number_scaler  = None
number_encoder = None

word_model   = None
word_encoder = None
word_scaler  = None

hands    = None   # MediaPipe Hands (alphabet + number)
holistic = None   # MediaPipe Holistic (word — captures both hands)
pose     = None   # MediaPipe Pose (kept for compatibility)

# Word frame collection state
_word_frame_buffer  = []
_word_is_collecting = False


# ─────────────────────────────────────────────────────────────────────
# SMOOTHER & VOTER  (alphabet + number)
# ─────────────────────────────────────────────────────────────────────
class LandmarkSmoother:
    """Temporal average over last N frames — kills MediaPipe jitter."""
    def __init__(self, window=5):
        self.buf = deque(maxlen=window)

    def smooth(self, landmarks_flat):
        self.buf.append(landmarks_flat)
        return np.mean(self.buf, axis=0)

    def reset(self):
        self.buf.clear()


class PredictionVoter:
    """Weighted majority vote — stops predictions flickering every frame."""
    def __init__(self, window=7):
        self.buf = deque(maxlen=window)

    def vote(self, prediction, confidence):
        self.buf.append((prediction, confidence))
        if len(self.buf) < 3:
            return None, 0.0
        votes = {}
        for pred, conf in self.buf:
            votes[pred] = votes.get(pred, 0) + conf
        best     = max(votes, key=votes.get)
        avg_conf = votes[best] / len(self.buf)
        return best, avg_conf

    def reset(self):
        self.buf.clear()


_alphabet_smoother = LandmarkSmoother(window=5)
_alphabet_voter    = PredictionVoter(window=7)
_number_smoother   = LandmarkSmoother(window=5)
_number_voter      = PredictionVoter(window=7)

# Alphabet hold/commit state
_last_committed_sign = None
_last_committed_time = 0.0
_sign_hold_start     = 0.0
_current_held_sign   = None

# Number hold/commit state
_num_last_committed_sign = None
_num_last_committed_time = 0.0
_num_sign_hold_start     = 0.0
_num_current_held_sign   = None


# ─────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────
def load_ai_models():
    global alphabet_model, alphabet_scaler, alphabet_encoder
    global number_model, number_scaler, number_encoder
    global word_model, word_encoder, word_scaler
    global hands, holistic, pose

    if not _ML_AVAILABLE:
        print("[recognition_controller] ML libraries not available — skipping model load.")
        print("[recognition_controller] Recognition WebSocket endpoints will return 'recognition_unavailable'.")
        return

    print("BASE_DIR:", BASE_DIR)
    print("Loading ALL AI models...")

    # \u2500\u2500 Alphabet \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n    print("\n--- Alphabet ---")
    print("  model  :", ALPHABET_MODEL,   "| exists:", os.path.exists(ALPHABET_MODEL))
    print("  scaler :", ALPHABET_SCALER,  "| exists:", os.path.exists(ALPHABET_SCALER))
    print("  encoder:", ALPHABET_ENCODER, "| exists:", os.path.exists(ALPHABET_ENCODER))
    alphabet_model   = tf.keras.models.load_model(ALPHABET_MODEL, custom_objects=_CUSTOM_OBJECTS)
    alphabet_scaler  = joblib.load(ALPHABET_SCALER)
    alphabet_encoder = joblib.load(ALPHABET_ENCODER)
    print("  Alphabet model loaded")

    # ── Number ────────────────────────────────────────────────────────
    print("\n--- Number ---")
    print("  model  :", NUMBER_MODEL,   "| exists:", os.path.exists(NUMBER_MODEL))
    print("  scaler :", NUMBER_SCALER,  "| exists:", os.path.exists(NUMBER_SCALER))
    print("  encoder:", NUMBER_ENCODER, "| exists:", os.path.exists(NUMBER_ENCODER))
    if os.path.exists(NUMBER_MODEL):
        number_model   = tf.keras.models.load_model(NUMBER_MODEL, custom_objects=_CUSTOM_OBJECTS)
        number_scaler  = joblib.load(NUMBER_SCALER)
        number_encoder = joblib.load(NUMBER_ENCODER)
        print("  Number model loaded")
    else:
        print("  Number model not found — skipping")

    # ── Word (new Transformer) ────────────────────────────────────────
    print("\n--- Word ---")
    print("  model  :", WORD_MODEL,   "| exists:", os.path.exists(WORD_MODEL))
    print("  encoder:", WORD_ENCODER, "| exists:", os.path.exists(WORD_ENCODER))
    print("  scaler :", WORD_SCALER,  "| exists:", os.path.exists(WORD_SCALER))
    if os.path.exists(WORD_MODEL):
        word_model   = tf.keras.models.load_model(WORD_MODEL, custom_objects=_CUSTOM_OBJECTS)
        word_encoder = joblib.load(WORD_ENCODER)
        word_scaler  = joblib.load(WORD_SCALER)
        print("  Word model loaded (Transformer)")
    else:
        print("  Word model not found — skipping")

    # ── MediaPipe HandLandmarker (Tasks API) ──────────────────────────
    HAND_TASK = os.path.join(BASE_DIR, "model_dev", "hand_landmarker.task")
    if not os.path.exists(HAND_TASK):
        print(f"  hand_landmarker.task not found at {HAND_TASK}")
        print("  Recognition features will be disabled.")
        hands = None
        holistic = None
        pose = None
    else:
        from mediapipe.tasks import python as _mp_tasks
        from mediapipe.tasks.python import vision as _mp_vision

        _base_opts_single = _mp_tasks.BaseOptions(model_asset_path=HAND_TASK)
        _single_opts = _mp_vision.HandLandmarkerOptions(
            base_options=_base_opts_single,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        hands = _mp_vision.HandLandmarker.create_from_options(_single_opts)

        _base_opts_dual = _mp_tasks.BaseOptions(model_asset_path=HAND_TASK)
        _dual_opts = _mp_vision.HandLandmarkerOptions(
            base_options=_base_opts_dual,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        holistic = _mp_vision.HandLandmarker.create_from_options(_dual_opts)
        pose = None
        print("  MediaPipe HandLandmarker initialised (Tasks API)")

    print("\nAll models loaded successfully!")


# ─────────────────────────────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────────────────────────────
def decode_base64_image(base64_image: str):
    if ',' in base64_image:
        base64_image = base64_image.split(',')[1]
    image_bytes = base64.b64decode(base64_image)
    nparr       = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def normalize_landmarks(hand_landmarks):
    """
    Scale + position invariant landmark features.
    1. Extract 21 (x,y,z) → shape (21,3)
    2. Center at wrist (index 0)
    3. Scale by wrist→middle-MCP distance (index 9)
    4. Flatten to 63 features

    hand_landmarks: List[NormalizedLandmark] from Tasks API HandLandmarker
    """
    lm  = np.array([[l.x, l.y, l.z] for l in hand_landmarks])
    lm -= lm[0]
    ref = np.linalg.norm(lm[9])
    if ref > 1e-6:
        lm /= ref
    return lm.flatten()


def extract_visual_landmarks(hand_landmarks):
    """Raw x,y for frontend skeleton drawing (un-normalized screen coords)."""
    return [{"x": lm.x, "y": lm.y} for lm in hand_landmarks]


def _apply_hold_cooldown(voted_label, voted_conf,
                         current_held, hold_start,
                         last_committed, last_committed_time,
                         voter):
    """
    Shared hold + cooldown logic for alphabet and number.
    User must hold a sign for HOLD_DURATION seconds to commit it.
    Same sign needs COOLDOWN_AFTER seconds before it can commit again.
    """
    now       = time.time()
    committed = False

    if voted_label != current_held:
        current_held = voted_label
        hold_start   = now

    held_for      = now - hold_start
    hold_progress = min(held_for / HOLD_DURATION, 1.0)

    if held_for >= HOLD_DURATION:
        since_last = now - last_committed_time
        same_sign  = (voted_label == last_committed)

        if not same_sign or since_last >= COOLDOWN_AFTER:
            committed           = True
            last_committed      = voted_label
            last_committed_time = now
            hold_start          = now
            voter.reset()

    return (committed, hold_progress,
            current_held, hold_start,
            last_committed, last_committed_time)


# ─────────────────────────────────────────────────────────────────────
# ALPHABET PREDICTOR
# ─────────────────────────────────────────────────────────────────────
def predict_alphabet(base64_image: str) -> dict:
    """
    WebSocket handler for /ws/predict/alphabet.

    Returns:
      sign          — current best prediction ("..." if uncertain)
      confidence    — float
      landmarks     — list of {x,y} for frontend skeleton
      committed     — bool: True = append this letter to input text
      hold_progress — float 0.0→1.0 for optional hold progress bar
    """
    global _last_committed_sign, _last_committed_time
    global _sign_hold_start, _current_held_sign

    # If MediaPipe or the alphabet model is not available, return an informative response
    if hands is None or alphabet_model is None or alphabet_scaler is None or alphabet_encoder is None:
        return {"sign": "...", "confidence": 0.0, "landmarks": None,
                "committed": False, "hold_progress": 0.0,
                "error": "recognition_unavailable", "reason": "mediapipe or alphabet model missing"}

    frame = decode_base64_image(base64_image)
    if frame is None:
        return {"sign": "...", "confidence": 0.0, "landmarks": None,
                "committed": False, "hold_progress": 0.0}

    # Convert to MediaPipe Image
    from mediapipe.tasks.python.vision.core import image as image_lib
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = image_lib.Image(image_format=image_lib.ImageFormat.SRGB, data=rgb_frame)

    result = hands.detect(mp_image)

    if not result.hand_landmarks or len(result.hand_landmarks) == 0:
        _alphabet_smoother.reset()
        _alphabet_voter.reset()
        _current_held_sign = None
        _sign_hold_start   = 0.0
        return {"sign": "...", "confidence": 0.0, "landmarks": None,
                "committed": False, "hold_progress": 0.0}

    hand_lm = result.hand_landmarks[0]
    visual  = extract_visual_landmarks(hand_lm)

    norm     = normalize_landmarks(hand_lm)
    smoothed = _alphabet_smoother.smooth(norm)
    scaled   = alphabet_scaler.transform(smoothed.reshape(1, -1))
    probs    = alphabet_model.predict(scaled, verbose=0)[0]

    confidence = float(np.max(probs))
    raw_label  = alphabet_encoder.inverse_transform([int(np.argmax(probs))])[0]
    voted_label, voted_conf = _alphabet_voter.vote(raw_label, confidence)

    if voted_label is None or voted_conf < ALPHABET_THRESHOLD:
        return {"sign": "...", "confidence": round(confidence, 4),
                "landmarks": visual, "committed": False, "hold_progress": 0.0}

    (committed, hold_progress,
     _current_held_sign, _sign_hold_start,
     _last_committed_sign, _last_committed_time) = _apply_hold_cooldown(
        voted_label, voted_conf,
        _current_held_sign, _sign_hold_start,
        _last_committed_sign, _last_committed_time,
        _alphabet_voter
    )

    return {
        "sign":          voted_label,
        "confidence":    round(voted_conf, 4),
        "landmarks":     visual,
        "committed":     committed,
        "hold_progress": round(hold_progress, 2)
    }


# ─────────────────────────────────────────────────────────────────────
# NUMBER PREDICTOR
# ─────────────────────────────────────────────────────────────────────
def predict_number(base64_image: str) -> dict:
    """
    WebSocket handler for /ws/predict/number.

    Returns:
      sign, confidence, landmarks, committed, hold_progress
    """
    global _num_last_committed_sign, _num_last_committed_time
    global _num_sign_hold_start, _num_current_held_sign

    # If MediaPipe or number model isn't available, return informative response
    if hands is None or number_model is None or number_scaler is None or number_encoder is None:
        return {"sign": "...", "confidence": 0.0, "landmarks": None,
                "committed": False, "hold_progress": 0.0,
                "error": "recognition_unavailable", "reason": "mediapipe or number model missing"}

    if number_model is None:
        return {"sign": "...", "confidence": 0.0, "landmarks": None,
                "committed": False, "hold_progress": 0.0}

    frame = decode_base64_image(base64_image)
    if frame is None:
        return {"sign": "...", "confidence": 0.0, "landmarks": None,
                "committed": False, "hold_progress": 0.0}

    # Convert to MediaPipe Image
    from mediapipe.tasks.python.vision.core import image as image_lib
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = image_lib.Image(image_format=image_lib.ImageFormat.SRGB, data=rgb_frame)

    result = hands.detect(mp_image)

    if not result.hand_landmarks or len(result.hand_landmarks) == 0:
        _number_smoother.reset()
        _number_voter.reset()
        _num_current_held_sign = None
        _num_sign_hold_start   = 0.0
        return {"sign": "...", "confidence": 0.0, "landmarks": None,
                "committed": False, "hold_progress": 0.0}

    hand_lm = result.hand_landmarks[0]
    visual  = extract_visual_landmarks(hand_lm)

    norm     = normalize_landmarks(hand_lm)
    smoothed = _number_smoother.smooth(norm)
    scaled   = number_scaler.transform(smoothed.reshape(1, -1))
    probs    = number_model.predict(scaled, verbose=0)[0]

    confidence = float(np.max(probs))
    raw_label  = str(number_encoder.inverse_transform([int(np.argmax(probs))])[0])
    voted_label, voted_conf = _number_voter.vote(raw_label, confidence)

    if voted_label is None or voted_conf < NUMBER_THRESHOLD:
        return {"sign": "...", "confidence": round(confidence, 4),
                "landmarks": visual, "committed": False, "hold_progress": 0.0}

    (committed, hold_progress,
     _num_current_held_sign, _num_sign_hold_start,
     _num_last_committed_sign, _num_last_committed_time) = _apply_hold_cooldown(
        voted_label, voted_conf,
        _num_current_held_sign, _num_sign_hold_start,
        _num_last_committed_sign, _num_last_committed_time,
        _number_voter
    )

    return {
        "sign":          voted_label,
        "confidence":    round(voted_conf, 4),
        "landmarks":     visual,
        "committed":     committed,
        "hold_progress": round(hold_progress, 2)
    }


# ─────────────────────────────────────────────────────────────────────
# WORD PREDICTOR — Transformer + MediaPipe Holistic
# ─────────────────────────────────────────────────────────────────────
def _extract_holistic_features(frame) -> np.ndarray:
    """
    Extract normalized both-hand features using HandLandmarker (Tasks API, 2 hands).
    Returns (126,) feature vector — 21 landmarks × 2 hands × 3 (x,y,z).
    Missing hand → zeros (model learned this means hand not visible).

    Handedness 'Left'/'Right' is from the camera's mirrored perspective:
    Tasks API reports 'Left' when the hand appears on the left side of the
    image (which is the person's right hand).  The convention here matches
    whatever was used during training data collection.
    """
    from mediapipe.tasks.python.vision.core import image as image_lib

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = image_lib.Image(image_format=image_lib.ImageFormat.SRGB, data=rgb_frame)
    result = holistic.detect(mp_image)

    left  = np.zeros((21, 3), dtype=np.float32)
    right = np.zeros((21, 3), dtype=np.float32)

    for i, hand_lms in enumerate(result.hand_landmarks):
        if i >= len(result.handedness):
            break
        category = result.handedness[i][0].category_name  # 'Left' or 'Right'
        arr = np.array([[l.x, l.y, l.z] for l in hand_lms], dtype=np.float32)
        arr -= arr[0]
        ref = np.linalg.norm(arr[9])
        if ref > 1e-6:
            arr /= ref
        if category == 'Left':
            left = arr
        else:
            right = arr

    return np.concatenate([left.flatten(), right.flatten()])  # (126,)


def _resample_sequence(seq, target_len=WORD_SEQ_LEN) -> np.ndarray:
    """
    Resample variable-length frame sequence to fixed length.
    Uses linear interpolation — same method used during training.
    """
    seq = np.array(seq, dtype=np.float32)
    if len(seq) == 0:
        return np.zeros((target_len, WORD_FEAT_DIM), dtype=np.float32)
    if len(seq) == target_len:
        return seq
    if len(seq) == 1:
        return np.tile(seq[0], (target_len, 1))

    x_old     = np.linspace(0, 1, len(seq))
    x_new     = np.linspace(0, 1, target_len)
    resampled = np.zeros((target_len, WORD_FEAT_DIM), dtype=np.float32)
    for i in range(WORD_FEAT_DIM):
        resampled[:, i] = interp1d(x_old, seq[:, i], kind='linear')(x_new)
    return resampled


def _run_word_prediction() -> dict:
    """
    Run Transformer model on buffered frames.
    Returns top-3 predictions always, sets ready=True above threshold.
    """
    seq    = _resample_sequence(_word_frame_buffer, WORD_SEQ_LEN)  # (30, 126)
    flat   = seq.reshape(WORD_SEQ_LEN, WORD_FEAT_DIM)
    scaled = word_scaler.transform(flat)
    seq_sc = scaled.reshape(1, WORD_SEQ_LEN, WORD_FEAT_DIM)

    probs = word_model.predict(seq_sc, verbose=0)[0]

    # Always return top-3 so frontend can show suggestions
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3 = [
        {
            "word":       word_encoder.inverse_transform([int(i)])[0],
            "confidence": round(float(probs[i]), 4)
        }
        for i in top3_idx
    ]

    best_conf = top3[0]["confidence"]
    best_word = top3[0]["word"]

    return {
        "sign":       best_word,
        "confidence": best_conf,
        "ready":      best_conf >= WORD_THRESHOLD,
        "top3":       top3       # frontend shows these as suggestion buttons
    }


def predict_word(base64_image: str) -> dict:
    """
    WebSocket handler for /ws/predict/word.

    Flow:
      - While hand is visible: collect frames into buffer
      - When hand disappears (sign complete): run prediction
      - Returns top-3 word suggestions so frontend can show buttons

    Returns:
      sign       — best predicted word ("..." while collecting)
      confidence — float
      ready      — bool: True = prediction complete, show result
      top3       — list of {word, confidence} for suggestion UI
      collecting — bool: True = currently recording a sign
      frames     — int: frames collected so far
    """
    global _word_frame_buffer, _word_is_collecting

    if word_model is None or holistic is None:
        return {"sign": "...", "confidence": 0.0, "ready": False,
                "top3": [], "collecting": False, "frames": 0,
                "error": "recognition_unavailable"}

    frame = decode_base64_image(base64_image)
    if frame is None:
        _word_frame_buffer.clear()
        _word_is_collecting = False
        return {"sign": "...", "confidence": 0.0, "ready": False,
                "top3": [], "collecting": False, "frames": 0}

    features    = _extract_holistic_features(frame)
    hand_present = np.any(features != 0)   # zeros = no hand detected

    if hand_present:
        # ── Collecting phase ──────────────────────────────────────────
        _word_is_collecting = True
        _word_frame_buffer.append(features)

        # Rolling window cap — keep most recent frames for long signs (~13s at 15fps)
        if len(_word_frame_buffer) > WORD_SEQ_LEN * 7:
            _word_frame_buffer = _word_frame_buffer[-(WORD_SEQ_LEN * 5):]

        return {
            "sign":       "...",
            "confidence": 0.0,
            "ready":      False,
            "top3":       [],
            "collecting": True,
            "frames":     len(_word_frame_buffer)
        }

    else:
        # ── Hand gone — predict if we have enough frames ───────────────
        if _word_is_collecting and len(_word_frame_buffer) >= WORD_MIN_FRAMES:
            result = _run_word_prediction()
            _word_frame_buffer.clear()
            _word_is_collecting = False
            return result

        # Not enough frames or wasn't collecting — reset silently
        _word_frame_buffer.clear()
        _word_is_collecting = False
        return {
            "sign":       "...",
            "confidence": 0.0,
            "ready":      False,
            "top3":       [],
            "collecting": False,
            "frames":     0
        }

