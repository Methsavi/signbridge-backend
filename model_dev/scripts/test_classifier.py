import cv2
import mediapipe as mp
import pickle
import numpy as np
import os

# --- PATHS ---
# Automatically find the path to the saved model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "sign_model.p")

# --- LOAD THE BRAIN ---
print("Loading model...")
try:
    with open(MODEL_PATH, 'rb') as f:
        model_dict = pickle.load(f)
        model = model_dict['model']
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print(f"❌ Error: Model file not found at {MODEL_PATH}")
    print("Did you run train_model.py?")
    exit()

# --- INITIALIZE CAMERA & HANDS ---
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print("Starting Camera... Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        continue

    # 1. Process Frame
    # Flip for selfie view, convert to RGB
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    # 2. If Hand Detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # --- EXTRACT FEATURES (Same format as data_collection.py) ---
            data_aux = []
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x)
                data_aux.append(landmark.y)

            # --- PREDICT ---
            # The model expects a list of lists (e.g., [[0.1, 0.2...]])
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = prediction[0]

            # --- DRAW UI ---
            # Draw the skeleton
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Draw a box and the text
            H, W, _ = frame.shape
            # Get simplified bounding box (just using min/max coords)
            x_vals = [lm.x for lm in hand_landmarks.landmark]
            y_vals = [lm.y for lm in hand_landmarks.landmark]
            x1, y1 = int(min(x_vals) * W) - 20, int(min(y_vals) * H) - 20
            x2, y2 = int(max(x_vals) * W) + 20, int(max(y_vals) * H) + 20

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)  # Black border
            cv2.putText(frame, predicted_character, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, predicted_character, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2, cv2.LINE_AA)

    # 3. Show Result
    cv2.imshow('SignBridge AI Test', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()