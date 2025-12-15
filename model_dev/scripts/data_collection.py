import cv2
import mediapipe as mp
import csv
import os

# SETTINGS
# 1. Get the directory where THIS script is located (.../model_dev/scripts)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Go up one level to get the 'model_dev' folder
MODEL_DEV_DIR = os.path.dirname(SCRIPT_DIR)

# 3. Define the exact path to the data file inside model_dev/data
DATA_FILE = os.path.join(MODEL_DEV_DIR, "data", "sign_data.csv")

print(f"Data will be saved to: {DATA_FILE}")

# INITIALIZATION
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # We limit to 1 hand for the prototype to keep it simple
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def save_to_csv(label, landmarks):
    """
    Saves the normalized coordinates (x, y) to the CSV file.
    Format: Label, x0, y0, ... x20, y20
    """
    # Create the row: Label first, then 42 numbers (21 points * x,y)
    row = [label]
    for landmark in landmarks.landmark:
        row.append(landmark.x)
        row.append(landmark.y)
    
    # Check if file exists to decide if we need to write headers
    file_exists = os.path.isfile(DATA_FILE)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    
    with open(DATA_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if new file (Label, x0, y0, x1, y1...)
        if not file_exists:
            header = ["label"]
            for i in range(21):
                header.extend([f"x{i}", f"y{i}"])
            writer.writerow(header)
            
        writer.writerow(row)
    print(f"Saved sample for: {label}")

# --- MAIN LOOP ---
def main():
    # 1. Ask user for the gesture name
    target_label = input("Enter the gesture name you want to record (e.g., 'Hello'): ").strip()
    if not target_label:
        print("No label entered. Exiting.")
        return

    cap = cv2.VideoCapture(0)
    print(f"\n--- RECORDING MODE: '{target_label}' ---")
    print("Press 's' to save a frame.")
    print("Press 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            continue

        # Flip and process
        image = cv2.flip(frame, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)

        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # UI Text
        cv2.putText(image, f"Recording: {target_label}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, "Press 's' to save", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('SignBridge Data Collection', image)

        key = cv2.waitKey(1) & 0xFF
        
        # SAVE DATA
        if key == ord('s'):
            if results.multi_hand_landmarks:
                save_to_csv(target_label, results.multi_hand_landmarks[0])
            else:
                print("No hand detected! Cannot save.")

        # QUIT
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()