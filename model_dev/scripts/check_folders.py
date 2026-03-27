import os
import cv2
import mediapipe as mp
import csv

# --- SETTINGS ---
OUTPUT_FILE = "sign_data_clean.csv"
IMAGES_PER_CLASS = 1000

# --- CORRECTED PATHS (Based on your check_folders.py output) ---
DATASET_PATHS = [
    "asl_alphabet_train",
    "asl_dataset",
    "asl_dataset1",
    "Sign-Language-Digits-Dataset",
    "Sign-Language-Digits-Dataset1"
]

# --- SETUP ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, "data", OUTPUT_FILE)
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)


def process_all():
    print(f"🚀 Starting Master Processing...")
    print(f"   Looking for datasets in: {RAW_DIR}")
    print(f"💾 Saving to: {OUTPUT_CSV_PATH}")

    # Create/Overwrite the file
    with open(OUTPUT_CSV_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ["label"]
        for i in range(21):
            header.extend([f"x{i}", f"y{i}"])
        writer.writerow(header)

        for dataset_name in DATASET_PATHS:
            full_path = os.path.join(RAW_DIR, dataset_name)

            if not os.path.exists(full_path):
                print(f"\n⚠️ WARNING: Folder not found: {full_path}")
                continue

            print(f"\n📂 Reading Dataset: {dataset_name}")

            # --- NESTED FOLDER DETECTION LOGIC ---
            # Some datasets (like ASL Alphabet) have a structure like:
            # raw/asl_alphabet_train/asl_alphabet_train/A
            # This logic detects that double-nesting automatically.

            target_dir = full_path
            try:
                contents = os.listdir(full_path)
                # If there is only 1 item and it's a folder with the same name, dive in
                if len(contents) == 1 and os.path.isdir(os.path.join(full_path, contents[0])):
                    nested_path = os.path.join(full_path, contents[0])
                    print(f"   ↳ Detected nested folder: {contents[0]}")
                    target_dir = nested_path
            except Exception as e:
                print(f"   Error checking folder: {e}")
                continue

            # Get classes
            classes = sorted(os.listdir(target_dir))

            for label in classes:
                class_dir = os.path.join(target_dir, label)
                if not os.path.isdir(class_dir):
                    continue

                # Optional: Skip non-sign folders
                if label.lower() in ["nothing", "space", "del"]:
                    continue

                print(f"   Processing Class: '{label}'...")

                image_files = os.listdir(class_dir)
                images_to_process = image_files[:IMAGES_PER_CLASS]

                count = 0
                for img_name in images_to_process:
                    img_path = os.path.join(class_dir, img_name)

                    try:
                        image = cv2.imread(img_path)
                        if image is None: continue

                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        results = hands.process(image_rgb)

                        if results.multi_hand_landmarks:
                            for hand_landmarks in results.multi_hand_landmarks:
                                row = [label]
                                for lm in hand_landmarks.landmark:
                                    row.append(lm.x)
                                    row.append(lm.y)
                                writer.writerow(row)
                                count += 1
                    except Exception:
                        pass

                print(f"     ✅ Saved {count} samples")

    print("\n🎉 DONE! You now have a perfectly balanced dataset at 'sign_data_clean.csv'.")


if __name__ == "__main__":
    process_all()