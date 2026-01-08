import os
import cv2
import mediapipe as mp
import csv
import time

# --- SETTINGS ---
# Process ALL images (Heavy workload!)
IMAGES_PER_CLASS_LIMIT = 3000

# --- PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # .../model_dev/scripts
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # .../model_dev

# Path to: model_dev/data/raw/asl_alphabet_train/asl_alphabet_train
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "asl_alphabet_train", "asl_alphabet_train")

# Path to: model_dev/data/sign_data.csv
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "sign_data.csv")

# --- INITIALIZE MEDIAPIPE ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,  # Must be True for images
    max_num_hands=1,
    min_detection_confidence=0.5
)


def process_dataset():
    start_time = time.time()

    # 1. Verify Dataset Path
    if not os.path.exists(RAW_DATA_DIR):
        print(f"❌ Error: Could not find dataset at: {RAW_DATA_DIR}")
        return

    print(f"🚀 Starting FULL processing... Limit: {IMAGES_PER_CLASS_LIMIT} images per class.")
    print(f"📂 Reading from: {RAW_DATA_DIR}")
    print(f"💾 Saving to: {OUTPUT_CSV}")
    print("☕ This will take a long time. Go grab a coffee!")

    # 2. Prepare CSV file
    file_exists = os.path.isfile(OUTPUT_CSV)

    # Open CSV in Append Mode ('a')
    with open(OUTPUT_CSV, mode='a', newline='') as f:
        writer = csv.writer(f)

        # Write header if new file
        if not file_exists:
            header = ["label"]
            for i in range(21):
                header.extend([f"x{i}", f"y{i}"])
            writer.writerow(header)

        # 3. Get sorted list of classes (A, B, C...)
        classes = sorted(os.listdir(RAW_DATA_DIR))

        # 4. Loop through each Class
        for label in classes:
            class_dir = os.path.join(RAW_DATA_DIR, label)

            if not os.path.isdir(class_dir):
                continue

            print(f"📂 Processing Class: '{label}'...")

            image_files = os.listdir(class_dir)
            images_to_process = image_files[:IMAGES_PER_CLASS_LIMIT]

            count = 0
            for img_name in images_to_process:
                img_path = os.path.join(class_dir, img_name)

                # Read Image
                image = cv2.imread(img_path)
                if image is None:
                    continue

                # Convert to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Process
                results = hands.process(image_rgb)

                # Save Data
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        row = [label]
                        for lm in hand_landmarks.landmark:
                            row.append(lm.x)
                            row.append(lm.y)

                        writer.writerow(row)
                        count += 1

            print(f"   ✅ Saved {count} samples for '{label}'")

    end_time = time.time()
    duration_min = (end_time - start_time) / 60
    print(f"\n🎉 Done! Processed in {duration_min:.2f} minutes.")


if __name__ == "__main__":
    process_dataset()