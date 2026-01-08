import pandas as pd
import os

# --- PATH SETTINGS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_FILE = os.path.join(BASE_DIR, "data", "sign_data.csv")


def check_data():
    print("--- 📊 DATA AUDIT REPORT ---")

    # 1. Check if file exists
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: File not found at {DATA_FILE}")
        return

    # 2. Load Data (This might take a few seconds for big files)
    print("Loading data... please wait.")
    try:
        df = pd.read_csv(DATA_FILE)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # 3. Basic Stats
    total_samples = len(df)
    unique_labels = df['label'].unique()
    label_counts = df['label'].value_counts()

    print(f"\n✅ Total Samples Recorded: {total_samples}")
    print(f"✅ Total Unique Gestures: {len(unique_labels)}")

    # 4. Detailed Breakdown
    print("\n--- 📝 Samples per Gesture ---")
    print(label_counts.to_string())  # Prints the full list

    # 5. Check for missing letters
    expected_alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    found_labels = [str(l) for l in unique_labels]

    missing = []
    for letter in expected_alphabet:
        if letter not in found_labels:
            missing.append(letter)

    if missing:
        print(f"\n⚠️ WARNING: The following letters are MISSING: {missing}")
    else:
        print("\n✨ SUCCESS: All A-Z letters are present!")


if __name__ == "__main__":
    check_data()