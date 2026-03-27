import pandas as pd
import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# IMPORTANT: Pointing to the NEW clean file
DATA_FILE = os.path.join(BASE_DIR, "data", "sign_data_clean.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "saved_models", "sign_model.p")


def train():
    print("--- 🧠 LOADING DATA ---")
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file not found at {DATA_FILE}")
        print("Did you run create_balanced_dataset.py?")
        return

    df = pd.read_csv(DATA_FILE)

    if df.empty:
        print("Error: CSV file is empty!")
        return

    print(f"Data Size: {len(df)} rows")
    print(f"Classes found: {df['label'].unique()}")

    # --- PREPARE ---
    X = df.drop('label', axis=1)
    y = df['label']

    # --- SPLIT ---
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

    # --- TRAIN ---
    print("🚀 Training Random Forest Model...")
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(x_train, y_train)

    # --- EVALUATE ---
    y_predict = model.predict(x_test)
    score = accuracy_score(y_test, y_predict)
    print(f"✅ Model Accuracy: {score * 100:.2f}%")

    # --- SAVE ---
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump({'model': model}, f)

    print(f"💾 Model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()