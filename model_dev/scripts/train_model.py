import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- PATHS ---
# We use os.path to make sure it works on Windows/Mac/Linux
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # points to model_dev
DATA_FILE = os.path.join(BASE_DIR, "data", "sign_data.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "saved_models", "sign_model.p")


def train():
    print("--- LOADING DATA ---")
    # 1. Load the dataset
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file not found at {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE)

    # Check if we have data
    if df.empty:
        print("Error: CSV file is empty!")
        return

    # 2. Separate Features (x,y coords) and Labels (Names)
    X = df.drop('label', axis=1)  # All columns except 'label' are features
    y = df['label']  # The 'label' column is the target

    # 3. Split into Training (80%) and Testing (20%) sets
    # This ensures we test the model on data it has never seen before
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

    print(f"Training with {len(x_train)} samples, Testing with {len(x_test)} samples.")

    # 4. Train the Model (Random Forest)
    print("--- TRAINING MODEL ---")
    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    # 5. Evaluate Accuracy
    y_predict = model.predict(x_test)
    score = accuracy_score(y_test, y_predict)

    print(f" Model Accuracy: {score * 100:.2f}%")

    # 6. Save the trained model
    # We use 'pickle' to save the python object to a file
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump({'model': model}, f)

    print(f" Model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()