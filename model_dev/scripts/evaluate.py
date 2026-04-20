import os

# --- MANDATORY ENVIRONMENT FIXES ---
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# --- PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "sign_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "signbridge_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "saved_models", "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "saved_models", "encoder.pkl")


def evaluate():
    print("📋 Starting Evaluation Process...")

    # 1. Load AI Components first to know which classes to look for
    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, ENCODER_PATH]):
        print("❌ Error: Model, Scaler, or Encoder files missing in saved_models folder.")
        return

    print("🧠 Loading Model and Scalers...")
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)

    known_classes = encoder.classes_
    print(f"✅ Model recognizes {len(known_classes)} classes.")

    # 2. Load and Filter Data
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: Data file not found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH, dtype={'label': str})

    # --- FIX: Filter out labels that the encoder doesn't know (like 'del', 'nothing') ---
    initial_count = len(df)
    df = df[df['label'].isin(known_classes)]
    filtered_count = len(df)

    if filtered_count < initial_count:
        print(f"🧹 Filtered out {initial_count - filtered_count} samples with unknown labels (e.g., 'del', 'space').")

    X = df.drop('label', axis=1).values
    y_true_labels = df['label'].values

    # 3. Preprocess Data
    X_scaled = scaler.transform(X)
    y_true_ids = encoder.transform(y_true_labels)

    # 4. Run Predictions
    print(f"🚀 Running predictions on {filtered_count} samples...")
    predictions_probs = model.predict(X_scaled, verbose=1)
    y_pred_ids = np.argmax(predictions_probs, axis=1)

    # 5. Generate Metrics
    print("\n" + "=" * 30)
    print("📊 CLASSIFICATION REPORT")
    print("=" * 30)
    print(classification_report(y_true_ids, y_pred_ids, target_names=known_classes))

    # 6. Confusion Matrix Visualization
    print("🎨 Generating Confusion Matrix...")
    cm = confusion_matrix(y_true_ids, y_pred_ids)

    plt.figure(figsize=(18, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=known_classes, yticklabels=known_classes)
    plt.title('SignBridge AI: Confusion Matrix (Filtered Data)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    chart_path = os.path.join(BASE_DIR, "evaluation_results.png")
    plt.savefig(chart_path)
    print(f"✅ Success! Performance chart saved to: {chart_path}")
    plt.show()


if __name__ == "__main__":
    evaluate()