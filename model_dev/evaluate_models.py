"""
SignBridge AI — Model Evaluation Summary
=========================================
Run this script from PyCharm (or terminal) to:
  • Load all three trained models
  • Inspect their architecture and output classes
  • Confirm inference works correctly
  • Print a training accuracy summary

Usage (from signbridge-backend directory):
    python model_dev/evaluate_models.py
"""

import os, io, json, re, zipfile, tempfile, time
import numpy as np
import joblib
import tensorflow as tf

# ── Keras version compatibility patch ─────────────────────────────────────────
# Models were saved with Keras 3 on Google Colab. Keras 3 adds a
# 'quantization_config' key to Dense / MultiHeadAttention configs.
# Older local Keras rejects this key during deserialization.
#
# Fix: read the .keras ZIP, strip every "quantization_config" key from the
# JSON config, write to a temp file, then load from that temp file.
# Weights are stored separately in model.weights.h5 and are unaffected.

def _strip_quantization_config(obj):
    """Recursively remove 'quantization_config' from any dict/list."""
    if isinstance(obj, dict):
        return {k: _strip_quantization_config(v)
                for k, v in obj.items()
                if k != "quantization_config"}
    elif isinstance(obj, list):
        return [_strip_quantization_config(item) for item in obj]
    return obj

def load_model_compat(path):
    """
    Load a .keras model saved with Keras 3 into an older Keras environment.
    Strips the 'quantization_config' field from the model config JSON so that
    the local Dense layer can deserialise it without errors.
    """
    with zipfile.ZipFile(path, "r") as zin:
        names = zin.namelist()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in names:
                data = zin.read(item)
                if item == "config.json":
                    cfg = json.loads(data.decode("utf-8"))
                    cfg = _strip_quantization_config(cfg)
                    data = json.dumps(cfg).encode("utf-8")
                zout.writestr(item, data)

    # Write patched zip to a temp file, load, delete
    tmp = tempfile.NamedTemporaryFile(suffix=".keras", delete=False)
    tmp.write(buf.getvalue())
    tmp.close()
    try:
        model = tf.keras.models.load_model(tmp.name)
    finally:
        os.unlink(tmp.name)
    return model

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    "alphabet": {
        "model":   os.path.join(BASE, "saved_models", "alphabet", "best_alphabet_model.keras"),
        "scaler":  os.path.join(BASE, "saved_models", "alphabet", "alphabet_scaler.pkl"),
        "encoder": os.path.join(BASE, "saved_models", "alphabet", "alphabet_encoder.pkl"),
        "input_shape":  (63,),      # 21 hand landmarks × 3 (x, y, z)
        "seq_model":    False,
        "trained_acc":  97.36,
        "dataset":      "ASL Alphabet Dataset (Kaggle)",
        "description":  "DNN — A–Z + del + space + nothing (29 classes)",
    },
    "number": {
        "model":   os.path.join(BASE, "saved_models", "number", "best_number_model.keras"),
        "scaler":  os.path.join(BASE, "saved_models", "number", "number_scaler.pkl"),
        "encoder": os.path.join(BASE, "saved_models", "number", "number_encoder.pkl"),
        "input_shape":  (63,),      # 21 hand landmarks × 3 (x, y, z)
        "seq_model":    False,
        "trained_acc":  None,       # fill in after checking your notebook output
        "dataset":      "ASL Digit Dataset (Kaggle)",
        "description":  "DNN — digits 0–9 (10 classes)",
    },
    "word": {
        "model":   os.path.join(BASE, "saved_models", "word", "best_word_model.keras"),
        "scaler":  os.path.join(BASE, "saved_models", "word", "word_scaler.pkl"),
        "encoder": os.path.join(BASE, "saved_models", "word", "word_encoder.pkl"),
        "input_shape":  (30, 126),  # 30 frames × 126 holistic landmarks
        "seq_model":    True,
        "trained_acc":  None,       # fill in after checking your notebook output
        "dataset":      "WLASL100 Processed Dataset (Kaggle)",
        "description":  "Transformer Encoder — WLASL100 word signs",
    },
}

# ── Helpers ────────────────────────────────────────────────────────────────────
SEP  = "=" * 60
DSEP = "-" * 60

def print_header(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def sizeof_fmt(num):
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num) < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} TB"

# ── Main ───────────────────────────────────────────────────────────────────────
def evaluate(name, cfg):
    print_header(f"{name.upper()} MODEL")

    print(f"  Description : {cfg['description']}")
    print(f"  Dataset     : {cfg['dataset']}")
    print()

    print("  Loading model ...", end=" ", flush=True)
    t0 = time.time()
    model = load_model_compat(cfg["model"])
    print(f"done  ({time.time()-t0:.2f}s)")

    print("  Loading scaler ...", end=" ", flush=True)
    scaler = joblib.load(cfg["scaler"])
    print("done")

    print("  Loading encoder ...", end=" ", flush=True)
    encoder = joblib.load(cfg["encoder"])
    print("done")

    # ── Architecture ──────────────────────────────────────────────────────────
    print()
    print(DSEP)
    print("  ARCHITECTURE")
    print(DSEP)
    model.summary(print_fn=lambda x: print(f"  {x}"))

    # ── Class info ────────────────────────────────────────────────────────────
    classes = list(encoder.classes_)
    print()
    print(DSEP)
    print("  OUTPUT CLASSES")
    print(DSEP)
    print(f"  Total classes : {len(classes)}")
    print(f"  Classes       : {classes}")

    # ── Parameter count & file size ───────────────────────────────────────────
    total_params = model.count_params()
    fsize = os.path.getsize(cfg["model"])
    print()
    print(f"  Total parameters : {total_params:,}")
    print(f"  Model file size  : {sizeof_fmt(fsize)}")

    # ── Dummy inference test ───────────────────────────────────────────────────
    print()
    print(DSEP)
    print("  INFERENCE SANITY CHECK  (random dummy input)")
    print(DSEP)

    np.random.seed(42)
    if cfg["seq_model"]:
        # Word model: scaler was fitted per-frame (126 features each).
        # Scale each of the 30 frames independently, then add batch dim.
        dummy_frames = np.random.randn(30, 126).astype(np.float32)
        dummy_scaled = scaler.transform(dummy_frames)   # (30, 126)
        dummy_input  = dummy_scaled.reshape(1, 30, 126) # (1, 30, 126)
    else:
        dummy_input = scaler.transform(
            np.random.randn(1, cfg["input_shape"][0]).astype(np.float32)
        )

    t0 = time.time()
    probs = model.predict(dummy_input, verbose=0)[0]
    elapsed = time.time() - t0

    top3_idx = np.argsort(probs)[-3:][::-1]
    print(f"  Inference time : {elapsed*1000:.1f} ms")
    print(f"  Top-3 predictions:")
    for rank, idx in enumerate(top3_idx, 1):
        label = encoder.classes_[idx]
        conf  = probs[idx] * 100
        print(f"    {rank}. {label:<15}  {conf:.2f}%")
    print(f"  ✓ Model runs correctly")

    # ── Training accuracy ─────────────────────────────────────────────────────
    print()
    print(DSEP)
    print("  TRAINING ACCURACY (from Jupyter notebook)")
    print(DSEP)
    if cfg["trained_acc"] is not None:
        print(f"  Test accuracy   : {cfg['trained_acc']:.2f}%")
    else:
        print(f"  Test accuracy   : (see your Jupyter notebook output)")
    print()


def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       SignBridge AI — Model Evaluation Summary           ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  TensorFlow version : {tf.__version__}")
    print(f"  NumPy version      : {np.__version__}")

    results = {}
    for name, cfg in MODELS.items():
        try:
            evaluate(name, cfg)
            results[name] = "✓ PASS"
        except Exception as e:
            print(f"\n  ✗ ERROR loading {name} model: {e}")
            results[name] = "✗ FAIL"

    # ── Summary table ─────────────────────────────────────────────────────────
    print_header("SUMMARY")
    print(f"  {'Model':<12}  {'Status':<10}  {'Trained Accuracy'}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*20}")
    for name, status in results.items():
        acc = MODELS[name]["trained_acc"]
        acc_str = f"{acc:.2f}%" if acc else "see notebook"
        print(f"  {name:<12}  {status:<10}  {acc_str}")
    print()
    print("  Note: Accuracy figures are from Colab training runs.")
    print("        To re-evaluate on test data, run the Jupyter notebooks")
    print("        in JupyterProject/models/ with the Kaggle datasets.")
    print()


if __name__ == "__main__":
    main()
