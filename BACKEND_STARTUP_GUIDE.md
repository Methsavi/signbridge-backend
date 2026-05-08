# SignBridge Backend - Windows Startup Guide

## What Was Fixed

Your backend had several Windows-specific issues that have been resolved:

### 1. **Missing Dependency: `python-multipart`**
   - FastAPI form/file upload endpoints require this package
   - **Status**: ✅ Added to `requirements.txt`

### 2. **Broken OpenCV Pin**
   - `opencv-python==4.7.0` doesn't exist for Python 3.13
   - **Status**: ✅ Updated to `opencv-python==4.7.0.72` (wheel available)

### 3. **Python 3.13 Scientific Stack Compatibility**
   - NumPy, TensorFlow, SciPy, scikit-learn versions needed wheel builds
   - **Status**: ✅ Pinned to Python 3.13-compatible releases:
     - `numpy>=1.26.4,<3.0`
     - `tensorflow==2.20.0`
     - `keras==3.10.0`
     - `scipy==1.15.2`
     - `scikit-learn==1.5.2`

### 4. **Keras Model Deserialization**
   - Saved models had `quantization_config` field unsupported by newer Keras
   - **Status**: ✅ Added compatibility monkeypatch in `recognition_controller.py`

### 5. **Unicode Errors in Console Output**
   - Emoji in print statements caused `UnicodeEncodeError` on Windows
   - **Status**: ✅ Replaced emoji with ASCII text in `app/main.py` and `recognition_controller.py`

### 6. **WebSocket Support**
   - Uvicorn needed explicit websocket library
   - **Status**: ✅ Installed `uvicorn[standard]` and `websockets`

### 7. **MediaPipe Integration**
   - Installed version uses Tasks API (requires model files)
   - Recognition endpoints gracefully disabled when models unavailable
   - **Status**: ✅ Backend runs with recognition disabled (models can be added later)

## All AI Models Load Successfully

Your backend now successfully loads:
- ✅ **Alphabet model** (`best_alphabet_model.keras` + scaler + encoder)
- ✅ **Number model** (`best_number_model.keras` + scaler + encoder)  
- ✅ **Word/Transformer model** (`best_word_model.keras` + scaler + encoder)

## How to Start the Backend

### Prerequisites
- Python 3.13
- Virtual environment activated
- MongoDB running (local or Atlas connection string in `.env`)
- Dependencies installed

### Step 1: Activate Virtual Environment
```powershell
cd "C:\3rd year projects\individual project\SignBridge\signbridge-backend"
.venv\Scripts\Activate.ps1
```

### Step 2: Create/Update `.env` File
```env
MONGODB_URI=mongodb+srv://YOUR_USER:YOUR_PASSWORD@YOUR_CLUSTER/database_name
FRONTEND_URL=http://localhost:5173
FRONTEND_VERCEL_URL=
```

### Step 3: Install Dependencies (One-Time)
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Start the Backend

**Option A: Using uvicorn directly**
```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

**Option B: Using app.main module**
```powershell
python -m app.main
```

### Step 5: Verify It's Running
Open in browser or curl:
```
http://127.0.0.1:8000/
```

Expected response:
```json
{"message": "SignBridge API is running!"}
```

## What's Working Now

- ✅ FastAPI server starts cleanly
- ✅ All Keras models load without errors
- ✅ MongoDB connection (if configured)
- ✅ REST API endpoints for users, feedback, features, dictionary
- ✅ WebSocket endpoints available (but recognition disabled without MediaPipe models)
- ✅ Profile image uploads via R2/S3
- ✅ Translation and TTS services

## Known Limitations

- **Recognition Features Disabled**: WebSocket endpoints `/ws/predict/alphabet`, `/ws/predict/number`, `/ws/predict/word` will return `"error": "recognition_unavailable"` because MediaPipe 0.10.35 requires external model files that aren't bundled.
  - **To enable**: Download MediaPipe's hand_landmarker.tflite and holistic_landmarker.tflite models and configure paths in `load_ai_models()`.

## Troubleshooting

### "Port 8000 already in use"
Kill the existing process:
```powershell
Get-Process | Where-Object {$_.ProcessName -like "*python*"} | Stop-Process
```

Then restart the server.

### "ModuleNotFoundError: No module named X"
Reinstall dependencies:
```powershell
pip install -r requirements.txt --force-reinstall
```

### "MongoDB connection failed"
- Ensure MongoDB is running: `net start MongoDB` (for local)
- Or verify your MONGODB_URI in `.env` is correct for Atlas

### "TensorFlow/Keras import errors"
The monkeypatch in `recognition_controller.py` handles most compatibility issues. If you still see errors:
1. Confirm Python 3.13: `python --version`
2. Reinstall TensorFlow: `pip install tensorflow==2.20.0 --force-reinstall`

## Files Modified

- `requirements.txt` - Updated dependency pins for Python 3.13
- `app/main.py` - Removed emoji from startup prints
- `app/controllers/recognition_controller.py` - Added Keras monkeypatch, removed emoji, disabled MediaPipe gracefully

## Next Steps

1. **Test the API**: Use Postman or the Swagger docs at `http://127.0.0.1:8000/docs`
2. **Enable Recognition** (Optional): Get MediaPipe model files and update paths in `load_ai_models()`
3. **Frontend Connection**: Configure `FRONTEND_URL` in `.env` to allow your frontend to connect

---

**Backend is now ready to run on Windows!** 🎉

