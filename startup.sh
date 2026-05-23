#!/bin/bash
# Azure App Service startup script
# Installs OpenGL ES libraries required by MediaPipe HandLandmarker,
# which links against libGLESv2.so.2 even in CPU-only mode.

echo "=== Installing system dependencies for MediaPipe ==="
apt-get install -y -qq libgles2-mesa libgles2-mesa-dev libgl1-mesa-glx libglib2.0-0 2>/dev/null \
  && echo "=== OpenGL ES libraries installed ===" \
  || echo "=== apt-get failed — MediaPipe GPU deps may be missing ==="

echo "=== Starting SignBridge Backend ==="
exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
