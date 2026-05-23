#!/bin/bash
# Oryx post-build hook — runs after "pip install -r requirements.txt" on Azure.
#
# Problem: mediapipe installs opencv-python (non-headless) as a dependency.
#          opencv-python requires libxcb.so.1 (X11) which does not exist on
#          Azure App Service Linux — causing "libxcb.so.1: cannot open shared
#          object file" at import time and disabling all ML features.
#
# Fix: uninstall opencv-python after the build; opencv-python-headless (already
#      in requirements.txt) provides the same cv2 module without X11.

echo "=== PostBuild Hook: Removing non-headless OpenCV if present ==="
pip uninstall -y opencv-python 2>/dev/null || echo "opencv-python was not installed — nothing to remove"
echo "=== PostBuild Hook: Done ==="
