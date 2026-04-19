import os

for root, dirs, files in os.walk("model_dev/saved_models"):
    print(root)
    for f in files:
        print("  ", f)