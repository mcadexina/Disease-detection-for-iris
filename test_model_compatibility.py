"""Test script to verify model compatibility with current NumPy version."""
import os
import sys
import pickle
import numpy as np

print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")

saved_models_dir = 'saved_models'
models_to_test = {
    'SVM (Gabor)': 'svm_gabor.pkl',
    'RF (Wavelet)': 'rf_wavelet.pkl',
}

print("\n" + "="*60)
print("Testing pickle model compatibility...")
print("="*60)

for model_name, filename in models_to_test.items():
    path = os.path.join(saved_models_dir, filename)
    if not os.path.exists(path):
        print(f"[FAIL] {model_name}: Model file not found at {path}")
        continue

    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"[OK] {model_name}: Successfully loaded")
        print(f"     Model type: {type(model).__name__}")
    except Exception as e:
        print(f"[FAIL] {model_name}: Failed to load")
        print(f"       Error: {type(e).__name__}: {e}")

print("\n" + "="*60)
print("Testing TensorFlow/Keras models...")
print("="*60)

try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")

    keras_models = {
        'Xception': 'xception_iris.h5',
        'ResNet50': 'resnet50_iris.h5',
        'MobileNet': 'mobilenet_iris.h5',
        'CNN': 'cnn_iris.h5',
    }

    for model_name, filename in keras_models.items():
        path = os.path.join(saved_models_dir, filename)
        if not os.path.exists(path):
            print(f"[WARN] {model_name}: Model file not found at {path}")
            continue

        try:
            model = tf.keras.models.load_model(path)
            print(f"[OK] {model_name}: Successfully loaded")
        except Exception as e:
            print(f"[FAIL] {model_name}: Failed to load")
            print(f"       Error: {type(e).__name__}: {str(e)[:80]}")

except ImportError as e:
    print(f"[FAIL] TensorFlow not available: {e}")

print("\n" + "="*60)
print("Compatibility Test Complete")
print("="*60)
