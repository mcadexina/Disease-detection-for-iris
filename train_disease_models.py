"""
train_disease_models.py
Full training pipeline for iris disease detection.

Classes : Healthy, Glaucoma, Myopia
Dataset : disease_model/ (three class folders)

Models trained
--------------
Deep learning (train/val/test split):
  1. Xception   → saved_models/xception_iris.h5
  2. ResNet50   → saved_models/resnet50_iris.h5
  3. MobileNet  → saved_models/mobilenet_iris.h5
  4. Custom CNN → saved_models/cnn_iris.h5

Machine learning (5-fold cross-validation):
  5. SVM  + Gabor features   → saved_models/svm_gabor.pkl
  6. RF   + Wavelet features → saved_models/rf_wavelet.pkl

All metrics are written to saved_models/metrics.json.

Usage:
  python train_disease_models.py [--models all|xception|resnet50|mobilenet|cnn|svm|rf]
"""

import os, json, argparse, pickle, time
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

# All paths are absolute and anchored to this script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure project root is importable regardless of CWD
import sys
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize

import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)

from models.disease_models import (
    create_xception_model, create_resnet50_model,
    create_mobilenet_model, create_custom_cnn_model,
    IMG_SIZE, IMG_SIZE_CNN,
)
from models.gabor_model import extract_features as gabor_features
from models.wavelet_model import extract_features as wavelet_features
from utils.evaluation import compute_multiclass_metrics

# ── Paths (absolute, anchored to this script's directory) ──────────────────
DATA_PATHS = {
    'Healthy':  os.path.join(BASE_DIR, 'disease_model', 'Healthy-20260312T025031Z-3-001', 'Healthy'),
    'Glaucoma': os.path.join(BASE_DIR, 'disease_model', 'Glaucoma-20260312T025028Z-3-001', 'Glaucoma'),
    'Myopia':   os.path.join(BASE_DIR, 'disease_model', 'Myopia-20260312T025034Z-3-001', 'Myopia'),
}
CLASSES = ['Healthy', 'Glaucoma', 'Myopia']
SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
METRICS_PATH     = os.path.join(SAVED_MODELS_DIR, 'metrics.json')

# ── Hyper-parameters ───────────────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS = 30
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════
def load_image_paths():
    """Return (paths, labels) lists across all classes."""
    paths, labels = [], []
    for idx, cls in enumerate(CLASSES):
        folder = DATA_PATHS[cls]
        if not os.path.isdir(folder):
            print(f"[WARN] Folder not found: {folder}")
            continue
        for fname in os.listdir(folder):
            if os.path.splitext(fname)[1].lower() in IMG_EXTENSIONS:
                paths.append(os.path.join(folder, fname))
                labels.append(idx)
    print(f"[DATA] Total images: {len(paths)} | Class distribution: "
          + " | ".join(f"{CLASSES[i]}={labels.count(i)}" for i in range(len(CLASSES))))
    return paths, labels


def load_images_for_dl(paths, labels, img_size, grayscale=False, max_per_class=None):
    """
    Load images into numpy arrays for deep-learning models.
    Returns (X, y) with X shape (N, H, W, C).
    """
    class_counts = {i: 0 for i in range(len(CLASSES))}
    X, y = [], []

    for path, label in tqdm(zip(paths, labels), total=len(paths), desc="Loading"):
        if max_per_class and class_counts[label] >= max_per_class:
            continue
        try:
            img = cv2.imread(path)
            if img is None:
                continue
            if grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, img_size)
                img = img.astype(np.float32) / 255.0
                img = img[..., np.newaxis]
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                img = img.astype(np.float32) / 255.0

            X.append(img)
            y.append(label)
            class_counts[label] += 1
        except Exception as e:
            print(f"[SKIP] {path}: {e}")

    return np.array(X), np.array(y)


def load_features_for_ml(paths, labels, method='gabor', max_per_class=None):
    """
    Extract hand-crafted features (Gabor or Wavelet) for ML models.
    Returns (X, y) where X is a 2-D feature matrix.
    """
    class_counts = {i: 0 for i in range(len(CLASSES))}
    X, y = [], []
    extract = gabor_features if method == 'gabor' else wavelet_features

    for path, label in tqdm(zip(paths, labels), total=len(paths),
                            desc=f"Extracting {method} features"):
        if max_per_class and class_counts[label] >= max_per_class:
            continue
        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            feat = extract(img)
            if feat is not None and len(feat) > 0:
                X.append(feat)
                y.append(label)
                class_counts[label] += 1
        except Exception as e:
            print(f"[SKIP] {path}: {e}")

    return np.array(X), np.array(y)


# ═══════════════════════════════════════════════════════════════════════════
# Training helpers
# ═══════════════════════════════════════════════════════════════════════════
def dl_callbacks(model_path, monitor='val_accuracy'):
    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
    return [
        ModelCheckpoint(model_path, monitor=monitor, save_best_only=True,
                        verbose=1, mode='max'),
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True,
                      verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4,
                          min_lr=1e-6, verbose=1),
    ]


def evaluate_dl_model(model, X_test, y_test):
    """Run model.predict and compute full metrics dict."""
    y_scores = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_scores, axis=1)
    return compute_multiclass_metrics(y_test, y_pred, y_scores, CLASSES)


def evaluate_ml_model(clf, X_test, y_test):
    """Run predict_proba and compute full metrics dict."""
    y_scores = clf.predict_proba(X_test)
    y_pred = np.argmax(y_scores, axis=1)
    return compute_multiclass_metrics(y_test, y_pred, y_scores, CLASSES)


# ═══════════════════════════════════════════════════════════════════════════
# Individual model trainers
# ═══════════════════════════════════════════════════════════════════════════
def train_deep_model(name, factory_fn, img_size, grayscale, model_path,
                       paths, labels, max_per_class=1500):
    print(f"\n{'='*60}\n[TRAIN] {name}\n{'='*60}")
    t0 = time.time()

    X, y = load_images_for_dl(paths, labels, img_size, grayscale,
                               max_per_class=max_per_class)
    if len(X) == 0:
        print(f"[ERROR] No data loaded for {name}. Skipping.")
        return None

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

    print(f"  Train: {len(X_tr)}  Val: {len(X_val)}  Test: {len(X_test)}")

    model = factory_fn()
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=dl_callbacks(model_path),
        verbose=1,
    )

    # Reload best checkpoint
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)

    metrics = evaluate_dl_model(model, X_test, y_test)
    metrics['training_time_s'] = round(time.time() - t0, 1)
    print(f"  Accuracy={metrics['accuracy']:.4f}  "
          f"FAR={metrics['far']:.4f}  FRR={metrics['frr']:.4f}  "
          f"EER={metrics['eer']:.4f}")
    return metrics


def train_ml_model(name, clf, method, model_path, paths, labels,
                   n_splits=5, max_per_class=1000):
    print(f"\n{'='*60}\n[TRAIN] {name}  ({n_splits}-fold CV)\n{'='*60}")
    t0 = time.time()

    X, y = load_features_for_ml(paths, labels, method=method,
                                 max_per_class=max_per_class)
    if len(X) == 0:
        print(f"[ERROR] No features extracted for {name}. Skipping.")
        return None

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y), 1):
        print(f"  Fold {fold}/{n_splits} …")
        clf_fold = clf.__class__(**clf.get_params())
        clf_fold.fit(X[tr_idx], y[tr_idx])
        m = evaluate_ml_model(clf_fold, X[val_idx], y[val_idx])
        fold_metrics.append(m)

    # Average across folds
    avg = {}
    for key in ['accuracy', 'precision', 'recall', 'f1', 'far', 'frr', 'eer']:
        avg[key] = round(float(np.mean([m[key] for m in fold_metrics])), 4)

    # Per-class average (store averages only, not full ROC arrays)
    avg['per_class'] = {}
    for cls in CLASSES:
        avg['per_class'][cls] = {
            k: round(float(np.mean([m['per_class'][cls][k]
                                    for m in fold_metrics])), 4)
            for k in ['far', 'frr', 'eer', 'roc_auc']
        }

    # Retrain final model on all data and save
    print("  Retraining final model on full dataset …")
    clf.fit(X, y)
    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)

    avg['training_time_s'] = round(time.time() - t0, 1)
    print(f"  CV Accuracy={avg['accuracy']:.4f}  "
          f"FAR={avg['far']:.4f}  FRR={avg['frr']:.4f}  "
          f"EER={avg['eer']:.4f}")
    return avg


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main(train_which='all'):
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    paths, labels = load_image_paths()
    paths = np.array(paths)
    labels = np.array(labels)

    # Load existing metrics so we don't overwrite previously trained models
    metrics_all = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics_all = json.load(f)

    # train_which is either the string 'all' or a set of key strings
    def _should_train(key):
        if train_which == 'all':
            return True
        return key in train_which

    # ── Xception ─────────────────────────────────────────────────────────
    if _should_train('xception'):
        m = train_deep_model(
            'Xception', create_xception_model,
            IMG_SIZE, grayscale=False,
            model_path=os.path.join(SAVED_MODELS_DIR, 'xception_iris.h5'),
            paths=paths, labels=labels,
        )
        if m:
            metrics_all['Xception'] = m

    # ── ResNet50 ──────────────────────────────────────────────────────────
    if _should_train('resnet50'):
        m = train_deep_model(
            'ResNet50', create_resnet50_model,
            IMG_SIZE, grayscale=False,
            model_path=os.path.join(SAVED_MODELS_DIR, 'resnet50_iris.h5'),
            paths=paths, labels=labels,
        )
        if m:
            metrics_all['ResNet50'] = m

    # ── MobileNet ─────────────────────────────────────────────────────────
    if _should_train('mobilenet'):
        m = train_deep_model(
            'MobileNet', create_mobilenet_model,
            IMG_SIZE, grayscale=False,
            model_path=os.path.join(SAVED_MODELS_DIR, 'mobilenet_iris.h5'),
            paths=paths, labels=labels,
        )
        if m:
            metrics_all['MobileNet'] = m

    # ── Custom CNN ────────────────────────────────────────────────────────
    if _should_train('cnn'):
        m = train_deep_model(
            'CNN', create_custom_cnn_model,
            IMG_SIZE_CNN, grayscale=True,
            model_path=os.path.join(SAVED_MODELS_DIR, 'cnn_iris.h5'),
            paths=paths, labels=labels,
        )
        if m:
            metrics_all['CNN'] = m

    # ── SVM + Gabor ───────────────────────────────────────────────────────
    if _should_train('svm'):
        clf_svm = SVC(kernel='rbf', C=10, gamma='scale',
                      probability=True, random_state=42)
        m = train_ml_model(
            'SVM (Gabor)', clf_svm, method='gabor',
            model_path=os.path.join(SAVED_MODELS_DIR, 'svm_gabor.pkl'),
            paths=paths, labels=labels,
        )
        if m:
            metrics_all['SVM (Gabor)'] = m

    # ── RF + Wavelet ──────────────────────────────────────────────────────
    if _should_train('rf'):
        clf_rf = RandomForestClassifier(
            n_estimators=200, max_depth=None,
            n_jobs=-1, random_state=42
        )
        m = train_ml_model(
            'RF (Wavelet)', clf_rf, method='wavelet',
            model_path=os.path.join(SAVED_MODELS_DIR, 'rf_wavelet.pkl'),
            paths=paths, labels=labels,
        )
        if m:
            metrics_all['RF (Wavelet)'] = m

    # Save combined metrics
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics_all, f, indent=2)
    print(f"\n[DONE] Metrics written to {METRICS_PATH}")


VALID_KEYS = {'all', 'xception', 'resnet50', 'mobilenet', 'cnn', 'svm', 'rf'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train iris disease models')
    parser.add_argument(
        '--models',
        default='all',
        help=(
            'Comma-separated list of models to train, or "all". '
            'Valid values: ' + ', '.join(sorted(VALID_KEYS))
        ),
    )
    args = parser.parse_args()

    # Accept "all" or a comma-separated subset, e.g. "svm,rf" or "cnn,resnet50"
    raw = [s.strip().lower() for s in args.models.split(',') if s.strip()]
    if 'all' in raw or set(raw) >= (VALID_KEYS - {'all'}):
        selected = 'all'
    else:
        invalid = set(raw) - VALID_KEYS
        if invalid:
            parser.error(
                f"Invalid model key(s): {invalid}. "
                f"Choose from: {', '.join(sorted(VALID_KEYS))}"
            )
        selected = set(raw)

    main(train_which=selected)
