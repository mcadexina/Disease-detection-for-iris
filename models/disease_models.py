"""
disease_models.py
All deep-learning model factories for iris disease detection:
  - Xception  (transfer learning)
  - ResNet50  (transfer learning)
  - MobileNetV2 (transfer learning)
  - Custom CNN

All models output 3 classes: Healthy, Glaucoma, Myopia.
Machine-learning models (SVM + RF) are defined in train_disease_models.py.
"""
import tensorflow as tf
from tensorflow.keras import layers, models, applications


NUM_CLASSES = 3
IMG_SIZE = (224, 224)        # Standard for Xception / ResNet50 / MobileNet
IMG_SIZE_CNN = (64, 64)      # Custom CNN uses smaller input
INPUT_SHAPE_DL = (*IMG_SIZE, 3)
INPUT_SHAPE_CNN = (*IMG_SIZE_CNN, 1)


def _add_classification_head(base, num_classes, dropout=0.5):
    """Attach a global-average-pool + dense head to a base model."""
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout / 2)(x)
    out = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    return models.Model(inputs=base.input, outputs=out)


def create_xception_model(num_classes=NUM_CLASSES, fine_tune_layers=30):
    """Xception backbone with ImageNet weights, custom classification head."""
    base = applications.Xception(
        weights='imagenet',
        include_top=False,
        input_shape=INPUT_SHAPE_DL,
    )
    # Freeze all layers initially
    base.trainable = True
    for layer in base.layers[:-fine_tune_layers]:
        layer.trainable = False

    model = _add_classification_head(base, num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def create_resnet50_model(num_classes=NUM_CLASSES, fine_tune_layers=20):
    """ResNet50 backbone with ImageNet weights, custom classification head."""
    base = applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=INPUT_SHAPE_DL,
    )
    base.trainable = True
    for layer in base.layers[:-fine_tune_layers]:
        layer.trainable = False

    model = _add_classification_head(base, num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def create_mobilenet_model(num_classes=NUM_CLASSES, fine_tune_layers=20):
    """MobileNetV2 backbone with ImageNet weights, custom classification head."""
    base = applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=INPUT_SHAPE_DL,
    )
    base.trainable = True
    for layer in base.layers[:-fine_tune_layers]:
        layer.trainable = False

    model = _add_classification_head(base, num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def create_custom_cnn_model(num_classes=NUM_CLASSES):
    """Lightweight custom CNN — grayscale 64×64 input."""
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, 3, padding='same', input_shape=INPUT_SHAPE_CNN),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        # Classifier
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu', name='feature_layer'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax', name='predictions'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def load_disease_model(model_path):
    """
    Load a saved Keras model from disk.
    Returns the model, or raises FileNotFoundError if the file is missing.
    """
    import os
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Run the training script first."
        )
    return tf.keras.models.load_model(model_path)


def preprocess_for_dl(image_np, img_size=IMG_SIZE):
    """
    Preprocess a numpy image array for DL models (Xception/ResNet50/MobileNet).
    Returns a float32 numpy array of shape (1, H, W, 3) in [0, 1].
    """
    import cv2
    import numpy as np

    if image_np.ndim == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 1:
        image_np = cv2.cvtColor(image_np[:, :, 0], cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

    img = cv2.resize(image_np, img_size).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def preprocess_for_cnn(image_np, img_size=IMG_SIZE_CNN):
    """
    Preprocess a numpy image array for the custom grayscale CNN.
    Returns a float32 numpy array of shape (1, H, W, 1) in [0, 1].
    """
    import cv2
    import numpy as np

    if image_np.ndim == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np.copy()

    img = cv2.resize(gray, img_size).astype(np.float32) / 255.0
    return np.expand_dims(img[..., np.newaxis], axis=0)
