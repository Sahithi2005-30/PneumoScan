"""
Pneumonia Detection Training Script - 3 Class Version
Classes: INVALID, NORMAL, PNEUMONIA
Dataset folder: chest_xray
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    CSVLogger
)
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

# ─── CONFIG ───────────────────────────────────────────────
IMG_SIZE       = 224
BATCH_SIZE     = 8
EPOCHS_PHASE1  = 2
EPOCHS_PHASE2  = 2
LABEL_SMOOTH   = 0.1

# You added INVALID inside the same chest_xray dataset
DATASET_PATH = "chest_xray"

train_dir = os.path.join(DATASET_PATH, "train")
val_dir   = os.path.join(DATASET_PATH, "val")
test_dir  = os.path.join(DATASET_PATH, "test")

print("Current working directory:", os.getcwd())
print("Dataset path:", os.path.abspath(DATASET_PATH))
print("Train directory:", os.path.abspath(train_dir))
print("Validation directory:", os.path.abspath(val_dir))
print("Test directory:", os.path.abspath(test_dir))

print("Train exists:", os.path.exists(train_dir))
print("Val exists:", os.path.exists(val_dir))
print("Test exists:", os.path.exists(test_dir))

if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Training folder not found: {os.path.abspath(train_dir)}")

if not os.path.exists(val_dir):
    raise FileNotFoundError(f"Validation folder not found: {os.path.abspath(val_dir)}")

os.makedirs("models", exist_ok=True)

# ─── DATA GENERATORS ──────────────────────────────────────
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    zoom_range=0.10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print(f"Train samples: {train_data.samples}")
print(f"Validation samples: {val_data.samples}")
print(f"Class indices: {train_data.class_indices}")

# Save class indices for app.py reference
with open("models/class_indices.txt", "w") as f:
    f.write(str(train_data.class_indices))

# ─── CLASS WEIGHTS ────────────────────────────────────────
classes = train_data.classes

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(classes),
    y=classes
)

class_weights = dict(enumerate(weights))
print(f"Class weights: {class_weights}")

# ─── MODEL BUILDER ────────────────────────────────────────
def build_model():
    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # Phase 1: freeze EfficientNet base
    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)

    # 3-class output
    output = Dense(3, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=output)
    return model, base

# ─── PHASE 1: TRAIN CLASSIFICATION HEAD ───────────────────
print("\n" + "=" * 55)
print("PHASE 1: Training classification head")
print("=" * 55)

model, base_model = build_model()

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH),
    metrics=["accuracy"]
)

callbacks_p1 = [
    EarlyStopping(
        patience=2,
        restore_best_weights=True,
        monitor="val_accuracy",
        mode="max"
    ),
    ReduceLROnPlateau(
        factor=0.5,
        patience=1,
        min_lr=1e-6,
        monitor="val_loss"
    ),
    ModelCheckpoint(
        "models/phase1_3class_best.h5",
        save_best_only=True,
        monitor="val_accuracy",
        mode="max"
    ),
    CSVLogger("models/training_3class_phase1.csv")
]

history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_PHASE1,
    class_weight=class_weights,
    callbacks=callbacks_p1
)

# ─── PHASE 2: FINE-TUNING ─────────────────────────────────
print("\n" + "=" * 55)
print("PHASE 2: Fine-tuning top 20 layers")
print("=" * 55)

base_model.trainable = True

# Freeze all layers except last 20
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH),
    metrics=["accuracy"]
)

callbacks_p2 = [
    EarlyStopping(
        patience=2,
        restore_best_weights=True,
        monitor="val_accuracy",
        mode="max"
    ),
    ReduceLROnPlateau(
        factor=0.5,
        patience=1,
        min_lr=1e-7,
        monitor="val_loss"
    ),
    ModelCheckpoint(
        "models/pneumonia_3class_model.h5",
        save_best_only=True,
        monitor="val_accuracy",
        mode="max"
    ),
    CSVLogger("models/training_3class_phase2.csv")
]

history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_PHASE2,
    class_weight=class_weights,
    callbacks=callbacks_p2
)

# Save final model
model.save("models/pneumonia_3class_model.h5")

print("\n" + "=" * 55)
print("3-class training complete!")
print("Model saved to: models/pneumonia_3class_model.h5")
print("Class indices:", train_data.class_indices)
print("=" * 55)

all_val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
print(f"Best Val Accuracy : {max(all_val_acc) * 100:.2f}%")