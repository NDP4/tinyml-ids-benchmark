#!/usr/bin/env python3
"""
05_dnn_training.py — DNN and 1D-CNN Training, Evaluation, and TFLite Export

Trains a Deep Neural Network (DNN) and a 1D Convolutional Neural Network (CNN)
for binary IDS classification on NSL-KDD. Models are exported as INT8 TFLite
and then as C byte-array headers for deployment via TensorFlow Lite Micro.

Usage:
    python python/05_dnn_training.py

Requirements:
    - TensorFlow >= 2.12
    - numpy, scikit-learn, matplotlib
    - Preprocessed data at data/preprocessed_top_10.npz
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "preprocessed_top_10.npz")
MODELS_DIR = os.path.join(BASE_DIR, "models", "nslkdd", "neural")
HEADERS_DIR = os.path.join(BASE_DIR, "models", "nslkdd", "neural")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(HEADERS_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("Loading preprocessed NSL-KDD data...")
data = np.load(DATA_PATH)
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']

print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"  y_train: {y_train.shape} (attack ratio: {y_train.mean():.3f})")
print(f"  y_test:  {y_test.shape} (attack ratio: {y_test.mean():.3f})")

NUM_FEATURES = X_train.shape[1]
assert NUM_FEATURES == 10, f"Expected 10 features, got {NUM_FEATURES}"

# ============================================================
# MODEL ARCHITECTURES
# ============================================================

def build_dnn_model(input_dim=10):
    """
    Compact DNN: Input(10) → Dense(32) → BN → Dropout → Dense(16) → BN → Dropout → Dense(8) → Dense(1)
    Target: <5KB TFLite INT8
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(8, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def build_cnn_model(input_dim=10):
    """
    1D-CNN: Input(10,1) → Conv1D(16,3) → Pool → Conv1D(8,3) → GAP → Dense(8) → Dense(1)
    Target: <4KB TFLite INT8
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim, 1)),
        tf.keras.layers.Conv1D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(8, 3, activation='relu', padding='same'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# ============================================================
# TRAINING
# ============================================================

EPOCHS = 100
BATCH_SIZE = 64
PATIENCE = 10

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=PATIENCE, restore_best_weights=True
)

# --- DNN ---
print("\n" + "=" * 60)
print("Training DNN...")
dnn_model = build_dnn_model(NUM_FEATURES)
dnn_model.summary()

dnn_history = dnn_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    verbose=1
)

# --- CNN ---
print("\n" + "=" * 60)
print("Training 1D-CNN...")
X_train_cnn = X_train.reshape(-1, NUM_FEATURES, 1)
X_test_cnn = X_test.reshape(-1, NUM_FEATURES, 1)

cnn_model = build_cnn_model(NUM_FEATURES)
cnn_model.summary()

cnn_history = cnn_model.fit(
    X_train_cnn, y_train,
    validation_split=0.2,
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    verbose=1
)

# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("Evaluating models on test set...")

for name, model, X_eval in [("DNN", dnn_model, X_test), ("CNN", cnn_model, X_test_cnn)]:
    y_prob = model.predict(X_eval, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n--- {name} ---")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Normal','Attack'])}")

# ============================================================
# SAVE KERAS MODELS
# ============================================================

dnn_path = os.path.join(MODELS_DIR, "dnn_model.keras")
cnn_path = os.path.join(MODELS_DIR, "cnn_model.keras")
dnn_model.save(dnn_path)
cnn_model.save(cnn_path)
print(f"Saved Keras models: {dnn_path}, {cnn_path}")

# ============================================================
# TFLITE CONVERSION (INT8 Post-Training Quantization)
# ============================================================

def representative_dataset_gen(X_data, num_samples=1000):
    """Generator for calibration during INT8 PTQ."""
    indices = np.random.choice(len(X_data), min(num_samples, len(X_data)), replace=False)
    for i in indices:
        yield [X_data[i:i+1].astype(np.float32)]


def convert_to_tflite_int8(model, X_cal, name):
    """Convert Keras model to INT8 TFLite."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_gen(X_cal)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    tflite_path = os.path.join(MODELS_DIR, f"{name}_model_int8.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"  {name} TFLite INT8: {len(tflite_model)} bytes ({len(tflite_model)/1024:.1f} KB)")
    return tflite_model, tflite_path


print("\n" + "=" * 60)
print("Converting to TFLite INT8...")

dnn_tflite, dnn_tflite_path = convert_to_tflite_int8(dnn_model, X_train, "dnn")
cnn_tflite, cnn_tflite_path = convert_to_tflite_int8(cnn_model, X_train_cnn, "cnn")

# ============================================================
# EXPORT AS C HEADER
# ============================================================

def tflite_to_c_header(tflite_bytes, header_name, output_path):
    """Convert TFLite model bytes to C header file."""
    array_name = f"{header_name}_model_data"
    hex_values = ', '.join(f'0x{b:02x}' for b in tflite_bytes)
    
    content = f"""// Auto-generated by 05_dnn_training.py
// Model: {header_name} (INT8 quantized)
// Size: {len(tflite_bytes)} bytes ({len(tflite_bytes)/1024:.1f} KB)

#ifndef {header_name.upper()}_MODEL_H
#define {header_name.upper()}_MODEL_H

#include <Arduino.h>

alignas(8) const unsigned char {array_name}[] PROGMEM = {{
  {hex_values}
}};

const unsigned int {array_name}_len = {len(tflite_bytes)};

#endif // {header_name.upper()}_MODEL_H
"""
    with open(output_path, 'w') as f:
        f.write(content)
    print(f"  Exported C header: {output_path}")


print("\nExporting C headers...")
tflite_to_c_header(dnn_tflite, "dnn", os.path.join(HEADERS_DIR, "dnn_model.h"))
tflite_to_c_header(cnn_tflite, "cnn", os.path.join(HEADERS_DIR, "cnn_model.h"))

# ============================================================
# SAVE METADATA
# ============================================================

metadata = {
    "dnn": {
        "architecture": "Dense(32)->BN->Drop(0.3)->Dense(16)->BN->Drop(0.2)->Dense(8)->Dense(1)",
        "params": int(dnn_model.count_params()),
        "tflite_int8_bytes": len(dnn_tflite),
        "epochs_trained": len(dnn_history.history['loss']),
    },
    "cnn": {
        "architecture": "Conv1D(16,3)->Pool->Conv1D(8,3)->GAP->Dense(8)->Drop(0.3)->Dense(1)",
        "params": int(cnn_model.count_params()),
        "tflite_int8_bytes": len(cnn_tflite),
        "epochs_trained": len(cnn_history.history['loss']),
    },
    "framework": "TensorFlow Lite Micro",
    "quantization": "INT8 PTQ",
    "features": 10,
    "dataset": "NSL-KDD"
}

meta_path = os.path.join(MODELS_DIR, "neural_metadata.json")
with open(meta_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"\nMetadata saved: {meta_path}")

print("\n" + "=" * 60)
print("DNN/CNN training and export complete!")
print("=" * 60)
