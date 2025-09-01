#!/usr/bin/env python3
from distutils.command.install import install

import numpy as np
import openpyxl
import pandas as pd
import tensorflow as tf

# ── CONFIGURATION ───────────────────────────────────────────────────────────────
MODEL_FILE = "cnn_binary_classifier_32px.keras"
EXCEL_OUT  = "Classification_Results.xlsx"

def load_and_preprocess_images(test_dir, target_size):
    """
    Reads all .pngs in test_dir, converts to grayscale, resizes to target_size,
    normalizes to [0,1], and returns:
      - X: Tensor shape (N, H, W, 1)
      - filenames: list of basenames
    """
    # tf.io.gfile.glob works like glob.glob without importing glob/os
    paths = sorted(tf.io.gfile.glob(test_dir.rstrip("/") + "/*.png"))
    imgs = []
    names = []
    for p in paths:
        data = tf.io.read_file(p)
        img  = tf.image.decode_png(data, channels=1)
        img  = tf.image.resize(img, target_size)
        img  = tf.cast(img, tf.float32) / 255.0
        imgs.append(img)
        # Extract just the filename
        names.append(p.split("/")[-1])
    if not imgs:
        raise ValueError(f"No PNG files found in '{test_dir}'")
    return tf.stack(imgs, axis=0), names

def main():
    # 1) Prompt user for test directory
    test_dir = input("Enter the test-images directory (path containing .png files): ").strip()
    # 2) Load the model
    print(f"Loading model from: {MODEL_FILE}")
    model = tf.keras.models.load_model(MODEL_FILE)

    # 3) Load & preprocess
    print(f"Loading and preprocessing images from: {test_dir}")
    X, filenames = load_and_preprocess_images(test_dir, target_size=model.input_shape[1:3])
    print(f"  → {len(filenames)} images loaded.")

    # 4) Predict
    print("Running inference...")
    probs = model.predict(X, verbose=0).flatten()
    preds = (probs >= 0.5).astype(int)

    # 5) Build DataFrame
    df = pd.DataFrame({
        "ImageFilename": filenames,
        "PredictedLabel": preds
    })

    # 6) Summary counts
    counts  = df["PredictedLabel"].value_counts().sort_index()
    summary = counts.rename_axis("Label").reset_index(name="TotalImages")

    # 7) Save to Excel
    print(f"Writing results to '{EXCEL_OUT}'")
    with pd.ExcelWriter(EXCEL_OUT) as writer:
        df.to_excel(writer,      sheet_name="Predictions", index=False)
        summary.to_excel(writer, sheet_name="Summary",     index=False)

    print("Done.")

if __name__ == "__main__":
    main()
