#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────────────
script_dir    = Path(__file__).resolve().parent
data_dir      = script_dir / "Celegans"   # expects Celegans/0, Celegans/1
batch_size    = 32
img_h, img_w  = 32, 32
seed          = 123
epochs        = 20
learning_rate = 5e-4

# ── Gather file paths & labels ─────────────────────────────────────────────────
class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
label_map   = {name: idx for idx, name in enumerate(class_names)}

all_paths  = []
all_labels = []
for class_name in class_names:
    for png_path in (data_dir / class_name).glob("*.png"):
        all_paths.append(str(png_path))
        all_labels.append(label_map[class_name])

all_paths  = np.array(all_paths)
all_labels = np.array(all_labels)

# ── Shuffle & split: 85% train, 15% val ────────────────────────────────────────
rng = np.random.RandomState(seed)
idx = rng.permutation(len(all_paths))
all_paths, all_labels = all_paths[idx], all_labels[idx]

n = len(all_paths)
n_train = int(0.85 * n)

train_paths, train_labels = all_paths[:n_train], all_labels[:n_train]
val_paths,   val_labels   = all_paths[n_train:],  all_labels[n_train:]

# ── Preprocessing function ─────────────────────────────────────────────────────
def load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_h, img_w])
    return img, label

# ── Dataset builder ────────────────────────────────────────────────────────────
def make_ds(paths, labels, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(buffer_size=len(paths), seed=seed)
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        aug = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
            layers.GaussianNoise(0.05),
        ], name="data_augmentation")
        ds = ds.map(lambda x, y: (aug(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# ── Create tf.data pipelines ────────────────────────────────────────────────────
train_ds = make_ds(train_paths, train_labels, augment=True)
val_ds   = make_ds(val_paths,   val_labels,   augment=False)
test_ds  = make_ds(all_paths,   all_labels,   augment=False)  # use all data for final test

# ── Build & compile model ───────────────────────────────────────────────────────
kernel_sizes = [(5,5), (3,3), (3,3)]

inputs = keras.Input(shape=(img_h, img_w, 1))
x = inputs

# Conv block 1
x = layers.Conv2D(64, kernel_sizes[0], padding="same", activation="relu")(x)
x = layers.MaxPooling2D(padding="same")(x)

# Conv block 2
x = layers.Conv2D(128, kernel_sizes[1], padding="same", activation="relu")(x)
x = layers.MaxPooling2D(padding="same")(x)

# Conv block 3
x = layers.Conv2D(128, kernel_sizes[2], padding="same", activation="relu")(x)
x = layers.MaxPooling2D(padding="same")(x)

x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs, name="cnn_5x5-3x3-3x3")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ── Train & validate ───────────────────────────────────────────────────────────
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    verbose=1
)

# ── Evaluate on full dataset (100%) ────────────────────────────────────────────
print("\nEvaluating on 100% of data (test):")
test_loss, test_acc = model.evaluate(test_ds, verbose=1)
print(f"Test loss: {test_loss:.4f} — Test accuracy: {test_acc:.4f}")

# ── Confusion matrix utilities ─────────────────────────────────────────────────
def compute_confusion_matrix(y_true, y_pred, labels=[0, 1]):
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for i, actual in enumerate(labels):
        for j, pred in enumerate(labels):
            cm[i, j] = np.sum((y_true == actual) & (y_pred == pred))
    return cm

def plot_confusion_matrix(cm, labels=[0, 1]):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel='Predicted Label',
        ylabel='True Label',
        title='Confusion Matrix'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j],
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()

# ── Compute & plot confusion matrix ────────────────────────────────────────────
y_true = []
y_pred = []

for x_batch, y_batch in test_ds:
    probs = model.predict(x_batch, verbose=0).flatten()
    preds = (probs >= 0.5).astype(int)
    y_true.extend(y_batch.numpy())
    y_pred.extend(preds)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

cm = compute_confusion_matrix(y_true, y_pred, labels=[0, 1])
print("\nConfusion Matrix:\n", cm)
plot_confusion_matrix(cm, labels=[0, 1])
