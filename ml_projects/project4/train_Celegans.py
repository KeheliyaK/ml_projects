import os
import glob
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import time

# Set a random seed for reproducibility.
np.random.seed(50)


# ============================
# 1. DATA LOADING FUNCTIONS
# ============================
def load_images_from_directory(directory, label, image_size=(28, 28)):
    """
    Loads all PNG images from a directory, converts them to grayscale,
    resizes them to a fixed size, and assigns the provided label.

    Args:
        directory (str): Directory containing PNG images.
        label (int): Label for these images (e.g., 0 or 1).
        image_size (tuple): Desired image size (H, W).

    Returns:
        tuple: (list of image arrays, list of labels)
    """
    image_paths = glob.glob(os.path.join(directory, '*.png'))
    images_list = []
    labels_list = []

    for path in image_paths:
        img = Image.open(path)
        img = img.convert('L')  # convert to grayscale
        img = img.resize(image_size)  # resize the image
        img_array = np.array(img)
        images_list.append(img_array)
        labels_list.append(label)

    return images_list, labels_list


# ============================
# 2. PREPROCESSING FUNCTIONS
# ============================
def preprocess_images(images, labels, num_classes=2):
    """
    Converts a list of images to a NumPy array, flattens each image,
    normalizes pixel values to [0,1], and one-hot encodes the labels.

    Args:
        images (list or np.array): List of image arrays (H x W).
        labels (list or np.array): List/array of labels.
        num_classes (int): Number of output classes (default: 2).

    Returns:
        X (np.array): Preprocessed feature matrix of shape (N, H*W).
        T (np.array): One-hot encoded labels of shape (N, num_classes).
    """
    X = np.array(images)  # shape: (N, H, W)
    N = X.shape[0]
    # Flatten images from (H, W) to (H*W,)
    X = X.reshape(N, -1).astype(np.float32)
    # Normalize pixel values to [0,1]
    X = X / 255.0

    # One-hot encode labels
    T = np.zeros((N, num_classes))
    T[np.arange(N), labels] = 1
    return X, T


def augment_with_bias(X):
    """
    Adds a bias term (column of ones) to the feature matrix.

    Args:
        X (np.array): Input features of shape (N, D).

    Returns:
        np.array: Augmented feature matrix of shape (N, D+1).
    """
    N = X.shape[0]
    ones = np.ones((N, 1), dtype=X.dtype)
    return np.hstack((X, ones))


def design_matrix_poly(X, degree):
    """
    Generates a polynomial design matrix from the input features.

    Each feature is raised element-wise to powers from 1 up to `degree`,
    and a bias term is included as the first column.

    Args:
        X (np.array): Input feature matrix of shape (N, D).
        degree (int): The degree of polynomial expansion.

    Returns:
        np.array: The design matrix of shape (N, 1 + D * degree).
    """
    N, D = X.shape
    phi = np.ones((N, 1))  # bias term
    for d in range(1, degree + 1):
        phi = np.hstack((phi, X ** d))
    return phi


# ============================
# 3. MODEL FUNCTIONS: SOFTMAX REGRESSION
# ============================
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def compute_loss(X, T, W, reg_lambda=0.0):
    """
    Computes the cross-entropy loss given data X, one-hot labels T,
    and weight matrix W. Includes L2 regularization (weight decay).

    Args:
        X (np.array): Feature matrix.
        T (np.array): One-hot encoded labels.
        W (np.array): Weight matrix.
        reg_lambda (float): Regularization strength.

    Returns:
        float: The regularized loss.
    """
    logits = np.dot(X, W)
    probs = softmax(logits)
    data_loss = -np.sum(T * np.log(probs + 1e-8)) / X.shape[0]
    reg_loss = 0.5 * reg_lambda * np.sum(W * W)
    return data_loss + reg_loss


def train_softmax_regression(X_train, T_train, X_val, T_val,
                             num_epochs=200, batch_size=64,
                             learning_rate=0.0001, momentum_decay=0.65,
                             reg_lambda=0.0005):
    """
    Trains a single-layer softmax regression model (with bias and polynomial features),
    using mini-batch gradient descent with momentum and weight decay.

    Args:
        X_train (np.array): Training features, shape (N_train, D).
        T_train (np.array): One-hot training labels.
        X_val (np.array): Validation features.
        T_val (np.array): One-hot validation labels.
        num_epochs (int): Number of epochs.
        batch_size (int): Mini-batch size.
        learning_rate (float): Learning rate.
        momentum_decay (float): Momentum decay factor.
        reg_lambda (float): Regularization strength (weight decay).

    Returns:
        tuple: (final weights, best weights according to validation loss, training time)
    """
    N, D = X_train.shape
    _, C = T_train.shape
    W = np.random.randn(D, C) * 0.01
    momentum = np.zeros_like(W)
    best_val_loss = float('inf')
    W_best = None

    # Optional real-time plotting
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    train_line, = ax.plot([], [], 'b-', label='Train Loss')
    val_line, = ax.plot([], [], 'r-', label='Val Loss')
    ax.legend()

    train_losses = []
    val_losses = []
    epochs = []

    start_time = time.time()
    for epoch in range(num_epochs):
        # Shuffle training data for each epoch
        indices = np.random.permutation(N)
        X_train = X_train[indices]
        T_train = T_train[indices]

        for i in range(0, N, batch_size):
            X_batch = X_train[i:i + batch_size]
            T_batch = T_train[i:i + batch_size]
            batch_N = X_batch.shape[0]

            # Forward pass
            logits = np.dot(X_batch, W)
            probs = softmax(logits)

            # Gradient computation with weight decay
            grad = np.dot(X_batch.T, (probs - T_batch)) / batch_N
            grad += reg_lambda * W

            # Momentum update
            momentum = momentum_decay * momentum + grad
            W -= learning_rate * momentum

        train_loss = compute_loss(X_train, T_train, W, reg_lambda=reg_lambda)
        val_loss = compute_loss(X_val, T_val, W, reg_lambda=reg_lambda)
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            W_best = W.copy()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epochs.append(epoch + 1)
        train_line.set_xdata(epochs)
        train_line.set_ydata(train_losses)
        val_line.set_xdata(epochs)
        val_line.set_ydata(val_losses)
        ax.relim()
        ax.autoscale_view(True, True, True)
        plt.draw()
        plt.pause(0.001)

    training_time = time.time() - start_time
    plt.ioff()
    plt.show()
    return W, W_best, training_time


def compute_accuracy(X, labels, W):
    logits = np.dot(X, W)
    probs = softmax(logits)
    pred_labels = np.argmax(probs, axis=1)
    return np.mean(pred_labels == labels)


# ============================
# 4. DATA SPLITTING AND TRAINING (NO GRID SEARCH)
# ============================
if __name__ == '__main__':
    # --- Load images from directories ---
    # Directories for each class (update these paths as needed)
    class0_dir = 'Celegans_ModelGen/0'  # Images labeled 0
    class1_dir = 'Celegans_ModelGen/1'  # Images labeled 1

    images0, labels0 = load_images_from_directory(class0_dir, label=0, image_size=(28, 28))
    images1, labels1 = load_images_from_directory(class1_dir, label=1, image_size=(28, 28))

    # Combine data from both classes
    all_images = images0 + images1
    all_labels = labels0 + labels1

    # Shuffle the combined data
    combined = list(zip(all_images, all_labels))
    random.shuffle(combined)
    all_images, all_labels = zip(*combined)

    # Preprocess the images and labels (flatten, normalize, one-hot encode)
    X, T = preprocess_images(all_images, all_labels, num_classes=2)
    total_images = X.shape[0]
    image_height, image_width = images0[0].shape  # assuming all images have same size

    print("Total images loaded:", total_images)
    print("Image size: {} x {}".format(image_width, image_height))

    # --- Data Splitting: 70% Train, 10% Validation, 20% Test ---
    indices = np.arange(total_images)
    np.random.shuffle(indices)

    test_size = int(0.2 * total_images)
    val_size = int(0.1 * total_images)
    train_size = total_images - test_size - val_size  # Should be ~70%

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    X_train = X[train_indices]
    T_train = T[train_indices]
    X_val = X[val_indices]
    T_val = T[val_indices]
    X_test = X[test_indices]
    T_test = T[test_indices]
    test_labels = np.argmax(T_test, axis=1)

    print(f"Training set size: {X_train.shape[0]} images")
    print(f"Validation set size: {X_val.shape[0]} images")
    print(f"Test set size: {X_test.shape[0]} images")

    # --- Augment data with a bias term ---
    X_train_aug = augment_with_bias(X_train)
    X_val_aug = augment_with_bias(X_val)
    X_test_aug = augment_with_bias(X_test)

    # --- Create polynomial design matrices with optimum hyperparameters ---
    # (Using the pre-determined optimum: poly_degree=8)
    poly_degree = 8
    X_train_poly = design_matrix_poly(X_train_aug, poly_degree)
    X_val_poly = design_matrix_poly(X_val_aug, poly_degree)
    X_test_poly = design_matrix_poly(X_test_aug, poly_degree)

    # --- Train the Model using optimum hyperparameters ---
    # Optimum hyperparameters:
    #   learning_rate = 0.0001, momentum_decay = 0.65, reg_lambda = 0.0005,
    #   num_epochs = 200, batch_size = 64
    print("\nStarting Training...")
    W_final, W_best, training_time = train_softmax_regression(
        X_train_poly, T_train, X_val_poly, T_val,
        num_epochs=200,
        batch_size=64,
        learning_rate=0.0001,
        momentum_decay=0.65,
        reg_lambda=0.0005
    )

    # --- Evaluate on Test Data ---
    print("\nEvaluating on Test Data...")
    test_start_time = time.time()
    test_accuracy = compute_accuracy(X_test_poly, test_labels, W_best)
    test_time = time.time() - test_start_time
    print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

    # --- Display Final Statistics ---
    print("\n----- FINAL STATISTICS -----")
    print(f"Image Size: {image_width} x {image_height}")
    print(f"Total Images: {total_images}")
    print(f"Training Set Size: {X_train.shape[0]} images")
    print(f"Validation Set Size: {X_val.shape[0]} images")
    print(f"Test Set Size: {X_test.shape[0]} images")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Testing Time: {test_time:.2f} seconds")

    # --- Save the Best Model Parameters for later use ---
    #np.savez('model_parameters_celegans.npz', W_best=W_best, poly_degree=poly_degree)
    #print("Model parameters saved to 'model_parameters_celegans.npz'")

import numpy as np
import matplotlib.pyplot as plt


def compute_confusion_matrix(y_true, y_pred, labels=[0, 1]):
    """
    Computes the confusion matrix for the provided true and predicted labels.

    Args:
        y_true (np.array): True labels (1D array).
        y_pred (np.array): Predicted labels (1D array).
        labels (list): List of label values. Default is [0, 1].

    Returns:
        np.array: A confusion matrix of shape (len(labels), len(labels)).
    """
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for i, actual in enumerate(labels):
        for j, pred in enumerate(labels):
            cm[i, j] = np.sum((y_true == actual) & (y_pred == pred))
    return cm


def plot_confusion_matrix(cm, labels=[0, 1]):
    """
    Plots the confusion matrix using matplotlib.

    Args:
        cm (np.array): Confusion matrix.
        labels (list): List of label names for the axes.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Set axis labels
    ax.set(xticks=np.arange(len(labels)),
           yticks=np.arange(len(labels)),
           xticklabels=labels, yticklabels=labels,
           xlabel='Predicted Label',
           ylabel='True Label',
           title='Confusion Matrix')

    # Rotate the tick labels and set alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()


# Example usage:
# Assuming you have computed the predictions for your test set:
#   - X_test_poly: the test features (after polynomial mapping and bias augmentation)
#   - W_best: the best weight matrix from training
#   - test_labels: true labels (1D array)
#
# Compute predictions:
logits_test = np.dot(X_test_poly, W_best)
probs_test = softmax(logits_test)
predicted_labels = np.argmax(probs_test, axis=1)

# Compute the confusion matrix
cm = compute_confusion_matrix(test_labels, predicted_labels, labels=[0, 1])
print("Confusion Matrix:")
print(cm)

# Plot the confusion matrix
plot_confusion_matrix(cm, labels=[0, 1])
