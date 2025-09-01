import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

# Set random seed for reproducibility.
np.random.seed(50)


# ---------------------------------------------------
# 1. Data Extraction Function (MNIST)
# ---------------------------------------------------
def load_mnist(images_path, labels_path):
    """
    Loads MNIST images and labels from IDX files.

    Args:
        images_path (str): Path to the images IDX file.
        labels_path (str): Path to the labels IDX file.

    Returns:
        tuple: (images, labels)
    """
    images = idx2numpy.convert_from_file(images_path)
    labels = idx2numpy.convert_from_file(labels_path)
    return images, labels


# ---------------------------------------------------
# 2. Preprocessing Functions
# ---------------------------------------------------
def preprocess_data(images, labels, num_classes=10):
    """
    Flattens images into a 1D feature vector, normalizes the pixel values to [0, 1],
    and one-hot encodes the labels.

    Args:
        images (np.array): Array of images (N, H, W).
        labels (np.array): Array of labels (N,).
        num_classes (int): Number of classes (10 for MNIST).

    Returns:
        X (np.array): Feature matrix, shape (N, H*W).
        T (np.array): One-hot encoded labels, shape (N, num_classes).
    """
    N = images.shape[0]
    X = images.reshape(N, -1).astype(np.float32) / 255.0
    T = np.zeros((N, num_classes))
    T[np.arange(N), labels] = 1
    return X, T


# ---------------------------------------------------
# 3. Polynomial Mapping Function (without bias)
# ---------------------------------------------------
def design_matrix_poly(X, degree):
    """
    Constructs a polynomial feature matrix for multi-dimensional input.
    Unlike before, we do NOT include the bias term since we will learn it separately.

    For each sample in X (shape (N, D)), this creates features by taking elementwise powers
    up to the given degree.

    Args:
        X (np.array): Original feature matrix of shape (N, D).
        degree (int): Polynomial degree.

    Returns:
        np.array: Polynomial design matrix of shape (N, D * degree).
    """
    # Compute polynomial features for each degree from 1 to degree.
    features = [X ** d for d in range(1, degree + 1)]
    # Concatenate along the feature (column) axis.
    phi = np.hstack(features)
    return phi


# ---------------------------------------------------
# 4. Model Functions: Softmax, Loss, Accuracy & Training
# ---------------------------------------------------
def softmax(z):
    """
    Compute softmax probabilities for each row in z.

    Args:
        z (np.array): Array of logits of shape (N, C)

    Returns:
        np.array: Softmax probabilities of shape (N, C).
    """
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def compute_loss(X, T, W, b, lambda_reg=0.0):
    """
    Compute the average cross-entropy loss with L2 (weight decay) regularization.

    Args:
        X (np.array): Feature matrix of shape (N, D).
        T (np.array): One-hot encoded labels of shape (N, C).
        W (np.array): Weight matrix of shape (D, C).
        b (np.array): Bias vector of shape (C,).
        lambda_reg (float): Regularization strength.

    Returns:
        float: The regularized loss.
    """
    logits = np.dot(X, W) + b  # add bias separately
    probs = softmax(logits)
    data_loss = -np.sum(T * np.log(probs + 1e-8)) / X.shape[0]
    reg_loss = (lambda_reg / 2) * np.sum(W * W)
    return data_loss + reg_loss


def compute_accuracy(X, labels, W, b):
    """
    Computes the classification accuracy.

    Args:
        X (np.array): Feature matrix (N, D)
        labels (np.array): True labels as integers (N,)
        W (np.array): Weight matrix (D, C)
        b (np.array): Bias vector (C,)

    Returns:
        float: Accuracy (fraction of correct predictions)
    """
    logits = np.dot(X, W) + b
    probs = softmax(logits)
    pred_labels = np.argmax(probs, axis=1)
    return np.mean(pred_labels == labels)


def train_softmax_regression(X_train, T_train, X_val, T_val,
                             num_epochs=80, batch_size=64,
                             learning_rate=0.01, momentum_decay=0.75,
                             lambda_reg=0.01):
    """
    Train a single-layer softmax regression model using mini-batch gradient descent with momentum
    and L2 regularization.

    Args:
        X_train (np.array): Training feature matrix, shape (N, D).
        T_train (np.array): Training one-hot labels, shape (N, C).
        X_val (np.array): Validation feature matrix, shape (N_val, D).
        T_val (np.array): Validation one-hot labels, shape (N_val, C).
        num_epochs (int): Number of training epochs.
        batch_size (int): Mini-batch size.
        learning_rate (float): Learning rate.
        momentum_decay (float): Momentum decay factor.
        lambda_reg (float): Weight decay regularization strength.

    Returns:
        tuple: (Final trained weights, Best weights based on validation loss, training time in seconds, final bias vector)
    """
    best_val_loss = float('inf')
    W_best = None
    b_best = None

    N, D = X_train.shape
    _, C = T_train.shape

    # Initialize weights and bias separately.
    W = np.random.randn(D, C) * 0.01
    b = np.zeros(C)
    momentum_W = np.zeros_like(W)
    momentum_b = np.zeros_like(b)

    start_time = time.time()

    for epoch in range(num_epochs):
        # Shuffle training data at each epoch.
        indices = np.random.permutation(N)
        X_shuffled = X_train[indices]
        T_shuffled = T_train[indices]

        for i in range(0, N, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            T_batch = T_shuffled[i:i + batch_size]

            # Forward pass: compute logits = X_batch * W + b.
            logits = np.dot(X_batch, W) + b
            probs = softmax(logits)

            # Compute gradient of loss wrt logits.
            grad_logits = (probs - T_batch) / X_batch.shape[0]
            # Gradients w.r.t. weights and bias.
            grad_W = np.dot(X_batch.T, grad_logits)  # shape (D, C)
            grad_b = np.sum(grad_logits, axis=0)  # shape (C,)
            # Add regularization gradient for W.
            grad_W += lambda_reg * W

            # Update momentum for weights and bias.
            momentum_W = momentum_decay * momentum_W + grad_W
            momentum_b = momentum_decay * momentum_b + grad_b

            # Update parameters.
            W -= learning_rate * momentum_W
            b -= learning_rate * momentum_b

        train_loss = compute_loss(X_train, T_train, W, b, lambda_reg)
        val_loss = compute_loss(X_val, T_val, W, b, lambda_reg)
        print(f"Epoch {epoch + 1}/{num_epochs}  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            W_best = W.copy()
            b_best = b.copy()

    training_time = time.time() - start_time
    return W, W_best, training_time, b_best


# ---------------------------------------------------
# 5. Main Execution
# ---------------------------------------------------
if __name__ == '__main__':
    # 5.1 Load MNIST training data.
    train_images_path = 'MNIST/train-images.idx3-ubyte'
    train_labels_path = 'MNIST/train-labels.idx1-ubyte'
    test_images_path = 'MNIST/t10k-images.idx3-ubyte'
    test_labels_path = 'MNIST/t10k-labels.idx1-ubyte'

    train_images, train_labels = load_mnist(train_images_path, train_labels_path)
    print("Train images shape:", train_images.shape)  # (60000, 28, 28)
    print("Train labels shape:", train_labels.shape)  # (60000,)

    # Preprocess training data.
    X, T = preprocess_data(train_images, train_labels)
    print("Feature matrix shape:", X.shape)  # (60000, 784)
    print("One-hot label matrix shape:", T.shape)  # (60000, 10)

    # Create a training/validation split (90% training, 10% validation).
    N = X.shape[0]
    split_index = int(0.9 * N)
    X_train, X_val = X[:split_index], X[split_index:]
    T_train, T_val = T[:split_index], T[split_index:]
    train_labels_split = train_labels[:split_index]
    val_labels_split = train_labels[split_index:]
    print("Training set:", X_train.shape, T_train.shape)
    print("Validation set:", X_val.shape, T_val.shape)

    # 5.2 Apply Polynomial Mapping (without bias).
    poly_degree = 3  # You can set to 3 for more nonlinearity.
    X_train_poly = design_matrix_poly(X_train, poly_degree)
    X_val_poly = design_matrix_poly(X_val, poly_degree)
    print(f"Using polynomial degree = {poly_degree}")
    print("Transformed training feature shape:", X_train_poly.shape)
    print("Transformed validation feature shape:", X_val_poly.shape)

    # 5.3 Train Softmax Regression with Regularization and separate bias.
    num_epochs = 75
    batch_size = 64
    learning_rate = 0.015
    momentum_decay = 0.7
    lambda_reg = 0.0000  # Regularization strength

    print("\nStarting training...")
    W_trained, W_best, training_time, b_best = train_softmax_regression(
        X_train_poly, T_train,
        X_val_poly, T_val,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        momentum_decay=momentum_decay,
        lambda_reg=lambda_reg
    )

    # 5.4 Evaluate on Validation and Test Data.
    val_accuracy = compute_accuracy(X_val_poly, val_labels_split, W_best, b_best)
    print(f"\nValidation Accuracy: {val_accuracy * 100:.2f}%")

    # Load test data.
    test_images, test_labels = load_mnist(test_images_path, test_labels_path)
    X_test, _ = preprocess_data(test_images, test_labels)
    # Apply the same polynomial mapping to test data.
    X_test_poly = design_matrix_poly(X_test, poly_degree)
    test_accuracy = compute_accuracy(X_test_poly, test_labels, W_best, b_best)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Save model parameters.
    np.savez('model_parameters.npz', W_best=W_best, b_best=b_best, poly_degree=poly_degree)
    print("Model parameters saved to 'model_parameters.npz'")
