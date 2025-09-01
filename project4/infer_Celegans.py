import os
import glob
import numpy as np
import pandas as pd
from PIL import Image


# ---------------------------
# Helper Functions
# ---------------------------
def load_unlabeled_images_from_directory(directory, image_size=(28, 28)):
    """
    Loads all PNG images from a directory, converts them to grayscale,
    resizes them to a fixed size, and returns a list of image arrays along with the filenames.

    Args:
        directory (str): Directory containing PNG images.
        image_size (tuple): Desired image size (height, width).

    Returns:
        tuple: (list of image arrays, list of filenames)
    """
    image_paths = glob.glob(os.path.join(directory, '*.png'))
    images_list = []
    filenames = []

    # Process each image file
    for path in sorted(image_paths):
        with Image.open(path) as img:
            # Convert image to grayscale and resize
            img = img.convert('L')
            img = img.resize(image_size)
            img_array = np.array(img)
        images_list.append(img_array)
        # Use the filename only (without full path)
        filenames.append(os.path.basename(path))

    return images_list, filenames


def preprocess_images(images):
    """
    Flattens and normalizes images.

    Args:
        images (list or np.array): List of image arrays of shape (H, W).

    Returns:
        np.array: Array of shape (N, H*W) with normalized pixel values.
    """
    X = np.array(images)  # (N, H, W)
    N = X.shape[0]
    # Flatten each image to a vector and normalize to [0,1]
    X = X.reshape(N, -1).astype(np.float32) / 255.0
    return X


def augment_with_bias(X):
    """
    Augments the feature matrix with a bias column.

    Args:
        X (np.array): Feature matrix of shape (N, D).

    Returns:
        np.array: Augmented matrix of shape (N, D+1).
    """
    N = X.shape[0]
    ones = np.ones((N, 1), dtype=X.dtype)
    return np.hstack((X, ones))


def design_matrix_poly(X, degree):
    """
    Constructs a polynomial design matrix from the input features.
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


def softmax(z):
    """
    Computes softmax probabilities in a numerically stable manner.

    Args:
        z (np.array): Logits of shape (N, C).

    Returns:
        np.array: Softmax probabilities with shape (N, C).
    """
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# ---------------------------
# Main Testing and Output
# ---------------------------
if __name__ == '__main__':
    # Prompt the user to enter the directory path containing the images
    images_dir = input("Please enter the directory path containing the images: ").strip()

    # Verify that the directory exists
    if not os.path.isdir(images_dir):
        print("The provided directory does not exist. Exiting.")
        exit(1)

    print(f"Processing images from directory: {images_dir}")

    # Load the images and obtain the filenames
    images, filenames = load_unlabeled_images_from_directory(images_dir, image_size=(28, 28))
    print(f"Loaded {len(images)} images.")

    # Preprocess the images: flatten and normalize
    X = preprocess_images(images)

    # Augment features with a bias term
    X_aug = augment_with_bias(X)

    # Load the saved model parameters (W_best and poly_degree)
    params = np.load('model_parameters_celegans.npz', allow_pickle=True)
    W_best = params['W_best']
    poly_degree = int(params['poly_degree'])
    print(f"Loaded model parameters with polynomial degree: {poly_degree}")

    # Build the polynomial design matrix for the test images
    X_poly = design_matrix_poly(X_aug, poly_degree)

    # Run the model: compute logits, softmax probabilities, and predicted labels
    logits = np.dot(X_poly, W_best)
    probabilities = softmax(logits)
    predicted_labels = np.argmax(probabilities, axis=1)

    # Create a DataFrame for predictions with image filenames and predicted labels
    predictions_df = pd.DataFrame({
        "ImageFilename": filenames,
        "PredictedLabel": predicted_labels
    })

    # Create a summary DataFrame that counts total images for each predicted label
    label_counts = predictions_df["PredictedLabel"].value_counts().sort_index()
    summary_df = label_counts.reset_index()
    summary_df.columns = ["Label", "TotalImages"]

    # Write the outputs to an Excel file with two sheets: Predictions and Summary
    excel_filename = "Celegans_Classification_Results.xlsx"
    with pd.ExcelWriter(excel_filename) as writer:
        predictions_df.to_excel(writer, sheet_name="Predictions", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    print(f"Classification results saved to Excel file: {excel_filename}")
