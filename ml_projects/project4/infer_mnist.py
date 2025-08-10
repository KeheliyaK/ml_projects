import os
from PIL import Image
import numpy as np
import pandas as pd


# ---------------------------
# Helper Functions
# ---------------------------
def load_tif_images(directory):
    """
    Loads .tif images from the specified directory.

    Assumes each image is 28x28 grayscale.
    Returns an array of images and a list of corresponding filenames.

    Args:
        directory (str): Path to the directory containing .tif images.

    Returns:
        tuple: (images, filenames)
            images: numpy array of shape (N, 28, 28)
            filenames: list of image filenames
    """
    images = []
    filenames = []
    for file in sorted(os.listdir(directory)):
        if file.lower().endswith('.tif'):
            file_path = os.path.join(directory, file)
            with Image.open(file_path) as img:
                # Convert to grayscale in case the image is not already
                img = img.convert('L')
                img_array = np.array(img)
            images.append(img_array)
            filenames.append(file)
    images = np.array(images)
    return images, filenames


def preprocess_data(images):
    """
    Flattens and normalizes images.

    Args:
        images (np.array): Array of images with shape (N, 28, 28).

    Returns:
        np.array: Flattened and normalized images with shape (N, 784).
    """
    N = images.shape[0]
    X = images.reshape(N, -1).astype(np.float32) / 255.0
    return X


def augment_with_bias(X):
    """
    Augments the feature matrix with a bias term (column of ones).

    Args:
        X (np.array): Feature matrix with shape (N, D).

    Returns:
        np.array: Augmented matrix of shape (N, D+1).
    """
    N = X.shape[0]
    ones = np.ones((N, 1), dtype=X.dtype)
    return np.hstack((X, ones))


def design_matrix_poly(X, degree):
    """
    Constructs a polynomial design matrix from the input features.

    For each sample in X (shape (N, D)), the function builds a new matrix that
    includes a bias term and the element-wise polynomial terms up to the specified degree.
    (Cross-terms are not included.)

    Args:
        X (np.array): Input feature matrix of shape (N, D).
        degree (int): The polynomial degree.

    Returns:
        np.array: Design matrix of shape (N, 1 + D * degree).
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
# Main Execution
# ---------------------------
if __name__ == '__main__':
    # Prompt the user to enter the directory path containing the images
    images_dir = input("Please enter the directory path containing the images: ").strip()

    # Verify that the directory exists
    if not os.path.isdir(images_dir):
        print("The provided directory does not exist. Exiting.")
        exit(1)

    print(f"Processing images from directory: {images_dir}")

    # Load images and filenames from the directory
    images, filenames = load_tif_images(images_dir)
    print(f"Loaded {images.shape[0]} images from directory: {images_dir}")

    # Preprocess the images: flatten and normalize
    X = preprocess_data(images)
    X_aug = augment_with_bias(X)

    # Load saved model parameters (W_best and poly_degree)
    params = np.load('model_parameters_MNIST.npz', allow_pickle=True)
    W_best = params['W_best']
    poly_degree = int(params['poly_degree'])

    # Build the polynomial design matrix for the test images
    X_poly = design_matrix_poly(X_aug, poly_degree)

    # Run the model: compute logits, softmax probabilities and predicted labels
    logits = np.dot(X_poly, W_best)
    probabilities = softmax(logits)
    predicted_labels = np.argmax(probabilities, axis=1)

    # Create a DataFrame for predictions with image filename and predicted label
    predictions_df = pd.DataFrame({
        "ImageFilename": filenames,
        "PredictedLabel": predicted_labels
    })

    # Create a summary DataFrame that counts the total number of images for each predicted label
    label_counts = predictions_df["PredictedLabel"].value_counts().sort_index()
    summary_df = label_counts.reset_index()
    summary_df.columns = ["Label", "TotalImages"]

    # Write the predictions and summary to an Excel file
    excel_filename = "MNIST_TIF_Classification_Results.xlsx"
    with pd.ExcelWriter(excel_filename) as writer:
        predictions_df.to_excel(writer, sheet_name="Predictions", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    print(f"Classification results saved to Excel file: {excel_filename}")
