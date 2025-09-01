import numpy as np
import matplotlib.pyplot as plt


# ------------------------
# Helper Functions
# ------------------------

def generate_dataset(N, noise_sigma=0.3):
    """
    Generates a dataset with N samples.
    x ~ U(0,1) and t = sin(2*pi*x) + noise, with noise ~ N(0, noise_sigma)
    """
    x = np.random.uniform(0, 1, (N, 1))
    t = np.sin(2 * np.pi * x) + np.random.normal(0, noise_sigma, (N, 1))
    return x, t


def design_matrix(x, m, s):
    """
    Build the design matrix using Gaussian RBFs.
    x is (N,1); m is the number of basis functions; s is the width.
    Centers are chosen evenly in [0,1].
    """
    centers = np.linspace(0, 1, m)  # shape (m,)
    # Using broadcasting: result is (N, m)
    phi = np.exp(-((x - centers) ** 2) / (2 * s ** 2))
    return phi


def fit_model(x, t, m=6, s=0.1, lamda=0.0):
    """
    Fit the model using regularized least squares (ridge regression).
    Returns the predicted values on the input x and the weights.
    """
    phi = design_matrix(x, m, s)  # (N, m)
    I = np.eye(phi.shape[1])
    # Compute regularized weights:
    w = np.linalg.inv(phi.T @ phi + lamda * I) @ (phi.T @ t)
    # Compute predictions:
    y = phi @ w
    return y, w


# ------------------------
# Experiment Function
# ------------------------

def run_experiment(lamda, L=100, N_train=25, m=6, s=0.1):
    """
    For a fixed lambda, run L iterations.
    In each iteration:
      - Generate a training set of N_train samples.
      - Fit the model with regularization.
      - Evaluate the model on a fixed evaluation set (for bias/variance)
        and on a fixed test set (of 1000 points for test error).
    Returns:
      bias_squared, variance, and test_error.
    """
    # Fixed evaluation set (for bias & variance) - use N_train points for evaluation.
    x_eval = np.linspace(0, 1, N_train).reshape(-1, 1)
    # Fixed test set (for test error)
    N_test = 1000
    x_test = np.linspace(0, 1, N_test).reshape(-1, 1)
    h_test = np.sin(2 * np.pi * x_test)  # true function on test set

    # Lists to store predictions on evaluation set and test set for each iteration.
    preds_eval = []
    preds_test = []

    for l in range(L):
        # Generate training set (each iteration uses different data)
        x_train, t_train = generate_dataset(N_train, noise_sigma=0.3)
        # Fit model with regularization:
        _, w = fit_model(x_train, t_train, m, s, lamda)
        # Evaluate on x_eval:
        phi_eval = design_matrix(x_eval, m, s)
        y_eval = phi_eval @ w
        preds_eval.append(y_eval.ravel())
        # Evaluate on test set:
        phi_test = design_matrix(x_test, m, s)
        y_test = phi_test @ w
        preds_test.append(y_test.ravel())

    # Convert to arrays: shapes (L, N_train) and (L, N_test)
    preds_eval = np.array(preds_eval)
    preds_test = np.array(preds_test)

    # Compute average prediction at each evaluation point (over L iterations)
    f_bar = np.mean(preds_eval, axis=0)  # shape: (N_train,)
    # True function values on evaluation set:
    h_eval = np.sin(2 * np.pi * x_eval).ravel()

    # Compute squared bias:
    bias_squared = np.mean((f_bar - h_eval) ** 2)

    # Compute variance: first compute variance at each evaluation point over L iterations, then average.
    variance = np.mean(np.mean((preds_eval - f_bar) ** 2, axis=0))

    # Compute test error: for each iteration, compute mean squared error on test set, then average.
    test_errors = [np.mean((preds_test[l] - np.sin(2 * np.pi * x_test).ravel()) ** 2) for l in range(L)]
    test_error = np.mean(test_errors)

    return bias_squared, variance, test_error


# ------------------------
# Main Loop over lambda values
# ------------------------

# Generate ln(lambda) values: we want ln(lambda) from -2.5 to 1.5 in steps of 0.25.
ln_lambda = np.arange(-2.5, 1.5 + 0.25, 0.25)  # or use linspace with appropriate number of points
lambda_values = np.exp(ln_lambda)  # corresponding lambda values

# Lists to store the computed metrics for each lambda.
bias_all = []
variance_all = []
test_error_all = []

# For reproducibility, you may set a seed here if desired.
np.random.seed(50)

for lamda in lambda_values:
    bias2, var, test_err = run_experiment(lamda, L=100, N_train=25, m=6, s=0.1)
    bias_all.append(bias2)
    variance_all.append(var)
    test_error_all.append(test_err)

# ------------------------
# Plotting the Results
# ------------------------

plt.figure(figsize=(10, 6))
plt.plot(ln_lambda, bias_all, 'ro-', label='Bias$^2$')
plt.plot(ln_lambda, variance_all, 'bo-', label='Variance')
plt.plot(ln_lambda, test_error_all, 'go-', label='Test Error')
plt.xlabel('ln(lambda)')
plt.ylabel('Error')
plt.title('Bias, Variance, and Test Error vs. ln(lambda)')
plt.legend()
plt.show()