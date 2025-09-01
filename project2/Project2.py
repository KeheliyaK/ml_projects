import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

pi = np.pi
N_train = 10
N_test = 100
N_train_2 = 100

# Generate first training set (N_train=10)
x_tr = np.random.uniform(0, 1, (N_train, 1))
epsilon_tr = np.random.normal(0, 0.3, (N_train, 1))
t_tr = np.sin(2 * pi * x_tr) + epsilon_tr


# Generate second training set (N_train_2=100)
x_tr_2 = np.random.uniform(0, 1, (N_train_2, 1))
epsilon_tr_2 = np.random.normal(0, 0.3, (N_train_2, 1))
t_tr_2 = np.sin(2 * pi * x_tr_2) + epsilon_tr_2

# Generate test set (N_test=100)
x_ts = np.random.uniform(0, 1, (N_test, 1))
epsilon_ts = np.random.normal(0, 0.3, (N_test, 1))
t_ts = np.sin(2 * pi * x_ts) + epsilon_ts

E_rms_tr_list = []
E_rms_ts_list = []

E_rms_tr_list_2 = []
E_rms_ts_list_2 = []

for i in range(10):
    # Building design matrices for current polynomial degree i (for N_train=10)
    phi_tr_matrix = np.hstack([x_tr ** j for j in range(i + 1)])
    phi_ts_matrix = np.hstack([x_ts ** j for j in range(i + 1)])

    # Building design matrices for N_train_2=100
    phi_tr_matrix_2 = np.hstack([x_tr_2 ** j for j in range(i + 1)])

    # Computing the weights:
    w = np.linalg.pinv(phi_tr_matrix) @ t_tr
    w2 = np.linalg.pinv(phi_tr_matrix_2) @ t_tr_2

    # Predictions
    y_tr = phi_tr_matrix @ w
    y_ts = phi_ts_matrix @ w
    y_tr_2 = phi_tr_matrix_2 @ w2
    y_ts_2 = phi_ts_matrix @ w2

    # Computing RMS errors
    E_rms_tr = np.sqrt(np.sum((y_tr - t_tr) ** 2) / N_train)
    E_rms_ts = np.sqrt(np.sum((y_ts - t_ts) ** 2) / N_test)

    E_rms_tr_2 = np.sqrt(np.sum((y_tr_2 - t_tr_2) ** 2) / N_train_2)
    E_rms_ts_2 = np.sqrt(np.sum((y_ts_2 - t_ts) ** 2) / N_test)

    E_rms_tr_list.append(E_rms_tr)
    E_rms_ts_list.append(E_rms_ts)

    E_rms_tr_list_2.append(E_rms_tr_2)
    E_rms_ts_list_2.append(E_rms_ts_2)

x_degrees = np.arange(10)
all_E_rms_tr = np.array(E_rms_tr_list)
all_E_rms_ts = np.array(E_rms_ts_list)
all_E_rms_tr_2 = np.array(E_rms_tr_list_2)
all_E_rms_ts_2 = np.array(E_rms_ts_list_2)


def custom_scale(xx, x_min=None, x_max=None):

    xx = np.array(xx)  # Ensure xx is a NumPy array

    if x_min is None:
        x_min = np.min(xx)
    if x_max is None:
        x_max = np.max(xx)


    norm = (xx - x_min) / (x_max - x_min)

    # Initialize `scaled` array
    scaled = np.copy(xx)

    # Apply condition using NumPy boolean indexing
    mask = xx > 0.79
    scaled[mask] = norm[mask] * (1 - x_min) + x_min

    return scaled


# Computing min and max for test errors )
test_min = np.min(all_E_rms_ts)
test_max = np.max(all_E_rms_ts)


scaled_E_rms_ts = custom_scale(all_E_rms_ts, test_min, test_max)

print('E_RMS for N=10')
print("M\t|Train \t\t|Test ")
for m in range(10):
    print(f"{m}\t|{all_E_rms_tr[m]:.4f}\t\t|{all_E_rms_ts[m]:.4f}")

print('E_RMS for N=100')
print("M\t|Train \t\t|Test")
for m in range(10):
    print(f"{m}\t|{all_E_rms_tr_2[m]:.4f}\t\t|{all_E_rms_ts_2[m]:.4f}")

# Plotting
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x_degrees, scaled_E_rms_ts, 'ro-', label='Test RMS (Scaled)')
plt.plot(x_degrees, all_E_rms_tr, 'bo-', label='Training RMS')
plt.xlabel("Polynomial Degree (M)")
plt.ylabel("E_rms")
plt.title("E_rms vs. M for N_train = 10")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x_degrees, all_E_rms_ts_2, 'ro-', label='Test RMS')
plt.plot(x_degrees, all_E_rms_tr_2, 'bo-', label='Training RMS')
plt.xlabel("Polynomial Degree (M)")
plt.ylabel("E_rms")
plt.title("E_{rms} vs. M for N_train = 100")
plt.legend()

plt.show()

