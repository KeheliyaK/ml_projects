# Project 5 — Neural Networks: *Modified* XOR Classification + Nonlinear Regression

This repository contains my solution for **Project 5**, implemented from scratch (no ML frameworks). It has two parts:

- **(a) Modified XOR classification** — one hidden layer with **2 tanh units**; plot the **training loss over time** and, after convergence, the **decision surface** together with the input data.
- **(b) Nonlinear regression** on noisy samples of `sin(2πx)`; compare **3 vs. 20 tanh units**; plot the **training loss** and the **final fit** with training RMSE on the figure.

## Files
- `Project5.py` — Main script implementing both parts (training loops, evaluation, and plotting).

## Requirements
- Python 3.8+
- `numpy`, `matplotlib`
  
Install:
```bash
pip install numpy matplotlib
```

---

## Part (a): *Modified* XOR classification

**Data (as in the assignment)** — two classes in 2D:
- Class $\mathcal{C}_1 = \{[-1,-1]^\top,\ [1,1]^\top\}$
- Class $\mathcal{C}_2 = \{[-1,\ 1]^\top,\ [1,-1]^\top\}$

This differs from the “usual” XOR labeling but is equivalent up to label swap; the network still needs **nonlinear** hidden units to separate the classes. In the script these are defined via `X_mat`, `X_xor`, `y_xor`.

**Network**: one hidden layer with **2 units** (`tanh` activation) and a `tanh` output mapped to class labels `{0,1}` via a 0.5 threshold.  
**Optimization**: plain gradient descent; **loss**: cross‑entropy.

> Note: the script also includes an example run with `n_hidden=3`; set `n_hidden=2` to match the spec exactly.

**Figures produced**
- **Training loss vs. epoch** (log‑scaled).
- **3D decision surface** $z=f(x_1,x_2)$ over a mesh grid with the four data points overlaid.

---

## Part (b): Nonlinear regression on noisy `sin(2πx)`

**Data generation**
```text
rng(100)            # Python equivalent: np.random.seed(100)
X = 2*rand(1,50)-1  # 50 points uniformly in [-1, 1]
T = sin(2*pi*X) + 0.3*randn(1,50)
```
The script reproduces this in NumPy.

**Network (`FFNNRegressor`)**
- 1 hidden layer, `tanh` activation, linear output; MSE loss.
- Mini‑batch gradient descent (set `batch_size=None` for full‑batch GD).

**Configurations compared**
- **3 tanh units**, 5000 iterations
- **20 tanh units**, 5000 iterations  
Common settings: learning rate `lr=0.1`, batch size `32` (can be changed).

**Figures produced**
- **Training loss vs. epoch** for each configuration.
- Final **fit plot**: scatter of data with the model prediction curve; the title shows **training RMSE**.

---

## Usage
From the folder containing `Project5.py`:
```bash
python Project5.py
```
Hyperparameters for both parts are set near the bottom of the file in the run blocks; adjust them to match the exact assignment (e.g., `n_hidden=2` for part a).

## License
This project is for academic coursework; reuse only with permission of the author/instructor.
