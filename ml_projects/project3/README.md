# Project 3 — Bias–Variance with Gaussian Basis + Ridge Regression

This repository contains my solution for **ECE 4332 / ECE 5332 – Project 3**.  
We study the **bias–variance trade‑off** by fitting a *linear regression with non‑linear features*
(Gaussian radial basis functions) to noisy samples of $t=\sin(2\pi x)+\epsilon$, while sweeping the
**ridge regularization** parameter $\lambda$.

## Files
- `Project3.py` — Main script. Generates data, fits models, computes   $bias^2$, **variance**, and **test error**, and plots them vs. $\ln(\lambda)$.


## Requirements
- Python 3.8+
- `numpy`, `matplotlib`  
Install: `pip install numpy matplotlib`

## Method (assignment spec)
- Draw **L = 100** independent training sets, each with **N = 25** points from $X\sim\mathcal{U}(0,1)$ and $t=\sin(2\pi X)+\epsilon$, with $\epsilon\sim\mathcal{N}(0,0.3^2)$.  
- For a grid of **$\lambda$** values, fit **linear regression with non‑linear models** using **Gaussian basis functions** with width **$s=0.1$** on each dataset.  
- Report $bias^2$,**variance**, and a **test error** measured on an i.i.d. test set of **1000** points.  

Formulas used:

$$
\bar f(x) = \frac{1}{L}\sum_{\ell=1}^{L} f^{(\ell)}(x)
$$

$$
\text{bias}^2 = \frac{1}{N}\sum_{n=1}^{N}\big(\bar f(x^{(n)}) - h(x^{(n)})\big)^2
\quad\text{with}\quad h(x)=\sin(2\pi x)
$$

$$
\text{variance} = \frac{1}{N}\sum_{n=1}^{N}\left[\frac{1}{L}\sum_{\ell=1}^{L}\big(f^{(\ell)}(x^{(n)})-\bar f(x^{(n)})\big)^2\right]
$$

## What the script does
- **Design matrix (Gaussian RBFs):** $\Phi_{ij}=\exp\!\big(-\tfrac{(x_i-c_j)^2}{2s^2}\big)$ with **$m=6$** centers evenly spaced in $[0,1]$ and **$s=0.1$**.  
- **Ridge solution:** $\mathbf {w} = (\Phi^\top\Phi + \lambda I)^{-1}\Phi^\top \mathbf t$.
- **Ensemble loop:** For each $\lambda$ (using **$\ln(\lambda)$ from -2.5 to 1.5 in steps of 0.25**), repeat **L=100** times:
  generate a new size‑25 training set, fit the model, store predictions on a fixed evaluation grid and on a 1000‑point test set.
- **Metrics:** Compute **bias$^2$**, **variance**, and **test MSE** by averaging across the ensemble.
- **Plot:** A single figure showing the three curves (**Bias$^2$**, **Variance**, **Test Error**) vs. $\ln(\lambda)$.  
- **Reproducibility:** `np.random.seed(50)` is set near the main loop.

## Usage
From the folder containing `Project3.py`:
```bash
python Project3.py
```
Feel free to tweak at the top of the script:
- `L` (ensemble size, default 100)
- `N_train` (default 25)
- `m` (number of RBF centers, default 6)
- `s` (RBF width, default 0.1)
- `ln_lambda` range (default -2.5 → 1.5 step 0.25)

## Notes
- If the plot window doesn’t appear, ensure you aren’t in a headless session; otherwise, replace `plt.show()` with `plt.savefig('bias_variance_vs_lambda.png', dpi=200)`.
- For stability at extreme $\lambda$ values, the code uses the closed‑form ridge solution.

## License
This project is for academic coursework; reuse only with permission of the author/instructor.
