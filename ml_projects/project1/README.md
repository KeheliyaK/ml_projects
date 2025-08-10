# Project 1 — Linear Regression on *carbig* (Weight → Horsepower)

This directory contains my submission for **ECE 4370 / ECE 5370 – Project 1**.  
The assignment requires implementing **simple linear regression** on MATLAB’s *carbig* dataset using **Weight** as the predictor (feature) and **Horsepower** as the target (label), with **two methods**:

1) **Closed‑form solution (Normal Equation)**  
2) **Gradient Descent**

The script reproduces the two required plots: a scatter of the data and a fitted regression line from each method.

---

## Files
- `Project1.py` — Main Python script implementing both solutions and plotting results.  
- `proj1Dataset.xlsx` — Excel version of MATLAB’s *carbig* dataset (Python users use this). *(Place in the same folder as `Project1.py`.)*
- `Proj1.pdf` — Assignment description (for reference).

> **Note**: The script expects the Excel file name to be `proj1Dataset.xlsx`. You can change the path via `data_path` at the top of `Project1.py`.

---

## Requirements
- Python 3.8+
- Packages:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `openpyxl` (Excel reader engine used by `pandas.read_excel`)

Install everything with:
```bash
pip install numpy pandas matplotlib openpyxl
```

---

## How it works

### 1) Data loading & preprocessing
- Reads the Excel file via:
  ```python
  df = pd.read_excel("proj1Dataset.xlsx", engine="openpyxl")
  ```
- Drops rows with missing values:
  ```python
  df1 = df.dropna()
  ```
- Selects **Weight** as `X` and **Horsepower** as `y` and builds a **design matrix** with an intercept:
  
X = [[1, weight_1], [1, weight_2], ..., [1, weight_m]]


### 2) Closed‑form solution (Normal Equation)
Uses the standard normal equation for simple linear regression:
\[
\mathbf{w}_{\text{closed}} = (X^\top X)^{-1} X^\top y
\]
where \(\mathbf{w} = [b,\; m]^\top\) gives intercept \(b\) and slope \(m\).  
Predictions: \(\hat{y} = X\,\mathbf{w}\).

### 3) Gradient Descent
Initializes weights randomly and iteratively updates:
\[
\mathbf{w} \leftarrow \mathbf{w} - \rho \cdot \frac{1}{m} X^\top (X\mathbf{w} - y)
\]
- **Learning rate**: `rho = 0.0001` (in the script)  
- **Convergence threshold**: `epsilon = 1e-5`  
- **Safety cap**: `max_iterations = 10_000_000`

You can modify these hyperparameters near the top of `Project1.py`.

---

## Output
Running the script produces a figure with **two side‑by‑side subplots**:

1) **Closed‑form solution**: scatter of (Weight, Horsepower) with the fitted line.  
2) **Gradient descent**: scatter of (Weight, Horsepower) with the fitted line learned by GD.

Both plots are titled with “MATLAB’s ‘carbig’ dataset” and label the axes as **Weight** (x) and **Horsepower** (y). The script also prints the initial random weights for GD and the learned parameters.

---

## Usage
From the folder containing `Project1.py` and `proj1Dataset.xlsx`:
```bash
python Project1.py
```

If your dataset is elsewhere or named differently, edit:
```python
data_path = "proj1Dataset.xlsx"
```
at the top of `Project1.py`.

---

## Notes & Tips
- If you see an error like *“ImportError: Missing optional dependency 'openpyxl'”*, install it with `pip install openpyxl`.
- If the plots look odd, check that your `Weight` and `Horsepower` columns exist and that `df.dropna()` didn’t remove all rows.
---

## License
This project is for academic coursework; reuse only with permission of the author/instructor.
