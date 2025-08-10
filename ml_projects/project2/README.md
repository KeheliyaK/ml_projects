# Project 2 — Polynomial Regression on noisy sin(2πx)

This repository contains my submission for **ECE 4370 / ECE 5370 – Project 2**.  
Goal: study model complexity and overfitting by fitting **polynomial linear regression** models of degree **M = 0…9** to noisy samples of  
\(t = \sin(2\pi x) + \epsilon\), with \(\epsilon \sim \mathcal{N}(0,\,0.3^2)\).  
I report **training** and **testing** RMS errors and plot how error varies with degree for **N_train=10** and **N_train=100**.

## Files
- `Project2.py` — Generates data, fits models with the Moore–Penrose pseudo-inverse, prints RMS tables, and plots results.
- *(No external data required.)*

## Requirements
- Python 3.8+
- `numpy`, `matplotlib`  
Install: `pip install numpy matplotlib`

## Method (matches assignment)
1) **Training set(s)**  
   - \(N^{	ext{Train}}=10\) and (repeat) \(N^{	ext{Train}}=100\)  
   - \(X^{	ext{Train}} \sim \mathcal{U}(0,1)\)  
   - \(t^{	ext{Train}} = \sin(2\pi X^{	ext{Train}}) + \epsilon,\; \epsilon \sim \mathcal{N}(0,0.3^2)\)

2) **Test set**  
   - \(N^{	ext{Test}}=100\)  
   - \(X^{	ext{Test}} \sim \mathcal{U}(0,1)\)  
   - \(t^{	ext{Test}} = \sin(2\pi X^{	ext{Test}}) + \epsilon,\; \epsilon \sim \mathcal{N}(0,0.3^2)\)

3) **Model / features**  
   - Polynomial basis up to degree \(M\): \(\Phi = [1, x, x^2, \ldots, x^M]\)  
   - Weights via least squares: \(\mathbf{w} = \Phi^\dagger \mathbf{t}\) (`np.linalg.pinv`)

4) **Evaluation**  
   - For each \(M=0,\ldots,9\), compute  
     
E_{\mathrm{RMS}}=\sqrt{J(\mathbf{w})/N},\quad J(\mathbf{w})=\sum_i (\hat t_i - t_i)^2

   - Record **train** and **test** errors for all 10 cases.

5) **Plots**  
   - One figure with **two subplots**:  
     - Left: \(N_{	ext{train}}=10\) — RMS vs. \(M\) (train & test).  
     - Right: \(N_{	ext{train}}=100\) — RMS vs. \(M\) (train & test).  
   - (Script applies light visual scaling to very large test errors so both curves remain readable.)

## Usage
```bash
python Project2.py
```
Defaults in the script:
- `np.random.seed(42)`
- `N_train = 10`, `N_train_2 = 100`, `N_test = 100`
- Degrees swept: `range(10)` → \(M=0…9\)

## Expected behavior
- With **N=10**, training error typically decreases with \(M\), while test error rises for large \(M\) (overfitting).  
- With **N=100**, the train/test curves are closer and the minimum test RMS occurs at a moderate \(M\).

## What to submit
Submit your `.py` file (and generated figure if required) per the course’s naming and upload instructions.
