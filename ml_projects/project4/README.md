# Project 4 — Logistic/Softmax Classifiers on MNIST and *C. elegans* Images

This project has **two independent parts**:

1) **MNIST (handwritten digits)** — train a **logistic regression** classifier and run inference.  
2) **C. elegans experimental images** — train a **softmax (multiclass) classifier** and run inference.

> ✅ You asked to **rename the scripts**. In this README I use the **new names** first and note the originals in parentheses.

---

## Repository layout (after renaming)
```
project4/
├── mnist_train.py         (was: biasMNIST.py)
├── infer_mnist.py         (was: outputMNIST.py)
├── train_Celegans.py      (was: CElegance.py)
├── infer_Celegans.py      (was: outputC.py)
└── README_Project4.md
```

**Rename commands (macOS/Linux):**
```bash
mv biasMNIST.py mnist_train.py
mv outputMNIST.py infer_mnist.py
mv CElegance.py train_Celegans.py
mv outputC.py infer_Celegans.py
```

---

## Requirements
- Python 3.8+
- Common packages: `numpy`, `scikit-learn`, `matplotlib` (and `pillow` if images are loaded via PIL)

Install:
```bash
pip install numpy scikit-learn matplotlib pillow
```

---

## Data

### MNIST
You can get MNIST from any of these sources (pick one):
- Official: <http://yann.lecun.com/exdb/mnist/>
- Kaggle mirror: <https://www.kaggle.com/datasets/oddrationale/mnist-in-csv>
- OpenML (Python API): <https://www.openml.org/d/554>

If your script downloads automatically (e.g., via `sklearn.datasets.fetch_openml("mnist_784")`), no manual download is needed.

### *C. elegans* dataset
Use your shared folder:  
<https://drive.google.com/drive/folders/1kD9KP6uFn1pzcIU_IQ0e8Rqskd6eScmL?usp=share_link>

After downloading, keep the folder structure your scripts expect (e.g., `train/` and `test/` subfolders or a CSV with paths).

---

## Part A — MNIST (Logistic Regression)

### Train
```bash
python mnist_train.py     --data_path ./data/mnist     --max_iter 1000     --test_split 0.2     --save_model ./mnist_logreg.joblib
```
Typical steps performed by the script:
- Load images/labels (from files or via OpenML).
- Flatten to 784‑dim vectors; scale to `[0,1]` or standardize.
- Fit **one‑vs‑rest logistic regression**.
- Save the model to `mnist_logreg.joblib` (or similar).

### Inference
```bash
python infer_mnist.py     --data_path ./data/mnist/test     --model ./mnist_logreg.joblib     --report ./mnist_report.txt
```
Expected outputs:
- Overall accuracy and per‑class metrics (precision/recall).
- Optionally a confusion matrix figure saved to disk.

---

## Part B — *C. elegans* (Softmax)

### Train
```bash
python train_Celegans.py     --data_path ./data/celegans     --epochs 50     --save_model ./celegans_softmax.joblib
```
Typical steps performed by the script:
- Load images and labels from your experimental dataset.
- Preprocess (grayscale/resize/flatten or simple features).
- Train a **multiclass softmax (multinomial logistic)** classifier.
- Save the trained model.

### Inference
```bash
python infer_Celegans.py     --data_path ./data/celegans/test     --model ./celegans_softmax.joblib     --report ./celegans_report.txt
```
Expected outputs:
- Accuracy (and optionally per‑class metrics) on the held‑out set.
- Example predictions visualized if the script supports it.

---

## Tips
- If training is slow or diverges, standardize features (`X = (X - mean) / std`) and lower the learning rate (if using a custom optimizer).
- For image folders, ensure consistent sizes; add a `--img_size` flag in your scripts or resize during loading.
- Keep random seeds for reproducibility: `np.random.seed(42)` and, if you later add PyTorch/TF, set their seeds as well.

## License
This project is for academic coursework; reuse only with permission of the author/instructor.
