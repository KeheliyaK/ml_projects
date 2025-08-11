# Project 6 — CNN Classification for Image Data

This project implements **Convolutional Neural Networks (CNNs)** to classify images.  
It’s an evolution of the previous *C. elegans* image classification task (Project 4), replacing logistic regression / softmax with a CNN for improved feature extraction and accuracy.

---

## Files
- `train_Celegans.py` — Trains the CNN on the dataset.
- `infer_Celegans.py` — Runs inference using a saved trained model.

---

## Requirements
- Python 3.8+
- `numpy`
- `matplotlib`
- `tensorflow` or `torch` (depending on your CNN implementation)
- `opencv-python` (if used for image pre-processing)

Install (example for TensorFlow-based code):
```bash
pip install numpy matplotlib tensorflow opencv-python
```

---

## Data
The model is trained on the **C. elegans** image dataset.

Download dataset from Google Drive:  
[📂 C. elegans Dataset](https://drive.google.com/drive/folders/1kD9KP6uFn1pzcIU_IQ0e8Rqskd6eScmL?usp=share_link)

After downloading, place the dataset in the expected folder path set in `finaltraining.py`.

---

## How it works

### Model Architecture (CNN)
The CNN is composed of:
1. **Convolutional Layer(s)** — extract spatial features from the input images.
2. **Activation (ReLU)** — introduce non-linearity.
3. **Pooling Layer(s)** — downsample the feature maps to reduce dimensionality.
4. **Fully Connected Layer(s)** — perform classification based on extracted features.
5. **Softmax Output Layer** — produces class probabilities.

Example configuration:
```text
Conv2D(filters=32, kernel_size=3, activation='relu')
MaxPooling2D(pool_size=2)
Conv2D(filters=64, kernel_size=3, activation='relu')
MaxPooling2D(pool_size=2)
Flatten()
Dense(128, activation='relu')
Dense(num_classes, activation='softmax')
```

### Training (`train_Celegans_CNN.py`)
- Loads the dataset and splits into train/validation sets.
- Normalizes image data.
- Defines the CNN architecture.
- Trains for the number of epochs set in the script.
- Saves the trained model to disk.

Run:
```bash
python finaltraining.py
```

### Inference (`infer_Celegans_CNN.py`)
- Loads the saved trained model.
- Reads and preprocesses new input images.
- Outputs predicted class and confidence score.

Run:
```bash
python outputCelegans.py
```

---

## License
This project is for academic coursework; reuse only with permission of the author/instructor.
