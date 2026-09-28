# Plant Disease Classification

A convolutional neural network (CNN) built with TensorFlow/Keras that classifies plant leaf images into disease categories. The model is trained on the **New Plant Diseases Dataset (Augmented)** image dataset, with one class per sub-folder of the training directory.

## How It Works

The training script (`main.py`) does the following:

1. Reads every image from the training folder (each sub-folder name is a class label).
2. Shuffles the file paths, then loads images in batches of 32 using OpenCV.
3. Resizes each image to **128 x 128** and scales pixel values to the 0-1 range.
4. Streams the batches into a `tf.data.Dataset` via a generator, so the full dataset is never held in memory.
5. Trains a CNN for up to **15 epochs**, then saves the model.

The number of output classes is not hard-coded. It is computed from the number of class folders in the training directory.

## Model Architecture

| Layer | Details |
|-------|---------|
| Input | 128 x 128 x 3 |
| Conv2D + MaxPooling2D | 32 filters, 3x3, ReLU, `same` padding, 2x2 pool |
| Conv2D + MaxPooling2D | 64 filters, 3x3, ReLU, `same` padding, 2x2 pool |
| Conv2D + MaxPooling2D | 128 filters, 3x3, ReLU, `same` padding, 2x2 pool |
| Flatten | - |
| Dense | 256 units, ReLU |
| Dropout | 0.5 |
| Dense (output) | one unit per class, Softmax |

**Compile settings:** Adam optimizer, `sparse_categorical_crossentropy` loss, accuracy metric.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Image size | 128 x 128 |
| Batch size | 32 |
| Epochs | 15 |
| Random seed | 42 |

Callbacks defined in the script: `EarlyStopping` (monitors `val_loss`, patience 5), `ModelCheckpoint` (saves the best model to `best_plant_disease_model.keras`), and `ReduceLROnPlateau` (halves the learning rate, patience 3).

## Project Structure

```
Plant-Disease-Classification/
├── data/                              # Dataset location (Kaggle download extracted here)
├── evaluation/                        # Evaluation-related files
├── src/plant_disease_classification/  # Python package (project entry point)
├── back.ipynb                         # Jupyter notebook
├── main.py                            # Data loading, model definition and training
├── best_plant_disease_model.keras     # Checkpoint saved by ModelCheckpoint
├── plant_disease_model.keras          # Saved model
├── model.keras                        # Final model (Keras format)
├── model.h5                           # Final model (HDF5 format)
├── pyproject.toml                     # Project metadata and dependencies
├── requirements.txt                   # pip-style dependency list
├── uv.lock                            # Locked dependencies for uv
└── .python-version                    # Python version for the project
```

## Tech Stack

- **Python** 3.11 or newer
- **TensorFlow / Keras**: model building and training
- **OpenCV**: image loading and resizing
- **NumPy, Pandas**: data handling
- **scikit-learn, Matplotlib, Seaborn**: evaluation and plotting
- **Pillow**: image handling
- **Streamlit**: listed as a dependency
- **Kaggle API**: dataset download
- **uv**: package management (`pyproject.toml` and `uv.lock`)
- **Ruff**: linting

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/himanshuchandrakar465/Plant-Disease-Classification.git
cd Plant-Disease-Classification
```

### 2. Install dependencies

Using **uv** (recommended, since the project ships a `uv.lock`):

```bash
uv sync
```

Or using **pip**:

```bash
pip install -r requirements.txt
```

### 3. Get the dataset

The training script expects the dataset at this path (Windows-style, relative to the project root):

```
data\archive\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train
```

Download the **New Plant Diseases Dataset (Augmented)** from Kaggle (for example with the Kaggle API, which is included in the dependencies) and extract it into `data/archive/` so the folder layout matches the path above. Each sub-folder of `train` must be one class containing that class's images.

### 4. Train the model

```bash
python main.py
```

When training finishes, the model is saved as `model.keras` and `model.h5`.

## Loading a Trained Model

```python
import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("model.keras")

img = cv2.imread("leaf.jpg")
img = cv2.resize(img, (128, 128)) / 255.0
pred = model.predict(np.expand_dims(img, axis=0))
class_index = int(np.argmax(pred))
```

Class indices follow the order in which `main.py` lists the class folders. Keep the same dataset folders to map an index back to its class name.

## Notes and Known Limitations

These come from reading the current code and are worth knowing before you build on it:

- **Callbacks are not applied.** `main.py` builds the callbacks list, but `cnn.fit()` is called without passing it, and no validation data is supplied. As written, early stopping, checkpointing and learning-rate reduction do not run during training.
- **No reported metrics.** This repository does not include verified accuracy or loss numbers, so none are claimed here. Add your own results after evaluating the model.
- **Windows-specific paths.** Paths and label parsing use backslashes (`\\`), so the script is written for Windows. On Linux or macOS the path handling needs adjusting.
- **Streamlit** is installed as a dependency, but no web app file is described in this README.

## Author

**himanshuchandrakar465**
