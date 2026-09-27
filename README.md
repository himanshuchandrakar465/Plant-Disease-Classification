<div align="center">

# 🌿 Plant Disease Classification

**A CNN that identifies plant leaf diseases from images.**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-planned-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/status-in--progress-yellow)

</div>

---

## 📖 Overview

This project trains a Convolutional Neural Network (CNN) to classify plant leaf images as **healthy** or **diseased**, and to identify *which* disease is present. It uses the [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) from Kaggle and a custom TensorFlow/Keras training pipeline.

## 📑 Table of Contents

- [Features](#-features)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Installation](#️-installation)
- [Dataset Setup](#-dataset-setup)
- [Usage](#-usage)
- [Roadmap](#-roadmap)
- [Author](#-author)
- [License](#-license)

## ✨ Features

- Custom data-loading pipeline (`tf.data.Dataset` generator) that streams and batches images instead of loading the full dataset into memory
- CNN built with Keras `Sequential` for multi-class leaf disease classification
- Training safeguards: `EarlyStopping`, `ModelCheckpoint`, and `ReduceLROnPlateau`
- Evaluation with a confusion matrix and per-class precision / recall / F1
- Managed with [`uv`](https://github.com/astral-sh/uv) (`pyproject.toml` + `uv.lock`), with a plain `requirements.txt` as a fallback

## 🧠 Model Architecture

| Layer | Details |
|---|---|
| Input | 128 × 128 × 3 |
| Conv2D + MaxPool | 32 filters, 3×3, ReLU |
| Conv2D + MaxPool | 64 filters, 3×3, ReLU |
| Conv2D + MaxPool | 128 filters, 3×3, ReLU |
| Flatten | — |
| Dense | 256 units, ReLU |
| Dropout | 0.5 |
| Dense (output) | `num_classes`, Softmax |

**Training config:** Adam optimizer · sparse categorical cross-entropy · batch size 32 · up to 15 epochs (early-stopped on validation loss).

## 📊 Results

<div align="center">
<img src="evaluation/evaluation.png" alt="Confusion matrix and per-class precision/recall/F1" width="850">
</div>

The model reaches **78.4% overall accuracy** on the evaluated classes, with per-class F1 scores mostly between 0.70–0.90. A few classes (e.g. *Cedar apple rust*) are harder to separate and show more confusion — a good next target for improvement.

## 🗂️ Project Structure

```
Plant-Disease-Classification/
├── data/
│   └── data.txt                     # where to get and place the dataset
├── evaluation/
│   └── evaluation.png               # confusion matrix + metrics
├── src/plant_disease_classification/
│   └── __init__.py                  # package entry point (scaffold)
├── main.py                          # data pipeline, model, training loop
├── best_plant_disease_model.keras   # best checkpoint (val_accuracy)
├── model.keras / model.h5           # final trained model
├── pyproject.toml / uv.lock         # uv-managed dependencies
├── requirements.txt                 # pip fallback
└── README.md
```

## ⚙️ Installation

```bash
git clone https://github.com/himanshuchandrakar465/Plant-Disease-Classification.git
cd Plant-Disease-Classification

# Option A — uv (recommended, matches pyproject.toml/uv.lock)
uv sync

# Option B — pip
pip install -r requirements.txt
```

## 📦 Dataset Setup

The dataset isn't bundled in the repo. Get it from Kaggle:

```bash
kaggle datasets download -d vipoooool/new-plant-diseases-dataset -p data --unzip
```

or download it manually from the [Kaggle page](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) and place it under `data/`.

> **Note:** `main.py` currently points to a hardcoded Windows-style path for the training folder — update the `folder` variable at the top of the script to match where you extract the dataset.

## 🚀 Usage

**Train the model:**

```bash
python main.py
```

This streams images in batches, trains the CNN, and saves:
- `best_plant_disease_model.keras` — best checkpoint by validation accuracy
- `model.keras` / `model.h5` — final model after training

## 🛣️ Roadmap

- [ ] Wire up the Streamlit app for interactive leaf-image predictions (dependency is already in place, UI isn't built yet)
- [ ] Make the dataset path OS-independent (`pathlib` instead of hardcoded `\\` splits)
- [ ] Add a proper `predict.py` / inference script
- [ ] Expand evaluation to the full class set and log training curves

## 👤 Author

**Himanshu Chandrakar** — aspiring AI/ML engineer working on deep learning and generative AI (PyTorch, TensorFlow, transformers, GANs).
[GitHub @himanshuchandrakar465](https://github.com/himanshuchandrakar465)

## 📄 License

No license has been added yet — until one is, all rights are reserved by default. Open an issue or reach out before reusing this code.
