from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import random

IMG_SIZE = (128, 128)  # resize every image to this shape
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42

folder = Path(
    r"data\archive\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train"
)


def get_all_path(folder):
    list_path = []
    for i in folder.iterdir():
        temp = Path(i)
        for j in temp.iterdir():
            list_path.append(j)
    return list_path


def embadding(folder):
    images = []
    label = []
    for i in get_all_path(folder):
        img = cv2.imread(str(i))
        if img is None:
            print(f"Warning: could not read {i}")
            continue
        images.append(img)
        label.append(str(i).split("\\")[5])
    # print(label)
    return np.array(images), label


def making_y(folder):
    list_y = []
    for i in folder.iterdir():
        list_y.append(str(i).split("\\")[-1])
    dict = {}
    for i, j in enumerate(list_y):
        dict[j] = i
    return dict


def making_x_and_y(folder):
    y_dict = making_y(folder)
    all_paths = get_all_path(folder)  # list of file paths, built once
    random.shuffle(all_paths)

    def batch(paths):
        images = []
        labels = []
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                print(f"Warning: could not read {p}")
                continue
            img = cv2.resize(img, IMG_SIZE)
            images.append(img)
            labels.append(y_dict[p.parent.name])  # class = parent folder name
        return np.array(images) / 255, np.array(labels)

    for i in range(0, len(all_paths), BATCH_SIZE):
        x, y = batch(all_paths[i : i + BATCH_SIZE])
        yield x, y


# wrap in tf.data.Dataset so it can restart each epoch (plain generators can't)
train_ds = tf.data.Dataset.from_generator(
    lambda: making_x_and_y(folder),
    output_signature=(
        tf.TensorSpec(shape=(None, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
    ),
)


def model():
    num_classes = len(list(folder.iterdir()))

    cnn = models.Sequential(
        [
            layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    cnn.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    cnn.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "best_plant_disease_model.keras",
            monitor="val_accuracy",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]

    return cnn, callbacks


cnn, callbacks = model()

history = cnn.fit(
    train_ds,
    epochs=EPOCHS,
)
cnn.save("model.keras")
cnn.save("model.h5")
