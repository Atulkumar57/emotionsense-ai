import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from model import EMOTIONS, IMG_SIZE


def load_fer2013(csv_path):
    df = pd.read_csv(csv_path)
    pixels = df["pixels"].apply(lambda p: np.array(p.split(), dtype="float32"))
    X = np.stack(pixels.values).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    y = to_categorical(df["emotion"].values, num_classes=len(EMOTIONS))
    usage = df["Usage"].values
    return X, y, usage


def split_train_val(X, y, usage):
    train_mask = usage == "Training"
    val_mask = usage == "PublicTest"
    return X[train_mask], y[train_mask], X[val_mask], y[val_mask]
