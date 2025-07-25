
import sys
import os
import keras
import numpy as np
import tensorflow as tf
import random
import keras as k
from keras import Sequential, layers

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.dataset import Dataset
from config import Config
from modelling.modules.attention.cbam import CBAM1D
from modelling.modules.attention.temporal_eca import TemporalECA
from modelling.modules.xception import XceptionBlock
from utils.set_seed import set_seeds

config = Config()
dataset = Dataset(config.random_seed, target="true_room")
set_seeds(config.random_seed)

hps = {
    "xception": {
        "num_filters": 16,
        "k_size": 7,
        "middle_blocks": 2,
        "downsample": False
    },
    "cbam": {
        "r_ratio": 8
    },
    "GRU": {
        "units": 64,
        "dropout": 0,
        "recurrent_dropout": 0
    },
    "Dense": {
        "units": 64,
        "activation": "relu",
        "kernel_regularizer": k.regularizers.l2(0.001),
        "kernel_constraint": k.constraints.max_norm(3.0)
    },
}

model = Sequential(
    [
        layers.Input(shape=(200, 7)),
        layers.Normalization(name="input_normalization"),
        XceptionBlock(
            **hps["xception"]
        ),
        CBAM1D(**hps["cbam"], name="cbam"),
        layers.GRU(**hps["GRU"], return_sequences=True),
        layers.GRU(**hps["GRU"], return_sequences=True),
        TemporalECA(name="temporal_eca"),
        layers.Dense(**hps["Dense"], name="dense_layer"),
        layers.Dropout(0.4, name="dropout_layer"),
        layers.Dense(4, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy"],
)

train_dataset = dataset.create_tf_dataset(
    config.processed_dataset_dir + "train/",
    batch_size=32,
)

for x, y in train_dataset.take(1):
    print("Sample X:", x.shape)
    print("Sample y:", y.shape)
    print("Sample y values:", y[0])
    break

val_dataset = dataset.create_tf_dataset(
    config.processed_dataset_dir + "val/",
    batch_size=32,
)

print(model.summary())

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    callbacks=[k.callbacks.EarlyStopping("val_loss", patience=8)],
)