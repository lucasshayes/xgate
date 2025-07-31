
import sys
import os
import numpy as np
import tensorflow as tf
import keras as k
from keras import Sequential, layers

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.dataset import Dataset
from config import Config
from modelling.modules.attention.feat_attention import feat_attention
from modelling.modules.attention.cbam import cbam_1d_block
from modelling.modules.attention.temporal_eca import temporal_eca_block
from modelling.modules.xception import xception_block
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
        "kernel_constraint": k.constraints.max_norm(3)
    },
}

# Normalize inputs
inputs = layers.Input(shape=(50, 7))
# x = feat_attention(inputs)
x = layers.Normalization(name="input_normalization")(inputs)
x = xception_block(x,
    **hps["xception"]
)
x = cbam_1d_block(x, **hps["cbam"], name="cbam")
x = layers.GRU(**hps["GRU"], return_sequences=True)(x)
x = layers.GRU(**hps["GRU"], return_sequences=True)(x)
x = temporal_eca_block(x, name="temporal_eca")
x = layers.Dense(**hps["Dense"], name="dense_layer")(x)
x = layers.Dropout(0.4, name="dropout_layer")(x)
outputs = layers.Dense(4, activation="softmax")(x)

model = k.Model(inputs=inputs, outputs=outputs, name="xgate_model")

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

test_dataset = dataset.create_tf_dataset(
    config.processed_dataset_dir + "test/",
    batch_size=32,
)

print(model.summary())


checkpoint_cb = k.callbacks.ModelCheckpoint(
    filepath="best_model.keras",     # or use save_weights_only=True
    monitor="val_loss",              # or "val_accuracy", etc.
    save_best_only=True,
)

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    callbacks=[k.callbacks.EarlyStopping("val_loss", patience=8), checkpoint_cb],
)

model = k.models.load_model("best_model.keras")

metrics = model.evaluate(test_dataset, return_dict=True)
print(metrics)