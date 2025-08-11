
import sys
import os
import numpy as np
import tensorflow as tf
import keras as k
from keras import layers

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.dataset import Dataset
from config import Config
from modelling.modules.attention.feat_attention import feat_attention
from modelling.modules.attention.cbam import cbam_1d_block
from modelling.modules.attention.temporal_eca import temporal_eca_block
from modelling.modules.xception import xception_block
from modelling.modules.rolling_extraction import rolling_extraction
from utils.set_seed import set_seeds

from evaluate import calc_attention_weights
from utils.plots import feat_attention_plot

config = Config()
dataset = Dataset(config.random_seed, target="true_room")
set_seeds(config.random_seed)

train_dataset = dataset.create_tf_dataset(
    config.processed_dataset_dir + "train/",
    batch_size=32,
)

val_dataset = dataset.create_tf_dataset(
    config.processed_dataset_dir + "val/",
    batch_size=32,
)

test_dataset = dataset.create_tf_dataset(
    config.processed_dataset_dir + "test/",
    batch_size=32,
)

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

inputs = layers.Input(shape=(200, 7))
x = rolling_extraction(inputs, 100) # Extracts mean and std features
x = feat_attention(x)
x = xception_block(x,
    **hps["xception"]
)
x = cbam_1d_block(x, **hps["cbam"], name="cbam")
x = layers.GRU(**hps["GRU"], return_sequences=True)(x)
x = layers.GRU(**hps["GRU"], return_sequences=True)(x)
x = temporal_eca_block(x, name="temporal_eca")
x = layers.Dense(**hps["Dense"], name="dense_layer")(x)
x = layers.Dropout(0.5, name="dropout_layer")(x)
outputs = layers.Dense(4, activation="softmax")(x)

model: k.Model = k.Model(inputs=inputs, outputs=outputs, name="xgate_model")

model.compile(
    optimizer=k.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy"],
)

checkpoint_cb = k.callbacks.ModelCheckpoint(
    filepath=config.model_checkpoints_dir + "best_model.keras",    
    monitor="val_loss",             
    save_best_only=True,
)

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    callbacks=[k.callbacks.EarlyStopping("val_loss", patience=8), checkpoint_cb],
)

model.summary()

def update_bn_stats(model, unlabeled_data):
    """Update batch normalization statistics.

    Args:
        model (tf.keras.Model): The Keras model with Batch Normalization layers.
        unlabeled_data (tf.data.Dataset): Unlabeled data.
    """
    for batch in unlabeled_data:
        _ = model(batch, training=True)  # updates BN running mean/var

test_features = np.asarray(list(test_dataset.map(lambda x, y: x)))
update_bn_stats(model, test_features)

model: k.Model = k.models.load_model("best_model.keras", compile=True)

metrics = model.evaluate(test_dataset, return_dict=True)
print(metrics)
