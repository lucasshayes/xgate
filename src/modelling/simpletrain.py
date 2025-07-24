
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.dataset import Dataset
from config import Config
from modelling.modules.attention.cbam import CBAM1D
from modelling.modules.attention.temporal_eca import TemporalECA
from modelling.modules.xception import XceptionBlock
from keras import Sequential, layers

config = Config()
dataset = Dataset(config.random_seed, target="true_room")

model = Sequential(
    [
        layers.Input(shape=(200, 7)),
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        CBAM1D(r_ratio=8, name="cbam"),
        layers.GRU(64, return_sequences=True),
        layers.GRU(128, return_sequences=True),
        TemporalECA(name="temporal_eca"),
        layers.GlobalAvgPool1D(name="global_avg_pool"),
        layers.Dense(64, activation="relu"),
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

val_dataset = dataset.create_tf_dataset(
    config.processed_dataset_dir + "val/",
    batch_size=32,
)

print(model.summary())

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)