
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
from modelling.modules.magnitude_extraction import magnitude_extraction
from utils.set_seed import set_seeds

def update_bn_stats(model: k.Model, unlabeled_data: tf.data.Dataset):
    """Update batch normalization statistics.

    Args:
        model (tf.keras.Model): The Keras model with Batch Normalization layers.
        unlabeled_data (tf.data.Dataset): Unlabeled data.
    """
    for batch in unlabeled_data:
        _ = model(batch, training=True)  # updates BN running mean/var

def ablation_study(hps: dict, experiment_names: list[str], train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset):
    """Perform ablation study on the model save all resultant models and weights.

    Args:
        hps (dict): Hyperparameters for the model.
        experiment_names (list[str]): List of experiment names to run.
        train_dataset (tf.data.Dataset): Training dataset.
        val_dataset (tf.data.Dataset): Validation dataset.
    """
    for name in experiment_names:
        print(f"Running ablation study for: {name}")
        model_path = config.model_checkpoints_dir + name + ".keras"
        
        inputs = layers.Input(shape=(200, 7))
        inputs = layers.Normalization(name="input_normalization")(inputs)
        # x = magnitude_extraction(inputs, indices=(0, 1, 2))
        x = rolling_extraction(inputs, 100) if not name == "no_rolling_extraction" else inputs
        x = feat_attention(x) if not (name == "no_feat_attention" or name == "no_attention") else x
        x = xception_block(x, **hps["xception"]) if not name == "no_xception" else x
        x = cbam_1d_block(x, **hps["cbam"], name="cbam") if not (name == "no_cbam" or name == "no_attention") else x
        x = layers.GRU(**hps["GRU"], return_sequences=True)(x) if not name == "no_gru" else x
        x = layers.GRU(**hps["GRU"], return_sequences=True)(x) if not name == "no_gru" else x
        x = temporal_eca_block(x, name="temporal_eca") if not (name == "no_eca" or name == "no_attention") else layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(**hps["Dense"], name="dense_layer")(x)
        x = layers.Dropout(0.5, name="dropout_layer")(x) 
        outputs = layers.Dense(4, activation="softmax")(x)
        
        model: k.Model = k.Model(inputs=inputs, outputs=outputs, name="xgate_model")
        
        model.compile(
            optimizer=k.optimizers.Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy"],
        )

        flops = get_flops(model)
        print(f"FLOPs for {name}: {flops / 1e6:.2f} million")
    
        checkpoint_cb = k.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
        )
        
        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=40,
            callbacks=[k.callbacks.EarlyStopping("val_loss", patience=8), checkpoint_cb],
        )

        test_features = np.asarray(list(test_dataset.map(lambda x, y: x)))
        update_bn_stats(model, test_features)

        model: k.Model = k.models.load_model(model_path, compile=True)

        metrics = model.evaluate(test_dataset, return_dict=True)
        print(metrics)

if __name__ == "__main__":
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

    ablation_study(hps, ["base", "no_cbam", "no_gru", "no_xception", "no_eca", "no_feat_attention", "no_attention", "no_rolling_extraction"], train_dataset, val_dataset)