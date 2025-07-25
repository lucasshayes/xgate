import os
import sys
import time
import json
import numpy as np
import keras.api as k
import keras_tuner as kt
from keras.api import Model
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import Config
from data.dataset import Dataset
from modelling.model import build_fused_model
from utils.callbacks import get_callbacks


class CustomHyperModel(kt.HyperModel):
    def __init__(self, config, dataset):
        # Initialize with config and dataset
        self.config = config
        self.dataset = dataset

    def build(self, hp: kt.HyperParameters) -> Model:
        """Builds the fused model with the passed hyperparameters and config.

        Args:
            hp (kt.HyperParameters): Object to attach hyper parameter search spaces to.

        Returns:
            Model: Model with search space hyperparameters.
        """
        k.backend.clear_session()

        # Model module toggle
        hp.Fixed("xception_bool", self.config.xception_enabled)
        hp.Fixed("cbam_bool", self.config.cbam_enabled)
        hp.Fixed("eca_bool", self.config.temporal_eca_enabled)
        # Downsample toggle
        hp.Fixed("downsample", self.config.downsample)
        # Temporal ECA params (original paper values)
        hp.Fixed("gamma", self.config.gamma)
        hp.Fixed("beta", self.config.beta)

        # Xception Params
        hp.Int("num_filters", min_value=16, max_value=32, step=8)
        hp.Int("kernel_size", min_value=3, max_value=7, step=2)
        hp.Int("middle_blocks", min_value=2, max_value=4)

        # CBAM reduction ratio
        hp.Int("r_ratio", min_value=8, max_value=16, step=8)

        # Model Params
        hp.Int("fc_units", min_value=64, max_value=128, step=32)
        hp.Int("gru_units", min_value=64, max_value=128, step=32)
        hp.Choice("learning_rate", values=[1e-3, 1e-4, 1e-5])
        hp.Choice("batch_size", values=[32, 64])

        # Dropout rates
        hp.Choice("gru_dropout", values=[0.0, 0.1])
        hp.Choice("xception_dropout", values=[0.25, 0.5])
        hp.Choice("fc_dropout", values=[0.25, 0.5])

        # Instantiate and compile model with hyperparameters
        model = build_fused_model(hp)
        return model

    def fit(self, hp: kt.HyperParameters, model: Model, *args, **kwargs):
        """Fits the model with the given hyperparameters and batch size
        Args:
            hp (kt.HyperParameters): Hyperparameters for the model.
            model (Model): The model to fit.
            *args: Additional positional arguments for model.fit.
            **kwargs: Additional keyword arguments for model.fit.
        Returns:
            History: The history of the training process.
        """
        batch_size = hp.get("batch_size")

        train_dataset = self.dataset.create_tf_dataset(
            self.config.processed_dataset_dir + "train/",
            batch_size=batch_size,
        )

        val_dataset = self.dataset.create_tf_dataset(
            self.config.processed_dataset_dir + "val/",
            batch_size=batch_size,
        )
        
        return model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=30,
            callbacks=[k.callbacks.EarlyStopping("val_loss", patience=3)],
            verbose=1,
        )


def clean_history(history):
    """Convert all NumPy types in history to native Python types."""
    return {metric: [float(x) for x in v] for metric, v in history.items()}


def train_model():
    """Performs hyperparameter tuning using the defined space in build_model, using the config.
    This config is produced from .env to tune the fused model appropriately before retraining
    and saving the best."""

    config = Config()
    dataset = Dataset(config.random_seed, target="true_room")
    
    # Check for required directories
    for base_dir in [
        config.model_exports_dir,
        config.reports_dir,
        config.model_checkpoints_dir,
        config.model_logs_dir,
    ]:
        os.makedirs(os.path.join(base_dir, config.experiment_name), exist_ok=True)

    # Tune the model according to val_loss
    tuner = kt.BayesianOptimization(
        CustomHyperModel(config, dataset),
        objective="val_loss",
        max_trials=200,
        directory=config.model_tuning_dir,
        project_name=config.experiment_name,
    )

    # Search the space for optimum parameters, use early stopping if no improvement.
    tuner.search()

    # Get best hyperparameters
    best_hps = tuner.get_best_hyperparameters(1)[0]

    # Build best model and retrain on full dataset with all callbacks
    model = tuner.hypermodel.build(best_hps)

    train_dataset = dataset.create_tf_dataset(
        config.processed_dataset_dir + "train/", batch_size=best_hps.get("batch_size")
    )
    val_dataset = dataset.create_tf_dataset(
        config.processed_dataset_dir + "val/", batch_size=best_hps.get("batch_size")
    )

    start_time = time.time()
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=50,
        callbacks=get_callbacks(
            config.experiment_name, 5, config.model_checkpoints_dir, config.model_logs_dir
        ),
        verbose=1,
    )
    end_time = time.time()

    # Clean history for saving
    cleaned_history = clean_history(history.history)

    # Save model
    model.save(config.model_exports_dir + config.experiment_name + "/tuned_model.keras")

    # Save model summary
    with open(
        config.reports_dir + config.experiment_name + "/tuned_summary.txt", "w", encoding="utf-8"
    ) as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    best_epoch_idx = int(np.argmin(history.history["val_loss"]))
    best_metrics = {metric: v[best_epoch_idx] for metric, v in cleaned_history.items()}

    # Save model report
    with open(config.reports_dir + config.experiment_name + "/tuned_desc.txt", "w") as f:
        f.write(f"Training time: {end_time - start_time:.2f} seconds\n")

        f.write("Hyperparameters: \n")
        json.dump(best_hps.values, f, indent=4)

        f.write(f"Metrics at best val_loss (Epoch {best_epoch_idx + 1}):\n")
        for metric, v in best_metrics.items():
            f.write(f"{metric}: {v:.4f}\n")

    # Save model historu
    with open(config.reports_dir + config.experiment_name + "/full_history.json", "w") as f:
        json.dump(cleaned_history, f, indent=4)



train_model()
