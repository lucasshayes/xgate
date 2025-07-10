import os
import sys
import keras.api as k
import keras_tuner as kt
from keras.api import Model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import Config
from data.dataset import Dataset
from modelling.model import FusedModel
from utils.callbacks import get_callbacks


def build_model(hp: kt.HyperParameters) -> Model:
    """Builds the fused model with the set hyperparameters.

    Args:
        hp (kt.HyperParameters): _description_

    Returns:
        Model: _description_
    """
    config = Config()

    # Model module toggle
    hp.Fixed("xception_bool", config.xception_enabled)
    hp.Fixed("cbam_bool", config.cbam_enabled)
    hp.Fixed("eca_bool", config.temporal_eca_enabled)
    # Downsample toggle
    hp.Fixed("downsample", config.downsample)
    # Temporal ECA params (original paper values)
    hp.Fixed("gamma", config.gamma)
    hp.fixed("beta", config.beta)

    # Xception Params
    hp.Int("num_filters", min_value=8, max_value=64, step=8)
    hp.Int("kernel_size", min_value=3, max_value=7, step=2)
    hp.Int("middle_blocks", min_value=1, max_value=3)

    # CBAM reduction ratio
    hp.Int("r_ratio", min_value=8, max_value=16, step=4)

    # Model Params
    hp.Int("fc_units", min_value=16, max_value=128, step=16)
    hp.Int("gru_units", min_value=16, max_value=128, step=16)
    hp.Choice("learning_rate", values=[0.01, 0.001, 0.0001])

    # Instantiate and compile model with hyperparameters
    model = FusedModel.build_and_compile(hp)
    return model


def train_model():
    """Performs hyperparameter tuning using the defined space in build_model, using the config.
    This config is produced from .env to tune the fused model appropriately before retraining
    and saving the best."""

    config = Config()
    dataset = Dataset(config.random_seed)

    # Create tensorflow datasets from the saved processed datasets
    train_dataset = dataset.create_tf_dataset(
        config.processed_dataset_dir + "train/",
        batch_size=config.batch_size,
    )
    val_dataset = dataset.create_tf_dataset(
        config.processed_dataset_dir + "val/",
        batch_size=config.batch_size,
    )

    # Tune the model according to mean squared error (loss)
    tuner = kt.Hyperband(
        hypermodel=build_model,
        objective="val_loss",
        max_epochs=30,
        directory=config.model_tuning_dir,
        project_name=".",
    )

    # Search the space for optimum parameters, use early stopping if no improvement.
    tuner.search(
        train_dataset,
        validation_data=val_dataset,
        epochs=30,
        callbacks=[k.callbacks.EarlyStopping(patience=2)],
    )

    # Get best hyperparameters
    best_hps = tuner.get_best_hyperparameters(1)[0]

    # Build best model and retrain on full dataset with all callbacks
    model = tuner.hypermodel.build(best_hps)
    model.fit(train_dataset, validation_data=val_dataset, epochs=50, callbacks=get_callbacks())

    # Save model and its summary
    with open(config.reports_dir + "tuned_summary.txt") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    model.save(config.model_exports_dir + "tuned_model.keras")


train_model()
