import os
import sys
import time
import json
import gc
import numpy as np
import optuna
import keras.api as k
from keras.api.callbacks import Callback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import Config
from data.dataset import Dataset
from modelling.model import FusedModel
from utils.callbacks import get_callbacks

class OptunaPruningCallback(Callback):
    def __init__(self, trial: optuna.Trial, start_epoch: int = 4, monitor: str = "val_mae"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_value = logs.get(self.monitor)

        if current_value is not None:
            self.trial.report(current_value, step=epoch)
            if epoch >= self.start_epoch and self.trial.should_prune():
                print(
                    f"[Trial {self.trial.number}] Pruning at epoch {epoch + 1} with {self.monitor}: {current_value}",
                    file=sys.stderr,
                )
                raise optuna.TrialPruned()


def clean_history(history: dict) -> dict:
    """
    Cleans the history dictionary to convert all values to float.
    Args:
        history (dict): History dictionary from model training.
    Returns:
        dict: Cleaned history with float values.
    """
    if not isinstance(history, dict):
        raise ValueError("History must be a dictionary.")

    return {metric: [float(x) for x in v] for metric, v in history.items()}


def objective(trial: optuna.Trial) -> float:
    
    config = Config()
    dataset = Dataset(config.random_seed)

    # Fixed flags from config
    trial.set_user_attr("xception_bool", config.xception_enabled)
    trial.set_user_attr("cbam_bool", config.cbam_enabled)
    trial.set_user_attr("eca_bool", config.temporal_eca_enabled)
    trial.set_user_attr("downsample", config.downsample)
    trial.set_user_attr("gamma", config.gamma)
    trial.set_user_attr("beta", config.beta)

    # Suggested hyperparameters
    hp = {
        "xception_bool": config.xception_enabled,
        "cbam_bool": config.cbam_enabled,
        "eca_bool": config.temporal_eca_enabled,
        "downsample": config.downsample,
        "gamma": config.gamma,
        "beta": config.beta,
        "num_filters": trial.suggest_int("num_filters", 16, 32, step=8),
        "kernel_size": trial.suggest_int("kernel_size", 3, 11, step=4),
        "middle_blocks": trial.suggest_int("middle_blocks", 1, 3),
        "r_ratio": trial.suggest_int("r_ratio", 8, 16, step=4),
        "fc_units": trial.suggest_int("fc_units", 64, 128, step=32),
        "gru_units": trial.suggest_int("gru_units", 128, 512, step=128),
        "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.001, 0.0001]),
        "xception_dropout": trial.suggest_float("xception_dropout", 0.2, 0.6, step=0.2),
        "gru_dropout": trial.suggest_float("gru_dropout", 0.2, 0.6, step=0.2),
        "fc_dropout": trial.suggest_float("fc_dropout", 0.2, 0.6, step=0.2),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32])
    }

    

    # Build and compile model
    model = FusedModel.build_and_compile(hp)
    
    # Load Datasets
    train_dataset = dataset.create_tf_dataset(
        config.processed_dataset_dir + "train/", hp.get("batch_size")
    )
    val_dataset = dataset.create_tf_dataset(config.processed_dataset_dir + "val/",  hp.get("batch_size"))

    # Fit with early stopping and pruning
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.epochs,
        callbacks=[
            OptunaPruningCallback(trial, monitor="val_mae"),
            k.callbacks.EarlyStopping("val_loss", patience=3),
        ],
        verbose=0,
    )
    
    k.backend.clear_session()
    gc.collect()
    
    return min(history.history["val_loss"])  # Return the best validation loss


def train_model():
    config = Config()

    for base_dir in [
        config.model_exports_dir,
        config.reports_dir,
        config.model_checkpoints_dir,
        config.model_logs_dir,
    ]:
        os.makedirs(os.path.join(base_dir, config.experiment_name), exist_ok=True)

    # Optimize
    study = optuna.create_study(
        direction="minimize",
        study_name=config.experiment_name,
        storage="sqlite:///" + config.model_tuning_dir + config.experiment_name + ".db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=200)

    # Best trial
    best_trial = study.best_trial
    best_params = best_trial.params
    best_batch_size = best_params["batch_size"]

    # Regenerate config and dataset
    dataset = Dataset(config.random_seed)
    best_hp = {
        **best_params,
        "xception_bool": config.xception_enabled,
        "cbam_bool": config.cbam_enabled,
        "eca_bool": config.temporal_eca_enabled,
        "downsample": config.downsample,
        "gamma": config.gamma,
        "beta": config.beta,
    }

    model = FusedModel.build_and_compile(best_hp)

    train_dataset = dataset.create_tf_dataset(
        config.processed_dataset_dir + "train/", best_batch_size
    )
    val_dataset = dataset.create_tf_dataset(config.processed_dataset_dir + "val/", best_batch_size)

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

    cleaned_history = clean_history(history.history)

    # Save model
    model.save(config.model_exports_dir + config.experiment_name + "/tuned_model.keras")

    # Save summary
    with open(config.reports_dir + config.experiment_name + "/tuned_summary.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    best_epoch_idx = int(np.argmin(history.history["val_loss"]))
    best_metrics = {m: v[best_epoch_idx] for m, v in cleaned_history.items()}

    # Save trial info
    with open(config.reports_dir + config.experiment_name + "/tuned_desc.txt", "w") as f:
        f.write(f"Training time: {end_time - start_time:.2f} seconds\n")
        f.write("Hyperparameters: \n")
        json.dump(best_params, f, indent=4)
        f.write(f"\nMetrics at best val_loss (Epoch {best_epoch_idx + 1}):\n")
        for m, v in best_metrics.items():
            f.write(f"{m}: {v:.4f}\n")

    with open(config.reports_dir + config.experiment_name + "/full_history.json", "w") as f:
        json.dump(cleaned_history, f, indent=4)

train_model()
