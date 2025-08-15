import os
import sys
import time
import json
import numpy as np
import keras as k
import keras_tuner as kt
from keras import Model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import Config
from data.cvDataset import Dataset
from modelling.model import build_fused_model
from utils.callbacks import get_callbacks
from utils.set_seed import set_seeds

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
        hp.Int("kernel_size", min_value=3, max_value=11, step=4)
        hp.Int("middle_blocks", min_value=2, max_value=4)

        # CBAM reduction ratio
        hp.Int("r_ratio", min_value=8, max_value=16, step=8)

        # Model Params
        hp.Int("fc_units", min_value=64, max_value=128, step=32)
        hp.Int("gru_units", min_value=64, max_value=128, step=32)
        hp.Choice("learning_rate", values=[1e-3, 1e-4])

        # Dropout rates
        hp.Choice("gru_dropout", values=[0.0, 0.1])
        hp.Choice("xception_dropout", values=[0.0, 0.2])
        hp.Choice("fc_dropout", values=[0.2, 0.4])

        # Instantiate and compile model with hyperparameters
        model = build_fused_model(hp)
        return model

class CrossValTuner(kt.BayesianOptimization):
    def __init__(self, hypermodel, config, dataset, **kwargs):
        super().__init__(hypermodel, **kwargs)
        self.config = config
        self.dataset = dataset
        self.fold_dirs = self._get_fold_dirs()

    def _get_fold_dirs(self):
        """Creates a list of fold directories for cross-validation."""
        base_dir = self.config.processed_dataset_dir
        fold_dirs = []
        
        for d in os.listdir(base_dir):
            if d.startswith("fold_"):
                fold_dirs.append(os.path.join(base_dir, d))
        
        print(f"Found {len(fold_dirs)} folds")
        return sorted(fold_dirs)

    def run_trial(self, trial, *args, **kwargs):
        """Override to use CV instead of single val set"""
        hp = trial.hyperparameters
        model = self.hypermodel.build(hp)
        
        fold_scores = []
        
        for i, fold_dir in enumerate(self.fold_dirs):
            print(f"Evaluating fold {i + 1}/{len(self.fold_dirs)}")
            train = self.dataset.create_tf_dataset(os.path.join(fold_dir, "train/"), batch_size=self.config.batch_size)
            val = self.dataset.create_tf_dataset(os.path.join(fold_dir, "val/"), batch_size=self.config.batch_size)
            
            # Reset weights
            model = self.hypermodel.build(hp)
            
            # Train on the fold
            history = model.fit(
                train,
                validation_data=val,
                epochs=30,
                callbacks=[k.callbacks.EarlyStopping("val_loss", patience=5)],
                verbose=0,
            )

            fold_scores.append(min(history.history["val_loss"]))

        mean_cv_score = np.mean(fold_scores)
        std_cv_score = np.std(fold_scores)
        print(f"Fold scores: {fold_scores}")
        print(f"  CV Score: {mean_cv_score:.4f} ± {std_cv_score:.4f}")
        
        return mean_cv_score

def clean_history(history):
    """Convert all NumPy types in history to native Python types."""
    return {metric: [float(x) for x in v] for metric, v in history.items()}

def train_model():
    """Performs hyperparameter tuning using the defined space in build_model, using the config.
    This config is produced from .env to tune the fused model appropriately before retraining
    and saving the best."""

    config = Config()
    dataset = Dataset(config.random_seed, target="true_room")
    set_seeds(config.random_seed)
    
    # Check for required directories
    for base_dir in [
        config.model_exports_dir,
        config.reports_dir,
        config.model_checkpoints_dir,
        config.model_logs_dir,
    ]:
        os.makedirs(os.path.join(base_dir, config.experiment_name), exist_ok=True)

    # Tune the model according to val_loss
    tuner = CrossValTuner(
        CustomHyperModel(config, dataset),
        config=config,
        dataset=dataset,
        objective="val_loss",
        max_trials=100,
        directory=config.model_tuning_dir,
        project_name=config.experiment_name,
    )
    
    # Run hyperparameter search
    print("Starting cross-val hyperparameter search")
    tuner.search()

    # Get best hyperparameters
    best_hps = tuner.get_best_hyperparameters(1)[0]
    print(f"Best hyperparameters: {best_hps.values}")

    # Build best model and retrain on full dataset with all callbacks
    print("Training final model with best  hyperparameters")
    model = tuner.hypermodel.build(best_hps)

    train = dataset.create_tf_dataset(
        config.processed_dataset_dir + "train/",
        batch_size=config.batch_size
    )
    test = dataset.create_tf_dataset(
        config.processed_dataset_dir + "test/",
        batch_size=config.batch_size,
        shuffle=False
    )

    start_time = time.time()
    history = model.fit(
        train,
        validation_data=test,
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
