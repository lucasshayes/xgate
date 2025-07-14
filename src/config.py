from dotenv import load_dotenv
import os


class Config:
    """
    Configuration class to manage environment variables and settings for the project.

    functions:
        - print_config: Prints the current configuration settings.

    attributes:
        - EXTERNAL_DATASET_DIR: Directory for external datasets.
        - RAW_DATASET_DIR: Directory for raw datasets.
        - PROCESSED_DATASET_DIR: Directory for processed datasets.
        - RANDOM_SEED: Random seed for reproducibility.
        - EXPERIMENT_NAME: Name of the experiment.
        - MODEL_CHECKPOINTS_DIR: Directory for model checkpoints.
        - MODEL_EXPORTS_DIR: Directory for model exports.
        - BASELINE: Whether to use baseline model.
        - CBAM_ENABLED: Whether to use CBAM (Convolutional Block Attention Module).
        - TEMPORAL_ECA_ENABLED: Whether to use Temporal ECA (Efficient Channel Attention).
        - BATCH_SIZE: Batch size for training.
        - EPOCHS: Number of epochs for training.
        - LEARNING_RATE: Learning rate for the optimizer.
        - RESULTS_DIR: Directory for saving results.
        - PLOT_DIR: Directory for saving plots.
    """

    def __init__(self):
        # Load environment variables from .env file
        load_dotenv(override=True)

        # Dataset
        # -- Directories
        self.external_dataset_dir = os.getenv("EXTERNAL_DATASET_DIR", "data/external/OWP-IMU/")
        self.raw_dataset_dir = os.getenv("RAW_DATASET_DIR", "data/raw/")
        self.processed_dataset_dir = os.getenv("PROCESSED_DATASET_DIR", "data/processed/")
        # -- Preprocessing Parameters
        self.sample_rate = int(os.getenv("SAMPLE_RATE", 200))
        self.window_size = int(os.getenv("WINDOW_SIZE", 4))
        self.step_size = int(os.getenv("STEP_SIZE", 2))
        self.test_size = float(os.getenv("TEST_SIZE", 0.2))
        self.val_size = float(os.getenv("VAL_SIZE", 0.2))

        # Model
        self.experiment_name = os.getenv("EXPERIMENT_NAME", "default_experiment")
        self.model_checkpoints_dir = os.getenv("MODEL_CHECKPOINTS_DIR", "models/checkpoints/")
        self.model_logs_dir = os.getenv("MODEL_LOGS_DIR", "models/logs/")
        self.model_exports_dir = os.getenv("MODEL_EXPORTS_DIR", "models/exports/")
        self.model_tuning_dir = os.getenv("MODEL_TUNING_DIR", "/models/tuning/")

        # -- Fused Hyperparameters
        self.xception_enabled = bool(os.getenv("XCEPTION_ENABLED", False))
        self.cbam_enabled = bool(os.getenv("CBAM_ENABLED", True))
        self.temporal_eca_enabled = bool(os.getenv("TEMPORAL_ECA_ENABLED", True))
        self.epochs = int(os.getenv("EPOCHS", 100))
        # -- Xception Fixed Params
        self.downsample = bool(os.getenv("DOWNSAMPLE", False))
        # -- CBAM fixed params
        self.gamma = int(os.getenv("GAMMA", 2))
        self.beta = int(os.getenv("BETA", 1))
        # Results
        self.reports_dir = os.getenv("REPORTS_DIR", "reports/")
        self.plot_dir = os.getenv("PLOT_DIR", "reports/figures/")

        # Misc
        self.random_seed = int(os.getenv("RANDOM_SEED", 42))

    def print_config(self):
        print("Configuration:")
        for attr, value in self.__dict__.items():
            print(f"{attr}: {value}")


# Example usage
if __name__ == "__main__":
    config = Config()
    config.print_config()
