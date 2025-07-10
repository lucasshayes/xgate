from keras import callbacks
import os


def get_callbacks(
    experiment_name: str, patience: int, model_checkpoints_dir: str, model_logs_dir: str
) -> list[callbacks.Callback]:
    """
    Get a list of callbacks for training the model.

    Args:
        experiment_name (str): Name of experiment (for saving)
        patience (int): Early stopping patience
        model_checpoints_dir (str): Where to store model checkpoints.
        model_logs_dir (str): Where to store model logs for tensorboard.
    Returns:
        list: A list of TensorFlow callbacks.
    """

    # Early stopping to prevent overfitting
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )

    # Model checkpoint to save the best model
    model_checkpoint = callbacks.ModelCheckpoint(
        filepath=model_checkpoints_dir + f"{experiment_name}_best_model.h5",
        monitor="val_loss",
        save_best_only=True,
    )

    # tensorboard callback for visualization
    tensorboard_callback = callbacks.TensorBoard(
        log_dir=os.path.join(model_logs_dir, experiment_name),
        histogram_freq=1,
        write_graph=True,
        write_images=True,
    )

    return [early_stopping, model_checkpoint, tensorboard_callback]
