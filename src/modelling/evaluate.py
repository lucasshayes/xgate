import json
import os
import sys
import keras as k
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config
from data.dataset import Dataset
from utils.plots import save_metrics, feat_attention_plot
from keras import backend as K

from modules.xception import xception_block
from modules.attention.cbam import cbam_1d_block
from modules.custom_layers import ReduceMean1D, ReduceMax1D

def calc_attention_weights(model, test_set):
    """Calculate attention weights from the model.

    Args:
        model (tf.keras.Model): The Keras model with attention layers.
        test_set (tf.data.Dataset): The test dataset.

    Returns:
        np.ndarray: The average attention weights.
    """
    attention_model = k.Model(inputs=model.input, outputs=model.get_layer("feat_attention_weights").output)

    for x_batch, y_batch in test_set.take(1):
        attention_weights = attention_model.predict(x_batch)
        break

    avg_attention = attention_weights.mean(axis=0)
    return avg_attention

def evaluate_model():
    """Evaluate the model and save metrics and plots.
    
    Raises:
        ValueError: If the model, test_set or target_classes is invalid.
    """
    config = Config()
    dataset = Dataset(config.random_seed, target="true_room")
    model_path = config.model_exports_dir + config.experiment_name + "/tuned_model.keras"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    custom_objects = {
        "ReduceMean1D": ReduceMean1D,
        "ReduceMax1D": ReduceMax1D,
        "cbam_1d_block": cbam_1d_block,
        "xception_block": xception_block
    }

    model = k.saving.load_model("best_model.keras", custom_objects=custom_objects)

    # Check model is loaded
    if model is None:
        raise ValueError("Failed to load the model")
    
    test_set = dataset.create_tf_dataset(config.processed_dataset_dir + "test/", config.batch_size, shuffle=False)

    # Check test_set is created
    if not isinstance(test_set, tf.data.Dataset):
        raise ValueError("Failed to create the test dataset, check processed dataset directory")
    
    # Evaluate the model, get metrics
    metrics = model.evaluate(test_set, return_dict=True)
    print("Evaluation Metrics:", metrics)
    
    # Generate predictions and prepare ground truth values
    preds = model.predict(test_set)
    gt = np.concatenate([y for _, y in test_set], axis=0)

    # Load target class labels from JSON file
    with open(config.processed_dataset_dir + "train/target_classes.json", 'r') as f:
        target_classes = json.load(f)
    
    if not target_classes:
        raise ValueError("Target classes are empty. Check the target_classes.json file.")
    
    sorted_classes = [""] * len(target_classes)
    for ohe_str, class_name in target_classes.items():
        ohe_list = eval(ohe_str)  # Convert string back to list
        class_index = ohe_list.index(1.0)  # Find position of 1.0
        sorted_classes[class_index] = class_name

    # Save plots and metrics
    save_metrics(gt, preds, sorted_classes, config.reports_dir + config.experiment_name + "/")
    print("Evaluation completed and metrics saved.")

    # Save attention weights plot
    attention_weights = calc_attention_weights(model, test_set)
    feat_attention_plot(attention_weights, ['ax', 'ay', 'az', 'bedroom', 'kitchen', 'living', 'stairs'], config.reports_dir + config.experiment_name + "/attention_weights.png")
    
if __name__ == "__main__":
    evaluate_model()
