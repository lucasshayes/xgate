import json
import os
import sys
import keras as k
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config
from data.dataset import Dataset
from modelling.modules.custom_layers import ReduceMean1D, ReduceMax1D
from utils.plots import save_metrics, feat_attention_plot

def extract_feat_attention(model, X, batch_size=32) -> np.ndarray:
    """Extract feature attention weights from the model.

    Args:
        model (tf.keras.Model): The Keras model with attention layers.
        X (np.ndarray): The input data.
        true_labels (np.ndarray): The ground truth labels.
        batch_size (int): The batch size for prediction.

    Returns:
        np.ndarray: The extracted room attention weights.
    """
    attention_model = k.Model(inputs=model.input, outputs=model.get_layer("feat_attention_weights").output)
    attention_weights = attention_model.predict(X, batch_size=batch_size, verbose=1)
    
    return attention_weights

def retrieve_preds(model_path: str, data_dir: str, batch_size: int) -> tuple[k.Model, np.ndarray, np.ndarray, np.ndarray]:
    """Retrieve model predictions for the test set.

    Args:
        model_path (str): The path to the Keras model.
        test_set (tf.data.Dataset): The test dataset.

    Returns:
        tuple[k.Model, np.ndarray, np.ndarray, np.ndarray]: The model, input data (X), ground truth labels (y), and predictions.
    """
    model = k.saving.load_model(model_path)

    # Check model is loaded
    if model is None:
        raise ValueError("Failed to load the model")

    test_set = dataset.create_tf_dataset(data_dir + "test/", batch_size, shuffle=False)

    # Check test_set is created
    if not isinstance(test_set, tf.data.Dataset):
        raise ValueError("Failed to create the test dataset, check processed dataset directory")
    
    # Evaluate the model, get metrics
    metrics = model.evaluate(test_set, return_dict=True)
    print("Evaluation Metrics:", metrics)
    
    # Generate predictions and prepare ground truth values
    preds = model.predict(test_set)
    
    X = np.concatenate([x.numpy() for x, y in test_set], axis=0)
    y = np.concatenate([y.numpy() for x, y in test_set], axis=0)

    return model, X, y, preds

def get_flops(model: k.Model):
    """Get the number of FLOPs for a Keras model.

    Args:
        model (k.Model): The Keras model.

    Returns:
        int: The number of FLOPs.
    """
    model_input = model.input
    if isinstance(model_input, list):
        model_input = model_input[0]
    model_func = tf.function(lambda x: model(x))
    sample_data = tf.TensorSpec([1] + list(model_input.shape[1:]), dtype=tf.float32)
    concrete_func = model_func.get_concrete_function(sample_data)
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    graph = frozen_func.graph
    flops = model_analyzer.profile(graph, run_meta=tf.compat.v1.RunMetadata(), options=ProfileOptionBuilder.float_operation(), cmd='op')
    return flops.total_float_ops

def ablation_eval(gt, preds):
    """Evaluate the model predictions against ground truth.

    Args:
        gt (np.ndarray): Ground truth labels.
        preds (np.ndarray): Model predictions.

    Returns:
        dict: Accuracy, AUC-ROC, Macro F1 Score and Average Precision
    """
    gt_labels = np.argmax(gt, axis=1)
    pred_labels = np.argmax(preds, axis=1)

    # Calculate metrics
    accuracy = np.mean(pred_labels == gt_labels)
    auc_roc = roc_auc_score(gt, preds, multi_class="ovr")
    macro_f1 = f1_score(gt_labels, pred_labels, average="macro")
    avg_precision = average_precision_score(gt, preds, average="macro")

    return {
        "accuracy": accuracy,
        "auc_roc": auc_roc,
        "macro_f1": macro_f1,
        "avg_precision": avg_precision
    }

def evaluate_model(model_path: str, output_dir: str, data_dir: str, batch_size: int):
    """Evaluate the model and save metrics and plots.
    
    Raises:
        ValueError: If the model, test_set or target_classes is invalid.
    """

    model, X, y, preds = retrieve_preds(model_path, data_dir, batch_size)

    # Load target class labels from JSON file
    with open(data_dir + "train/target_classes.json", 'r') as f:
        target_classes = json.load(f)
    
    if not target_classes:
        raise ValueError("Target classes are empty. Check the target_classes.json file.")
    
    sorted_classes = [""] * len(target_classes)
    for ohe_str, class_name in target_classes.items():
        ohe_list = eval(ohe_str)  # Convert string back to list
        class_index = ohe_list.index(1.0)  # Find position of 1.0
        sorted_classes[class_index] = class_name

    # Save plots and metrics
    save_metrics(y, preds, sorted_classes, output_dir + "/")
    print("Evaluation completed and metrics saved.")

    if any([layer.name == "feat_attention_weights" for layer in model.layers]):
        attention_weights = extract_feat_attention(model, X)
        feat_attention_plot(attention_weights, y, ['ax', 'ay', 'az', 'bedroom', 'kitchen', 'living', 'stairs'], sorted_classes, output_dir + "/attention_weights.png")

def evaluate_models(model_titles: list[str], model_dir: str, output_dir: str, data_dir: str, batch_size: int):
    """Evaluate multiple models and save their metrics and plots.

    Args:
        model_titles (list[str]): List of model titles to evaluate.
        model_dir (str): Directory containing the model files.
        output_dir (str): Directory to save the evaluation results.
        data_dir (str): Directory containing the input data.
        batch_size (int): Batch size for evaluation.

    Raises:
        FileNotFoundError: If the model directory or data directory does not exist.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    ablation_list = []

    for title in model_titles:
        model_path = os.path.join(model_dir, title + ".keras")
        if not os.path.exists(model_path):
            print(f"Model {title} does not exist at {model_path}. Skipping.")
            continue

        title_output = output_dir + title + "/"
        os.makedirs(title_output, exist_ok=True)
        model, X, gt, preds = retrieve_preds(model_path, data_dir, batch_size)
        metrics = ablation_eval(gt, preds)
        model_dict = {"model": title, **metrics, "flops": get_flops(model), "params": model.count_params()}
        ablation_list.append(model_dict)

    df = pd.DataFrame(ablation_list)
    print(df)

if __name__ == "__main__":
    config = Config()
    dataset = Dataset(config.random_seed, target="true_room")
    
    # evaluate_models(
    #     model_titles=["base", "no_attention", "no_cbam", "no_eca", "no_feat_attention", "no_gru", "no_xception", "no_rolling_extraction"],
    #     model_dir=config.model_checkpoints_dir,
    #     output_dir=config.reports_dir,
    #     data_dir=config.processed_dataset_dir,
    #     batch_size=32
    # )
    
    evaluate_model(
        model_path=config.model_checkpoints_dir + "base.keras",
        output_dir=config.reports_dir + "base/",
        data_dir=config.processed_dataset_dir,
        batch_size=32
    )
