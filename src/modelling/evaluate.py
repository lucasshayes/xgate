import json
import os
import sys
import keras as k
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config
from data.dataset import Dataset
from utils.plots import cm_plot, report_plot
from keras import backend as K

from modules.xception import xception_block
from modules.attention.cbam import cbam_1d_block
from modules.custom_layers import ReduceMean1D, ReduceMax1D


def evaluate_model():
    print(K.backend())
    config = Config()
    dataset = Dataset(config.random_seed, target="true_room")
    model_path = config.model_exports_dir + config.experiment_name + "/tuned_model.keras"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    custom_objects = {
        "ReduceMean1D": ReduceMean1D,
        "ReduceMax1D": ReduceMax1D,
        "cbam_1d_block": cbam_1d_block,
        "xception_block": xception_block,
    }

    model = k.saving.load_model(model_path, custom_objects=custom_objects)

    # Check model is loaded
    if model is None:
        raise ValueError("Failed to load the model")
    
    test_set = dataset.create_tf_dataset(config.processed_dataset_dir + "test/", config.batch_size, shuffle=False)

    # Check test_set is created
    if test_set is None:
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
    
    sorted_classes = [""] * len(target_classes)
    for ohe_str, class_name in target_classes.items():
        ohe_list = eval(ohe_str)  # Convert string back to list
        class_index = ohe_list.index(1.0)  # Find position of 1.0
        sorted_classes[class_index] = class_name
    
    # Convert predictions and ground truth to class labels
    y_pred = preds.argmax(axis=1)
    y_true = gt.argmax(axis=1)
    
    # Create and plot confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    cm_plot(cm, labels=sorted_classes, title="Confusion Matrix", dir=config.reports_dir + config.experiment_name + "/")

    # Create and save classification report
    report = classification_report(y_true, y_pred, target_names=sorted_classes, output_dict=True)
    with open(config.reports_dir + config.experiment_name + "/classification_report.json", "w") as f:
        json.dump(report, f, indent=4)
    
    report_plot(report, labels=sorted_classes, dir=config.reports_dir + config.experiment_name +"/")
   



evaluate_model()
