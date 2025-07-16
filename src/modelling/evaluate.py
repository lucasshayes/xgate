import os
import sys
import keras.api as k
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config
from data.dataset import Dataset
from utils.metrics import p_ecdf, euclidean_distance, ecdf


def evaluate_model():
    config = Config()
    dataset = Dataset(config.random_seed)
    model = k.saving.load_model(
        config.model_exports_dir + config.experiment_name + "/tuned_model.keras"
    )
    test_set = Dataset.create_tf_dataset(config.processed_dataset_dir + "train/", config.batch_size)

    metrics = model.evaluate(test_set, return_dict=True)

    preds = model.predict(test_set)
    gt = np.concatenate([y for _, y in test_set], axis=0)

    euclidean_errors = euclidean_distance(gt, preds)
    sorted_errors, ecdf_values = ecdf(euclidean_errors)

    ecdf99 = p_ecdf(sorted_errors, ecdf_values, 0.99)
    ecdf50 = p_ecdf(sorted_errors, ecdf_values, 0.50)


evaluate_model()
