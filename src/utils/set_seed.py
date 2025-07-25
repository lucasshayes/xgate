import keras as k
import tensorflow as tf
import os
import numpy as np
import random


def set_seeds(seed: int):
    k.utils.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if k.backend.backend() == "tensorflow":
        tf.config.experimental.enable_op_determinism()
        tf.random.set_seed(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
    os.environ['PYTHONHASHSEED'] = str(seed)