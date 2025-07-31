import os
import sys
import keras
import numpy as np
from keras import layers
from keras import ops

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from custom_layers import ReduceMean1D

def temporal_eca_block(inputs, gamma=2, b=1, name="temporal_eca"):
    channel = inputs.shape[-1]
    # Calculate kernel size based on channel size
    t = int(abs((np.log2(channel) + b) / gamma))
    k_size = t if t % 2 else t + 1
    
    # Pooling over the temporal dimension
    squeeze = ReduceMean1D(axis=-1, keepdims=True, name=f"{name}_squeeze")(inputs)
    # Excitation step (1D convolution))
    excite = layers.Conv1D(
        filters=1,
        kernel_size=k_size,
        activation="sigmoid",
        padding="same",
        use_bias=False,
        name=f"{name}_excite_conv"
    )(squeeze)
    # Scale to inputs
    scaled = layers.Multiply(name=f"{name}_scale")([inputs, excite])
    # Final pooling to reduce dimensions
    weighted_pool = ReduceMean1D(axis=1, keepdims=False, name=f"{name}_weighted_pool")(scaled)
    return weighted_pool

if __name__ == "__main__":
    # Create a simple model for testing
    inputs = layers.Input(shape=(50, 7))
    x = temporal_eca_block(inputs, gamma=2, b=1, name="temporal_eca")
    model = keras.Model(inputs, x)
    model.summary()