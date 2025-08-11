from keras import layers
import keras as k

def rolling_extraction(x, window: int = 5):
    """Applies a domain shift operation to the input tensor. Using a rolling mean and standard deviation to normalize the input.

    Args:
        x (tf.Tensor): Input tensor of shape (batch_size, time_steps, features).
        window (int, optional): Size of the rolling window. Defaults to 5.

    Returns:
        tf.Tensor: Output tensor of shape (batch_size, time_steps, features * 3).
    """
    rolling_mean = layers.AveragePooling1D(pool_size=window, strides=1, padding="same", name="rolling_mean")(x)
    
    x_sq = k.ops.square(x)
    rolling_mean_x2 = layers.AveragePooling1D(pool_size=window, strides=1, padding="same", name="rolling_std")(x_sq)
    
    variance = k.ops.maximum(rolling_mean_x2 - k.ops.square(rolling_mean), k.backend.epsilon())
    rolling_std = k.ops.sqrt(variance)
    
    return k.layers.Concatenate(name="concat")([x, rolling_mean, rolling_std])