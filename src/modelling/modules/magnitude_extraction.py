from keras import layers
import keras as k

def magnitude_extraction(x, indices=(0, 1, 2)):
    """Computes accelerometer magnitude features from X, Y, Z channels 
    and concatenates them to the input.

    Args:
        x (tf.Tensor): Input tensor of shape (batch_size, time_steps, features).
        indices (tuple, optional): Indices of accelerometer X, Y, Z 
            in the feature dimension. Defaults to (0, 1, 2).

    Returns:
        tf.Tensor: Output tensor of shape (batch_size, time_steps, features + 2),
            where the two new features are magnitude and dynamic magnitude.
    """
    # Extract accelerometer channels
    acc_x = x[..., indices[0]]
    acc_y = x[..., indices[1]]
    acc_z = x[..., indices[2]]

    # Compute magnitude: sqrt(x² + y² + z²)
    mag = k.ops.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    mag_norm = layers.LayerNormalization(axis=-1, name="mag_layernorm")(mag)
    # Compute dynamic magnitude (gravity removed)
    mag_dyn = k.ops.abs(mag - 9.81)
    mag_dyn_norm = layers.LayerNormalization(axis=-1, name="mag_dyn_layernorm")(mag_dyn)

    # Concatenate new features to original input
    return layers.Concatenate(name="acc_magnitude_concat")(
        # Match dimensions (batch, time_steps) x2 -> (batch, time_steps, 1) x2 -> (batch, time_steps, features + 2)
        [x, k.ops.expand_dims(mag_norm, -1), k.ops.expand_dims(mag_dyn_norm, -1)]
    )