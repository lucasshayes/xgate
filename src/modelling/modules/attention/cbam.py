import os
import sys
from keras import layers
import keras 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from custom_layers import ReduceMean1D, ReduceMax1D

def cbam_1d_block(inputs, r_ratio=8, name="cbam"):
    """Convolutional Block Attention Module (CBAM) across Feature and Temporal axis

    Args:
        inputs (tf.Tensor): Input tensor of shape (batch_size, time_steps, channels).
        r_ratio (int, optional): Reduction ratio for channel attention. Defaults to 8.
        name (str, optional): Name of the layer. Defaults to "cbam".

    Returns:
        tf.Tensor: Output tensor of shape (batch_size, time_steps, channels).
    """
    channel = inputs.shape[-1]
    
    # Feature (channel) attention
    # Global average pooling and max pooling along temporal dimension
    avg_pool = ReduceMean1D(axis=1, keepdims=True, name=f"{name}_feature_avg")(inputs)
    max_pool = ReduceMax1D(axis=1, keepdims=True, name=f"{name}_feature_max")(inputs)
    
    # Shared MLP for both avg and max pooled features
    # Dense layers for channel attention
    avg_mlp = layers.Dense(channel // r_ratio, activation="relu", name=f"{name}_avg_dense1")(avg_pool)
    avg_mlp = layers.Dense(channel, name=f"{name}_avg_dense2")(avg_mlp)
    
    max_mlp = layers.Dense(channel // r_ratio, activation="relu", name=f"{name}_max_dense1")(max_pool)
    max_mlp = layers.Dense(channel, name=f"{name}_max_dense2")(max_mlp)
    
    # Combine and apply sigmoid
    channel_att = layers.Add(name=f"{name}_feature_add")([avg_mlp, max_mlp])
    channel_att = layers.Activation("sigmoid", name=f"{name}_feature_sigmoid")(channel_att)
    
    # Apply channel attention
    channel_refined = layers.Multiply(name=f"{name}_feature_mul")([inputs, channel_att])
    
    # Temporal (spatial) attention
    # Average and max pooling along channel dimension
    temporal_avg = ReduceMean1D(axis=-1, name=f"{name}_temporal_avg")(channel_refined)
    temporal_max = ReduceMax1D(axis=-1, name=f"{name}_temporal_max")(channel_refined)
    
    # Concatenate avg and max features
    temporal_concat = layers.Concatenate(axis=-1, name=f"{name}_temporal_concat")([temporal_avg, temporal_max])
    
    # Conv1D for spatial attention
    temporal_att = layers.Conv1D(
        filters=1,
        kernel_size=7,
        padding="same",
        activation="sigmoid",
        use_bias=False,
        name=f"{name}_temporal_conv"
    )(temporal_concat)

    # Apply temporal attention
    cbam_output = layers.Multiply(name=f"{name}_temporal_mul")([channel_refined, temporal_att])
    
    # Residual connection
    output = layers.Add(name=f"{name}_residual_add")([inputs, cbam_output])
    
    return output

# Example usage:
if __name__ == "__main__":
    # Create a simple model for testing
    inputs = layers.Input(shape=(200, 64))
    x = cbam_1d_block(inputs, r_ratio=8, name="cbam")
    outputs = layers.Dense(10, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    model.summary()
    