from keras import layers
import keras as k

def feat_attention(inputs):
    """SENet style Feature Attention Mechanism.

    Args:
        inputs (Tensor): Input tensor of shape (batch_size, seq_length, features).

    Returns:
        Tensor: Output tensor with applied feature attention.
    """
    # Pools over and reduces temporal dimension
    avg_pool = layers.GlobalAveragePooling1D(name="feat_attention_avg_pool")(inputs)
    # Softmax to get attention weights
    # Could instead use SENet style learnable weights using 2 dense layers with r ratio. A/B test accuracy.
    attn_weights = layers.Softmax(axis=-1, name="feat_attention_weights")(avg_pool)
    # Reshape to match the input shape
    attn_weights = layers.Reshape((1, inputs.shape[-1]), name="feat_attention_reshape")(attn_weights)
    # Multiply the input by the attention weights
    x = layers.Multiply(name="feat_attention_multiply")([inputs, attn_weights])
    return x