from keras import layers

def feat_attention(inputs):
    # Avg pool across the feature dimension
    avg_pool = layers.GlobalAveragePooling1D(name="feat_attention_avg_pool")(inputs)
    # Softmax to get attention weights
    attn_weights = layers.Softmax(axis=-1, name="feat_attention_weights")(avg_pool)
    # Reshape to match the input shape
    attn_weights = layers.Reshape((1, 7), name="feat_attention_reshape")(attn_weights)
    # Multiply the input by the attention weights
    x = layers.Multiply(name="feat_attention_multiply")([inputs, attn_weights])
    return x