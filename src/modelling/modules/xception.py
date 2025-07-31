import keras as k
from keras import layers

def xception_block(inputs, num_filters=32, k_size=3, middle_blocks=2, downsample=False, name="xception"):
    # Block 1
    x = layers.Conv1D(
        filters=num_filters, kernel_size=k_size, padding="same", 
        strides=2, kernel_initializer="he_normal", 
        kernel_constraint=k.constraints.max_norm(3),
        name=f"{name}_conv1d_1"
    )(inputs)
    x = layers.BatchNormalization(name=f"{name}_bn_1")(x)
    x = layers.ReLU(name=f"{name}_ReLU_1")(x)
    
    # Block 2
    x = layers.Conv1D(
        filters=num_filters*2, kernel_size=k_size, padding="same", 
        kernel_initializer="he_normal", kernel_constraint=k.constraints.max_norm(3),
        name=f"{name}_conv1d_2"
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn_2")(x)
    x = layers.ReLU(name=f"{name}_ReLU_2")(x)

    # Setup residual connection
    residual = x
    
    # Separable convolutions
    x = layers.SeparableConv1D(
        filters=num_filters * 4, kernel_size=3, padding="same",
        depthwise_initializer="he_normal", pointwise_initializer="he_normal",
        depthwise_constraint=k.constraints.max_norm(3),
        pointwise_constraint=k.constraints.max_norm(3),
        name=f"{name}_sep_conv1"
    )(x)
    x = layers.BatchNormalization(name=f"{name}_sep_bn1")(x)
    x = layers.ReLU(name=f"{name}_ReLU_3")(x)
    
    x = layers.SeparableConv1D(
        filters=num_filters * 4, kernel_size=3, padding="same",
        depthwise_initializer="he_normal", pointwise_initializer="he_normal",
        depthwise_constraint=k.constraints.max_norm(3),
        pointwise_constraint=k.constraints.max_norm(3),
        name=f"{name}_sep_conv2"
    )(x)
    x = layers.BatchNormalization(name=f"{name}_sep_bn2")(x)

    # Residual layers
    if residual.shape[-1] != num_filters * 4:
        residual = layers.Conv1D(
            filters=num_filters*4, kernel_size=1, padding="same",
            kernel_initializer='he_normal', kernel_constraint=k.constraints.max_norm(3),
            name=f"{name}_residual_conv"
        )(residual)
        residual = layers.BatchNormalization(name=f"{name}_residual_bn")(residual)
    
    # Add residual connection
    x = layers.Add(name=f"{name}_add")([x, residual])
    
    # Middle blocks
    for i in range(middle_blocks):
        x = middle_block(x, num_filters * 4, i, name=name)
    
    # Optional downsampling
    if downsample:
        x = layers.AveragePooling1D(pool_size=2, name=f"{name}_downsample")(x)

    return x

def middle_block(inputs, filters, i, name="xception"):

    # Setup residual connection
    residual = inputs
    
    # Three separable convolutions
    x = layers.ReLU(name=f"{name}_middle_{i}_relu_1")(inputs)
    x = layers.SeparableConv1D(
        filters, 3, padding="same", use_bias=False,
        depthwise_initializer="he_normal", pointwise_initializer="he_normal",
        depthwise_constraint=k.constraints.max_norm(3),
        pointwise_constraint=k.constraints.max_norm(3),
        name=f"{name}_middle_{i}_conv1d_1"
    )(x)
    x = layers.BatchNormalization(name=f"{name}_middle_{i}_bn_1")(x)

    x = layers.ReLU(name=f"{name}_middle_{i}_relu_2")(x)
    x = layers.SeparableConv1D(
        filters, 3, padding="same", use_bias=False,
        depthwise_initializer="he_normal", pointwise_initializer="he_normal",
        depthwise_constraint=k.constraints.max_norm(3),
        pointwise_constraint=k.constraints.max_norm(3),
        name=f"{name}_middle_{i}_conv1d_2"
    )(x)
    x = layers.BatchNormalization(name=f"{name}_middle_{i}_bn_2")(x)

    x = layers.ReLU(name=f"{name}_middle_{i}_relu_3")(x)
    x = layers.SeparableConv1D(
        filters, 3, padding="same", use_bias=False,
        depthwise_initializer='he_normal', pointwise_initializer='he_normal',
        depthwise_constraint=k.constraints.max_norm(3),
        pointwise_constraint=k.constraints.max_norm(3),
        name=f"{name}_middle_{i}_conv1d_3"
    )(x)
    x = layers.BatchNormalization(name=f"{name}_middle_{i}_bn_3")(x)

    # Residual layers
    if residual.shape[-1] != filters:
        residual = layers.Conv1D(
            filters, 1, padding="same", use_bias=False,
            kernel_initializer='he_normal', kernel_constraint=k.constraints.max_norm(3),
            name=f"{name}_middle_{i}_residual_conv"
        )(residual)
        residual = layers.BatchNormalization(name=f"{name}_middle_{i}_residual_bn")(residual)

    # Add residual connection
    x = layers.Add(name=f"{name}_middle_{i}_residual")([x, residual])
    return x

# Example usage:
if __name__ == "__main__":
    # Create a simple model for testing
    inputs = layers.Input(shape=(200, 7))
    x = xception_block(inputs, num_filters=32, middle_blocks=2)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    model = k.Model(inputs, outputs)
    model.summary()
    