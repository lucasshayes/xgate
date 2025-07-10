import tensorflow as tf
from keras import layers, activations
from keras import Sequential


class CBAM1D(layers.Layer):
    def __init__(self, r_ratio=8, name="cbam", **kwargs):
        super(CBAM1D, self).__init__(name=name, **kwargs)
        self.r_ratio = r_ratio

    def build(self, input_shape):
        channel = input_shape[-1]

        # Channel attention MLP
        self.shared_mlp = Sequential(
            [
                layers.Dense(channel // self.r_ratio, activation="relu"),
                layers.Dense(channel),
            ],
            name=f"{self.name}_shared_mlp",
        )

        # Spatial attention conv layer (no bias)
        self.spatial_conv = layers.Conv1D(
            filters=1,
            kernel_size=7,
            padding="same",
            activation="sigmoid",
            use_bias=False,
            name=f"{self.name}_spatial_conv",
        )

    def call(self, inputs):
        # Feature (channel) attention
        avg_pool = tf.reduce_mean(inputs, axis=1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=1, keepdims=True)

        avg_out = self.shared_mlp(avg_pool)
        max_out = self.shared_mlp(max_pool)

        channel_att = activations.sigmoid(avg_out + max_out)
        channel_refined = inputs * channel_att

        # Temporal (spatial) attention
        avg_pool = tf.reduce_mean(channel_refined, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(channel_refined, axis=-1, keepdims=True)

        concat = tf.concat([avg_pool, max_pool], axis=-1)
        spatial_att = self.spatial_conv(concat)

        return channel_att * spatial_att

    def compute_output_shape(self, input_shape):
        return input_shape
