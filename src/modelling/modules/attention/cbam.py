import tensorflow as tf
from keras import layers
from keras import Sequential


class CBAM1D(layers.Layer):
    def __init__(self, r_ratio=8, name="cbam", **kwargs):
        super(CBAM1D, self).__init__(name=name, **kwargs)
        self.r_ratio = r_ratio

    def build(self, input_shape):
        channel = input_shape[-1]
        
        # Feature (channel) attention layers
        self.feature_avg = layers.Lambda(
            lambda x: tf.reduce_mean(x, axis=-1, keepdims=True),
            name=f"{self.name}_feature_avg",
        )
        
        self.feature_max = layers.Lambda(
            lambda x: tf.reduce_max(x, axis=-1, keepdims=True),
            name=f"{self.name}_feature_max",
        )
        
        self.shared_mlp = Sequential(
            [
                layers.Dense(channel // self.r_ratio, activation="relu"),
                layers.Dense(channel),
            ],
            name=f"{self.name}_shared_mlp",
        )
        
        self.feature_add = layers.Add(name=f"{self.name}_feature_add")
        self.feature_sigmoid = layers.Activation("sigmoid", name=f"{self.name}_feature_sigmoid") 
        self.feature_mul = layers.Multiply(name=f"{self.name}_feature_mul")

        # Temporal (spatial) attention layers
        self.temporal_avg = layers.Lambda(
            lambda x: tf.reduce_mean(x, axis=1, keepdims=True),
            name=f"{self.name}_temporal_avg",
        )
        
        self.temporal_max = layers.Lambda(
            lambda x: tf.reduce_max(x, axis=1, keepdims=True),
            name=f"{self.name}_temporal_max",
        )
        
        self.temporal_concat = layers.Concatenate(axis=-1, name=f"{self.name}_concat")
        
        self.temporal_conv = layers.Conv1D(
            filters=1,
            kernel_size=7,
            padding="same",
            activation="sigmoid",
            use_bias=False,
            name=f"{self.name}_spatial_conv",
        )
        
        # Final layers
        self.final_mul = layers.Multiply(name=f"{self.name}_final_mul")
        self.res_add = layers.Add(name=f"{self.name}_residual_add")

    def call(self, inputs):
        # Feature (channel) attention
        avg_pool = self.feature_avg(inputs)
        max_pool = self.feature_max(inputs)

        avg_out = self.shared_mlp(avg_pool)
        max_out = self.shared_mlp(max_pool)

        channel_att = self.feature_sigmoid(self.feature_add([avg_out, max_out]))
        channel_refined = self.feature_mul([inputs, channel_att])

        # Temporal (spatial) attention
        avg_pool = self.temporal_avg(channel_refined)
        max_pool = self.temporal_max(channel_refined)

        concat = self.temporal_concat([avg_pool, max_pool])
        spatial_att = self.temporal_conv(concat)

        cbam_final = self.final_mul([channel_refined, spatial_att])
        return self.res_add([inputs, cbam_final])

    def compute_output_shape(self, input_shape):
        return input_shape
