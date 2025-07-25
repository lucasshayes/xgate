import numpy as np
from keras import layers
from keras import ops


class TemporalECA(layers.Layer):
    def __init__(self, gamma=2, b=1, name="temporal_eca", **kwargs):
        super(TemporalECA, self).__init__(name=name, **kwargs)
        self.gamma = gamma
        self.b = b

    def build(self, input_shape):
        channel = input_shape[-1]

        # Calculate kernel size based on channel size
        t = int(abs((np.log2(channel) + self.b) / self.gamma))
        k_size = t if t % 2 else t + 1

        # Summarizes each time step
        self.squeeze = layers.Lambda(
            lambda x: ops.mean(x, axis=-1, keepdims=True),
            name=f"{self.name}_squeeze",
        )
        # Conv over time
        self.excite = layers.Conv1D(
            filters=1,
            kernel_size=k_size,
            activation="sigmoid",
            padding="same",
            use_bias=False,
            name=f"{self.name}_conv",
        )

        self.scale = layers.Multiply(name=f"{self.name}_scale")

        self.weighted_pool = layers.Lambda(
            lambda x: ops.mean(x, axis=1, keepdims=False),
            name=f"{self.name}_weighted_pool",
        )

    def call(self, inputs):
        squeeze = self.squeeze(inputs)
        excite = self.excite(squeeze)
        scaled = self.scale([inputs, excite])
        weighted_pool = self.weighted_pool(scaled)
        return weighted_pool

    def compute_output_shape(self, input_shape):
        return input_shape
