from keras import layers
import keras.api as k


class XceptionBlock(layers.Layer):
    def __init__(
        self,
        num_filters=32,
        k_size=3,
        middle_blocks=2,
        downsample=False,
        name="xception",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.num_filters = num_filters
        self.k_size = k_size
        self.middle_blocks = middle_blocks
        self.downsample_enabled = downsample
        self.block_name = name

        self.middle_blocks_layers = []
        self.downsample = None

    def build(self, input_shape):
        name = self.block_name
        num_filters = self.num_filters
        k_size = self.k_size

        self.conv1 = layers.Conv1D(
            filters=num_filters, kernel_size=k_size, padding="same", name=f"{name}_conv1d_1"
        )
        self.bn1 = layers.BatchNormalization(name=f"{name}_bn_1")
        self.relu1 = layers.ReLU(name=f"{name}_ReLU_1")

        self.conv2 = layers.Conv1D(
            filters=num_filters * 2, kernel_size=k_size, padding="same", name=f"{name}_conv1d_2"
        )
        self.bn2 = layers.BatchNormalization(name=f"{name}_bn_2")
        self.relu2 = layers.ReLU(name=f"{name}_ReLU_2")

        self.middle_blocks_layers = [
            MiddleBlock(num_filters * 4, i, name=name) for i in range(self.middle_blocks)
        ]

        if self.downsample_enabled:
            self.downsample = layers.AveragePooling1D(pool_size=2, name=f"{name}_downsample")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        for block in self.middle_blocks_layers:
            x = block(x)

        if self.downsample:
            x = self.downsample(x)

        return x

    def compute_output_shape(self, input_shape):
        channels = self.num_filters * 4
        length = input_shape[1] // 2 if self.downsample_enabled else input_shape[1]

        return (input_shape[0], length, channels)


class MiddleBlock(layers.Layer):
    def __init__(self, filters, i, name="xception", **kwargs):
        super().__init__(name=f"{name}_middle_block_{i}", **kwargs)
        self.filters = filters
        self.name = name
        self.i = i

    def build(self, input_shape):
        i = self.i
        filters = self.filters
        name = self.name

        self.relu1 = layers.ReLU(name=f"{name}_ReLU_{i * 3}")
        self.sepconv1 = layers.SeparableConv1D(
            filters, 3, padding="same", use_bias=False, name=f"{name}_conv1d_{(i * 3) + 2}"
        )
        self.bn1 = layers.BatchNormalization(name=f"{name}_bn_{(i * 3) + 2}")
        self.relu2 = layers.ReLU(name=f"{name}_ReLU_{(i * 3) + 1}")
        self.sepconv2 = layers.SeparableConv1D(
            filters, 3, padding="same", use_bias=False, name=f"{name}_conv1d_{(i * 3) + 3}"
        )
        self.bn2 = layers.BatchNormalization(name=f"{name}_bn_{(i * 3) + 3}")
        self.relu3 = layers.ReLU(name=f"{name}_ReLU_{(i * 3) + 2}")
        self.sepconv3 = layers.SeparableConv1D(
            filters, 3, padding="same", use_bias=False, name=f"{name}_conv1d_{(i * 3) + 4}"
        )
        self.bn3 = layers.BatchNormalization(name=f"{name}_bn_{(i * 3) + 4}")
        self.add = layers.Add(name=f"{name}_residual_{i}")

        if input_shape[-1] != self.filters:
            self.res_conv = layers.Conv1D(
                self.filters, 1, padding="same", use_bias=False, name=f"{name}_residual_conv_{i}"
            )
            self.res_bn = layers.BatchNormalization(name=f"{name}_residual_bn_{i}")
        else:
            self.res_conv = None

    def call(self, inputs):
        x = self.relu1(inputs)
        x = self.sepconv1(x)
        x = self.bn1(x)

        x = self.relu2(x)
        x = self.sepconv2(x)
        x = self.bn2(x)

        x = self.relu3(x)
        x = self.sepconv3(x)
        x = self.bn3(x)

        res = inputs
        if self.res_conv:
            res = self.res_conv(inputs)
            res = self.res_bn(res)

        x = self.add([x, res])
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.filters)


if __name__ == "__main__":
    xception = XceptionBlock()
    dummy_input = k.random.normal((1, 200, 10))
    output = xception(dummy_input)
    print(output.shape)
