from keras import layers, Model
from keras_tuner import HyperParameters
import keras.api as k
from modules.xception import XceptionBlock
from modules.attention.cbam import CBAM1D
from modules.attention.temporal_eca import TemporalECA


class FusedModel(Model):
    def __init__(self, hp: HyperParameters, name="fused_model", **kwargs):
        """
        Initializes the FusedModel with a base model and an attention module.

        Args:
            xception (bool): Whether to include the Xception block.
            cbam (bool): Whether to include the CBAM attention module.
            eca (bool): Whether to include the ECA attention module.
            name (str): Name of the model.
        """

        super(FusedModel, self).__init__(name=name, **kwargs)
        self.hp = hp
        self.xception_bool = hp.get("xception_bool")
        self.cbam_bool = hp.get("cbam_bool")
        self.eca_bool = hp.get("eca_bool")
        self.xception_dropout_rate = hp.get("xception_dropout")
        self.gru_dropout_rate = hp.get("gru_dropout")
        self.fc_dropout_rate = hp.get("fc_dropout")

        if self.xception_bool:
            self.xception = XceptionBlock(
                hp.get("num_filters"),
                hp.get("kernel_size"),
                hp.get("middle_blocks"),
                hp.get("downsample"),
            )

        self.cbam = CBAM1D(r_ratio=hp.get("r_ratio"))
        self.eca = TemporalECA(hp.get("gamma"), hp.get("beta"))

        self.fc = layers.Dense(
            units=hp.get("fc_units"),
            activation="relu",
            name="fc_layer",
        )
        self.gru_1 = layers.GRU(
            units=hp.get("gru_units"),
            return_sequences=True,
            name="gru_layer",
        )
        self.gru_2 = layers.GRU(
            units=hp.get("gru_units"),
            return_sequences=True,
            name="gru_layer",
        )
        self.bn = layers.BatchNormalization(
            name="batch_normalization",
        )
        self.out = layers.Dense(
            units=3,
            name="output_layer",
        )
        self.xception_dropout = (
            layers.Dropout(self.xception_dropout_rate, name="xception_dropout_layer")
            if self.xception_dropout_rate > 0
            else None
        )
        self.gru_dropout = (
            layers.Dropout(self.gru_dropout_rate, name="gru_dropout_layer")
            if self.gru_dropout_rate > 0
            else None
        )
        self.fc_dropout = (
            layers.Dropout(self.fc_dropout_rate, name="fc_dropout_layer")
            if self.fc_dropout_rate > 0
            else None
        )

    def build(self, input_shape):
        super().build(input_shape)

    @classmethod
    def build_and_compile(cls, hp: HyperParameters):
        """Builds and compiles the FusedModel with hyperparameters.

        Args:
            hp (HyperParameters): Hyperparameters for the model.

        Returns:
            FusedModel: An instance of the FusedModel.
        """

        model = cls(hp=hp)

        model.compile(
            optimizer=k.optimizers.Adam(learning_rate=hp.get("learning_rate")),
            loss=k.losses.MeanSquaredError(),
            metrics=[k.metrics.MeanAbsoluteError(), k.metrics.RootMeanSquaredError()],
        )

        return model

    def call(self, inputs, training=False):
        """
        Forward pass through the model.

        Args:
            inputs: Input data for the model.

        Returns:
            Output after applying the base model and attention module.
        """
        x = inputs

        if self.xception_bool:
            x = self.xception(x, training=training)
        else:
            x = self.fc(x)

        if self.xception_dropout and training:
            x = self.xception_dropout(x, training=training)

        if self.cbam_bool:
            x = self.cbam(x)

        x = self.gru_1(x)

        if self.gru_dropout and training:
            x = self.gru_dropout(x, training=training)

        x = self.gru_2(x)

        if self.gru_dropout and training:
            x = self.gru_dropout(x, training=training)
            
        if self.eca_bool:
            x = self.eca(x)

        x = self.bn(x, training=training)
        x = self.fc(x)
        
        if self.fc_dropout and training:
            x = self.fc_dropout(x, training=training)
            
        return self.out(x)


if __name__ == "__main__":
    hp = HyperParameters()

    # Xception Params
    hp.Fixed("xception_bool", True)
    hp.Fixed("num_filters", 32)
    hp.Fixed("kernel_size", 3)
    hp.Fixed("middle_blocks", 2)
    hp.Fixed("downsample", False)

    # CBAM Params
    hp.Fixed("cbam_bool", True)
    hp.Fixed("r_ratio", 8)

    # Temporal ECA params
    hp.Fixed("eca_bool", True)
    hp.Fixed("gamma", 2)
    hp.Fixed("beta", 1)

    # Fused Params
    hp.Fixed("gru_units", 64)
    hp.Fixed("fc_units", 64)
    model = FusedModel(hp)
    dummy_input = k.random.normal((1, 200, 11))
    _ = model(dummy_input)
    model.summary()
