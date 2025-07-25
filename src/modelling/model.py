from keras import layers, Model
from keras_tuner import HyperParameters
import keras.api as k
from modules.xception import XceptionBlock
from modules.attention.cbam import CBAM1D
from modules.attention.temporal_eca import TemporalECA


def build_fused_model(hp: HyperParameters):
    """
    Build and compile the FusedModel using Functional API with the passed hyperparameters.
    """

    inputs = k.Input(shape=(200, 7)) 
    x = inputs

    x = layers.Normalization(name="input_normalization")(x)
    
    # Xception or FC branch
    if hp.get("xception_bool"):
        x = XceptionBlock(
            hp.get("num_filters"),
            hp.get("kernel_size"),
            hp.get("middle_blocks"),
            hp.get("downsample"),
        )(x)
        if hp.get("xception_dropout") > 0:
            x = layers.Dropout(hp.get("xception_dropout"), name="xception_dropout_layer")(x)
    else:
        x = layers.Dense(
            units=hp.get("fc_units"),
            activation="relu",
            name="fc_layer",
        )(x)

    # CBAM attention
    if hp.get("cbam_bool"):
        x = CBAM1D(r_ratio=hp.get("r_ratio"))(x)
    
    # First GRU layer
    x = layers.GRU(
        units=hp.get("gru_units"),
        return_sequences=True,
        name="gru_layer_1",
        recurrent_dropout=hp.get("gru_dropout"),
        dropout=hp.get("gru_dropout"),
        kernel_constraint=k.constraints.max_norm(2),
        recurrent_constraint=k.constraints.max_norm(2),
        kernel_initializer="orthogonal",
        recurrent_initializer="orthogonal",
    )(x)

    # Second GRU layer
    x = layers.GRU(
        units=hp.get("gru_units"),
        return_sequences=True,
        name="gru_layer_2",
        recurrent_dropout=hp.get("gru_dropout"),
        dropout=hp.get("gru_dropout"),
        kernel_constraint=k.constraints.max_norm(2),
        recurrent_constraint=k.constraints.max_norm(2),
        kernel_initializer="orthogonal",
        recurrent_initializer="orthogonal",
    )(x)
    
    # Temporal ECA attention
    if hp.get("eca_bool"):
        x = TemporalECA(hp.get("gamma"), hp.get("beta"))(x)

    # Fully connected dense layer
    x = layers.Dense(
        units=hp.get("fc_units"),
        activation="relu",
        kernel_initializer="he_normal",
        kernel_constraint=k.constraints.max_norm(2),
        name="fc_layer_2",
    )(x)
    if hp.get("fc_dropout") > 0:
        x = layers.Dropout(hp.get("fc_dropout"), name="fc_dropout_layer")(x)

    # Output layer
    outputs = layers.Dense(
        units=4,
        activation="softmax", 
        name="output_layer",
        kernel_initializer=k.initializers.RandomNormal(stddev=0.1),
        kernel_constraint=k.constraints.max_norm(2.0)
    )(x)

    model = k.Model(inputs=inputs, outputs=outputs, name="fused_model")

    model.compile(
        optimizer=k.optimizers.Adam(learning_rate=hp.get("learning_rate"), clipnorm=0.5),
        loss=k.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["sparse_categorical_accuracy"],
    )

    return model


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
    hp.Fixed("xception_dropout", 0.2)
    hp.Fixed("gru_dropout", 0.1)
    hp.Fixed("fc_dropout", 0.1)
    hp.Fixed("learning_rate", 1e-3)

    model = build_fused_model(hp)

    dummy_input = k.random.normal((1, 200, 7))
    _ = model(dummy_input)
    model.summary()
