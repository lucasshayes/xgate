from keras import layers, Model
from keras_tuner import HyperParameters
import keras as k
from modules.xception import xception_block
from modules.attention.cbam import cbam_1d_block
from modules.attention.temporal_eca import temporal_eca_block
from modules.attention.feat_attention import feat_attention
from modules.rolling_extraction import rolling_extraction
from utils.set_seed import set_seeds
from config import Config
def build_fused_model(hp: HyperParameters):
    """
    Build and compile the FusedModel using Functional API with the passed hyperparameters.
    """
    config = Config()
    k.utils.set_random_seed(config.random_seed)

    inputs = k.Input(shape=(200, 7))
    x = inputs

    x = layers.Normalization(name="input_normalization")(x)
    x = rolling_extraction(x, 100)
    x = feat_attention(x)
    
    # Xception or FC branch
    if hp.get("xception_bool"):
        x = xception_block(
            x,
            num_filters=hp.get("num_filters"),
            k_size=hp.get("kernel_size"),
            middle_blocks=hp.get("middle_blocks"),
            downsample=hp.get("downsample"),
        )
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
        x = cbam_1d_block(x, r_ratio=hp.get("r_ratio"))
    
    # First GRU layer
    x = layers.GRU(
        units=hp.get("gru_units"),
        return_sequences=True,
        name="gru_layer_1",
        recurrent_dropout=hp.get("gru_dropout"),
        dropout=hp.get("gru_dropout")
    )(x)

    # Second GRU layer
    x = layers.GRU(
        units=hp.get("gru_units"),
        return_sequences=True,
        name="gru_layer_2",
        recurrent_dropout=hp.get("gru_dropout"),
        dropout=hp.get("gru_dropout")
    )(x)
    
    # Temporal ECA attention
    if hp.get("eca_bool"):
        x = temporal_eca_block(x, gamma=hp.get("gamma"), b=hp.get("beta"), name="temporal_eca")

    # Fully connected dense layer
    x = layers.Dense(
        units=hp.get("fc_units"),
        activation="relu",
        kernel_initializer="he_normal",
        kernel_constraint=k.constraints.max_norm(3),
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
        kernel_constraint=k.constraints.max_norm(3)
    )(x)

    model = k.Model(inputs=inputs, outputs=outputs, name="fused_model")

    model.compile(
        optimizer=k.optimizers.Adam(learning_rate=hp.get("learning_rate")),
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
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

    dummy_input = k.random.normal((1, 50, 7))
    _ = model(dummy_input)
    model.summary()
