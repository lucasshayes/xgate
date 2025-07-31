import keras 
from keras import layers
import tensorflow as tf

@keras.saving.register_keras_serializable()
class ReduceMean1D(layers.Layer):
    def __init__(self, axis=1, keepdims=True, name="reduce_mean_1d", **kwargs):
        super(ReduceMean1D, self).__init__(name=name, **kwargs)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=self.axis, keepdims=self.keepdims)

    def get_config(self):
        config = super().get_config()
        config.update({
            "axis": self.axis,
            "keepdims": self.keepdims
        })
        return config

@keras.saving.register_keras_serializable()
class ReduceMax1D(layers.Layer):
    def __init__(self, axis=1, keepdims=True, name="reduce_max_1d", **kwargs):
        super(ReduceMax1D, self).__init__(name=name, **kwargs)
        self.axis = axis
        self.keepdims = keepdims
    
    def call(self, inputs):
        return tf.reduce_max(inputs, axis=self.axis, keepdims=self.keepdims)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "axis": self.axis,
            "keepdims": self.keepdims
        })
        return config