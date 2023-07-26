from keras_cv.backend import keras
from keras_cv.backend import ops


class MLPBlock(keras.layers.Layer):
    def __init__(self, embedding_dim, mlp_dim, activation=keras.activations.gelu):
        super().__init__()
        self.dense_layer1 = keras.layers.Dense(mlp_dim)
        self.dense_layer2 = keras.layers.Dense(embedding_dim)
        self.activation = keras.layers.Activation(activation)

    def call(self, x):
        return self.dense_layer2(self.activation(self.dense_layer1(x)))


class LayerNormalization(keras.layers.Layer):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="weight",
            shape=(input_shape[-1],),
            initializer="ones",
            trainable=True
        )
        self.bias = self.add_weight(
            name="weight",
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True
        )

    def call(self, x):
        u = ops.mean(x, axis=-1, keepdims=True)
        s = ops.mean(ops.square(x - u), axis=-1, keepdims=True)
        x = (x - u) / ops.sqrt(s + self.epsilon)
        x = self.weight * x + self.bias
        return x
    
    def compute_output_shape(self, input_shape):
        return input_shape
