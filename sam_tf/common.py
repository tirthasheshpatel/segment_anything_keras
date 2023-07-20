import tensorflow as tf
import keras
from keras import layers


class MLPBlock(layers.Layer):
    def __init__(self, embedding_dim, mlp_dim, activation=keras.activations.gelu):
        super().__init__()
        self.dense_layer1 = layers.Dense(mlp_dim)
        self.dense_layer2 = layers.Dense(embedding_dim)
        self.activation = layers.Activation(activation)
        
    def call(self, x):
        return self.dense_layer2(self.activation(self.dense_layer1(x)))


class LayerNormalization(layers.Layer):
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
        u = tf.reduce_mean(x, axis=-1, keepdims=True)
        s = tf.reduce_mean(tf.square(x - u), axis=-1, keepdims=True)
        x = (x - u) / tf.sqrt(s + self.epsilon)
        x = self.weight * x + self.bias
        return x
