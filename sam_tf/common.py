from keras_cv.backend import keras


@keras.utils.register_keras_serializable(package="keras_cv")
class MLPBlock(keras.layers.Layer):
    def __init__(self, embedding_dim, mlp_dim, activation="gelu", **kwargs):
        super().__init__(**kwargs)
        self.dense_layer1 = keras.layers.Dense(mlp_dim)
        self.dense_layer2 = keras.layers.Dense(embedding_dim)
        self.activation_layer = keras.layers.Activation(activation)

        self.embedding_dim = embedding_dim
        self.mlp_dim = mlp_dim
        self.activation = activation

        self.built = False

    def build(self, input_shape=None):
        self.dense_layer1.build([self.embedding_dim])
        self.dense_layer2.build([self.mlp_dim])

        self.built = True

    def call(self, x):
        return self.dense_layer2(self.activation_layer(self.dense_layer1(x)))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "mlp_dim": self.mlp_dim,
                "activation": self.activation,
            }
        )
