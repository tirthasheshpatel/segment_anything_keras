import math

from keras_cv.backend import keras
from keras_cv.backend import ops
from sam_tf.common import LayerNormalization


class RandomFrequencyPositionalEmbeddings(keras.layers.Layer):
    def __init__(self, *, num_positional_features, scale, **kwargs):
        super().__init__(**kwargs)
        self.num_positional_features = num_positional_features
        self.scale = scale
        self.positional_encoding_gaussian_matrix = self.scale * ops.random.normal(
            shape=(2, self.num_positional_features), dtype=self.dtype
        )

    def __positional_encodings(self, coords):
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * math.pi * coords
        return ops.concatenate([ops.sin(coords), ops.cos(coords)], axis=-1)

    def call(self, size):
        H, W = size
        H, W = ops.cast(H, "int64"), ops.cast(W, "int64")
        grid = ops.ones(shape=(H, W), dtype=self.dtype)
        y_embed = ops.cumsum(grid, axis=0) - 0.5
        x_embed = ops.cumsum(grid, axis=1) - 0.5
        y_embed = y_embed / ops.cast(H, "float32")
        x_embed = x_embed / ops.cast(W, "float32")
        return self.__positional_encodings(ops.stack([x_embed, y_embed], axis=-1))

    def call_with_coords(self, coords_input, image_size):
        coords_normalized = ops.stack(
            [
                coords_input[..., 0] / image_size[1],
                coords_input[..., 1] / image_size[0],
            ],
            axis=-1,
        )
        return self.__positional_encodings(coords_normalized)


class PromptEncoder(keras.models.Model):
    def __init__(
        self,
        *,
        embed_dim,
        image_embedding_size,
        input_image_size,
        mask_in_chans,
        activation=keras.activations.gelu,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        # convert the image_embedding_size to a tensor since keras core
        # expects the input type to be a symbolic/concrete tensor.
        self.image_embedding_size = ops.convert_to_tensor(
            image_embedding_size, dtype="int32"
        )
        self.input_image_size = ops.convert_to_tensor(
            input_image_size, dtype="int32"
        )
        self.positional_embedding_layer = RandomFrequencyPositionalEmbeddings(
            num_positional_features=self.embed_dim // 2, scale=1
        )

        self.foreground_point_embed = keras.layers.Embedding(1, embed_dim)
        self.background_point_embed = keras.layers.Embedding(1, embed_dim)
        self.top_left_corner_embed = keras.layers.Embedding(1, embed_dim)
        self.bottom_right_corner_embed = keras.layers.Embedding(1, embed_dim)
        self.not_a_point_embed = keras.layers.Embedding(1, embed_dim)

        self.mask_input_size = (
            4 * image_embedding_size[0],
            4 * image_embedding_size[1],
        )
        self.mask_downscaler = keras.models.Sequential(
            [
                keras.layers.Conv2D(mask_in_chans // 4, kernel_size=2, strides=2),
                LayerNormalization(),
                keras.layers.Activation(activation),
                keras.layers.Conv2D(mask_in_chans, kernel_size=2, strides=2),
                LayerNormalization(),
                keras.layers.Activation(activation),
                keras.layers.Conv2D(embed_dim, kernel_size=1),
            ]
        )
        self.mask_downscaler.build(
            [None, 4 * image_embedding_size[0], 4 * image_embedding_size[1], 1]
        )
        self.no_mask_embed = keras.layers.Embedding(1, embed_dim)

        # Build embeddings layers: I don't like this, maybe we could just have embedding matrices
        # directly.
        for layer in [
            self.foreground_point_embed,
            self.background_point_embed,
            self.top_left_corner_embed,
            self.bottom_right_corner_embed,
            self.not_a_point_embed,
            self.no_mask_embed,
        ]:
            layer.build([])

    def get_dense_pe(self):
        return self.positional_embedding_layer(self.image_embedding_size)[
            None, ...
        ]

    def __embed_points(self, points, labels, pad):
        points = points + 0.5
        if pad:
            padding_point = ops.zeros((points.shape[0], 1, 2), dtype=self.dtype)
            padding_label = -ops.ones((labels.shape[0], 1), dtype=self.dtype)
            points = ops.concatenate([points, padding_point], axis=1)
            labels = ops.concatenate([labels, padding_label], axis=1)
        point_embeddings = self.positional_embedding_layer.call_with_coords(
            points, self.input_image_size
        )
        labels = ops.broadcast_to(labels[..., None], point_embeddings.shape)
        point_embeddings = ops.where(
            labels == 0,
            point_embeddings + self.background_point_embed.weights[0],
            point_embeddings + self.foreground_point_embed.weights[0],
        )
        point_embeddings = ops.where(
            labels == -1,
            # TODO: for whatever reason, ops.broadcast_to doesn't work here, so
            #       we instead use zeros_like to broadcast to the correct shape.
            self.not_a_point_embed.weights[0] + ops.zeros_like(point_embeddings),
            point_embeddings,
        )
        return point_embeddings

    def __embed_box(self, box):
        box = box + 0.5
        coords = ops.reshape(box, (-1, 2, 2))
        corner_embedding = self.positional_embedding_layer.call_with_coords(
            coords, self.input_image_size
        )
        top_left_embedding = (
            corner_embedding[:, 0, :] + self.top_left_corner_embed.weights[0]
        )
        bottom_right_embedding = (
            corner_embedding[:, 1, :] + self.bottom_right_corner_embed.weights[0]
        )
        corner_embedding = ops.stack(
            [top_left_embedding, bottom_right_embedding], axis=1
        )
        return corner_embedding

    def __embed_mask(self, mask):
        mask_embedding = self.mask_downscaler(mask)
        return mask_embedding

    def call(self, points=None, labels=None, box=None, mask=None):
        if points is not None:
            B = points.shape[0]
        elif box is not None:
            B = box.shape[0]
        elif mask is not None:
            B = mask.shape[0]
        else:
            raise ValueError("At least one of the inputs must not be None.")
        sparse_embeddings = ops.zeros((B, 0, self.embed_dim), dtype=self.dtype)
        if points is not None:
            if labels is None:
                raise ValueError("`labels` must also be provided with `points`")
            point_embeddings = self.__embed_points(points, labels, pad=(box is None))
            sparse_embeddings = ops.concatenate([sparse_embeddings, point_embeddings], axis=1)
        if box is not None:
            box_embeddings = self.__embed_box(box)
            sparse_embeddings = ops.concatenate([sparse_embeddings, box_embeddings], axis=1)
        if mask is not None:
            dense_embeddings = self.__embed_mask(mask)
        else:
            dense_embeddings = ops.broadcast_to(
                ops.reshape(
                    self.no_mask_embed.weights[0], (1, 1, 1, self.embed_dim)
                ),
                shape=(
                    B,
                    self.image_embedding_size[0],
                    self.image_embedding_size[1],
                    self.embed_dim,
                ),
            )
        return sparse_embeddings, dense_embeddings
