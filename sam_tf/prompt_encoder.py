import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models


class RandomFrequencyPositionalEmbeddings(layers.Layer):
    def __init__(self, *, num_positional_features, scale, **kwargs):
        super().__init__(**kwargs)
        self.num_positional_features = num_positional_features
        self.scale = scale
        self.positional_encoding_gaussian_matrix = self.scale * tf.random.normal(
            shape=(2, self.num_positional_features), dtype=self.dtype
        )

    def __positional_encodings(self, coords):
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * math.pi * coords
        return tf.concat([tf.math.sin(coords), tf.math.cos(coords)], axis=-1)

    def call(self, size):
        H, W = size
        H, W = tf.cast(H, tf.int64), tf.cast(W, tf.int64)
        grid = tf.ones(shape=(H, W), dtype=self.dtype)
        y_embed = tf.cumsum(grid, axis=0) - 0.5
        x_embed = tf.cumsum(grid, axis=1) - 0.5
        y_embed = y_embed / tf.cast(H, tf.float32)
        x_embed = x_embed / tf.cast(W, tf.float32)
        return self.__positional_encodings(tf.stack([x_embed, y_embed], axis=-1))

    def call_with_coords(self, coords_input, image_size):
        coords_normalized = tf.stack(
            [
                coords_input[..., 0] / image_size[1],
                coords_input[..., 1] / image_size[0],
            ],
            axis=-1,
        )
        return self.__positional_encodings(coords_normalized)


class PromptEncoder(models.Model):
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
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        self.positional_embedding_layer = RandomFrequencyPositionalEmbeddings(
            num_positional_features=self.embed_dim // 2, scale=1
        )

        self.foreground_point_embed = layers.Embedding(1, embed_dim)
        self.background_point_embed = layers.Embedding(1, embed_dim)
        self.top_left_corner_embed = layers.Embedding(1, embed_dim)
        self.bottom_right_corner_embed = layers.Embedding(1, embed_dim)
        self.not_a_point_embed = layers.Embedding(1, embed_dim)

        self.mask_input_size = (
            4 * image_embedding_size[0],
            4 * image_embedding_size[1],
        )
        self.mask_downscaler = models.Sequential(
            [
                layers.Conv2D(mask_in_chans // 4, kernel_size=2, strides=2),
                layers.LayerNormalization(),
                layers.Activation(activation),
                layers.Conv2D(mask_in_chans, kernel_size=2, strides=2),
                layers.LayerNormalization(),
                layers.Activation(activation),
                layers.Conv2D(embed_dim, kernel_size=1),
            ]
        )
        self.mask_downscaler.build(
            [None, 4 * image_embedding_size[0], 4 * image_embedding_size[1], 1]
        )
        self.no_mask_embed = layers.Embedding(1, embed_dim)

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
            tf.newaxis, ...
        ]

    def __embed_points(self, points, labels, pad):
        points = points + 0.5
        if pad:
            padding_point = tf.zeros((points.shape[0], 1, 2), dtype=self.dtype)
            padding_label = -tf.ones((labels.shape[0], 1), dtype=self.dtype)
            points = tf.concat([points, padding_point], axis=1)
            labels = tf.concat([labels, padding_label], axis=1)
        point_embeddings = self.positional_embedding_layer.call_with_coords(
            points, self.input_image_size
        )
        labels = tf.broadcast_to(labels[..., tf.newaxis], point_embeddings.shape)
        point_embeddings = tf.where(
            labels == 0,
            point_embeddings + self.background_point_embed.weights[0],
            point_embeddings + self.foreground_point_embed.weights[0],
        )
        point_embeddings = tf.where(
            labels == -1,
            tf.broadcast_to(self.not_a_point_embed.weights[0], point_embeddings.shape),
            point_embeddings,
        )
        return point_embeddings

    def __embed_box(self, box):
        box = box + 0.5
        coords = tf.reshape(box, shape=(-1, 2, 2))
        corner_embedding = self.positional_embedding_layer.call_with_coords(
            coords, self.input_image_size
        )
        top_left_embedding = (
            corner_embedding[:, 0, :] + self.top_left_corner_embed.weights[0]
        )
        bottom_right_embedding = (
            corner_embedding[:, 1, :] + self.bottom_right_corner_embed.weights[0]
        )
        corner_embedding = corner_embedding + tf.stack(
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
        sparse_embeddings = tf.zeros((B, 0, self.embed_dim), dtype=self.dtype)
        if points is not None:
            if labels is None:
                raise ValueError("`labels` must also be provided with `points`")
            point_embeddings = self.__embed_points(points, labels, pad=(box is None))
            sparse_embeddings = tf.concat([sparse_embeddings, point_embeddings], axis=1)
        if box is not None:
            box_embeddings = self.__embed_box(box)
            sparse_embeddings = tf.concat([sparse_embeddings, box_embeddings], axis=1)
        if mask is not None:
            dense_embeddings = self.__embed_mask(mask)
        else:
            dense_embeddings = tf.broadcast_to(
                tf.reshape(
                    self.no_mask_embed.weights[0], shape=(1, 1, 1, self.embed_dim)
                ),
                shape=(
                    B,
                    self.image_embedding_size[0],
                    self.image_embedding_size[1],
                    self.embed_dim,
                ),
            )
        return sparse_embeddings, dense_embeddings
