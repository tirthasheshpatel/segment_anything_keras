import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
import keras_cv


def get_rel_pos(query_size, key_size, rel_pos):
    max_rel_dist = 2 * max(query_size, key_size) - 1
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = tf.image.resize(
            images=tf.reshape(
                rel_pos, shape=(1, rel_pos.shape[0], rel_pos.shape[1], 1)
            ),
            size=(max_rel_dist, rel_pos.shape[1]),
            method="bilinear",
        )
        rel_pos_resized = tf.squeeze(rel_pos_resized, axis=(0, -1))
    else:
        rel_pos_resized = rel_pos
    query_coordinates = tf.range(query_size, dtype=tf.float32)[:, tf.newaxis] * max(
        key_size / query_size, 1.0
    )
    key_coordinates = tf.range(key_size, dtype=tf.float32)[tf.newaxis, :] * max(
        query_size / key_size, 1.0
    )
    relative_coordinates = (query_coordinates - key_coordinates) + (key_size - 1) * max(
        query_size / key_size, 1.0
    )
    relative_coordinates = tf.cast(relative_coordinates, dtype=tf.int64)
    return tf.gather(rel_pos_resized, relative_coordinates)


def add_decomposed_rel_pos(
    attention_map, queries, rel_pos_h, rel_pos_w, query_size, key_size
):
    query_height, query_width = query_size
    key_height, key_width = key_size
    rel_heights = get_rel_pos(query_height, key_height, rel_pos_h)
    rel_widths = get_rel_pos(query_width, key_width, rel_pos_w)

    B, _, C = queries.shape
    rel_queries = tf.reshape(queries, shape=(B, query_height, query_width, C))
    rel_heights = tf.einsum("bhwc,hkc->bhwk", rel_queries, rel_heights)
    rel_widths = tf.einsum("bhwc,wkc->bhwk", rel_queries, rel_widths)

    attention_map = tf.reshape(
        attention_map, shape=(B, query_height, query_width, key_height, key_width)
    )
    attention_map = attention_map + rel_heights[..., :, tf.newaxis]
    attention_map = attention_map + rel_widths[..., tf.newaxis, :]
    attention_map = tf.reshape(
        attention_map, shape=(B, query_height * query_width, key_height * key_width)
    )
    return attention_map


class MultiHeadAttentionWithRelativePE(layers.Layer):
    def __init__(self, *, num_heads, key_dim, use_bias, input_size, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.scale = self.key_dim**0.5

        self.queries = layers.Dense(key_dim * self.num_heads, use_bias=use_bias)
        self.keys = layers.Dense(key_dim * self.num_heads, use_bias=use_bias)
        self.values = layers.Dense(key_dim * self.num_heads, use_bias=use_bias)
        self.projection = layers.Dense(key_dim * self.num_heads)

        self.input_size = input_size
        self.rel_pos_h = self.add_weight(
            name="rel_pos_h",
            shape=(2 * self.input_size[0] - 1, self.key_dim),
            initializer="zeros",
            trainable=True,
        )
        self.rel_pos_w = self.add_weight(
            name="rel_pos_w",
            shape=(2 * self.input_size[0] - 1, self.key_dim),
            initializer="zeros",
            trainable=True,
        )

    def call(self, x):
        B, H, W, C = x.shape
        queries = tf.transpose(
            tf.reshape(self.queries(x), shape=(B, H * W, self.num_heads, self.key_dim)),
            perm=(0, 2, 1, 3),
        )
        queries = tf.reshape(queries, shape=(B * self.num_heads, H * W, self.key_dim))
        keys = tf.transpose(
            tf.reshape(self.keys(x), shape=(B, H * W, self.num_heads, self.key_dim)),
            perm=(0, 2, 1, 3),
        )
        keys = tf.reshape(keys, shape=(B * self.num_heads, H * W, self.key_dim))
        values = tf.transpose(
            tf.reshape(self.values(x), shape=(B, H * W, self.num_heads, self.key_dim)),
            perm=(0, 2, 1, 3),
        )
        values = tf.reshape(values, shape=(B * self.num_heads, H * W, self.key_dim))
        attention_map = (queries * self.scale) @ tf.transpose(keys, perm=(0, 2, 1))
        attention_map = add_decomposed_rel_pos(
            attention_map, queries, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
        )
        attention_map = tf.math.softmax(attention_map, axis=-1)
        x = tf.reshape(
            attention_map @ values, shape=(B, self.num_heads, H, W, self.key_dim)
        )
        x = tf.transpose(x, perm=(0, 2, 3, 1, 4))
        x = tf.reshape(x, shape=(B, H, W, C))
        x = self.projection(x)

        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    pad_height = (window_size - H % window_size) % window_size
    pad_width = (window_size - W % window_size) % window_size
    if pad_height > 0 or pad_width > 0:
        x = tf.pad(x, ((0, 0), (0, pad_height), (0, pad_width), (0, 0)))
    H_padded, W_padded = H + pad_height, W + pad_width
    x = tf.reshape(
        x,
        shape=(
            B,
            H_padded // window_size,
            window_size,
            W_padded // window_size,
            window_size,
            C,
        ),
    )
    windows = tf.reshape(
        tf.transpose(x, perm=(0, 1, 3, 2, 4, 5)),
        shape=(-1, window_size, window_size, C),
    )
    return windows, (H_padded, W_padded)


def window_unpartition(windows, window_size, HW_padded, HW):
    H_padded, W_padded = HW_padded
    H, W = HW
    B = windows.shape[0] // ((H // window_size) * (W // window_size))
    x = tf.reshape(
        windows,
        shape=(
            B,
            H_padded // window_size,
            W_padded // window_size,
            window_size,
            window_size,
            -1,
        ),
    )
    x = tf.reshape(
        tf.transpose(x, perm=(0, 1, 3, 2, 4, 5)), shape=(B, H_padded, W_padded, -1)
    )
    return x[:, :H, :W, :]


class WindowedTransformerEncoder(layers.Layer):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        *,
        project_dim,
        mlp_dim,
        num_heads,
        use_bias,
        use_rel_pos,
        window_size,
        input_size,
        activation=keras.activations.gelu,
        layer_norm_epsilon=1e-6,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.project_dim = project_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.layer_norm_epsilon = layer_norm_epsilon
        self.mlp_units = [mlp_dim, project_dim]
        self.window_size = window_size
        self.use_rel_pos = use_rel_pos

        self.layer_norm1 = layers.LayerNormalization(epsilon=self.layer_norm_epsilon)
        self.layer_norm2 = layers.LayerNormalization(epsilon=self.layer_norm_epsilon)
        if use_rel_pos:
            self.attention = MultiHeadAttentionWithRelativePE(
                num_heads=self.num_heads,
                key_dim=self.project_dim // self.num_heads,
                use_bias=use_bias,
                input_size=input_size,
            )
        else:
            self.attention = layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.project_dim // self.num_heads,
                use_bias=use_bias,
            )
        self.dense1 = layers.Dense(self.mlp_units[0])
        self.activation1 = activation
        self.dense2 = layers.Dense(self.mlp_units[1])

    def call(self, x):
        shortcut = x
        x = self.layer_norm1(x)

        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            # Window Partition
            x, HW_padded = window_partition(x, self.window_size)
            if self.use_rel_pos:
                x = self.attention(x)
            else:
                x = self.attention(x, x)
            # Reverse Window Partition
            x = window_unpartition(x, self.window_size, HW_padded, (H, W))
        else:
            if self.use_rel_pos:
                x = self.attention(x)
            else:
                x = self.attention(x, x)

        x = shortcut + x
        x = x + self.dense2(self.activation1(self.dense1(self.layer_norm2(x))))

        return x


class ImageEncoder(models.Model):
    def __init__(
        self,
        *,
        img_size=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=1280,
        depth=32,
        mlp_dim=1280 * 4,
        num_heads=16,
        out_chans=256,
        use_bias=True,
        use_rel_pos=False,
        window_size=0,
        global_attention_indices=[7, 15, 23, 31],
        layer_norm_epsilon=1e-6,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.out_chans = out_chans
        self.use_bias = use_bias
        self.use_rel_pos = use_rel_pos
        self.window_size = window_size
        self.global_attention_indices = global_attention_indices
        self.layer_norm_epsilon = layer_norm_epsilon

        self.patch_embed = keras_cv.layers.PatchingAndEmbedding(
            project_dim=embed_dim, patch_size=patch_size
        )
        self.transformer_blocks = []
        for i in range(depth):
            block = WindowedTransformerEncoder(
                project_dim=embed_dim,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                use_bias=use_bias,
                use_rel_pos=use_rel_pos,
                window_size=window_size if i in global_attention_indices else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.transformer_blocks.append(block)
        self.transformer_blocks = models.Sequential(self.transformer_blocks)
        self.bottleneck = models.Sequential(
            [
                layers.Conv2D(filters=out_chans, kernel_size=1, use_bias=False),
                layers.LayerNormalization(epsilon=layer_norm_epsilon),
                layers.Conv2D(
                    filters=out_chans, kernel_size=3, padding="same", use_bias=False
                ),
                layers.LayerNormalization(epsilon=layer_norm_epsilon),
            ]
        )

    def call(self, x):
        B, _, _, _ = x.shape
        x = self.patch_embed(x)
        # remove the class token and reshape to an image for downscaling
        x = tf.reshape(
            x[:, 1:, :],
            shape=(
                B,
                self.img_size // self.patch_size,
                self.img_size // self.patch_size,
                self.embed_dim,
            ),
        )
        x = self.transformer_blocks(x)
        return self.bottleneck(x)
