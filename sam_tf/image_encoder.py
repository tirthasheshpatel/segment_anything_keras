from keras_cv.backend import keras
from keras_cv.backend import ops

from sam_tf.common import LayerNormalization, MLPBlock


def get_rel_pos(query_size, key_size, rel_pos):
    max_rel_dist = 2 * max(query_size, key_size) - 1
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = ops.image.resize(
            images=ops.reshape(
                rel_pos, (1, rel_pos.shape[0], rel_pos.shape[1], 1)
            ),
            size=(max_rel_dist, rel_pos.shape[1]),
            method="bilinear",
        )
        rel_pos_resized = ops.squeeze(rel_pos_resized, axis=(0, -1))
    else:
        rel_pos_resized = rel_pos
    query_coordinates = ops.arange(query_size, dtype="float32")[:, None] * max(
        key_size / query_size, 1.0
    )
    key_coordinates = ops.arange(key_size, dtype="float32")[None, :] * max(
        query_size / key_size, 1.0
    )
    relative_coordinates = (query_coordinates - key_coordinates) + (key_size - 1) * max(
        query_size / key_size, 1.0
    )
    relative_coordinates = ops.cast(relative_coordinates, dtype="int64")
    return ops.take(rel_pos_resized, relative_coordinates, 0)


def add_decomposed_rel_pos(
    attention_map, queries, rel_pos_h, rel_pos_w, query_size, key_size
):
    query_height, query_width = query_size
    key_height, key_width = key_size
    rel_heights = get_rel_pos(query_height, key_height, rel_pos_h)
    rel_widths = get_rel_pos(query_width, key_width, rel_pos_w)

    B, _, C = queries.shape
    rel_queries = ops.reshape(queries, (B, query_height, query_width, C))
    rel_heights = ops.einsum("bhwc,hkc->bhwk", rel_queries, rel_heights)
    rel_widths = ops.einsum("bhwc,wkc->bhwk", rel_queries, rel_widths)

    attention_map = ops.reshape(
        attention_map, (B, query_height, query_width, key_height, key_width)
    )
    attention_map = attention_map + rel_heights[..., :, None]
    attention_map = attention_map + rel_widths[..., None, :]
    attention_map = ops.reshape(
        attention_map, (B, query_height * query_width, key_height * key_width)
    )
    return attention_map


@keras.saving.register_keras_serializable(package="keras_cv")
class MultiHeadAttentionWithRelativePE(keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        key_dim,
        use_bias=True,
        use_rel_pos=False,
        input_size=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.scale = self.key_dim**-0.5
        self.use_bias = use_bias

        self.qkv = keras.layers.Dense(key_dim * self.num_heads * 3, use_bias=self.use_bias)
        self.projection = keras.layers.Dense(key_dim * self.num_heads)

        self.input_size = input_size
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            if input_size is None:
                raise ValueError(
                    "Input size must be provided if using relative positional encoding."
                )
            self.rel_pos_h = self.add_weight(
                name="rel_pos_h",
                shape=(2 * self.input_size[0] - 1, self.key_dim),
                initializer="zeros",
                trainable=True,
            )
            self.rel_pos_w = self.add_weight(
                name="rel_pos_w",
                shape=(2 * self.input_size[1] - 1, self.key_dim),
                initializer="zeros",
                trainable=True,
            )

    def call(self, x):
        B, H, W, C = x.shape
        qkv = ops.transpose(
            ops.reshape(self.qkv(x), (B, H * W, 3, self.num_heads, self.key_dim)),
            axes=(2, 0, 3, 1, 4),
        )
        qkv = ops.reshape(qkv, (3, B * self.num_heads, H * W, self.key_dim))
        # TODO: remove this once unstack is added in keras core
        if keras.backend.backend() == "tensorflow":
            import tensorflow as tf
            queries, keys, values = tf.unstack(qkv, axis=0)
            del tf
        elif keras.backend.backend() == "torch":
            queries, keys, values = qkv.unbind(0)
        elif keras.backend.backend() == "jax":
            import jax
            queries, keys, values = [
                jax.lax.index_in_dim(qkv, i, 0, keepdims=False)
                for i in range(qkv.shape[0])
            ]
            del jax
        attention_map = (queries * self.scale) @ ops.transpose(keys, axes=(0, 2, 1))

        if self.use_rel_pos:
            attention_map = add_decomposed_rel_pos(
                attention_map, queries, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )
        attention_map = ops.softmax(attention_map, axis=-1)
        x = ops.reshape(
            attention_map @ values, (B, self.num_heads, H, W, self.key_dim)
        )
        x = ops.transpose(x, axes=(0, 2, 3, 1, 4))
        x = ops.reshape(x, (B, H, W, C))
        x = self.projection(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "use_bias": self.use_bias,
            "use_rel_pos": self.use_rel_pos,
            "input_size": self.input_size,
        })
        return config


def window_partition(x, window_size):
    B, H, W, C = x.shape
    pad_height = (window_size - H % window_size) % window_size
    pad_width = (window_size - W % window_size) % window_size
    if pad_height > 0 or pad_width > 0:
        x = ops.pad(x, ((0, 0), (0, pad_height), (0, pad_width), (0, 0)))
    H_padded, W_padded = H + pad_height, W + pad_width
    x = ops.reshape(
        x,
        (
            B,
            H_padded // window_size,
            window_size,
            W_padded // window_size,
            window_size,
            C,
        ),
    )
    windows = ops.reshape(
        ops.transpose(x, axes=(0, 1, 3, 2, 4, 5)),
        (-1, window_size, window_size, C),
    )
    return windows, (H_padded, W_padded)


def window_unpartition(windows, window_size, HW_padded, HW):
    H_padded, W_padded = HW_padded
    H, W = HW
    B = windows.shape[0] // ((H_padded // window_size) * (W_padded // window_size))
    x = ops.reshape(
        windows,
        (
            B,
            H_padded // window_size,
            W_padded // window_size,
            window_size,
            window_size,
            -1,
        ),
    )
    x = ops.reshape(
        ops.transpose(x, axes=(0, 1, 3, 2, 4, 5)), (B, H_padded, W_padded, -1)
    )
    return x[:, :H, :W, :]


@keras.utils.register_keras_serializable(package="keras_cv")
class WindowedTransformerEncoder(keras.layers.Layer):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        project_dim,
        mlp_dim,
        num_heads,
        use_bias=True,
        use_rel_pos=False,
        window_size=0,
        input_size=None,
        activation=keras.activations.gelu,
        layer_norm_epsilon=1e-6,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.project_dim = project_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.use_bias = use_bias
        self.input_size = input_size
        self.activation = activation
        self.layer_norm_epsilon = layer_norm_epsilon
        self.window_size = window_size
        self.use_rel_pos = use_rel_pos

        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=self.layer_norm_epsilon)
        self.layer_norm2 = keras.layers.LayerNormalization(epsilon=self.layer_norm_epsilon)
        self.attention = MultiHeadAttentionWithRelativePE(
            num_heads=self.num_heads,
            key_dim=self.project_dim // self.num_heads,
            use_bias=use_bias,
            use_rel_pos=use_rel_pos,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.mlp_block = MLPBlock(project_dim, mlp_dim, activation)

    def call(self, x):
        shortcut = x
        x = self.layer_norm1(x)
        # Window Partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]

            x, HW_padded = window_partition(x, self.window_size)

        x = self.attention(x)
        # Reverse Window Partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, HW_padded, (H, W))

        x = shortcut + x
        x = x + self.mlp_block(self.layer_norm2(x))

        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "project_dim": self.project_dim,
            "mlp_dim": self.mlp_dim,
            "num_heads": self.num_heads,
            "use_bias": self.use_bias,
            "use_rel_pos": self.use_rel_pos,
            "window_size": self.window_size,
            "input_size": self.input_size,
            "activation": self.activation,
            "layer_norm_epsilon": self.layer_norm_epsilon,
        })
        return config


@keras.utils.register_keras_serializable(package="keras_cv")
class PatchingAndEmbedding(keras.layers.Layer):
    def __init__(self, kernel_size=(16, 16), strides=(16, 16), embed_dim=768, **kwargs):
        super().__init__(**kwargs)

        self.projection = keras.layers.Conv2D(
            embed_dim, kernel_size=kernel_size, strides=strides
        )
        
        self.kernel_size = kernel_size
        self.strides = strides
        self.embed_dim = embed_dim

    def call(self, x):
        x = self.projection(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "embed_dim": self.embed_dim
        })
        return config


@keras.utils.register_keras_serializable(package="keras_cv")
class ImageEncoder(keras.layers.Layer):
    def __init__(
        self,
        img_size=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=1280,
        depth=32,
        mlp_dim=1280 * 4,
        num_heads=16,
        out_chans=256,
        use_bias=True,
        use_abs_pos=True,
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
        self.use_abs_pos = use_abs_pos
        self.window_size = window_size
        self.global_attention_indices = global_attention_indices
        self.layer_norm_epsilon = layer_norm_epsilon

        self.patch_embed = PatchingAndEmbedding(
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            embed_dim=embed_dim,
        )
        if self.use_abs_pos:
            self.pos_embed = self.add_weight(
                name="pos_embed",
                shape=(
                    1,
                    self.img_size // self.patch_size,
                    self.img_size // self.patch_size,
                    self.embed_dim,
                ),
                initializer="zeros",
                trainable=True,
            )
        else:
            self.pos_embed = None
        self.transformer_blocks = []
        for i in range(depth):
            block = WindowedTransformerEncoder(
                project_dim=embed_dim,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                use_bias=use_bias,
                use_rel_pos=use_rel_pos,
                window_size=window_size if i not in global_attention_indices else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.transformer_blocks.append(block)
        self.transformer_blocks = keras.models.Sequential(self.transformer_blocks)
        self.bottleneck = keras.models.Sequential(
            [
                keras.layers.Conv2D(filters=out_chans, kernel_size=1, use_bias=False),
                LayerNormalization(epsilon=layer_norm_epsilon),
                keras.layers.Conv2D(
                    filters=out_chans, kernel_size=3, padding="same", use_bias=False
                ),
                LayerNormalization(epsilon=layer_norm_epsilon),
            ]
        )

    def call(self, x):
        B, _, _, _ = x.shape
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.transformer_blocks(x)
        return self.bottleneck(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "img_size": self.img_size,
            "patch_size": self.patch_size,
            "in_chans": self.in_chans,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "mlp_dim": self.mlp_dim,
            "num_heads": self.num_heads,
            "out_chans": self.out_chans,
            "use_bias": self.use_bias,
            "use_abs_pos": self.use_abs_pos,
            "use_rel_pos": self.use_rel_pos,
            "window_size": self.window_size,
            "global_attention_indices": self.global_attention_indices,
            "layer_norm_epsilon": self.layer_norm_epsilon,
        })
        return config
