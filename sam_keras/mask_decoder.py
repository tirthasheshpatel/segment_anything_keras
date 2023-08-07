# Author: Tirth Patel (tirthasheshpatel@gmail.com)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from keras_cv.backend import keras
from keras_cv.backend import ops

from sam_keras.common import MLPBlock


@keras.utils.register_keras_serializable(package="keras_cv")
class MultiHeadAttentionWithDownsampling(keras.layers.Layer):
    """Multi-Head Attention with downsampling.

    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.

    This layer first downscales the features of input queries, keys, and
    values using a dense layer. Multi-head attention is then performed
    and the attention map is projected back (upscaled) to the number of
    input features.

    Args:
        num_heads (int): Number of attention heads.
        key_dim (int): Size of each attention head for query, key, and
            value.
        downsample_rate (int, optional): The factor by which to downscale the
            input features i.e. the input features of size `key_dim` are
            projected down to `key_dim // downsample_rate`.
    """

    def __init__(self, num_heads, key_dim, downsample_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.downsample_rate = downsample_rate
        self.internal_dims = key_dim // downsample_rate

        # Downsample
        self.query_proj = keras.layers.Dense(
            self.internal_dims * self.num_heads
        )
        self.key_proj = keras.layers.Dense(self.internal_dims * self.num_heads)
        self.value_proj = keras.layers.Dense(
            self.internal_dims * self.num_heads
        )

        # Upsample
        self.out_proj = keras.layers.Dense(self.key_dim * self.num_heads)

        self.query_proj.build([None, None, self.num_heads * self.key_dim])
        self.key_proj.build([None, None, self.num_heads * self.key_dim])
        self.value_proj.build([None, None, self.num_heads * self.key_dim])
        self.out_proj.build([None, None, self.internal_dims * self.num_heads])

        self.built = True

    def __separate_heads(self, x):
        B, N, C = x.shape
        x = ops.reshape(x, (B, N, self.num_heads, C // self.num_heads))
        return ops.transpose(x, axes=(0, 2, 1, 3))

    def __recombine_heads(self, x):
        B, N_H, N_T, C_PH = x.shape
        x = ops.transpose(x, axes=(0, 2, 1, 3))
        return ops.reshape(x, (B, N_T, N_H * C_PH))

    def call(self, query, value, key):
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        # Separate into heads
        query = self.__separate_heads(query)
        key = self.__separate_heads(key)
        value = self.__separate_heads(value)

        # Attention
        C_PH = query.shape[-1]
        out = query @ ops.transpose(key, (0, 1, 3, 2))
        out = out / ops.sqrt(ops.cast(C_PH, dtype=self.dtype))
        out = ops.softmax(out, axis=-1)

        # Get output
        attention_map = out @ value
        attention_map = self.__recombine_heads(attention_map)
        return self.out_proj(attention_map)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "downsample_rate": self.downsample_rate,
            }
        )
        return config


@keras.utils.register_keras_serializable(package="keras_cv")
class TwoWayMultiHeadAttention(keras.layers.Layer):
    """Two-way multi-head attention layer.

    Args:
        num_heads (int): Number of attention heads.
        key_dim (int): Size of each attention head for query, key, and
            value.
        mlp_dim (int): Number of hidden dims to use in the mlp block.
        skip_first_layer_pe (bool): A boolean indicating whether to skip the
            first layer positional embeddings.
        attention_downsample_rate (int, optional): The downsample rate to use
            in the attention layers. Defaults to 2.
        activation (str, optional): The activation for the mlp block's output
            layer. Defaults to "relu".
    """

    def __init__(
        self,
        num_heads,
        key_dim,
        mlp_dim,
        skip_first_layer_pe,
        attention_downsample_rate=2,
        activation="relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.mlp_dim = mlp_dim
        self.skip_first_layer_pe = skip_first_layer_pe
        self.attention_downsample_rate = attention_downsample_rate
        self.activation = activation

        self.self_attention = MultiHeadAttentionWithDownsampling(
            num_heads=num_heads, key_dim=key_dim
        )
        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.cross_attention_token_to_image = (
            MultiHeadAttentionWithDownsampling(
                num_heads=num_heads,
                key_dim=key_dim,
                downsample_rate=attention_downsample_rate,
            )
        )
        self.layer_norm2 = keras.layers.LayerNormalization(epsilon=1e-5)

        self.mlp_block = MLPBlock(key_dim * num_heads, mlp_dim, activation)

        self.layer_norm3 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.cross_attention_image_to_token = (
            MultiHeadAttentionWithDownsampling(
                num_heads=num_heads,
                key_dim=key_dim,
                downsample_rate=attention_downsample_rate,
            )
        )
        self.layer_norm4 = keras.layers.LayerNormalization(epsilon=1e-5)

        self.layer_norm1.build([None, None, self.num_heads * self.key_dim])
        self.layer_norm2.build([None, None, self.num_heads * self.key_dim])
        self.layer_norm3.build([None, None, self.num_heads * self.key_dim])
        self.layer_norm4.build([None, None, self.num_heads * self.key_dim])

        self.built = True

    def call(self, queries, keys, query_pe, key_pe):
        if self.skip_first_layer_pe:
            queries = self.self_attention(
                query=queries, value=queries, key=queries
            )
        else:
            queries_with_pe = queries + query_pe
            attention_map = self.self_attention(
                query=queries_with_pe, key=queries_with_pe, value=queries
            )
            queries = queries + attention_map
        queries = self.layer_norm1(queries)

        queries_with_pe = queries + query_pe
        keys_with_pe = keys + key_pe
        attention_map = self.cross_attention_token_to_image(
            query=queries_with_pe, key=keys_with_pe, value=keys
        )
        queries = queries + attention_map
        queries = self.layer_norm2(queries)

        mlp_out = self.mlp_block(queries)
        queries = queries + mlp_out
        queries = self.layer_norm3(queries)

        queries_with_pe = queries + query_pe
        keys_with_pe = keys + key_pe
        attention_map = self.cross_attention_image_to_token(
            query=keys_with_pe, key=queries_with_pe, value=queries
        )
        keys = keys + attention_map
        keys = self.layer_norm4(keys)

        return queries, keys

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "mlp_dim": self.mlp_dim,
                "skip_first_layer_pe": self.skip_first_layer_pe,
                "attention_downsample_rate": self.attention_downsample_rate,
                "activation": self.activation,
            }
        )
        return config


@keras.utils.register_keras_serializable(package="keras_cv")
class TwoWayTransformer(keras.layers.Layer):
    """A two-way cross-attention transformer decoder.

    A transformer decoder that attends to an input image using
    queries whose positional embedding is supplied.

    The transformer decoder design is shown in [1]_. Each decoder layer
    performs 4 steps: (1) self-attention on the tokens, (2) cross-attention
    from tokens (as queries) to the image embedding, (3) a point-wise MLP
    updates each token, and (4) cross-attention from the image embedding (as
    queries) to tokens. This last step updates the image embedding with prompt
    information. Each self/cross-attention and MLP has a residual connection
    and layer normalization.

    To ensure the decoder has access to critical geometric information the
    positional encodings are added to the image embedding whenever they
    participate in an attention layer. Additionally, the entire original
    prompt tokens (including their positional encodings) are re-added to the
    updated tokens whenever they participate in an attention layer. This
    allows for a strong dependence on both the prompt token's geometric
    location and type.

    Args:
        depth (int): The depth of the attention blocks (the number
            of attention blocks to use).
        embedding_dim (int): The number of features of the input image and
            point embeddings.
        num_heads (int): Number of heads to use in the attention layers.
        mlp_dim (int): The number of units in the hidden layer of the MLP
            block used in the attention layers.
        activation (str, optional): The activation of the MLP block's output
            layer used in the attention layers. Defaults to "relu".
        attention_downsample_rate (int, optional): The downsample rate of the
            attention layers. Defaults to 2.

    References:
        - [Segment Anything](https://arxiv.org/abs/2304.02643)
    """

    def __init__(
        self,
        depth,
        embedding_dim,
        num_heads,
        mlp_dim,
        activation="relu",
        attention_downsample_rate=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.activation = activation
        self.attention_downsample_rate = attention_downsample_rate
        self.layers = []
        for i in range(depth):
            self.layers.append(
                TwoWayMultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=embedding_dim // num_heads,
                    mlp_dim=mlp_dim,
                    skip_first_layer_pe=(i == 0),
                    attention_downsample_rate=attention_downsample_rate,
                    activation=activation,
                )
            )
        self.final_attention_token_to_image = (
            MultiHeadAttentionWithDownsampling(
                num_heads=num_heads,
                key_dim=embedding_dim // num_heads,
                downsample_rate=attention_downsample_rate,
            )
        )
        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5)

        self.final_layer_norm.build([None, None, self.embedding_dim])

        self.built = True

    def call(self, image_embedding, image_pe, point_embedding):
        B, H, W, C = image_embedding.shape
        image_embedding = ops.reshape(image_embedding, (B, H * W, C))
        B, H, W, C = image_pe.shape
        image_pe = ops.reshape(image_pe, (B, H * W, C))
        queries = point_embedding
        keys = image_embedding

        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        queries_with_pe = queries + point_embedding
        keys_with_pe = keys + image_pe
        attention_map = self.final_attention_token_to_image(
            query=queries_with_pe, key=keys_with_pe, value=keys
        )
        queries = queries + attention_map
        queries = self.final_layer_norm(queries)

        return queries, keys

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depth": self.depth,
                "embedding_dim": self.embedding_dim,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "activation": self.activation,
                "attention_downsample_rate": self.attention_downsample_rate,
            }
        )
        return config


@keras.utils.register_keras_serializable(package="keras_cv")
class MLP(keras.layers.Layer):
    """A MLP block with architecture
    `input_dim -> [hidden_dim] * (num_layers - 1) -> output_dim`.

    Args:
        hidden_dim (int): The number of units in the hidden layers.
        output_dim (int): The number of units in the output layer.
        num_layers (int): The total number of dense layers to use.
    """
    def __init__(self, hidden_dim, output_dim, num_layers, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.dense_net = []
        for hidden_dim in h:
            self.dense_net.append(keras.layers.Dense(hidden_dim))
            self.dense_net.append(keras.layers.Activation("relu"))
        self.dense_net.append(keras.layers.Dense(output_dim))
        self.dense_net = keras.models.Sequential(self.dense_net)

        self.built = False

    def build(self, input_shape):
        self.dense_net.build(input_shape)

        self.built = True

    def call(self, x):
        return self.dense_net(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "num_layers": self.num_layers,
            }
        )
        return config


@keras.utils.register_keras_serializable(package="keras_cv.models")
class MaskDecoder(keras.models.Model):
    """Mask decoder for the segment anything model.

    This lightweight module efficiently maps the image embedding and a set of
    prompt embeddings to an output mask. Before applying the transformer
    decoder, the layer first inserts into the set of prompt embeddings a
    learned output token embedding that will be used at the decoder's output.
    For simplicity, these embeddings (not including the image embedding) are
    collectively called "tokens".

    The image embeddings, positional image embeddings, and tokens are passed
    through a transformer decoder. After running the decoder, the layer
    upsamples the updated image embedding by 4x with two transposed
    convolutional layers (now it's downscaled 4x relative to the input
    image). Then, the tokens attend once more to the image embedding and
    the updated output token embedding are passed to a small 3-layer MLP that
    outputs a vector matching the channel dimension of the upscaled image
    embedding. Finally, a mask is predicted with a spatially point-wise
    product between the upscaled image embedding and the MLP's output.

    Args:
        transformer_dim (int): The number of input features to the transformer
            decoder.
        transformer (keras.layers.Layer): A transformer decoder.
        num_multimask_outputs (int): Number of multimask outputs. The model
            would generate these many extra masks when `multimask_output` is
            `True`.
        iou_head_depth (int): The depth of the dense net used to predict the
            IoU confidence score.
        iou_head_hidden_dim (int): The number of units in the hidden layers
            used in the dense net to predict the IoU confidence score.
        activation (str, optional): Activation to use in the mask upscaler
            network. Defaults to "gelu".

    References:
        - [Segment Anything](https://arxiv.org/abs/2304.02643)
    """
    def __init__(
        self,
        transformer_dim,
        transformer,
        num_multimask_outputs,
        iou_head_depth,
        iou_head_hidden_dim,
        activation="gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_head_depth = iou_head_depth
        self.iou_head_hidden_dim = iou_head_hidden_dim
        self.activation = activation

        self.iou_token = keras.layers.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = keras.layers.Embedding(
            self.num_mask_tokens, transformer_dim
        )

        self.output_upscaling = keras.models.Sequential(
            [
                keras.layers.Conv2DTranspose(
                    transformer_dim // 4, kernel_size=2, strides=2
                ),
                keras.layers.LayerNormalization(epsilon=1e-6),
                keras.layers.Activation(activation),
                keras.layers.Conv2DTranspose(
                    transformer_dim // 8, kernel_size=2, strides=2
                ),
                keras.layers.Activation(activation),
            ]
        )

        self.output_hypernetworks_mlps = [
            MLP(transformer_dim, transformer_dim // 8, 3)
            for _ in range(self.num_mask_tokens)
        ]

        self.iou_prediction_head = MLP(
            iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        self.iou_token.build(None)
        self.mask_tokens.build(None)

        self.output_upscaling.build([None, None, None, self.transformer_dim])

        for mlp in self.output_hypernetworks_mlps:
            mlp.build([None, self.transformer_dim])

        self.iou_prediction_head.build([None, self.transformer_dim])

        self.built = True

    def call(
        self,
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
        multimask_output,
    ):
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        if multimask_output:
            return masks[:, 1:, :, :], iou_pred[:, 1:]
        return masks[:, :1, :, :], iou_pred[:, :1]

    def predict_masks(
        self,
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
    ):
        output_tokens = ops.concatenate(
            [self.iou_token.weights[0], self.mask_tokens.weights[0]], axis=0
        )
        output_tokens = ops.broadcast_to(
            output_tokens[None, ...],
            shape=(
                sparse_prompt_embeddings.shape[0],
                output_tokens.shape[0],
                output_tokens.shape[1],
            ),
        )
        tokens = ops.concatenate(
            [output_tokens, sparse_prompt_embeddings], axis=1
        )

        source = ops.broadcast_to(
            image_embeddings,
            shape=(
                tokens.shape[0],
                image_embeddings.shape[1],
                image_embeddings.shape[2],
                image_embeddings.shape[3],
            ),
        )
        source = source + dense_prompt_embeddings
        positional_source = ops.broadcast_to(
            image_pe,
            shape=(
                tokens.shape[0],
                image_embeddings.shape[1],
                image_embeddings.shape[2],
                image_embeddings.shape[3],
            ),
        )
        B, H, W, C = source.shape

        hidden_state, source = self.transformer(
            source, positional_source, tokens
        )
        iou_token_out = hidden_state[:, 0, :]
        mask_tokens_out = hidden_state[:, 1 : (1 + self.num_mask_tokens), :]

        source = ops.reshape(source, (B, H, W, C))
        upscaled_embeddings = self.output_upscaling(source)
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = ops.stack(hyper_in_list, axis=1)
        B, H, W, C = upscaled_embeddings.shape
        upscaled_embeddings = ops.reshape(
            ops.transpose(upscaled_embeddings, axes=(0, 3, 1, 2)),
            (B, C, H * W),
        )
        masks = ops.reshape(
            hyper_in @ upscaled_embeddings, (B, self.num_mask_tokens, H, W)
        )

        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "transformer_dim": self.transformer_dim,
                "transformer": keras.saving.serialize_keras_object(
                    self.transformer
                ),
                "num_multimask_outputs": self.num_multimask_outputs,
                "iou_head_depth": self.iou_head_depth,
                "iou_head_hidden_dim": self.iou_head_hidden_dim,
                "activation": self.activation,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config.update(
            {"transformer": keras.layers.deserialize(config["transformer"])}
        )
        return super().from_config(config)
