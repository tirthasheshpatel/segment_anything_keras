import tensorflow as tf
import keras
from keras import layers
from keras import models

from sam_tf.common import LayerNormalization, MLPBlock


class AttentionWithDownsampling(layers.Layer):
    def __init__(self, *, num_heads, key_dim, downsample_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.downsample_rate = downsample_rate
        self.internal_dims = key_dim // downsample_rate

        # Downsample
        self.query_proj = layers.Dense(self.internal_dims * self.num_heads)
        self.key_proj = layers.Dense(self.internal_dims * self.num_heads)
        self.value_proj = layers.Dense(self.internal_dims * self.num_heads)

        # Upsample
        self.out_proj = layers.Dense(self.key_dim * self.num_heads)

        # XXX: embedding_dim = key_dim * num_heads

    def __separate_heads(self, x):
        B, N, C = x.shape
        x = tf.reshape(x, (B, N, self.num_heads, C // self.num_heads))
        return tf.transpose(x, perm=(0, 2, 1, 3))

    def __recombine_heads(self, x):
        B, N_H, N_T, C_PH = x.shape
        x = tf.transpose(x, perm=(0, 2, 1, 3))
        return tf.reshape(x, (B, N_T, N_H * C_PH))

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
        out = query @ tf.transpose(key, (0, 1, 3, 2))
        out = out / tf.sqrt(tf.cast(C_PH, dtype=self.dtype))
        out = tf.math.softmax(out, axis=-1)

        # Get output
        attention_map = out @ value
        attention_map = self.__recombine_heads(attention_map)
        return self.out_proj(attention_map)


class TwoWayAttention(layers.Layer):
    def __init__(
        self,
        *,
        num_heads,
        key_dim,
        mlp_dim,
        skip_first_layer_pe,
        attention_downsample_rate=2,
        activation=keras.activations.relu,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.mlp_dim = mlp_dim
        self.skip_first_layer_pe = skip_first_layer_pe
        self.attention_downsample_rate = attention_downsample_rate
        self.activation = activation

        self.self_attention = AttentionWithDownsampling(
            num_heads=num_heads, key_dim=key_dim
        )
        self.layer_norm1 = layers.LayerNormalization()
        self.cross_attention_token_to_image = AttentionWithDownsampling(
            num_heads=num_heads,
            key_dim=key_dim,
            downsample_rate=attention_downsample_rate,
        )
        self.layer_norm2 = layers.LayerNormalization()
        
        self.mlp_block = MLPBlock(key_dim * num_heads, mlp_dim, activation)

        self.layer_norm3 = layers.LayerNormalization()
        self.cross_attention_image_to_token = AttentionWithDownsampling(
            num_heads=num_heads,
            key_dim=key_dim,
            downsample_rate=attention_downsample_rate,
        )
        self.layer_norm4 = layers.LayerNormalization()

    def call(self, queries, keys, query_pe, key_pe):
        if self.skip_first_layer_pe:
            queries = self.self_attention(query=queries, value=queries, key=queries)
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


class TwoWayTransformer(layers.Layer):
    def __init__(
        self,
        *,
        depth,
        embedding_dim,
        num_heads,
        mlp_dim,
        activation=keras.activations.relu,
        attention_downsample_rate=2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = []
        for i in range(depth):
            self.layers.append(
                TwoWayAttention(
                    num_heads=num_heads,
                    key_dim=embedding_dim // num_heads,
                    mlp_dim=mlp_dim,
                    skip_first_layer_pe=(i == 0),
                    attention_downsample_rate=attention_downsample_rate,
                    activation=activation,
                )
            )
        self.final_attention_token_to_image = AttentionWithDownsampling(
            num_heads=num_heads,
            key_dim=embedding_dim // num_heads,
            downsample_rate=attention_downsample_rate,
        )
        self.final_layer_norm = layers.LayerNormalization()

    def call(self, image_embedding, image_pe, point_embedding):
        B, H, W, C = image_embedding.shape
        image_embedding = tf.reshape(image_embedding, shape=(B, H * W, C))
        B, H, W, C = image_pe.shape
        image_pe = tf.reshape(image_pe, shape=(B, H * W, C))
        queries = point_embedding
        keys = image_embedding

        for layer in self.layers:
            queries, keys = layer(
                queries=queries, keys=keys, query_pe=point_embedding, key_pe=image_pe
            )

        queries_with_pe = queries + point_embedding
        keys_with_pe = keys + image_pe
        attention_map = self.final_attention_token_to_image(
            query=queries_with_pe, key=keys_with_pe, value=keys
        )
        queries = queries + attention_map
        queries = self.final_layer_norm(queries)

        return queries, keys


class MLP(models.Model):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.dense_net = []
        for hidden_dim in h:
            self.dense_net.append(layers.Dense(hidden_dim))
            self.dense_net.append(layers.Activation(keras.activations.relu))
        self.dense_net.append(layers.Dense(output_dim))
        self.dense_net = models.Sequential(self.dense_net)

    def call(self, x):
        return self.dense_net(x)


class MaskDecoder(models.Model):
    def __init__(
        self,
        *,
        transformer_dim,
        transformer,
        num_multimask_outputs,
        iou_head_depth,
        iou_head_hidden_dim,
        activation=keras.activations.gelu,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = layers.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = layers.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = models.Sequential(
            [
                layers.Conv2DTranspose(transformer_dim // 4, kernel_size=2, strides=2),
                LayerNormalization(),
                layers.Activation(activation),
                layers.Conv2DTranspose(transformer_dim // 8, kernel_size=2, strides=2),
                layers.Activation(activation),
            ]
        )

        self.output_hypernetworks_mlps = [
            MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
            for _ in range(self.num_mask_tokens)
        ]

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        # Don't like this; maybe we can just use raw learnable weight matrices.
        self.iou_token.build([])
        self.mask_tokens.build([])

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
        output_tokens = tf.concat(
            [self.iou_token.weights[0], self.mask_tokens.weights[0]], axis=0
        )
        output_tokens = tf.broadcast_to(
            output_tokens[tf.newaxis, ...],
            shape=(
                sparse_prompt_embeddings.shape[0],
                output_tokens.shape[0],
                output_tokens.shape[1],
            ),
        )
        tokens = tf.concat([output_tokens, sparse_prompt_embeddings], axis=1)

        # TODO: is this the same as torch.repeat_interleave?
        source = tf.broadcast_to(
            image_embeddings,
            shape=(
                sparse_prompt_embeddings.shape[0],
                image_embeddings.shape[1],
                image_embeddings.shape[2],
                image_embeddings.shape[3],
            ),
        )
        source = source + dense_prompt_embeddings
        # TODO: is this the same as torch.repeat_interleave?
        positional_source = tf.broadcast_to(image_pe, shape=image_embeddings.shape)
        B, H, W, C = source.shape

        hidden_state, source = self.transformer(source, positional_source, tokens)
        iou_token_out = hidden_state[:, 0, :]
        mask_tokens_out = hidden_state[:, 1 : (1 + self.num_mask_tokens), :]

        source = tf.reshape(source, shape=(B, H, W, C))
        upscaled_embeddings = self.output_upscaling(source)
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = tf.stack(hyper_in_list, axis=1)
        B, H, W, C = upscaled_embeddings.shape
        upscaled_embeddings = tf.reshape(
            tf.transpose(upscaled_embeddings, perm=(0, 3, 1, 2)), shape=(B, C, H * W)
        )
        masks = tf.reshape(
            hyper_in @ upscaled_embeddings, shape=(B, self.num_mask_tokens, H, W)
        )
        # masks = tf.transpose(masks, perm=(0, 2, 3, 1))

        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred
