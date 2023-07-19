import numpy as np
import tensorflow as tf

from sam_tf.image_encoder import (
    ImageEncoder,
    MultiHeadAttentionWithRelativePE,
    WindowedTransformerEncoder,
)
from sam_tf.prompt_encoder import PromptEncoder
from sam_tf.mask_decoder import MaskDecoder, TwoWayAttention, TwoWayTransformer


tf.debugging.disable_traceback_filtering()


def test_multi_head_attention_with_relative_pe():
    attention_with_rel_pe = MultiHeadAttentionWithRelativePE(
        num_heads=16, key_dim=1280 // 16, use_bias=True, input_size=(64, 64)
    )
    x = tf.ones(shape=(1, 64, 64, 1280))
    x_out = attention_with_rel_pe(x)
    assert x_out.shape == (1, 64, 64, 1280)


def test_windowed_transformer_encoder():
    windowed_transformer_encoder = WindowedTransformerEncoder(
        project_dim=1280,
        mlp_dim=1280 * 4,
        num_heads=16,
        use_bias=True,
        use_rel_pos=True,
        window_size=14,
        input_size=(64, 64),
    )
    x = tf.ones((1, 64, 64, 1280))
    x_out = windowed_transformer_encoder(x)
    assert x_out.shape == (1, 64, 64, 1280)
    assert tf.reduce_all(x_out == 1)


def get_image_encoder():
    image_encoder = ImageEncoder(
        img_size=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=1280,
        depth=32,
        mlp_dim=1280 * 4,
        num_heads=16,
        out_chans=256,
        use_bias=True,
        use_rel_pos=True,
        window_size=14,
        global_attention_indices=[7, 15, 23, 31],
    )
    return image_encoder


def test_image_encoder():
    image_encoder = get_image_encoder()
    x = tf.ones((1, 1024, 1024, 3))
    x_out = image_encoder(x)
    assert x_out.shape == (1, 64, 64, 256)


def get_points_labels_box_mask(B):
    prompt_encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(1024, 1024),
        mask_in_chans=16,
    )

    points = tf.convert_to_tensor(
        np.random.randint(0, 1023, (B, 10, 2)), dtype=tf.float32
    )
    labels = tf.convert_to_tensor(1 * (np.random.rand(B, 10) > 0.5), dtype=tf.int64)
    box = tf.constant(
        [
            [
                [[10, 10], [500, 500]],
                [[20, 20], [500, 500]],
                [[30, 30], [500, 500]],
                [[40, 40], [500, 500]],
                [[50, 50], [500, 500]],
                [[60, 60], [500, 500]],
                [[70, 70], [500, 500]],
            ]
        ],
        dtype=tf.float32,
    )
    box = box[:, :B, ...]
    input_mask = tf.convert_to_tensor(
        1.0 * (np.random.rand(B, 256, 256, 1) > 0.5), dtype=tf.float32
    )

    return prompt_encoder, points, labels, box, input_mask


def test_prompt_encoder():
    prompt_encoder, points, labels, box, input_mask = get_points_labels_box_mask(7)

    sparse_embeddings, dense_embeddings = prompt_encoder(
        points=points, labels=labels, box=box, mask=input_mask
    )

    num_parameters = sum(np.prod(x.shape) for x in prompt_encoder.trainable_weights)

    assert sparse_embeddings.shape == (7, 12, 256)
    assert dense_embeddings.shape == (7, 64, 64, 256)
    assert num_parameters == 6220


def test_two_way_attention():
    prompt_encoder, points, labels, box, input_mask = get_points_labels_box_mask(1)
    image_encoder = get_image_encoder()

    image = tf.ones((1, 1024, 1024, 3))
    image_embeddings = image_encoder(image)

    sparse_embeddings, _ = prompt_encoder(
        points=points, labels=labels, box=box, mask=input_mask
    )

    two_way_attention = TwoWayAttention(
        num_heads=8, key_dim=256 // 8, mlp_dim=2048, skip_first_layer_pe=False
    )
    queries, keys = two_way_attention(
        queries=sparse_embeddings,
        keys=tf.reshape(image_embeddings, shape=(1, 64 * 64, 256)),
        query_pe=sparse_embeddings,
        key_pe=tf.reshape(prompt_encoder.get_dense_pe(), shape=(1, 64 * 64, 256)),
    )

    assert queries.shape == (1, 12, 256)
    assert keys.shape == (1, 64 * 64, 256)


def test_two_way_transformer():
    image_encoder = get_image_encoder()
    prompt_encoder, points, labels, box, input_mask = get_points_labels_box_mask(1)
    sparse_embeddings, dense_embeddings = prompt_encoder(
        points=points, labels=labels, box=box, mask=input_mask
    )
    image = tf.ones((1, 1024, 1024, 3))
    image_embeddings = image_encoder(image)
    two_way_transformer = TwoWayTransformer(
        depth=2, embedding_dim=256, num_heads=8, mlp_dim=2048
    )
    queries, keys = two_way_transformer(
        image_embedding=image_embeddings,
        image_pe=prompt_encoder.get_dense_pe(),
        point_embedding=sparse_embeddings,
    )
    assert queries.shape == (1, 12, 256)
    assert keys.shape == (1, 64 * 64, 256)


def test_mask_decoder():
    image_encoder = get_image_encoder()
    prompt_encoder, points, labels, box, input_mask = get_points_labels_box_mask(1)
    sparse_embeddings, dense_embeddings = prompt_encoder(
        points=points, labels=labels, box=box, mask=input_mask
    )
    image = tf.ones((1, 1024, 1024, 3))
    image_embeddings = image_encoder(image)
    mask_decoder = MaskDecoder(
        transformer_dim=256,
        transformer=TwoWayTransformer(
            depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8
        ),
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )
    masks, iou_pred = mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings[:1, ...],
        dense_prompt_embeddings=dense_embeddings[:1, ...],
        multimask_output=True,
    )
    num_parameters = sum(np.prod(x.shape) for x in mask_decoder.trainable_variables)
    assert masks.shape == (1, 3, 256, 256)
    assert iou_pred.shape == (1, 3)
    assert num_parameters == 4058340
