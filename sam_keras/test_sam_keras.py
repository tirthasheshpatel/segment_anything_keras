# Author: Tirth Patel (tirthasheshpatel@gmail.com)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import pytest
import numpy as np

from keras_cv.backend import keras
from keras_cv.backend import ops

from sam_keras.image_encoder import (
    ImageEncoder,
    MultiHeadAttentionWithRelativePE,
    WindowedTransformerEncoder,
)
from sam_keras.prompt_encoder import PromptEncoder
from sam_keras.mask_decoder import MaskDecoder, TwoWayMultiHeadAttention, TwoWayTransformer


keras.src.utils.traceback_utils.disable_traceback_filtering()


class TestSAM:
    def test_multi_head_attention_with_relative_pe(self):
        attention_with_rel_pe = MultiHeadAttentionWithRelativePE(
            num_heads=16,
            key_dim=1280 // 16,
            use_bias=True,
            input_size=(64, 64),
        )
        x = np.ones(shape=(1, 64, 64, 1280))
        x_out = ops.convert_to_numpy(attention_with_rel_pe(x))
        assert x_out.shape == (1, 64, 64, 1280)

    def test_windowed_transformer_encoder(self):
        windowed_transformer_encoder = WindowedTransformerEncoder(
            project_dim=1280,
            mlp_dim=1280 * 4,
            num_heads=16,
            use_bias=True,
            use_rel_pos=True,
            window_size=14,
            input_size=(64, 64),
        )
        x = np.ones((1, 64, 64, 1280))
        x_out = ops.convert_to_numpy(windowed_transformer_encoder(x))
        assert x_out.shape == (1, 64, 64, 1280)
        assert np.all(x_out == 1)

    @pytest.mark.extra_large
    def test_image_encoder(self):
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
        x = np.ones((1, 1024, 1024, 3))
        x_out = ops.convert_to_numpy(image_encoder(x))
        num_parameters = sum(
            np.prod(tuple(x.shape)) for x in image_encoder.trainable_variables
        )
        assert x_out.shape == (1, 64, 64, 256)
        assert num_parameters == 637_026_048

        # saving test
        path = os.path.join(
            tempfile.gettempdir(), "sam_tf_image_encoder.keras"
        )
        image_encoder.save(path)
        loaded_model = keras.saving.load_model(path)
        x_out_loaded = ops.convert_to_numpy(loaded_model(x))
        np.testing.assert_equal(x_out, x_out_loaded)

    def get_points_labels_box_mask(self, B):
        prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(64, 64),
            input_image_size=(1024, 1024),
            mask_in_chans=16,
        )

        points = ops.convert_to_tensor(
            np.random.randint(0, 1023, (B, 10, 2)), dtype="float32"
        )
        labels = ops.convert_to_tensor(
            1 * (np.random.rand(B, 10) > 0.5), dtype="int64"
        )
        box = ops.array(
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
            dtype="float32",
        )
        box = box[:, :B, ...]
        input_mask = ops.convert_to_tensor(
            1.0 * (np.random.rand(B, 256, 256, 1) > 0.5), dtype="float32"
        )

        return prompt_encoder, points, labels, box, input_mask

    def test_prompt_encoder(self):
        (
            prompt_encoder,
            points,
            labels,
            box,
            input_mask,
        ) = self.get_points_labels_box_mask(7)

        sparse_embeddings, dense_embeddings = prompt_encoder(
            points=points, labels=labels, box=box, mask=input_mask
        )

        num_parameters = sum(
            np.prod(tuple(x.shape)) for x in prompt_encoder.trainable_weights
        )

        sparse_embeddings = ops.convert_to_numpy(sparse_embeddings)
        dense_embeddings = ops.convert_to_numpy(dense_embeddings)

        assert sparse_embeddings.shape == (7, 12, 256)
        assert dense_embeddings.shape == (7, 64, 64, 256)
        assert num_parameters == 6_220

        # saving test
        path = os.path.join(
            tempfile.gettempdir(), "sam_tf_prompt_encoder.keras"
        )
        prompt_encoder.save(path)
        loaded_model = keras.saving.load_model(path)
        sparse_embeddings_loaded, dense_embeddings_loaded = loaded_model(
            points=points, labels=labels, box=box, mask=input_mask
        )
        sparse_embeddings_loaded = ops.convert_to_numpy(
            sparse_embeddings_loaded
        )
        dense_embeddings_loaded = ops.convert_to_numpy(dense_embeddings_loaded)
        pegm_ref = ops.convert_to_numpy(
            prompt_encoder.positional_embedding_layer.positional_encoding_gaussian_matrix
        )
        pegm_loaded = ops.convert_to_numpy(
            loaded_model.positional_embedding_layer.positional_encoding_gaussian_matrix
        )
        np.testing.assert_equal(pegm_ref, pegm_loaded)
        np.testing.assert_equal(sparse_embeddings, sparse_embeddings_loaded)
        np.testing.assert_equal(dense_embeddings, dense_embeddings_loaded)

    def test_two_way_multi_head_attention(self):
        (
            prompt_encoder,
            points,
            labels,
            box,
            input_mask,
        ) = self.get_points_labels_box_mask(1)
        image_embeddings = np.random.randn(1, 64, 64, 256).astype(np.float32)

        sparse_embeddings, _ = prompt_encoder(
            points=points, labels=labels, box=box, mask=input_mask
        )

        two_way_attention = TwoWayMultiHeadAttention(
            num_heads=8,
            key_dim=256 // 8,
            mlp_dim=2048,
            skip_first_layer_pe=False,
        )
        queries, keys = two_way_attention(
            queries=sparse_embeddings,
            keys=ops.reshape(image_embeddings, (1, 64 * 64, 256)),
            query_pe=sparse_embeddings,
            key_pe=ops.reshape(
                prompt_encoder.get_dense_pe(), (1, 64 * 64, 256)
            ),
        )

        queries, keys = map(ops.convert_to_numpy, [queries, keys])

        assert queries.shape == (1, 12, 256)
        assert keys.shape == (1, 64 * 64, 256)

    def test_two_way_transformer(self):
        (
            prompt_encoder,
            points,
            labels,
            box,
            input_mask,
        ) = self.get_points_labels_box_mask(1)
        sparse_embeddings, _ = prompt_encoder(
            points=points, labels=labels, box=box, mask=input_mask
        )
        image_embeddings = np.random.randn(1, 64, 64, 256)
        two_way_transformer = TwoWayTransformer(
            depth=2, embedding_dim=256, num_heads=8, mlp_dim=2048
        )
        queries, keys = two_way_transformer(
            image_embedding=image_embeddings,
            image_pe=prompt_encoder.get_dense_pe(),
            point_embedding=sparse_embeddings,
        )
        queries, keys = map(ops.convert_to_numpy, [queries, keys])
        assert queries.shape == (1, 12, 256)
        assert keys.shape == (1, 64 * 64, 256)

    def test_mask_decoder(self):
        (
            prompt_encoder,
            points,
            labels,
            box,
            input_mask,
        ) = self.get_points_labels_box_mask(1)
        sparse_embeddings, dense_embeddings = prompt_encoder(
            points=points, labels=labels, box=box, mask=input_mask
        )
        image_embeddings = np.random.randn(1, 64, 64, 256)
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
        num_parameters = sum(
            np.prod(tuple(x.shape)) for x in mask_decoder.trainable_variables
        )
        masks, iou_pred = map(ops.convert_to_numpy, [masks, iou_pred])
        assert masks.shape == (1, 3, 256, 256)
        assert iou_pred.shape == (1, 3)
        assert num_parameters == 4_058_340

        # saving test
        path = os.path.join(tempfile.gettempdir(), "sam_tf_mask_decoder.keras")
        mask_decoder.save(path)
        loaded_model = keras.saving.load_model(path)
        masks_loaded, iou_pred_loaded = loaded_model(
            image_embeddings=image_embeddings,
            image_pe=prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings[:1, ...],
            dense_prompt_embeddings=dense_embeddings[:1, ...],
            multimask_output=True,
        )
        masks_loaded = ops.convert_to_numpy(masks_loaded)
        iou_pred_loaded = ops.convert_to_numpy(iou_pred_loaded)
        np.testing.assert_equal(masks, masks_loaded)
        np.testing.assert_equal(iou_pred, iou_pred_loaded)
