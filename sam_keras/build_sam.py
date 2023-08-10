# Author: Tirth Patel (tirthasheshpatel@gmail.com)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from sam_keras.image_encoder import ImageEncoder
from sam_keras.prompt_encoder import PromptEncoder
from sam_keras.mask_decoder import TwoWayTransformer, MaskDecoder
from sam_keras.sam import SegmentAnythingModel


huge_config = {
    "img_size": 1024,
    "patch_size": 16,
    "in_chans": 3,
    "embed_dim": 1280,
    "depth": 32,
    "mlp_dim": 1280 * 4,
    "num_heads": 16,
    "out_chans": 256,
    "use_bias": True,
    "use_rel_pos": True,
    "window_size": 14,
    "global_attention_indices": [7, 15, 23, 31]
}

large_config = {
    "img_size": 1024,
    "patch_size": 16,
    "in_chans": 3,
    "embed_dim": 1024,
    "depth": 24,
    "mlp_dim": 1024 * 4,
    "num_heads": 16,
    "out_chans": 256,
    "use_bias": True,
    "use_rel_pos": True,
    "window_size": 14,
    "global_attention_indices": [5, 11, 17, 23]
}

base_config = {
    "img_size": 1024,
    "patch_size": 16,
    "in_chans": 3,
    "embed_dim": 768,
    "depth": 12,
    "mlp_dim": 768 * 4,
    "num_heads": 12,
    "out_chans": 256,
    "use_bias": True,
    "use_rel_pos": True,
    "window_size": 14,
    "global_attention_indices": [2, 5, 8, 11]
}


def build_sam(config):
    image_encoder = ImageEncoder(**config)
    embed_size = config["img_size"] // config["patch_size"]
    prompt_encoder = PromptEncoder(
        embed_dim=config["out_chans"],
        image_embedding_size=(embed_size, embed_size),
        input_image_size=(config["img_size"], config["img_size"]),
        mask_in_chans=16
    )
    mask_decoder = MaskDecoder(
        transformer_dim=config["out_chans"],
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=config["out_chans"],
            mlp_dim=2048,
            num_heads=8
        ),
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=256
    )
    return SegmentAnythingModel(image_encoder, prompt_encoder, mask_decoder)


def build_sam_huge():
    return build_sam(huge_config)


def build_sam_large():
    return build_sam(large_config)


def build_sam_base():
    return build_sam(base_config)
