# Author: Tirth Patel (tirthasheshpatel@gmail.com)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from keras import ops
from PIL import Image
from .prompter import SAMPrompter


__all__ = ["ResizeLongestSide", "SAMPredictor"]


class ResizeLongestSide:
    def __init__(self, target_length):
        self.target_length = int(target_length)

    def apply_image(self, image):
        image = np.asarray(image)
        if len(image.shape) != 3:
            raise ValueError("`image` must be of shape `(H, W, C)`.")
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1])
        return np.asarray(
            Image.fromarray(image).resize(
                target_size[::-1], resample=Image.Resampling.BILINEAR
            )
        )

    def apply_coords(self, coords, original_size):
        coords = ops.convert_to_tensor(coords)
        if len(coords.shape) != 3 and coords.shape[-1] != 2:
            raise ValueError(
                f"`coords` must be of shape `(B, N, 2)` but got `{coords.shape}`"
            )
        old_h, old_w = tuple(original_size)
        new_h, new_w = self.get_preprocess_shape(original_size[0], original_size[1])
        coords = ops.cast(coords, "float32")
        coords_x = coords[..., 0] * (new_w / old_w)
        coords_y = coords[..., 1] * (new_h / old_h)
        return ops.stack([coords_x, coords_y], axis=-1)

    def apply_boxes(self, boxes, original_size):
        boxes = ops.convert_to_tensor(boxes)
        B, N = boxes.shape[0:2]
        if len(boxes.shape) != 3 and boxes.shape[-1] != 4:
            raise ValueError(
                f"`boxes` must of shape `(B, N, 4)` but got `{boxes.shape}`"
            )
        boxes = self.apply_coords(ops.reshape(boxes, (B, N, 2, 2)), original_size)
        return boxes

    def get_preprocess_shape(self, old_h, old_w):
        scale = self.target_length * 1.0 / max(old_h, old_w)
        new_h = old_h * scale
        new_w = old_w * scale
        return int(new_h + 0.5), int(new_w + 0.5)


class SAMPredictor:
    mask_threshold = 0.0

    def __init__(
        self,
        model,
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    ):
        self.model = model
        self.pixel_mean = ops.convert_to_tensor(pixel_mean, dtype="float32")
        self.pixel_std = ops.convert_to_tensor(pixel_std, dtype="float32")
        self.img_size = model.backbone.input.shape[1]
        self.transform = ResizeLongestSide(self.img_size)
        self.prompter = SAMPrompter(self.model.prompt_encoder, self.model.mask_decoder)
        self.reset_image()

    def set_image(self, image, **kwargs):
        input_image = self.transform.apply_image(image)
        input_image_tensor = ops.convert_to_tensor(input_image, dtype="float32")
        input_image_tensor = input_image_tensor[None, :, :, :]

        self.set_tensor_image(input_image_tensor, image.shape[:2], **kwargs)

    def set_tensor_image(self, transformed_image, original_image_size, **kwargs):
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method. Expects the input
        image to be already transformed to the format expected by the model.

        Arguments:
          transformed_image (tensor): The input image, with shape
            1x3xHxW, which has been transformed with ResizeLongestSide.
          original_image_size (tuple(int, int)): The size of the image
            before transformation, in (H, W) format.
        """
        self.reset_image()

        self.original_size = tuple(original_image_size)
        self.input_size = tuple(transformed_image.shape[-2:])
        self.unprocessed_image = transformed_image
        input_image = self.preprocess_images(transformed_image)
        self.features = self.model.backbone.predict(input_image, **kwargs)
        self.is_image_set = True

    def _broadcast_batch(self, B, *args):
        res = []
        for arg in args:
            res.append(
                ops.broadcast_to(arg, (B,) + arg.shape[1:]) if arg is not None else arg
            )
        return res

    def predict(
        self, batched_input, multimask_output=True, return_logits=True, **kwargs
    ):
        batched_input = batched_input.copy()

        if self.is_image_set:
            batched_input["image"] = self.unprocessed_image
            batched_input["original_size"] = self.original_size
            images = self.features
        else:
            images = self.preprocess_images(batched_input["image"])

        points = batched_input.get("point_coords", None)
        labels = batched_input.get("point_labels", None)
        boxes = batched_input.get("boxes", None)
        masks = batched_input.get("mask_inputs", None)

        if points is not None and boxes is None:
            pad_point = ops.zeros((points.shape[0], 1, 2), dtype="float32")
            pad_label = -ops.ones((labels.shape[0], 1), dtype="float32")
            points = ops.concatenate([points, pad_point], axis=1)
            labels = ops.concatenate([labels, pad_label], axis=1)

        B = max(
            [
                images.shape[0],
                points.shape[0] if points is not None else 0,
                labels.shape[0] if labels is not None else 0,
                boxes.shape[0] if boxes is not None else 0,
                masks.shape[0] if masks is not None else 0,
            ]
        )

        images, points, labels, boxes, masks = self._broadcast_batch(
            B, images, points, labels, boxes, masks
        )

        model_input = {"images": images}

        if points is not None:
            model_input["points"] = points
            model_input["labels"] = labels
        if boxes is not None:
            model_input["boxes"] = boxes
        if masks is not None:
            model_input["masks"] = masks

        if self.is_image_set:
            outs = self.prompter.predict(model_input, **kwargs)
        else:
            outs = self.model.predict(model_input, **kwargs)
        low_res_masks, iou_scores = outs["masks"], outs["iou_pred"]
        if multimask_output:
            low_res_masks = low_res_masks[:, 1:, :, :]
            iou_scores = iou_scores[:, 1:]
        else:
            low_res_masks = low_res_masks[:, :1, :, :]
            iou_scores = iou_scores[:, :1]
        masks = self.postprocess_masks(
            low_res_masks,
            input_size=batched_input["image"].shape[1:3],
            original_size=batched_input["original_size"],
        )
        if not return_logits:
            masks = ops.cast(masks > self.mask_threshold, dtype="float32")
        batched_output = {
            "masks": masks,
            "iou_predictions": iou_scores,
            "low_res_masks": low_res_masks,
        }
        return batched_output

    def postprocess_masks(self, masks, input_size, original_size):
        masks = ops.image.resize(
            ops.transpose(masks, axes=(0, 2, 3, 1)),
            size=(self.img_size, self.img_size),
            interpolation="bilinear",
        )
        masks = masks[..., : input_size[0], : input_size[1], :]
        masks = ops.image.resize(masks, size=original_size, interpolation="bilinear")
        return ops.transpose(masks, axes=(0, 3, 1, 2))

    def preprocess_images(self, x):
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[1:3]
        pad_h = self.img_size - h
        pad_w = self.img_size - w
        x = ops.pad(x, [(0, 0), (0, pad_h), (0, pad_w), (0, 0)])
        # KerasCV now rescales the images and normalizes them.
        # Just unnormalize such that when KerasCV normalizes them
        # again, the padded values map to 0.
        x = x * self.pixel_std + self.pixel_mean
        return x

    def get_image_embedding(self):
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        return self.features

    def reset_image(self):
        """Resets the currently set image."""
        self.is_image_set = False
        self.unprocessed_image = None
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None
