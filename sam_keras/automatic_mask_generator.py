# Author: Tirth Patel (tirthasheshpatel@gmail.com)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import keras
from keras import ops

# TODO: KerasCV made the non_max_suppression layer internal since 0.7.0 release.
#       Instead of trying to access the internals, copy-paste the code for
#       non_max_suppression in this repo and use that instead.
try:
    from keras_cv.layers.object_detection.non_max_suppression import non_max_suppression
except ImportError:
    from keras_cv.src.layers.object_detection.non_max_suppression import (
        non_max_suppression,
    )

from sam_keras.amg_utils import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    box_xyxy_to_yxyx,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_tensor,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)


__all__ = ["SAMAutomaticMaskGenerator"]


def _box_area(boxes):
    return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])


def _batched_nms(boxes, scores, iou_threshold, max_output_size):
    if keras.config.backend() == "torch":
        from torchvision.ops import batched_nms

        idx = batched_nms(
            boxes,
            scores,
            ops.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=iou_threshold,
        )
        del batched_nms
    elif keras.config.backend() == "tensorflow":
        import tensorflow as tf

        idx = tf.image.non_max_suppression(
            boxes=box_xyxy_to_yxyx(boxes),
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
        )
        del tf
    elif keras.config.backend() == "jax":
        from sam_keras import jax_nms

        idx = jax_nms.non_max_suppression_padded(
            boxes=box_xyxy_to_yxyx(boxes)[None, ...],
            scores=scores[None, ...],
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
        )
        del jax_nms
    else:
        idx, num_valid = non_max_suppression(
            boxes=box_xyxy_to_yxyx(boxes),
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
        )
        idx = idx[0][:num_valid]
    return idx


class SAMAutomaticMaskGenerator:
    def __init__(
        self,
        predictor,
        points_per_side=32,
        points_per_batch=64,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        stability_score_offset=1.0,
        box_nms_thresh=0.7,
        crop_n_layers=0,
        crop_nms_thresh=0.7,
        crop_overlap_ratio=512 / 1500,
        crop_n_points_downscale_factor=1,
        point_grids=None,
        min_mask_region_area=0,
        output_mode="binary_mask",
        max_output_masks=100,
    ) -> None:
        """
        Using a SAM model, generates masks for the entire image.
        Generates a grid of point prompts over the image, then filters
        low quality and duplicate masks. The default settings are chosen
        for SAM with a ViT-H backbone.

        Arguments:
          predictor (Sam): The SAM model to use for mask prediction.
          points_per_side (int or None): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2. If None, 'point_grids' must provide explicit
            point sampling.
          points_per_batch (int): Sets the number of points run simultaneously
            by the model. Higher numbers may be faster but use more GPU memory.
          pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
          stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
          stability_score_offset (float): The amount to shift the cutoff when
            calculated the stability score.
          box_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.
          crop_n_layers (int): If >0, mask prediction will be run again on
            crops of the image. Sets the number of layers to run, where each
            layer has 2**i_layer number of image crops.
          crop_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks between different crops.
          crop_overlap_ratio (float): Sets the degree to which crops overlap.
            In the first crop layer, crops will overlap by this fraction of
            the image length. Later layers with more crops scale down this overlap.
          crop_n_points_downscale_factor (int): The number of points-per-side
            sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
          point_grids (list(np.ndarray) or None): A list over explicit grids
            of points used for sampling, normalized to [0,1]. The nth grid in the
            list is used in the nth crop layer. Exclusive with points_per_side.
          min_mask_region_area (int): If >0, postprocessing will be applied
            to remove disconnected regions and holes in masks with area smaller
            than min_mask_region_area. Requires opencv.
          output_mode (str): The form masks are returned in. Can be 'binary_mask',
            'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
            For large resolutions, 'binary_mask' may consume large amounts of
            memory.
          max_output_masks (int): Maximum number of masks to generate.
        """

        if not ((points_per_side is None) ^ (point_grids is None)):
            raise ValueError(
                "Exactly one of points_per_side or point_grid must be provided."
            )
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")

        if output_mode not in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ]:
            raise ValueError(f"Unknown output_mode {output_mode}.")
        if output_mode == "coco_rle":
            from pycocotools import mask as mask_utils  # type: ignore # noqa: F401

        if min_mask_region_area > 0:
            import cv2  # type: ignore # noqa: F401

        self.predictor = predictor
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode
        self.max_output_masks = max_output_masks

    def generate(self, image, **kwargs):
        """
        Generates masks for the given image.

        Arguments:
          image (np.ndarray): The image to generate masks for, in HWC uint8 format.

        Returns:
           list(dict(str, any)): A list over records for masks. Each record is
             a dict containing the following keys:
               segmentation (dict(str, any) or np.ndarray): The mask. If
                 output_mode='binary_mask', is an array of shape HW. Otherwise,
                 is a dictionary containing the RLE.
               bbox (list(float)): The box around the mask, in XYWH format.
               area (int): The area in pixels of the mask.
               predicted_iou (float): The model's own prediction of the mask's
                 quality. This is filtered by the pred_iou_thresh parameter.
               point_coords (list(list(float))): The point coordinates input
                 to the model to generate this mask.
               stability_score (float): A measure of the mask's quality. This
                 is filtered on using the stability_score_thresh parameter.
               crop_box (list(float)): The crop of the image used to generate
                 the mask, given in XYWH format.
        """

        # Generate masks
        mask_data = self._generate_masks(image, **kwargs)

        # Filter small disconnected regions and holes in masks
        if self.min_mask_region_area > 0:
            mask_data = self.postprocess_small_regions(
                mask_data,
                self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )

        # Encode masks
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [
                coco_encode_rle(rle) for rle in mask_data["rles"]
            ]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
        else:
            mask_data["segmentations"] = mask_data["rles"]

        # Write mask records
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": ops.convert_to_numpy(
                    box_xyxy_to_xywh(mask_data["boxes"][idx])
                ).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                "crop_box": ops.convert_to_numpy(
                    box_xyxy_to_xywh(mask_data["crop_boxes"][idx])
                ).tolist(),
            }
            curr_anns.append(ann)

        return curr_anns

    def _generate_masks(self, image, **kwargs):
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )

        # Iterate over image crops
        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(
                image, crop_box, layer_idx, orig_size, **kwargs
            )
            data.cat(crop_data)

        # Remove duplicate masks between crops
        if len(crop_boxes) > 1:
            # Prefer masks from smaller crops
            scores = 1 / _box_area(data["crop_boxes"])
            boxes = ops.cast(data["boxes"], "float32")
            scores = ops.cast(scores, "float32")
            keep_by_nms = _batched_nms(
                boxes,
                scores,
                iou_threshold=self.crop_nms_thresh,
                max_output_size=self.max_output_masks,
            )
            data.filter(keep_by_nms)

        data.to_numpy()

        return data

    def _process_crop(self, image, crop_box, crop_layer_idx, orig_size, **kwargs):
        # Crop the image and calculate embeddings
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]
        self.predictor.set_image(cropped_im, **kwargs)

        # Get points for this crop
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale

        # Generate masks for this crop in batches
        data = MaskData()
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(
                points, cropped_im_size, crop_box, orig_size, **kwargs
            )
            data.cat(batch_data)
            del batch_data
        self.predictor.reset_image()

        # Remove duplicates within this crop.
        keep_by_nms = _batched_nms(
            ops.cast(data["boxes"], "float32"),
            ops.cast(data["iou_preds"], "float32"),
            iou_threshold=self.box_nms_thresh,
            max_output_size=self.max_output_masks,
        )
        data.filter(keep_by_nms)

        # Return to the original image frame
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = ops.convert_to_tensor(
            [crop_box for _ in range(len(data["rles"]))]
        )

        return data

    def _process_batch(self, points, im_size, crop_box, orig_size, **kwargs):
        orig_h, orig_w = orig_size

        # Run model on this batch
        transformed_points = self.predictor.transform.apply_coords(points, im_size)
        in_points = ops.convert_to_tensor(transformed_points)[:, None, :]
        B = in_points.shape[0]
        in_labels = ops.ones(B, dtype="int32")[:, None]
        in_points = ops.concatenate(
            [in_points, ops.zeros((B, 1, 2), dtype=in_points.dtype)], axis=1
        )
        in_labels = ops.concatenate(
            [in_labels, -ops.ones((B, 1), dtype=in_labels.dtype)], axis=1
        )
        out = self.predictor.predict(
            dict(point_coords=in_points, point_labels=in_labels),
            multimask_output=True,
            **kwargs,
        )
        masks, iou_preds = out["masks"], out["iou_predictions"]

        # Serialize predictions and store in MaskData
        masks, iou_preds, points = map(
            ops.convert_to_tensor, [masks, iou_preds, points]
        )
        data = MaskData(
            masks=ops.reshape(masks, (-1, *masks.shape[2:])),
            iou_preds=ops.reshape(iou_preds, (-1, *iou_preds.shape[2:])),
            points=ops.convert_to_tensor(ops.repeat(points, masks.shape[1], axis=0)),
        )
        del masks

        # Filter by predicted IoU
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            data.filter(keep_mask)

        # Calculate stability score
        data["stability_score"] = calculate_stability_score(
            data["masks"], self.predictor.mask_threshold, self.stability_score_offset
        )
        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            data.filter(keep_mask)

        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > self.predictor.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        # Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(
            data["boxes"], crop_box, [0, 0, orig_w, orig_h]
        )
        if not ops.all(keep_mask):
            data.filter(keep_mask)

        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_tensor(data["masks"])
        del data["masks"]

        return data

    def postprocess_small_regions(
        self, mask_data: MaskData, min_area: int, nms_thresh: float
    ) -> MaskData:
        """
        Removes small disconnected regions and holes in masks, then reruns
        box NMS to remove any new duplicates.

        Edits mask_data in place.

        Requires open-cv as a dependency.
        """
        if len(mask_data["rles"]) == 0:
            return mask_data

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = rle_to_mask(rle)

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(ops.convert_to_tensor(mask)[None, ...])
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = ops.concatenate(new_masks, axis=0)
        scores = ops.convert_to_tensor(scores, "float32")
        boxes = batched_mask_to_box(masks)
        keep_by_nms = _batched_nms(
            ops.cast(boxes, "float32"),
            scores,
            iou_threshold=nms_thresh,
            max_output_size=self.max_output_masks,
        )

        # We update the boxes directly in the loop below.
        # Copy the boxes data since, for the tensorflow backend, Keras 3 returns
        # readonly arrays which can't be mutated in-place.
        mask_data["boxes"] = mask_data["boxes"].copy()

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_tensor = masks[i_mask][None, ...]
                mask_data["rles"][i_mask] = mask_to_rle_tensor(mask_tensor)[0]
                mask_data["boxes"][i_mask] = ops.convert_to_numpy(
                    boxes[i_mask]
                )  # update res directly
        mask_data.filter(keep_by_nms)

        return mask_data
