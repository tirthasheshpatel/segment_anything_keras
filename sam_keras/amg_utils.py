# Author: Tirth Patel (tirthasheshpatel@gmail.com)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from keras import ops

import math
from copy import deepcopy
from itertools import product


class MaskData:
    """
    A structure for storing masks and their related data in batched format.
    Implements basic filtering and concatenation.
    """

    def __init__(self, **kwargs):
        for v in kwargs.values():
            if (
                not isinstance(v, list)
                and not ops.is_tensor(v)
                and not isinstance(v, np.ndarray)
            ):
                raise ValueError(
                    "`MaskData` only supports `list`, tensors, and numpy arrays."
                )
        self._stats = dict(**kwargs)

    def __setitem__(self, key, item):
        if (
            not isinstance(item, list)
            and not ops.is_tensor(item)
            and not isinstance(item, np.ndarray)
        ):
            raise ValueError(
                "`MaskData` only supports `list`, tensors, and numpy arrays."
            )
        self._stats[key] = item

    def __delitem__(self, key):
        del self._stats[key]

    def __getitem__(self, key):
        return self._stats[key]

    def items(self):
        return self._stats.items()

    def filter(self, keep):
        for k, v in self._stats.items():
            if v is None:
                self._stats[k] = None
            elif ops.is_tensor(v):
                if "bool" in str(keep.dtype):
                    self._stats[k] = v[keep]
                else:
                    self._stats[k] = ops.take(v, keep, 0)
            elif isinstance(v, np.ndarray):
                self._stats[k] = v[ops.convert_to_numpy(keep)]
            elif isinstance(v, list) and "bool" in str(keep.dtype):
                self._stats[k] = [a for i, a in enumerate(v) if keep[i]]
            elif isinstance(v, list):
                self._stats[k] = [v[i] for i in keep]
            else:
                raise TypeError(f"MaskData key {k} has an unsupported type {type(v)}.")

    def cat(self, new_stats):
        for k, v in new_stats.items():
            if k not in self._stats or self._stats[k] is None:
                self._stats[k] = deepcopy(v)
            elif ops.is_tensor(v):
                self._stats[k] = ops.concatenate([self._stats[k], v], axis=0)
            elif isinstance(v, np.ndarray):
                self._stats[k] = np.concatenate([self._stats[k], v], axis=0)
            elif isinstance(v, list):
                self._stats[k] = self._stats[k] + v
            else:
                raise TypeError(f"MaskData key {k} has an unsupported type {type(v)}.")

    def to_numpy(self):
        for k, v in self._stats.items():
            if ops.is_tensor(v):
                self._stats[k] = ops.convert_to_numpy(v)


def _isclose(x1, x2, atol, rtol):
    x1 = ops.convert_to_numpy(x1)
    x2 = ops.convert_to_numpy(x2)
    return ops.convert_to_tensor(np.isclose(x1, x2, rtol=rtol, atol=atol))


def is_box_near_crop_edge(boxes, crop_box, orig_box, atol=20.0):
    """Filter masks at the edge of a crop, but not at the edge of the original image."""
    crop_box_torch = ops.convert_to_tensor(crop_box, dtype="float32")
    orig_box_torch = ops.convert_to_tensor(orig_box, dtype="float32")
    boxes = ops.cast(uncrop_boxes_xyxy(boxes, crop_box), dtype="float32")
    near_crop_edge = _isclose(boxes, crop_box_torch[None, :], atol=atol, rtol=0)
    near_image_edge = _isclose(boxes, orig_box_torch[None, :], atol=atol, rtol=0)
    near_crop_edge = near_crop_edge & (~near_image_edge)
    return ops.any(near_crop_edge, axis=1)


def box_xyxy_to_xywh(box_xyxy, axis=-1):
    box_xyxy = ops.moveaxis(box_xyxy, axis, 0)
    box_xywh = ops.stack(
        [
            box_xyxy[0],
            box_xyxy[1],
            box_xyxy[2] - box_xyxy[0],
            box_xyxy[3] - box_xyxy[1],
        ],
        axis=0,
    )
    return ops.moveaxis(box_xywh, 0, axis)


def box_xyxy_to_yxyx(box_xyxy, axis=-1):
    box_xyxy = ops.moveaxis(box_xyxy, axis, 0)
    box_yxyx = ops.stack(
        [
            box_xyxy[1],
            box_xyxy[0],
            box_xyxy[3],
            box_xyxy[2],
        ],
        axis=0,
    )
    return ops.moveaxis(box_yxyx, 0, axis)


def batch_iterator(batch_size: int, *args):
    if not len(args) > 0 or not all(len(a) == len(args[0]) for a in args):
        raise ValueError("Batched iteration must have inputs of all the same size.")
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]


def mask_to_rle_tensor(tensor):
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    tensor = ops.convert_to_numpy(tensor)
    # Put in fortran order and flatten h,w
    b, h, w = tensor.shape
    tensor = np.reshape(np.transpose(tensor, axes=(0, 2, 1)), (b, w * h))

    # Compute change indices
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = np.stack(np.nonzero(diff), 1)

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = np.concatenate(
            [
                np.array([0], dtype=cur_idxs.dtype),
                cur_idxs + 1,
                np.array([h * w], dtype=cur_idxs.dtype),
            ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == False else [0]
        counts.extend(list(btw_idxs))
        out.append({"size": [h, w], "counts": counts})
    return out


def rle_to_mask(rle):
    """Compute a binary mask from an uncompressed RLE."""
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity ^= True
    mask = mask.reshape(w, h)
    return mask.transpose()


def area_from_rle(rle):
    return sum(rle["counts"][1::2])


def calculate_stability_score(masks, mask_threshold, threshold_offset):
    """
    Computes the stability score for a batch of masks. The stability
    score is the IoU between the binary masks obtained by thresholding
    the predicted mask logits at high and low values.
    """
    # One mask is always contained inside the other.
    # Save memory by preventing unnecessary cast to torch.int64
    intersections = ops.sum(
        ops.sum(
            ops.cast(masks > (mask_threshold + threshold_offset), dtype="float32"), -1
        ),
        -1,
    )
    unions = ops.sum(
        ops.sum(
            ops.cast(masks > (mask_threshold - threshold_offset), dtype="float32"), -1
        ),
        -1,
    )
    return intersections / unions


def build_point_grid(n_per_side):
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points


def build_all_layer_point_grids(n_per_side, n_layers, scale_per_layer):
    """Generates point grids for all crop layers."""
    points_by_layer = []
    for i in range(n_layers + 1):
        n_points = int(n_per_side / (scale_per_layer**i))
        points_by_layer.append(build_point_grid(n_points))
    return points_by_layer


def generate_crop_boxes(im_size, n_layers, overlap_ratio):
    """
    Generates a list of crop boxes of different sizes. Each layer
    has (2**i)**2 boxes for the ith layer.
    """
    crop_boxes, layer_idxs = [], []
    im_h, im_w = im_size
    short_side = min(im_h, im_w)

    # Original image
    crop_boxes.append([0, 0, im_w, im_h])
    layer_idxs.append(0)

    def crop_len(orig_len, n_crops, overlap):
        return int(math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops))

    for i_layer in range(n_layers):
        n_crops_per_side = 2 ** (i_layer + 1)
        overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))

        crop_w = crop_len(im_w, n_crops_per_side, overlap)
        crop_h = crop_len(im_h, n_crops_per_side, overlap)

        crop_box_x0 = [int((crop_w - overlap) * i) for i in range(n_crops_per_side)]
        crop_box_y0 = [int((crop_h - overlap) * i) for i in range(n_crops_per_side)]

        # Crops in XYWH format
        for x0, y0 in product(crop_box_x0, crop_box_y0):
            box = [x0, y0, min(x0 + crop_w, im_w), min(y0 + crop_h, im_h)]
            crop_boxes.append(box)
            layer_idxs.append(i_layer + 1)

    return crop_boxes, layer_idxs


def uncrop_boxes_xyxy(boxes, crop_box):
    x0, y0, _, _ = crop_box
    boxes = ops.cast(boxes, "float32")
    offset = ops.convert_to_tensor([[x0, y0, x0, y0]], dtype="float32")
    # Check if boxes has a channel dimension
    if len(boxes.shape) == 3:
        offset = offset[:, None, ...]
    return boxes + offset


def uncrop_points(points, crop_box):
    x0, y0, _, _ = crop_box
    points = ops.cast(points, "float32")
    offset = ops.convert_to_tensor([[x0, y0]], dtype="float32")
    # Check if points has a channel dimension
    if len(points.shape) == 3:
        offset = offset[:, None, ...]
    return points + offset


def uncrop_masks(masks, crop_box, orig_h, orig_w):
    x0, y0, x1, y1 = crop_box
    if x0 == 0 and y0 == 0 and x1 == orig_w and y1 == orig_h:
        return masks
    # Coordinate transform masks
    pad = [(0, 0)] * len(masks.shape[:-2])
    pad_x, pad_y = orig_w - (x1 - x0), orig_h - (y1 - y0)
    pad = pad + [(y0, pad_y - y0), (x0, pad_x - x0)]
    return ops.pad(masks, pad)


def remove_small_regions(mask, area_thresh, mode):
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    """
    import cv2

    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # Row 0 is background label
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # If every region is below threshold, keep largest
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True


def coco_encode_rle(uncompressed_rle):
    from pycocotools import mask as mask_utils

    h, w = uncompressed_rle["size"]
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    rle["counts"] = rle["counts"].decode("utf-8")  # Necessary to serialize with json
    return rle


def batched_mask_to_box(masks):
    """
    Calculates boxes in XYXY format around masks. Return [0,0,0,0] for
    an empty mask. For input shape C1xC2x...xHxW, the output shape is C1xC2x...x4.
    """
    if ops.size(masks) == 0:
        return ops.zeros((*masks.shape[:-2], 4))

    # Normalize shape to CxHxW
    shape = masks.shape
    h, w = shape[-2:]
    if len(shape) > 2:
        masks = ops.reshape(masks, (-1, *masks.shape[-2:]))
    else:
        masks = masks[None, ...]

    # Get top and bottom edges
    in_height = ops.max(masks, axis=-1)
    in_height_coords = (
        ops.cast(in_height, "float32") * ops.arange(h, dtype="float32")[None, :]
    )
    bottom_edges = ops.max(in_height_coords, axis=-1)
    in_height_coords = in_height_coords + h * ops.cast(~in_height, "float32")
    top_edges = ops.min(in_height_coords, axis=-1)

    # Get left and right edges
    in_width = ops.max(masks, axis=-2)
    in_width_coords = (
        ops.cast(in_width, "float32") * ops.arange(w, dtype="float32")[None, :]
    )
    right_edges = ops.max(in_width_coords, axis=-1)
    in_width_coords = in_width_coords + w * ops.cast(~in_width, "float32")
    left_edges = ops.min(in_width_coords, axis=-1)

    # If the mask is empty the right edge will be to the left of the left edge.
    # Replace these boxes with [0, 0, 0, 0]
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = ops.stack([left_edges, top_edges, right_edges, bottom_edges], axis=-1)
    out = out * ops.cast((~empty_filter)[..., None], "float32")

    # Return to original shape
    if len(shape) > 2:
        out = ops.reshape(out, (*shape[:-2], 4))
    else:
        out = out[0]

    return out
