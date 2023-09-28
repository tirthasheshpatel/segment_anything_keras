# Taken from https://github.com/mlperf/training_results_v0.7/blob/3dbb53064a6b79354c68a6832414b6536fee1a75/Google/benchmarks/ssd/implementations/ssd-research-JAX-tpu-v3-4096/nms.py
# See https://github.com/google/flax/discussions/1929#discussioncomment-2378312
#
#                                  Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/
#
#    TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
#
#    1. Definitions.
#
#       "License" shall mean the terms and conditions for use, reproduction,
#       and distribution as defined by Sections 1 through 9 of this document.
#
#       "Licensor" shall mean the copyright owner or entity authorized by
#       the copyright owner that is granting the License.
#
#       "Legal Entity" shall mean the union of the acting entity and all
#       other entities that control, are controlled by, or are under common
#       control with that entity. For the purposes of this definition,
#       "control" means (i) the power, direct or indirect, to cause the
#       direction or management of such entity, whether by contract or
#       otherwise, or (ii) ownership of fifty percent (50%) or more of the
#       outstanding shares, or (iii) beneficial ownership of such entity.
#
#       "You" (or "Your") shall mean an individual or Legal Entity
#       exercising permissions granted by this License.
#
#       "Source" form shall mean the preferred form for making modifications,
#       including but not limited to software source code, documentation
#       source, and configuration files.
#
#       "Object" form shall mean any form resulting from mechanical
#       transformation or translation of a Source form, including but
#       not limited to compiled object code, generated documentation,
#       and conversions to other media types.
#
#       "Work" shall mean the work of authorship, whether in Source or
#       Object form, made available under the License, as indicated by a
#       copyright notice that is included in or attached to the work
#       (an example is provided in the Appendix below).
#
#       "Derivative Works" shall mean any work, whether in Source or Object
#       form, that is based on (or derived from) the Work and for which the
#       editorial revisions, annotations, elaborations, or other modifications
#       represent, as a whole, an original work of authorship. For the purposes
#       of this License, Derivative Works shall not include works that remain
#       separable from, or merely link (or bind by name) to the interfaces of,
#       the Work and Derivative Works thereof.
#
#       "Contribution" shall mean any work of authorship, including
#       the original version of the Work and any modifications or additions
#       to that Work or Derivative Works thereof, that is intentionally
#       submitted to Licensor for inclusion in the Work by the copyright owner
#       or by an individual or Legal Entity authorized to submit on behalf of
#       the copyright owner. For the purposes of this definition, "submitted"
#       means any form of electronic, verbal, or written communication sent
#       to the Licensor or its representatives, including but not limited to
#       communication on electronic mailing lists, source code control systems,
#       and issue tracking systems that are managed by, or on behalf of, the
#       Licensor for the purpose of discussing and improving the Work, but
#       excluding communication that is conspicuously marked or otherwise
#       designated in writing by the copyright owner as "Not a Contribution."
#
#       "Contributor" shall mean Licensor and any individual or Legal Entity
#       on behalf of whom a Contribution has been received by Licensor and
#       subsequently incorporated within the Work.
#
#    2. Grant of Copyright License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       copyright license to reproduce, prepare Derivative Works of,
#       publicly display, publicly perform, sublicense, and distribute the
#       Work and such Derivative Works in Source or Object form.
#
#    3. Grant of Patent License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       (except as stated in this section) patent license to make, have made,
#       use, offer to sell, sell, import, and otherwise transfer the Work,
#       where such license applies only to those patent claims licensable
#       by such Contributor that are necessarily infringed by their
#       Contribution(s) alone or by combination of their Contribution(s)
#       with the Work to which such Contribution(s) was submitted. If You
#       institute patent litigation against any entity (including a
#       cross-claim or counterclaim in a lawsuit) alleging that the Work
#       or a Contribution incorporated within the Work constitutes direct
#       or contributory patent infringement, then any patent licenses
#       granted to You under this License for that Work shall terminate
#       as of the date such litigation is filed.
#
#    4. Redistribution. You may reproduce and distribute copies of the
#       Work or Derivative Works thereof in any medium, with or without
#       modifications, and in Source or Object form, provided that You
#       meet the following conditions:
#
#       (a) You must give any other recipients of the Work or
#           Derivative Works a copy of this License; and
#
#       (b) You must cause any modified files to carry prominent notices
#           stating that You changed the files; and
#
#       (c) You must retain, in the Source form of any Derivative Works
#           that You distribute, all copyright, patent, trademark, and
#           attribution notices from the Source form of the Work,
#           excluding those notices that do not pertain to any part of
#           the Derivative Works; and
#
#       (d) If the Work includes a "NOTICE" text file as part of its
#           distribution, then any Derivative Works that You distribute must
#           include a readable copy of the attribution notices contained
#           within such NOTICE file, excluding those notices that do not
#           pertain to any part of the Derivative Works, in at least one
#           of the following places: within a NOTICE text file distributed
#           as part of the Derivative Works; within the Source form or
#           documentation, if provided along with the Derivative Works; or,
#           within a display generated by the Derivative Works, if and
#           wherever such third-party notices normally appear. The contents
#           of the NOTICE file are for informational purposes only and
#           do not modify the License. You may add Your own attribution
#           notices within Derivative Works that You distribute, alongside
#           or as an addendum to the NOTICE text from the Work, provided
#           that such additional attribution notices cannot be construed
#           as modifying the License.
#
#       You may add Your own copyright statement to Your modifications and
#       may provide additional or different license terms and conditions
#       for use, reproduction, or distribution of Your modifications, or
#       for any such Derivative Works as a whole, provided Your use,
#       reproduction, and distribution of the Work otherwise complies with
#       the conditions stated in this License.
#
#    5. Submission of Contributions. Unless You explicitly state otherwise,
#       any Contribution intentionally submitted for inclusion in the Work
#       by You to the Licensor shall be under the terms and conditions of
#       this License, without any additional terms or conditions.
#       Notwithstanding the above, nothing herein shall supersede or modify
#       the terms of any separate license agreement you may have executed
#       with Licensor regarding such Contributions.
#
#    6. Trademarks. This License does not grant permission to use the trade
#       names, trademarks, service marks, or product names of the Licensor,
#       except as required for reasonable and customary use in describing the
#       origin of the Work and reproducing the content of the NOTICE file.
#
#    7. Disclaimer of Warranty. Unless required by applicable law or
#       agreed to in writing, Licensor provides the Work (and each
#       Contributor provides its Contributions) on an "AS IS" BASIS,
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#       implied, including, without limitation, any warranties or conditions
#       of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
#       PARTICULAR PURPOSE. You are solely responsible for determining the
#       appropriateness of using or redistributing the Work and assume any
#       risks associated with Your exercise of permissions under this License.
#
#    8. Limitation of Liability. In no event and under no legal theory,
#       whether in tort (including negligence), contract, or otherwise,
#       unless required by applicable law (such as deliberate and grossly
#       negligent acts) or agreed to in writing, shall any Contributor be
#       liable to You for damages, including any direct, indirect, special,
#       incidental, or consequential damages of any character arising as a
#       result of this License or out of the use or inability to use the
#       Work (including but not limited to damages for loss of goodwill,
#       work stoppage, computer failure or malfunction, or any and all
#       other commercial damages or losses), even if such Contributor
#       has been advised of the possibility of such damages.
#
#    9. Accepting Warranty or Additional Liability. While redistributing
#       the Work or Derivative Works thereof, You may choose to offer,
#       and charge a fee for, acceptance of support, warranty, indemnity,
#       or other liability obligations and/or rights consistent with this
#       License. However, in accepting such obligations, You may act only
#       on Your own behalf and on Your sole responsibility, not on behalf
#       of any other Contributor, and only if You agree to indemnify,
#       defend, and hold each Contributor harmless for any liability
#       incurred by, or claims asserted against, such Contributor by reason
#       of your accepting any such warranty or additional liability.
#
#    END OF TERMS AND CONDITIONS
#
#    APPENDIX: How to apply the Apache License to your work.
#
#       To apply the Apache License to your work, attach the following
#       boilerplate notice, with the fields enclosed by brackets "[]"
#       replaced with your own identifying information. (Don't include
#       the brackets!)  The text should be enclosed in the appropriate
#       comment syntax for the file format. We also recommend that a
#       file or class name and description of purpose be included on the
#       same "printed page" as the copyright notice for easier
#       identification within third-party archives.
#
#    Copyright 2018 The MLPerf Authors
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


"""Non-max Suppression example.

This script does non-max suppression used in models like SSD
"""

from jax import lax
import jax.numpy as jnp

_NMS_TILE_SIZE = 256


def _bbox_overlap(boxes, gt_boxes):
    """Find Bounding box overlap.

    Args:
      boxes: first set of bounding boxes
      gt_boxes: second set of boxes to compute IOU

    Returns:
      iou: Intersection over union matrix of all input bounding boxes
    """
    bb_y_min, bb_x_min, bb_y_max, bb_x_max = jnp.split(
        ary=boxes, indices_or_sections=4, axis=2
    )
    gt_y_min, gt_x_min, gt_y_max, gt_x_max = jnp.split(
        ary=gt_boxes, indices_or_sections=4, axis=2
    )

    # Calculates the intersection area.
    i_xmin = jnp.maximum(bb_x_min, jnp.transpose(gt_x_min, [0, 2, 1]))
    i_xmax = jnp.minimum(bb_x_max, jnp.transpose(gt_x_max, [0, 2, 1]))
    i_ymin = jnp.maximum(bb_y_min, jnp.transpose(gt_y_min, [0, 2, 1]))
    i_ymax = jnp.minimum(bb_y_max, jnp.transpose(gt_y_max, [0, 2, 1]))
    i_area = jnp.maximum((i_xmax - i_xmin), 0) * jnp.maximum((i_ymax - i_ymin), 0)

    # Calculates the union area.
    bb_area = (bb_y_max - bb_y_min) * (bb_x_max - bb_x_min)
    gt_area = (gt_y_max - gt_y_min) * (gt_x_max - gt_x_min)
    # Adds a small epsilon to avoid divide-by-zero.
    u_area = bb_area + jnp.transpose(gt_area, [0, 2, 1]) - i_area + 1e-8

    # Calculates IoU.
    iou = i_area / u_area

    return iou


def _self_suppression(in_args):
    iou, _, iou_sum = in_args
    batch_size = iou.shape[0]
    can_suppress_others = jnp.reshape(
        jnp.max(iou, 1) <= 0.5, [batch_size, -1, 1]
    ).astype(iou.dtype)
    iou_suppressed = (
        jnp.reshape(
            (jnp.max(can_suppress_others * iou, 1) <= 0.5).astype(iou.dtype),
            [batch_size, -1, 1],
        )
        * iou
    )
    iou_sum_new = jnp.sum(iou_suppressed, [1, 2])
    return iou_suppressed, jnp.any(iou_sum - iou_sum_new > 0.5), iou_sum_new


def _cross_suppression(in_args):
    boxes, box_slice, iou_threshold, inner_idx = in_args
    batch_size = boxes.shape[0]
    new_slice = lax.dynamic_slice(
        boxes, [0, inner_idx * _NMS_TILE_SIZE, 0], [batch_size, _NMS_TILE_SIZE, 4]
    )
    iou = _bbox_overlap(new_slice, box_slice)
    ret_slice = (
        jnp.expand_dims((jnp.all(iou < iou_threshold, [1])).astype(box_slice.dtype), 2)
        * box_slice
    )
    return boxes, ret_slice, iou_threshold, inner_idx + 1


def _suppression_loop_body(in_args):
    """Process boxes in the range [idx*_NMS_TILE_SIZE, (idx+1)*_NMS_TILE_SIZE).

    Args:
      in_args: A tuple of arguments: boxes, iou_threshold, output_size, idx

    Returns:
      boxes: updated boxes.
      iou_threshold: pass down iou_threshold to the next iteration.
      output_size: the updated output_size.
      idx: the updated induction variable.
    """
    boxes, iou_threshold, output_size, idx = in_args
    num_tiles = boxes.shape[1] // _NMS_TILE_SIZE
    batch_size = boxes.shape[0]

    # Iterates over tiles that can possibly suppress the current tile.
    box_slice = lax.dynamic_slice(
        boxes, [0, idx * _NMS_TILE_SIZE, 0], [batch_size, _NMS_TILE_SIZE, 4]
    )

    def _loop_cond(in_args):
        _, _, _, inner_idx = in_args
        return inner_idx < idx

    _, box_slice, _, _ = lax.while_loop(
        _loop_cond, _cross_suppression, (boxes, box_slice, iou_threshold, 0)
    )

    # Iterates over the current tile to compute self-suppression.
    iou = _bbox_overlap(box_slice, box_slice)
    mask = jnp.expand_dims(
        jnp.reshape(jnp.arange(_NMS_TILE_SIZE), [1, -1])
        > jnp.reshape(jnp.arange(_NMS_TILE_SIZE), [-1, 1]),
        0,
    )
    iou *= (jnp.logical_and(mask, iou >= iou_threshold)).astype(iou.dtype)

    def _loop_cond2(in_args):
        _, loop_condition, _ = in_args
        return loop_condition

    suppressed_iou, _, _ = lax.while_loop(
        _loop_cond2, _self_suppression, (iou, True, jnp.sum(iou, [1, 2]))
    )
    suppressed_box = jnp.sum(suppressed_iou, 1) > 0
    box_slice *= jnp.expand_dims(1.0 - suppressed_box.astype(box_slice.dtype), 2)

    # Uses box_slice to update the input boxes.
    mask = jnp.reshape(
        (jnp.equal(jnp.arange(num_tiles), idx)).astype(boxes.dtype), [1, -1, 1, 1]
    )
    boxes = jnp.tile(
        jnp.expand_dims(box_slice, 1), [1, num_tiles, 1, 1]
    ) * mask + jnp.reshape(boxes, [batch_size, num_tiles, _NMS_TILE_SIZE, 4]) * (
        1 - mask
    )
    boxes = jnp.reshape(boxes, [batch_size, -1, 4])

    # Updates output_size.
    output_size += jnp.sum(jnp.any(box_slice > 0, [2]).astype(jnp.int32), [1])
    return boxes, iou_threshold, output_size, idx + 1


def non_max_suppression_padded(scores, boxes, max_output_size, iou_threshold):
    """A wrapper that handles non-maximum suppression.

    Assumption:
      * The boxes are sorted by scores unless the box is a dot (all coordinates
        are zero).
      * Boxes with higher scores can be used to suppress boxes with lower scores.

    The overal design of the algorithm is to handle boxes tile-by-tile:

    boxes = boxes.pad_to_multiply_of(tile_size)
    num_tiles = len(boxes) // tile_size
    output_boxes = []
    for i in range(num_tiles):
      box_tile = boxes[i*tile_size : (i+1)*tile_size]
      for j in range(i - 1):
        suppressing_tile = boxes[j*tile_size : (j+1)*tile_size]
        iou = _bbox_overlap(box_tile, suppressing_tile)
        # if the box is suppressed in iou, clear it to a dot
        box_tile *= _update_boxes(iou)
      # Iteratively handle the diagnal tile.
      iou = _box_overlap(box_tile, box_tile)
      iou_changed = True
      while iou_changed:
        # boxes that are not suppressed by anything else
        suppressing_boxes = _get_suppressing_boxes(iou)
        # boxes that are suppressed by suppressing_boxes
        suppressed_boxes = _get_suppressed_boxes(iou, suppressing_boxes)
        # clear iou to 0 for boxes that are suppressed, as they cannot be used
        # to suppress other boxes any more
        new_iou = _clear_iou(iou, suppressed_boxes)
        iou_changed = (new_iou != iou)
        iou = new_iou
      # remaining boxes that can still suppress others, are selected boxes.
      output_boxes.append(_get_suppressing_boxes(iou))
      if len(output_boxes) >= max_output_size:
        break

    Args:
      scores: a tensor with a shape of [batch_size, anchors].
      boxes: a tensor with a shape of [batch_size, anchors, 4].
      max_output_size: a scalar integer `Tensor` representing the maximum number
        of boxes to be selected by non max suppression.
      iou_threshold: a float representing the threshold for deciding whether boxes
        overlap too much with respect to IOU.
    Returns:
      nms_scores: a tensor with a shape of [batch_size, anchors]. It has same
        dtype as input scores.
      nms_proposals: a tensor with a shape of [batch_size, anchors, 4]. It has
        same dtype as input boxes.
    """
    batch_size = boxes.shape[0]
    num_boxes = boxes.shape[1]
    pad = int(jnp.ceil(float(num_boxes) / _NMS_TILE_SIZE)) * _NMS_TILE_SIZE - num_boxes
    boxes = jnp.pad(boxes.astype(jnp.float32), [[0, 0], [0, pad], [0, 0]])
    scores = jnp.pad(scores.astype(jnp.float32), [[0, 0], [0, pad]])
    num_boxes += pad

    def _loop_cond(in_args):
        unused_boxes, unused_threshold, output_size, idx = in_args
        return jnp.logical_and(
            jnp.min(output_size) < max_output_size, idx < num_boxes // _NMS_TILE_SIZE
        )

    selected_boxes, _, output_size, _ = lax.while_loop(
        _loop_cond,
        _suppression_loop_body,
        (boxes, iou_threshold, jnp.zeros([batch_size], jnp.int32), 0),
    )
    idx = num_boxes - lax.top_k(
        jnp.any(selected_boxes > 0, [2]).astype(jnp.int32)
        * jnp.expand_dims(jnp.arange(num_boxes, 0, -1), 0),
        max_output_size,
    )[0].astype(jnp.int32)
    idx = jnp.minimum(idx, num_boxes - 1)
    idx = jnp.reshape(
        idx + jnp.reshape(jnp.arange(batch_size) * num_boxes, [-1, 1]), [-1]
    )
    return idx[idx < num_boxes - pad]
