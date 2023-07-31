import numpy as np
from keras_cv.backend import ops
from PIL import Image


# Re-implementation of https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/utils/transforms.py#L16 in TensorFlow
class ResizeLongestSide:
    def __init__(self, target_length):
        self.target_length = target_length

    def apply_image(self, image):
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1])
        return np.array(Image.fromarray(image).resize(target_size[::-1], resample=Image.Resampling.BILINEAR))

    def apply_coords(self, coords, original_size):
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(original_size[0], original_size[1])
        coords = ops.cast(coords, "float32")
        coords_x = coords[..., 0] * (new_w / old_w)
        coords_y = coords[..., 1] * (new_h / old_h)
        return ops.stack([coords_x, coords_y], axis=-1)

    def apply_boxes(self, boxes, original_size):
        boxes = self.apply_coords(
            ops.reshape(
                boxes, (-1, 2, 2)
            ),
            original_size
        )
        return boxes

    def get_preprocess_shape(self, old_h, old_w):
        scale = self.target_length * 1.0 / max(old_h, old_w)
        new_h = old_h * scale
        new_w = old_w * scale
        return int(new_h + 0.5), int(new_w + 0.5)


class SegmentAnythingModel:
    mask_threshold = 0.0

    def __init__(
        self,
        image_encoder,
        prompt_encoder,
        mask_decoder,
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.pixel_mean = ops.array(pixel_mean, dtype="float32")
        self.pixel_std = ops.array(pixel_std, dtype="float32")
        self.transform = ResizeLongestSide(image_encoder.img_size)
        self.reset_image()

    def set_image(self, image):
        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)
        input_image_tensor = ops.convert_to_tensor(input_image, dtype="float32")
        input_image_tensor = input_image_tensor[None, :, :, :]

        self.set_tensor_image(input_image_tensor, image.shape[:2])

    def set_tensor_image(self, transformed_image, original_image_size):
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method. Expects the input
        image to be already transformed to the format expected by the model.

        Arguments:
          transformed_image (torch.Tensor): The input image, with shape
            1x3xHxW, which has been transformed with ResizeLongestSide.
          original_image_size (tuple(int, int)): The size of the image
            before transformation, in (H, W) format.
        """
        self.reset_image()

        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        self.unprocessed_image = transformed_image
        input_image = self.preprocess_images(transformed_image)
        self.features = self.image_encoder(input_image)
        self.is_image_set = True

    def predict(self, batched_input, multimask_output=True, return_logits=True):
        if isinstance(batched_input, list):
            images = ops.concatenate(
                [self.preprocess_images(x["image"]) for x in batched_input], axis=0
            )
            features = self.image_encoder(images)
        else:
            batched_input["image"] = self.unprocessed_image
            batched_input["original_size"] = self.original_size
            features = self.features
            batched_input = [batched_input]

        image_encodings = ops.unstack(features, axis=0)

        outputs = []
        for image_record, image_encoded in zip(batched_input, image_encodings):
            if "point_coords" in image_record:
                points = image_record["point_coords"]
                labels = image_record["point_labels"]
            else:
                points, labels = None, None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                labels=labels,
                box=image_record.get("boxes", None),
                mask=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_scores = self.mask_decoder(
                image_embeddings=image_encoded[None, ...],
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[1:3],
                original_size=image_record["original_size"],
            )
            if not return_logits:
                masks = ops.cast(masks > self.mask_threshold, dtype="float32")
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_scores,
                    "low_res_masks": low_res_masks,
                }
            )
        return outputs

    def postprocess_masks(self, masks, input_size, original_size):
        masks = ops.image.resize(
            ops.transpose(masks, axes=(0, 2, 3, 1)),
            size=(self.image_encoder.img_size, self.image_encoder.img_size),
            interpolation="bilinear",
        )
        masks = masks[..., : input_size[0], : input_size[1], :]
        masks = ops.image.resize(
            masks, size=original_size, interpolation="bilinear"
        )
        return ops.transpose(masks, axes=(0, 3, 1, 2))

    def preprocess_images(self, x):
        x = (x - self.pixel_mean) / self.pixel_std

        h, w = x.shape[1:3]
        pad_h = self.image_encoder.img_size - h
        pad_w = self.image_encoder.img_size - w
        x = ops.pad(x, [(0, 0), (0, pad_h), (0, pad_w), (0, 0)])
        return x

    def get_image_embedding(self):
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert self.features is not None, "Features must exist if an image has been set."
        return self.features

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.unprocessed_image = None
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None
