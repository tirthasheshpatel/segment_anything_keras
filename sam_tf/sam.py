from keras_cv.backend import keras
from keras_cv.backend import ops


class SegmentAnythingModel(keras.models.Model):
    mask_threshold = 0.0
    image_format = "RGB"

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
        self.pixel_mean = ops.array(pixel_mean, dtype=self.dtype)
        self.pixel_std = ops.array(pixel_std, dtype=self.dtype)

    def call(self, batched_input, multimask_output=True):
        images = ops.concatenate(
            [self.preprocess_images(x["image"]) for x in batched_input], axis=0
        )
        # TODO: remove this atrocity once unstack is added in keras core
        image_encodings = self.image_encoder(images)
        if keras.backend.backend() == "tensorflow":
            import tensorflow as tf

            image_encodings = tf.unstack(image_encodings, axis=0)
            del tf
        elif keras.backend.backend() == "torch":
            image_encodings = image_encodings.unbind(0)
        elif keras.backend.backend() == "jax":
            import jax

            image_encodings = [
                jax.lax.index_in_dim(image_encodings, i, 0, keepdims=False)
                for i in range(image_encodings.shape[0])
            ]
            del jax

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
            # masks = masks > self.mask_threshold
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
