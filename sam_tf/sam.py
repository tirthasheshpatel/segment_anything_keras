import tensorflow as tf
from tensorflow.keras import models


class SegmentAnythingModel(models.Model):
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
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std

    def call(self, x, points, labels, box, mask):
        image_encoded = self.image_encoder(x)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points, labels=labels, box=box, mask=mask
        )
        masks, iou_scores = self.mask_decoder(
            image_embeddings=image_encoded,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return self.postprocess_masks(masks), iou_scores

    def postprocess_masks(self, masks):
        return tf.transpose(
            tf.image.resize(
                tf.transpose(masks, perm=(0, 2, 3, 1)),
                size=(1024, 1024),
                method="bilinear",
            ),
            perm=(0, 3, 1, 2),
        )

    def preprocess_images(self, x):
        return (x - self.pixel_mean) / self.pixel_std
