# Author: Tirth Patel (tirthasheshpatel@gmail.com)

import numpy as np
import keras
from keras import ops


__all__ = ["SAMPrompter"]


class SAMPrompter(keras.Model):
    def __init__(
        self, prompt_encoder, mask_decoder, feature_shape=(64, 64, 256), **kwargs
    ):
        # Define the prompt encoder inputs -- Prompts
        prompt_inputs = {
            "points": keras.Input(shape=[None, 2], name="points"),
            "labels": keras.Input(shape=[None], name="labels"),
            "boxes": keras.Input(shape=[None, 2, 2], name="boxes"),
            "masks": keras.Input(shape=[None, None, None, 1], name="masks"),
        }

        # All Inputs -- Features + Prompts
        all_inputs = {"images": keras.Input(feature_shape, name="images")}
        all_inputs.update(prompt_inputs)

        # Build the prompt encoder
        prompt_embeddings = prompt_encoder(prompt_inputs)

        # Define the mask decoder inputs
        mask_decoder_inputs = {
            "image_embeddings": all_inputs["images"],
            "image_pe": prompt_embeddings["dense_positional_embeddings"],
            "sparse_prompt_embeddings": prompt_embeddings["sparse_embeddings"],
            "dense_prompt_embeddings": prompt_embeddings["dense_embeddings"],
        }

        # Build the mask decoder
        outputs = mask_decoder(mask_decoder_inputs)

        super().__init__(inputs=all_inputs, outputs=outputs, **kwargs)

        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

    def predict_step(self, *args, **kwargs):
        if len(args) == 2:
            args = (args[0], _add_placeholder_prompts(args[-1]))
        else:
            args = (_add_placeholder_prompts(args[0]),)

        return super().predict_step(*args, **kwargs)


def _add_placeholder_prompts(inputs):
    """Adds placeholder prompt inputs for a call to SAM.

    Because SAM is a functional subclass model, all inputs must be specified in
    calls to the model. However, prompt inputs are all optional, so we have to
    add placeholders when they're not specified by the user.
    """
    inputs = inputs.copy()

    # Get the batch shape based on the image input
    B = ops.shape(inputs["images"])[0]

    # The type of the placeholders must match the existing inputs with respect
    # to whether or not they are tensors (as opposed to Numpy arrays).
    zeros = ops.zeros if ops.is_tensor(inputs["images"]) else np.zeros

    # Fill in missing inputs.
    if "points" not in inputs:
        inputs["points"] = zeros((B, 0, 2))
    if "labels" not in inputs:
        inputs["labels"] = zeros((B, 0))
    if "boxes" not in inputs:
        inputs["boxes"] = zeros((B, 0, 2, 2))
    if "masks" not in inputs:
        inputs["masks"] = zeros((B, 0, 256, 256, 1))

    return inputs
