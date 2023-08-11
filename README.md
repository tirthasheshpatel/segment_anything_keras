# Segment Anything Model in Multi-Backend Keras

This is an implementation of the Segment Anything model in Keras Core.

The model currently runs in TensorFlow, JAX, PyTorch, and NumPy, and supports inference.

Check out the demos for using the model and porting over the weights from PyTorch:

- [Predictor demo](Segment_Anything_multi_backend_Keras_Demo.ipynb)
- [Atomatic Mask Generator demo](Segment_Anything_Automatic_Mask_Generator_Demo.ipynb)

Note on JAX automatic mask generator support: Since JAX doesn't offer an optimized implementation of the NMS op, it is slower than other backends.

**Note: This project will be merged with KerasCV in the future.**

## Install the package

```shell
pip install git+https://github.com/tirthasheshpatel/segment_anything_keras.git
```

## Port Weights

```python
# Use TensorFlow backend, choose any you want
import os
os.environ['KERAS_BACKEND'] = "tensorflow"

# torch model
import torch
from segment_anything.build_sam import build_sam_vit_h
from segment_anything.modeling import Sam
from segment_anything import sam_model_registry, SamPredictor
from sam_keras import port_weights
from sam_keras import build_sam_huge

# Define the huge model in Keras Core
model = build_model_huge()

# Create a predictor to port the weights from PyTorch to TensorFlow
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

# Port the PyTorch model's weights to the multi-backend Keras model
port_weights(model, predictor.model)

# Save the image_encoder, prompt_encoder and the mask_decoder
model.image_encoder.save_weights("sam_vitdet_huge.weights.h5")
model.prompt_encoder.save_weights("sam_prompt_encoder.weights.h5")
model.mask_decoder.save_weights("sam_mask_decoder.weights.h5")
```

## Load the model in any backend

Once the model is saved, it can be loaded into any backend!
For example, in the above example, the model is saved in TensorFlow
but it can be loaded in JAX, PyTorch, and NumPy!

```python
import os
os.environ['KERAS_BACKEND'] = "jax"

from sam_keras import build_sam_huge

# Define the huge model in Keras Core
model = build_model_huge()

# Load the image_encoder, prompt_encoder and the mask_decoder
model.image_encoder.load_weights("sam_vitdet_huge.weights.h5")
model.prompt_encoder.load_weights("sam_prompt_encoder.weights.h5")
model.mask_decoder.load_weights("sam_mask_decoder.weights.h5")
```
