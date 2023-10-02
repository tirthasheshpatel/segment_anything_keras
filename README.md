# Segment Anything Model in Multi-Backend Keras

This is an implementation of the Segment Anything predictor and automatic mask
generator in Keras Core.

The demos uses KerasCV's Segment Anything model. Note that we depend on the
KerasCV's source directly until v0.7.0 has branched.

- [Predictor demo](Segment_Anything_multi_backend_Keras_Demo.ipynb)
- [Atomatic Mask Generator demo](Segment_Anything_Automatic_Mask_Generator_Demo.ipynb)

## Install the package

```shell
pip install git+https://github.com/tirthasheshpatel/segment_anything_keras.git
```

Install the required dependencies:

```shell
pip install Pillow numpy keras-core git+https://github.com/keras-team/keras-cv.git
```

Install TensorFlow, JAX, or PyTorch, whichever backend you'd like to use.

To get all the dependencies and all the backends to run the demos, do:

```shell
pip install -r requirements.txt
```

## Getting the pretrained Segment Anything Model

```python
# Use TensorFlow backend, choose any you want
import os
os.environ['KERAS_BACKEND'] = "tensorflow"

from keras_cv.models import SegmentAnythingModel
from sam_keras import SAMPredictor

# Get the huge model trained on the SA-1B dataset.
# Other available options are:
#   - "sam_base_sa1b"
#   - "sam_large_sa1b"
model = SegmentAnythingModel.from_preset("sam_huge_sa1b")

# Create the predictor
predictor = SAMPredictor(model)

# Now you can use the predictor just like the one on the original repo.
# The only difference is list of input dicts isn't supported; instead
# pass each input dict separately to the `predict` method.
```

## Notes

Right now JAX and TensorFlow have large compile-time overhead. Prompt encoder
recompiles each time a different combination of prompts (points only,
points + boxes, boxes only, etc) is passed. To avoid this, compile the model
with `run_eagerly=True`.

## Benchmarks

TODO
