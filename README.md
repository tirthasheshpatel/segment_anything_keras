# Segment Anything Model in Multi-Backend Keras

This is an implementation of the Segment Anything predictor and automatic mask
generator in Keras 3.

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
pip install Pillow numpy keras-nightly git+https://github.com/keras-team/keras-cv.git
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

All the benchmarks were run in Colab with following configurations:

- For A100: 40 GB GPU RAM, 83.5 GB CPU RAM
- For V100: 16 GB GPU RAM, 51 GB CPU RAM

| Model                 | Device   | End-To-End Huge              | End-to-End Large              |  End-to-End Base              | Fixed Image              |
| --------------------- | -------- | ---------------------------- | ----------------------------- | ----------------------------- | ------------------------ |
| PyTorch Native        | A100     | 445 ms ± 4.76 ms             | 272 ms ± 3.73 ms              | 126 ms ± 624 µs               | 8.54 ms ± 53.2 µs        |
| PyTorch (Keras 3)     | A100     | 482 ms ± 1.86 ms             | 293 ms ± 1.82 ms              | 146 ms ± 907 µs               | 36.4 ms ± 424 µs         |
| TensorFlow (Keras 3)  | A100     | 197 ms ± 2.12 ms             | 158 ms ± 1.05 ms              | 124 ms ± 577 µs               | 76.1 ms ± 515 µs         |
| **JAX (Keras 3)**     | **A100** | **125 ms ± 476 µs**          | **84.8 ms ± 193 µs**          | **44.2 ms ± 210 µs**          | **6.78 ms ± 135 µs**     |
| PyTorch Native        | V100     | 585 ms ± 3.67 ms             | 339 ms ± 1.2 ms               | 153 ms ± 575 µs               | 8.54 ms ± 266 µs         |
| PyTorch (Keras 3)     | V100     | 616 ms ± 1.22 ms             | 365 ms ± 2.52 ms              | 153 ms ± 575 µs               | 37.6 ms ± 1.09 ms        |
| TensorFlow (Keras 3)  | V100     | 585 ms ± 4.91 ms             | 380 ms ± 2.71 ms              | 205 ms ± 3.25 ms              | 79 ms ± 1.72 ms          |
| Jax (Keras 3)         | V100     | 545 ms ± 3.02 ms             | 313 ms ± 1.07 ms              | 125 ms ± 441 µs               | 7.17 ms ± 101 µs         |
