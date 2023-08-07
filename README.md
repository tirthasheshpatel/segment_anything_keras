# Segment Anything Model in Multi-Backend Keras

This is an implementation of the Segment Anything model in Keras Core.

The model currently runs in TensorFlow, JAX, PyTorch, and NumPy, and supports inference.

Check out the demos for using the model and porting over the weights from PyTorch:

- [Predictor demo](Segment_Anything_multi_backend_Keras_Demo.ipynb)
- [Atomatic Mask Generator demo](Segment_Anything_Automatic_Mask_Generator_Demo.ipynb)

Note on JAX automatic mask generator support: Since JAX doesn't offer an optimized implementation of the NMS op, it is slower than other backends.

**Note: This project will be merged with KerasCV in the future.**
