from functools import wraps

from keras_cv.backend import keras


def _torch_no_grad(func):
    @wraps(func)
    def func_wrapper(*args, **kwargs):
        if keras.backend.backend() == "torch":
            import torch
            with torch.no_grad():
                return func(*args, **kwargs)
        return func(*args, **kwargs)
    return func_wrapper
