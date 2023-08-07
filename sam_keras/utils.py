# Author: Tirth Patel (tirthasheshpatel@gmail.com)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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
