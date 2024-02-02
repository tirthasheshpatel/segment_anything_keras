# Author: Tirth Patel (tirthasheshpatel@gmail.com)

# Copyright (c) Tirth Patel.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

setup(
    name="sam_keras",
    version="0.0.1",
    # we also require keras-cv but since we depend on its source,
    # we expect the user to install it separately.
    install_requires=["numpy", "Pillow", "keras-cv"],
    packages=find_packages(),
    extras_require={
        "all": ["matplotlib", "pycocotools", "opencv-python", "tensorflow", "torch",
                "torchvision", "torchaudio", "jax", "jaxlib"],
    },
)
