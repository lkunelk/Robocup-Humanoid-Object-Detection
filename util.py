import numpy as np


def torch_to_cv(img):
    assert img.shape[0] <= 3  # first dim is color channel
    return np.moveaxis(img, 0, -1)


def cv_to_torch(img):
    assert img.shape[2]  # last dim is color channel
    return np.moveaxis(img, 2, 0)
