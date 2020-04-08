import numpy as np
import torch


def torch_to_cv(img):
    if isinstance(img, np.ndarray):
        return np.moveaxis(img, 0, -1)

    img = np.array(img.detach().cpu())
    if len(img.shape) >= 3:
        assert img.shape[0] <= 3  # first dim is color channel
        return np.moveaxis(img, 0, -1)
    else:
        assert len(img.shape) == 2  # gray scale
        return np.squeeze(img)


def cv_to_torch(img):
    assert img.shape[2]  # last dim is color channel
    return np.moveaxis(img, 2, 0)
