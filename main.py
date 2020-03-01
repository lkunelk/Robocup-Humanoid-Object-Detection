import torch
import numpy as np
from model import CNN
import my_dataset
import matplotlib.pyplot as plt
import cv2

l = my_dataset.initialize_loader()
img, y = next(iter(l))
my_dataset.display_image(img[0], y[0])
