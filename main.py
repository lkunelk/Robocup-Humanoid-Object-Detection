import torch
import numpy as np
import model, my_dataset
import matplotlib.pyplot as plt
import cv2

l = my_dataset.initialize_loader()
img, y = next(iter(l))
print(img.shape)
plt.figure()
plt.imshow(img[0].permute(1,2,0))
plt.show()
