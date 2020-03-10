import torch
import numpy as np
from model import CNN, init_weights
from my_dataset import initialize_loader, display_image
import train
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':
    model = CNN()
    model.apply(init_weights)
    print(model.conv1[0].weight)
    train.train(model)