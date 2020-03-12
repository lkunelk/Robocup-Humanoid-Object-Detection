import os
import torch
import numpy as np
from model import CNN, init_weights
from my_dataset import initialize_loader, display_image
import train
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':
    model = CNN(
        kernel=3,
        num_features=16,
        dropout=0.1)

    # Save directory
    output_folder = 'outputs'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model.apply(init_weights)

    train.train(
        model,
        learn_rate=0.01,
        batch_size=64,
        epochs=20)

    torch.save(model.state_dict(), 'outputs/model')