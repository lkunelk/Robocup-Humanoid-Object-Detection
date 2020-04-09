import os
import torch
import numpy as np
from model import CNN, init_weights, find_bounding_boxes
from my_dataset import initialize_loader, display_image, stream_image, draw_bounding_boxes
from train import Trainer
import matplotlib.pyplot as plt
import PIL
import cv2


def train_model():
    model = CNN(
        kernel=3,
        num_features=10,
        dropout=0.2)

    # Save directory
    output_folder = 'outputs'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model.apply(init_weights)
    # model.load_state_dict(torch.load('outputs/model'))

    trainer = Trainer(model,
                      learn_rate=0.01,
                      batch_size=64,
                      epochs=20,
                      output_folder='outputs')
    trainer.train()

    torch.save(model.state_dict(), 'outputs/model')


def display_dataset():
    [trainl, _, _], [traind, _, testd] = initialize_loader(6, num_workers=1, shuffle=False)
    traind.visualize_images(delay=10)


if __name__ == '__main__':
    train_model()
