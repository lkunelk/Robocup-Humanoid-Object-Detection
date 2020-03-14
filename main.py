import os
import torch
import numpy as np
from model import CNN, init_weights, find_bounding_boxes
from my_dataset import initialize_loader, display_image
import train
import matplotlib.pyplot as plt
import cv2

def train_model():
    model = CNN(
        kernel=3,
        num_features=16,
        dropout=0.2)

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

def test_bounding_box():
    train, valid, test = initialize_loader(10)

    model = CNN()
    model.load_state_dict(torch.load('outputs/model'))
    model.eval()

    for batch, mask, bbx in train:
        output, _ = model(batch.float())
        display_image(pred=output[0])
        break

    img = np.zeros((152, 200, 1))
    center = (75, 40)
    size = (20, 30)
    img = cv2.ellipse(img, center, size, 0, 0, 360, (1), -1)
    cv2.imshow('hi?', img)
    cv2.waitKey(0)
    img = np.rollaxis(img, 2)
    bbx = find_bounding_boxes(torch.Tensor(img))
    print(bbx)

if __name__ == '__main__':
    train_model()