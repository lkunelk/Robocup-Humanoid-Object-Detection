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
        batch_size=8,
        epochs=20)

    torch.save(model.state_dict(), 'outputs/model')

def test_bounding_box():
    # train, valid, test = initialize_loader(10)
    #
    # model = CNN()
    # model.load_state_dict(torch.load('outputs/model'))
    # model.eval()
    #
    # for batch, mask in train:
    #     _, output = model(batch.float())
    #     display_image(pred=output[0])
    #     break

    img = np.zeros((150, 200, 1))
    center = (75, 40)
    size = (20, 30)
    img = cv2.ellipse(img, center, size, 0, 0, 360, (255), -1)
    cv2.imshow('hi?', img)
    cv2.waitKey(0)
    img = img.astype(np.uint8)
    print(img.shape, img.dtype)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)
    x, y, w, h = cv2.boundingRect(contours[0])

    # img = np.rollaxis(img, 2)
    # bbx = find_bounding_boxes(torch.Tensor(img))
    print(x, y, w, h)

if __name__ == '__main__':
    test_bounding_box()