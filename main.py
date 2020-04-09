import os
import torch
import numpy as np
from model import CNN, init_weights, find_bounding_boxes
from my_dataset import initialize_loader, display_image, draw_bounding_boxes
from train import Trainer
import matplotlib.pyplot as plt
import PIL


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


def test_bounding_box():
    train, valid, test = initialize_loader(10, shuffle=False)

    model = CNN()
    model.load_state_dict(torch.load('outputs/model'))
    model.eval()

    img = None

    for batch, mask in train:
        inp = batch[0]
        _, img = model(batch.float())
        break

    img = img[0][1:2].detach().numpy()  # get ball classification output as (1xHxW) numpy array
    bbxs = find_bounding_boxes(img)
    inp = inp.detach().numpy()
    inp = draw_bounding_boxes(inp, bbxs, 255)
    display_image([
        (inp, None, 'Input'),
        (img, 'gray', 'Output')])


if __name__ == '__main__':
    train_model()
