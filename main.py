import torch
import numpy as np
from model import CNN
from my_dataset import initialize_loader, display_image
import train
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':
    # ltrain, ltest = initialize_loader()
    # for d in ltest:
    #     display_image(d[0][0], d[1][0])
    #     break
    # model = CNN()
    # train.train(model, 0.1)

    model = CNN()
    x = torch.Tensor(np.ones((1,3,16,16)))
    out = model.forward(x)
    print(out.shape)