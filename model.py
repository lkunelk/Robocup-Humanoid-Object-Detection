import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 1, (2, 2))
        )

    def forward(self, x):
        x = self.layer1(x)
        return x
