import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, kernel=3, num_features=16):
        super(CNN, self).__init__()

        pad = kernel // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, num_features, kernel, padding=pad),
            nn.BatchNorm2d(num_features)
        )

        self.maxpool1 = nn.Sequential(
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(num_features, 2 * num_features, kernel, padding=pad),
            nn.BatchNorm2d(2 * num_features),
            nn.LeakyReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * num_features, 2 * num_features, kernel, padding=pad),
            nn.BatchNorm2d(2 * num_features),
            nn.LeakyReLU()
        )

        self.maxpool2 = nn.Sequential(
            nn.MaxPool2d(2)
        )

        # concat 16 + 32
        self.conv4 = nn.Sequential(
            nn.Conv2d(3 * num_features, 4 * num_features, kernel, padding=pad),
            nn.BatchNorm2d(4 * num_features),
            nn.LeakyReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(4 * num_features, 4 * num_features, kernel, padding=pad),
            nn.BatchNorm2d(4 * num_features),
            nn.LeakyReLU()
        )

        # concat 48 + 32
        self.conv6 = nn.Sequential(
            nn.Conv2d(7 * num_features, 8 * num_features, kernel, padding=pad),
            nn.BatchNorm2d(8 * num_features),
            nn.LeakyReLU()
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(8 * num_features, 8 * num_features, kernel, padding=pad),
            nn.BatchNorm2d(8 * num_features),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        # concat 48 + 128
        self.conv8 = nn.Sequential(
            nn.Conv2d(11 * num_features, 4 * num_features, kernel, padding=pad),
            nn.BatchNorm2d(4 * num_features),
            nn.LeakyReLU()
        )

        self.conv9 = nn.Sequential(
            nn.Conv2d(4 * num_features, 2 * num_features, kernel, padding=pad),
            nn.BatchNorm2d(2 * num_features),
            nn.LeakyReLU()
        )

        self.conv10 = nn.Sequential(
            nn.Conv2d(2 * num_features, 2 * num_features, kernel, padding=pad),
            nn.BatchNorm2d(2 * num_features),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        # concat 16 + 32
        self.conv11 = nn.Sequential(
            nn.Conv2d(3 * num_features, 1 * num_features, kernel, padding=pad),
            nn.BatchNorm2d(1 * num_features),
            nn.LeakyReLU()
        )

        self.conv12 = nn.Sequential(
            nn.Conv2d(1 * num_features, 1 * num_features, kernel, padding=pad),
            nn.BatchNorm2d(1 * num_features),
            nn.LeakyReLU()
        )

        self.conv13 = nn.Sequential(
            nn.Conv2d(1 * num_features, 1, kernel, padding=pad),
            nn.LeakyReLU()
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        max1_out = self.maxpool1(conv1_out)
        x = self.conv2(max1_out)
        x = self.conv3(x)

        cat1 = torch.cat((max1_out, x), 1)
        max2 = self.maxpool2(cat1)
        x = self.conv4(max2)
        x = self.conv5(x)
        x = torch.cat((max2, x), 1)
        x = self.conv6(x)

        # Decoding part
        x = self.conv7(x)

        cat2 = torch.cat((cat1, x), 1)
        x = self.conv8(cat2)
        x = self.conv9(x)
        x = self.conv10(x)

        x = torch.cat((conv1_out, x), 1)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)

        return x
