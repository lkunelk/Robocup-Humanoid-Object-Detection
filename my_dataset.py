import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import os
import cv2
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from PIL import Image

train_path = '/home/robosoccer/Nam/bit-bots-ball-dataset-2018/train'
negative_path = '/home/robosoccer/Nam/bit-bots-ball-dataset-2018/negative'
test_path = '/home/robosoccer/Nam/bit-bots-ball-dataset-2018/test'


def initialize_loader(batch_size):
    transform = torchvision.transforms.Resize((152, 200))

    train_folders = [os.path.join(train_path, folder) for folder in os.listdir(train_path)]
    test_folders = [os.path.join(test_path, folder) for folder in os.listdir(test_path)]

    full_dataset = MyDataSet(train_folders, transform=transform, train=True)
    test_dataset = MyDataSet(test_folders, transform=transform, train=False)

    train_size = int(0.8 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=4,
                              shuffle=True,
                              drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              num_workers=4,
                              shuffle=True,
                              drop_last=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             num_workers=4,
                             shuffle=True,
                             drop_last=True)

    return train_loader, valid_loader, test_loader


def display_image(img, mask, y):
    '''
    :param img: torch tensor channelxWxH
    :param y: grayscale mask representing ball location
    :return: None
    '''
    img = img.numpy()
    img = np.rollaxis(img, 0, 3)  # HxWxchannel

    y = y.detach().numpy().reshape((152, 200))

    mask = mask.numpy().reshape((152, 200))
    print(np.amax(mask))

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(img)
    ax[0, 1].imshow(y, cmap='gray')
    ax[1, 0].imshow(mask, cmap='gray')

    plt.show()


def read_image(path):
    img = Image.open(path)
    # img = resize(img, (150, 200, 3))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = np.rollaxis(img, 2)  # flip to channel*W*H
    return img


class MyDataSet(Dataset):
    def __init__(self, file_paths, transform=None, train=False):
        self.file_paths = file_paths
        self.valid_filenames = []
        self.transform = transform

        # add paths for train data with labels
        for path in file_paths:
            # print(path)
            # find txt file with labels
            file_labels = None
            for file in os.listdir(path):
                if '.txt' in file:
                    file_labels = os.path.join(path, file)

            # store full path with label for each image
            with open(file_labels) as labels:
                for i, line in enumerate(labels):
                    if i > 5:  # ignore first few metadata lines
                        label, img, _, _, x1, y1, x2, y2, _, _, _, _ = line.split('|')
                        assert label == 'label::ball'
                        img_path = os.path.join(path, img)
                        self.valid_filenames.append([
                            img_path,
                            [(int(x1), int(y1)), (int(x2), int(y2))]
                        ])
            break  # DEBUG only read one folder for testing
        # add paths for negative examples (no ball in picture)
        # if train:
        #     for file in os.listdir(negative_path):
        #         img_path = os.path.join(negative_path, file)
        #         self.valid_filenames.append((img_path, [(0, 0), (0, 0)]))

    def __len__(self):
        return len(self.valid_filenames)

    def __getitem__(self, index):
        img_path, label = self.valid_filenames[index]
        img = read_image(img_path)

        if self.transform:
            img = self.transform(img)
        img = np.array(img)

        mask = np.zeros((152, 200, 1))
        pt1 = np.array(label[0]) / 4
        pt2 = np.array(label[1]) / 4
        center = tuple(((pt1 + pt2) / 2).astype(np.int))
        size = tuple(((pt2 - pt1) / 2).astype(np.int))
        if not size == (0, 0):
            mask = cv2.ellipse(mask, center, size, 0, 0, 360, (1), -1)

        mask = np.rollaxis(mask, 2)
        img = np.rollaxis(img, 2)  # flip to channel*W*H
        return img, mask
