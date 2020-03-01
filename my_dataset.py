from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import torch

train_path = '/home/nam/Desktop/bit-bots-ball-dataset-2018/train'
test_path = None


def initialize_loader(train_batch_size=64):
    folders = [os.path.join(train_path, folder) for folder in os.listdir(train_path)]
    train_dataset = MyDataSet(folders)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, num_workers=4, shuffle=True, drop_last=True)
    return train_loader


def read_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.rollaxis(img, 2) # flip to channel*W*H
    return img


class MyDataSet(Dataset):
    def __init__(self, file_paths):
        print(train_path)
        self.file_paths = file_paths
        self.valid_filenames = []
        for path in file_paths:
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
                        self.valid_filenames.append((img_path, [int(x1), int(y1), int(x2), int(y2)]))

    def __len__(self):
        return len(self.valid_filenames)

    def __getitem__(self, index):
        img_path, label = self.valid_filenames[index]
        img = read_image(img_path)
        label = np.array(label)
        return img, label
