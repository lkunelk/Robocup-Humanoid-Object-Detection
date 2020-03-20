import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import enum
import os
import cv2
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from PIL import Image

TESTING = False

train_path = '../robot-dataset'  # '../bit-bots-ball-dataset-2018/train'
negative_path = '../bit-bots-ball-dataset-2018/negative'
test_path = '../bit-bots-ball-dataset-2018/test'


def initialize_loader(batch_size):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(152),
        torchvision.transforms.CenterCrop((152, 200)),
    ])

    train_folders = [os.path.join(train_path, folder) for folder in os.listdir(train_path)]
    test_folders = [os.path.join(test_path, folder) for folder in os.listdir(test_path)]

    full_dataset = MyDataSet(train_folders, transform=transform, train=True)
    test_dataset = MyDataSet(test_folders, transform=transform, train=False)

    train_size = int(0.8 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=64,
                              shuffle=True,
                              drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              num_workers=64,
                              shuffle=True,
                              drop_last=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             num_workers=64,
                             shuffle=True,
                             drop_last=True)

    return train_loader, valid_loader, test_loader


def display_image(img=None, mask=None, y=None, pred=None):
    '''
    :param img: torch tensor channelxWxH
    :param mask: ground truth label (0, 1)
    :param y: grayscale output from model
    :param pred: y with bounding box
    :return: None
    '''
    fig, ax = plt.subplots(3, 2)
    if img is not None:
        img = np.moveaxis(img.numpy(), 0, -1)  # HxWxchannel
        ax[0, 0].set_title('Input')
        ax[0, 0].imshow(img)

    if y is not None:
        y = np.moveaxis(y.detach().numpy(), 0, -1)
        ax[0, 1].set_title('Output')
        ax[0, 1].imshow(y)

    if mask is not None:
        mask = mask.numpy()
        ax[1, 0].set_title('Mask')
        ax[1, 0].imshow(mask)

    if pred is not None:
        p = pred.detach().numpy()[0]
        ax[1, 1].set_title('Prediction: other')
        ax[1, 1].imshow(p, cmap='gray')

    if pred is not None:
        p = pred.detach().numpy()[1]
        ax[2, 0].set_title('Prediction: ball')
        ax[2, 0].imshow(p, cmap='gray')

    if pred is not None:
        p = pred.detach().numpy()[2]
        ax[2, 1].set_title('Prediction: robot')
        ax[2, 1].imshow(p, cmap='gray')

    plt.show()


def read_image(path):
    return Image.open(path)


class Label(enum.Enum):
    OTHER = 0
    BALL = 1
    ROBOT = 2


class MyDataSet(Dataset):
    def __init__(self, folder_paths, transform=None, train=False):
        self.folder_paths = folder_paths  # folders of the images
        self.img_paths = []  # all individual images
        self.bounding_boxes = {}  # image paths and their labels
        self.transform = transform

        # add paths for train data with labels
        for path in folder_paths:
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

                        img_path = os.path.join(path, img)

                        if label == 'label::ball':
                            label = Label.BALL
                        if label == 'label::robot':
                            label = Label.ROBOT

                        if img_path not in self.img_paths:
                            self.bounding_boxes[img_path] = []
                            self.img_paths.append(img_path)

                        self.bounding_boxes[img_path].append(
                            [label, int(x1), int(y1), int(x2), int(y2)])

            if TESTING:
                break  # keep dataset small
        # add paths for negative examples (no ball in picture)
        # if train:
        #     for file in os.listdir(negative_path):
        #         img_path = os.path.join(negative_path, file)
        #         self.valid_filenames.append((img_path, [(0, 0), (0, 0)]))

    def __len__(self):
        return len(self.bounding_boxes)

    def __getitem__(self, index):
        '''
        :param index: index of data point
        :return: img ndarray (3 x w x h) RGB image
                 mask ndarray (w x h) segmentation classification of each pixel
        '''
        img_path = self.img_paths[index]
        bounding_boxes = self.bounding_boxes[img_path]
        img = read_image(img_path)

        height, width, _ = np.array(img).shape
        # the final mask will have no channels but we need 3 to convert to PIL image to apply transformation
        mask = np.zeros((height, width, 3))
        for bb in bounding_boxes:
            label = bb[0]
            pt1 = np.array(bb[1:3])
            pt2 = np.array(bb[3:5])

            center = tuple(((pt1 + pt2) / 2).astype(np.int))
            size = tuple(((pt2 - pt1) / 2).astype(np.int))

            if not size == (0, 0):
                if label == Label.BALL:
                    mask = cv2.ellipse(mask, center, size, 0, 0, 360, (label.value), -1)
                if label == Label.ROBOT:
                    mask = cv2.rectangle(mask, tuple(pt1), tuple(pt2), (label.value), -1)

        mask = Image.fromarray(mask.astype('uint8'))

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        img = np.array(img)
        mask = np.array(mask)
        img = np.moveaxis(img, -1, 0)  # flip to channel*W*H
        mask = np.moveaxis(mask, -1, 0)[0]  # get rid of channel dimension
        return img, mask
