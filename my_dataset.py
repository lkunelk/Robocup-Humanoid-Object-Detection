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
import util

TESTING = True

train_path = '../bit-bots-ball-dataset-2018/train'
valid_path = '../bit-bots-ball-dataset-2018/valid'
negative_path = '../bit-bots-ball-dataset-2018/negative'
test_path = '../bit-bots-ball-dataset-2018/test'


def initialize_loader(batch_size, shuffle=True):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(150, interpolation=Image.NEAREST),
        torchvision.transforms.CenterCrop((150, 200)),
    ])

    train_folders = [os.path.join(train_path, folder) for folder in os.listdir(train_path)]
    valid_folders = [os.path.join(valid_path, folder) for folder in os.listdir(valid_path)]
    test_folders = [os.path.join(test_path, folder) for folder in os.listdir(test_path)]

    train_dataset = MyDataSet(train_folders, transform=transform)
    valid_dataset = MyDataSet(valid_folders, transform=transform)
    test_dataset = MyDataSet(test_folders, transform=transform)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=64,
                              shuffle=shuffle)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              num_workers=64,
                              shuffle=shuffle)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             num_workers=64,
                             shuffle=shuffle)

    print('train dataset: # images {}, # robots {}, # balls {}'.format(
        len(train_dataset),
        train_dataset.num_robot_labels,
        train_dataset.num_ball_labels
    ))

    return (train_loader, valid_loader, test_loader), (train_dataset, valid_dataset, test_dataset)


def draw_bounding_boxes(img, bbxs, colour):
    '''
    :param img: rgb torch image
    :param bbxs:
    :param colour:
    :return:
    '''
    img = util.torch_to_cv(img)
    img = img.copy()  # cv2 seems to like copies to draw rectangles on

    for bbx in bbxs:
        if isinstance(bbx, str):
            return util.cv_to_torch(img)
        pt0 = (int(bbx[0]), int(bbx[1]))
        pt1 = (int(bbx[2]), int(bbx[3]))
        img = cv2.rectangle(img, pt0, pt1, colour, 1)
    return util.cv_to_torch(img)


def display_image(to_plot):
    '''
    :param to_plot: list of tuples of the form (img [(cxhxw) numpy array], cmap [str], title [str])
    '''
    fig, ax = plt.subplots(3, 2, figsize=(8, 10))
    for i, plot_info in enumerate(to_plot):
        img = util.torch_to_cv(plot_info[0])
        cmap = plot_info[1]
        title = plot_info[2]

        ax[i // 2, i % 2].imshow(img, cmap=cmap)
        ax[i // 2, i % 2].set_title(title)
    plt.show()


def read_image(path):
    return Image.open(path)


class Label(enum.Enum):
    '''
    the values correspond to which output neuron should be activated
    '''
    OTHER = 0
    BALL = 1
    ROBOT = 2


class MyDataSet(Dataset):
    def __init__(self, folder_paths, transform=None, train=False):
        self.folder_paths = folder_paths  # folders of the images
        self.img_paths = []  # all individual images
        self.bounding_boxes = {}  # image paths and their labels
        self.transform = transform

        # statistics
        self.num_ball_labels = 0
        self.num_robot_labels = 0

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
                            self.num_ball_labels += 1
                        if label == 'label::robot':
                            label = Label.ROBOT
                            self.num_robot_labels += 1

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
        return img, mask, img_path

    def get_bounding_boxes(self, img_path):
        return self.bounding_boxes[img_path]