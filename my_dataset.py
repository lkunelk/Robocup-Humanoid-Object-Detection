import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import enum
import os
import copy
import cv2
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import util

train_path = '../bit-bots-ball-dataset-2018/train'
valid_path = '../bit-bots-ball-dataset-2018/valid'
negative_path = '../bit-bots-ball-dataset-2018/negative'
test_path = '../bit-bots-ball-dataset-2018/test'


def initialize_loader(batch_size, num_workers=64, shuffle=True):
    train_folders = [os.path.join(train_path, folder) for folder in os.listdir(train_path)]
    valid_folders = [os.path.join(valid_path, folder) for folder in os.listdir(valid_path)]
    test_folders = [os.path.join(test_path, folder) for folder in os.listdir(test_path)]
    print(valid_folders)

    train_folders = ['../bit-bots-ball-dataset-2018/valid/bitbots-set00-05']

    train_dataset = MyDataSet(train_folders, (150, 200))
    valid_dataset = MyDataSet(valid_folders, (150, 200))
    test_dataset = MyDataSet(test_folders, (150, 200))

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=shuffle)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=shuffle)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=shuffle)

    print('train dataset: # images {:>6}, # robots {:>6}, # balls {:>6}'.format(
        len(train_dataset),
        train_dataset.num_robot_labels,
        train_dataset.num_ball_labels
    ))

    print('valid dataset: # images {:>6}, # robots {:>6}, # balls {:>6}'.format(
        len(valid_dataset),
        valid_dataset.num_robot_labels,
        valid_dataset.num_ball_labels
    ))

    print('test dataset:  # images {:>6}, # robots {:>6}, # balls {:>6}'.format(
        len(test_dataset),
        test_dataset.num_robot_labels,
        test_dataset.num_ball_labels
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


def stream_image(img, wait, scale):
    img = util.torch_to_cv(img)
    width, height, _ = img.shape
    img = cv2.resize(img, (height * scale, width * scale))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('my_window', img)
    cv2.waitKey(wait)


def read_image(path):
    # using opencv imread crashes Pytorch DataLoader for some reason
    return Image.open(path)


class Label(enum.Enum):
    '''
    the values correspond to which output neuron should be activated
    '''
    OTHER = 0
    BALL = 1
    ROBOT = 2


class MyDataSet(Dataset):
    def __init__(self, folder_paths, target_dim):

        self.folder_paths = folder_paths  # folders of the images
        self.img_paths = []  # all individual images
        self.bounding_boxes = {}  # image paths and their labels
        self.target_height = target_dim[0]
        self.target_width = target_dim[1]
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.target_height, interpolation=Image.BILINEAR),
            torchvision.transforms.CenterCrop((self.target_height, self.target_width)),
        ])

        # label statistics
        self.num_ball_labels = 0
        self.num_robot_labels = 0

        # add paths for train data with labels
        for path in folder_paths:
            for file in os.listdir(path):
                if '.txt' in file:
                    print('reading:', file)
                    file_labels = os.path.join(path, file)
                    self.read_labels(path, file_labels)

    def read_labels(self, path, file_labels):
        # store full path with label for each image
        with open(file_labels) as labels:
            for i, line in enumerate(labels):
                if i <= 5:  # ignore first few metadata lines
                    continue

                try:
                    label, img, _, _, x1, y1, x2, y2, _, _, _, _ = line.split('|')
                except:
                    # ignore unknown format
                    continue
                img_path = os.path.join(path, img)

                if label == 'label::ball':
                    label = Label.BALL
                    self.num_ball_labels += 1
                elif label == 'label::robot':
                    label = Label.ROBOT
                    self.num_robot_labels += 1
                else:
                    print('Unexpected Label:', label)

                img_path = os.path.join(path, img)
                if img_path not in self.img_paths:
                    self.bounding_boxes[img_path] = []
                    self.img_paths.append(img_path)

                self.bounding_boxes[img_path].append([int(x1), int(y1), int(x2), int(y2), label])

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
            pt1, pt2, label = np.array(bb[0:2]), np.array(bb[2:4]), bb[4]

            center = tuple(((pt1 + pt2) / 2).astype(np.int))
            size = tuple(((pt2 - pt1) / 2).astype(np.int))

            if not size == (0, 0):
                if label == Label.BALL:
                    mask = cv2.ellipse(mask, center, size, 0, 0, 360, (label.value), -1)
                if label == Label.ROBOT:
                    mask = cv2.rectangle(mask, tuple(pt1), tuple(pt2), (label.value), -1)

        mask = Image.fromarray(mask.astype('uint8'))

        # Apply transformations to get desired dimensions
        img = np.array(self.transform(img))
        mask = np.array(self.transform(mask))

        # flip to channel*W*H - how Pytorch expects it
        img = np.moveaxis(img, -1, 0)
        mask = np.moveaxis(mask, -1, 0)[0]  # get rid of channel dimension

        return img, mask, index

    def get_bounding_boxes(self, index):
        img_path = self.img_paths[index]
        img = read_image(img_path)

        # we need to scale bounding boxes since we applied a transformation
        height, width, _ = np.shape(img)
        height_scale = self.target_height / height
        width_offset = (width * height_scale - self.target_width) / 2
        bbxs = copy.deepcopy(self.bounding_boxes[img_path])
        for bbx in bbxs:
            bbx[0] = int(bbx[0] * height_scale - width_offset)
            bbx[1] = int(bbx[1] * height_scale)
            bbx[2] = int(bbx[2] * height_scale - width_offset)
            bbx[3] = int(bbx[3] * height_scale)
        return bbxs

    def visualize_images(self, delay=10, scale=4):
        self.img_paths = list(sorted(self.img_paths))  # we want names to be sorted so that they are displayed in order
        for ind in range(len(self)):
            img, _, _ = self[ind]
            bbxs = self.get_bounding_boxes(ind)
            img = draw_bounding_boxes(img, bbxs, 255)
            stream_image(img, delay, scale)
            print(self.img_paths[ind])
