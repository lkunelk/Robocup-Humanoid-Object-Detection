import os
import copy
import cv2
import enum
import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import util

train_path = '../bit-bots-ball-dataset-2018/train'
test_path = '../bit-bots-ball-dataset-2018/test'


def initialize_loader(batch_size, num_workers=64, shuffle=True):
    train_folders = [os.path.join(train_path, folder) for folder in os.listdir(train_path)]
    # train_folders = ['../bit-bots-ball-dataset-2018/train/bitbots-set00-05',
    #                  '../bit-bots-ball-dataset-2018/train/sequences-jasper-euro-ball-1',
    #                  '../bit-bots-ball-dataset-2018/train/sequences-euro-ball-robot-1',
    #                  '../bit-bots-ball-dataset-2018/train/bitbots-set00-07',
    #                  '../bit-bots-ball-dataset-2018/train/bitbots-set00-04',
    #                  '../bit-bots-ball-dataset-2018/train/bitbots-set00-10',
    #                  '../bit-bots-ball-dataset-2018/train/imageset_352',
    #                  '../bit-bots-ball-dataset-2018/train/imageset_168',
    #                  '../bit-bots-ball-dataset-2018/train/bitbots-set00-08',
    #                  '../bit-bots-ball-dataset-2018/train/imageset_61',
    #                  '../bit-bots-ball-dataset-2018/train/sequences-misc-ball-1']
    # valid_folders = [os.path.join(valid_path, folder) for folder in os.listdir(valid_path)]
    test_folders = [os.path.join(test_path, folder) for folder in os.listdir(test_path)]

    full_dataset = MyDataSet(train_folders, (150, 200))
    test_dataset = MyDataSet(test_folders, (150, 200))

    train_size = int(0.8 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

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

    print('full dataset: # images {:>6}, # robots {:>6}, # balls {:>6}'.format(
        len(full_dataset),
        full_dataset.num_robot_labels,
        full_dataset.num_ball_labels
    ))

    # print('valid dataset: # images {:>6}, # robots {:>6}, # balls {:>6}'.format(
    #     len(valid_dataset),
    #     valid_dataset.num_robot_labels,
    #     valid_dataset.num_ball_labels
    # ))

    print('test dataset:  # images {:>6}, # robots {:>6}, # balls {:>6}'.format(
        len(test_dataset),
        test_dataset.num_robot_labels,
        test_dataset.num_ball_labels
    ))

    return (train_loader, valid_loader, test_loader), (full_dataset, test_dataset)


class Label(enum.Enum):
    '''
    Defines output layers of model
    '''
    BALL = 0
    ROBOT = 1
    OTHER = 2


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
        img = util.read_image(img_path)

        height, width, _ = np.array(img).shape
        # the final mask will have no channels but we need 3 to convert to PIL image to apply transformation
        mask = np.ones((height, width, 3)) * Label.OTHER.value
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
        img = util.read_image(img_path)

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
            img = util.draw_bounding_boxes(img, bbxs, 255)
            util.stream_image(img, delay, scale)
