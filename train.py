import os
import time
import enum
import numpy as np
import torch
from model import find_batch_bounding_boxes
from my_dataset import initialize_loader, display_image, draw_bounding_boxes, Label
import matplotlib.pyplot as plt


class Trainer:
    class ErrorType(enum.Enum):
        TRUE_POSITIVE = 0
        FALSE_POSITIVE = 1
        TRUE_NEGATIVE = 2
        FALSE_NEGATIVE = 3

    def __init__(self, model, learn_rate, batch_size, epochs, output_folder):
        self.model = model
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.output_folder = output_folder
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
        weight = torch.tensor([0.1, 0.2, 1.0])  # weigh importance of the label during training
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight.cuda())

        self.train_losses = []
        self.train_ious = []
        self.train_radius_losses = []

        self.valid_losses = []
        self.valid_ious = []
        self.valid_radius_losses = []

        self.valid_stats = {'BALL': [], 'ROBOT': []}

        loaders, datasets = initialize_loader(batch_size)
        self.train_loader, self.valid_loader, self.test_loader = loaders
        self.train_dataset, self.test_dataset = datasets
        print('Datasets Loaded! # of batches train:{} valid:{} test:{}'.format(
            len(self.train_loader), len(self.valid_loader), len(self.test_loader)))

    def train_epoch(self, epoch):
        self.model.train()
        start_epoch = time.time()
        batchload_times = []
        losses = []
        t_readimg = time.time()
        for images, masks, img_paths in self.train_loader:
            batchload_times.append(time.time() - t_readimg)

            images = images.cuda()
            masks = masks.cuda()

            self.optimizer.zero_grad()
            _, logits = self.model(images.float())
            loss = self.criterion(logits, masks.long())
            loss.backward()
            self.optimizer.step()
            losses.append(loss.data.item())

            t_readimg = time.time()
        self.train_losses.append(sum(losses) / len(losses))

        time_elapsed = time.time() - start_epoch
        print('Epoch [{:2d}/{:2d}]: Train Loss: {: 4.6f}, Avg. Batch Load (s): {:.4f}, Epoch (s): {: 4.2f}'.format(
            epoch + 1,
            self.epochs,
            self.train_losses[-1],
            sum(batchload_times) / len(batchload_times),
            time_elapsed))

    def test_model(self, test_type):
        dataset, loader = None, None
        if test_type == 'valid':
            dataset, loader = self.train_dataset, self.valid_loader
        elif test_type == 'test':
            dataset, loader = self.test_dataset, self.test_loader

        self.model.eval()
        start_valid = time.time()
        losses = []
        stats = {Label.BALL: [0, 0, 0, 0], Label.ROBOT: [0, 0, 0, 0]}
        for images, masks, indexes in loader:
            images = images.cuda()
            masks = masks.cuda()
            outputs, logits = self.model(images.float())
            loss = self.criterion(logits, masks.long())
            losses.append(loss.data.item())

            bbxs = find_batch_bounding_boxes(outputs)
            self.update_batch_stats(stats, bbxs, masks, dataset, indexes)

            # Show sample image with bounding boxes to get feel for what model is learning
            for i in range(1):
                img = draw_bounding_boxes(images[i], bbxs[i][Label.BALL.value], (255, 0, 0))  # balls
                img = draw_bounding_boxes(img, bbxs[i][Label.ROBOT.value], (0, 0, 255))  # robots

                display_image([
                    (img, None, 'Input'),
                    (masks[i], None, 'Truth'),
                    (outputs[i], None, 'Prediction'),
                    (outputs[i][Label.OTHER.value], 'gray', 'Background'),
                    (outputs[i][Label.BALL.value], 'gray', 'Ball'),
                    (outputs[i][Label.ROBOT.value], 'gray', 'Robot')
                ])
                print('ball', bbxs[i][Label.BALL.value])
                print('robot', bbxs[i][Label.ROBOT.value])
                input('wait:')

        self.valid_losses.append(np.sum(losses) / len(losses))
        time_elapsed = time.time() - start_valid

        print('{:>20} Loss: {: 4.6f}, , {} time (s): {: 4.2f}'.format(
            test_type,
            self.valid_losses[-1],

            test_type,
            time_elapsed))
        print('{:>20} ball tp:{:6d}, fp:{:6d}, tn:{:6d}, fn:{:6d}'.format(
            '',
            stats[0][self.ErrorType.TRUE_POSITIVE.value],
            stats[0][self.ErrorType.FALSE_POSITIVE.value],
            stats[0][self.ErrorType.TRUE_NEGATIVE.value],
            stats[0][self.ErrorType.FALSE_NEGATIVE.value],
        ))
        print('{:>20} robot tp:{:6d}, fp:{:6d}, tn:{:6d}, fn:{:6d}'.format(
            '',
            stats[1][self.ErrorType.TRUE_POSITIVE.value],
            stats[1][self.ErrorType.FALSE_POSITIVE.value],
            stats[1][self.ErrorType.TRUE_NEGATIVE.value],
            stats[1][self.ErrorType.FALSE_NEGATIVE.value],
        ))

    def train(self):
        print('Starting Training')
        start_train = time.time()

        self.model.cuda()
        for epoch in range(self.epochs):
            self.train_epoch(epoch)
            self.test_model('valid')

        self.test_model('test')

        time_elapsed = time.time() - start_train
        print('Finished training in: {: 4.2f}min'.format(time_elapsed / 60))

        self.plot_losses()

    def update_batch_stats(self, stats, batch_bounding_boxes, batch_masks, dataset, batch_img_indexes):
        """
        calculate true/false positive/negative
        the predicted center of bounding box needs to fall on the ground truth prediction
        """
        for batch_ind, bounding_boxes in enumerate(batch_bounding_boxes):
            mask = batch_masks[batch_ind]
            img_index = batch_img_indexes[batch_ind]
            for pred_class in [Label.BALL, Label.ROBOT]:
                for bbx in bounding_boxes[pred_class.value]:
                    x_center = int((bbx[0] + bbx[2]) / 2)
                    y_center = int((bbx[1] + bbx[3]) / 2)
                    if mask[y_center][x_center] == pred_class.value:
                        bbx.append('tp')
                        stats[pred_class][self.ErrorType.TRUE_POSITIVE.value] += 1
                    else:
                        bbx.append('fp')
                        stats[pred_class][self.ErrorType.FALSE_POSITIVE.value] += 1

                # TODO implement tn, fn
                # true_bounding_boxes = dataset.get_bounding_boxes(img_index)
                # if not true_bounding_boxes and not bbxs:
                #     stats[pred_class - 1][self.ErrorType.TRUE_NEGATIVE.value] += 1
                # elif true_bounding_boxes and not bbxs:
                #     for _ in true_bbxs:
                #         stats[pred_class - 1][self.ErrorType.FALSE_NEGATIVE.value] += 1

    def plot_losses(self):
        plt.figure()
        plt.plot(self.train_losses, "ro-", label="Train")
        plt.plot(self.valid_losses, "go-", label="Validation")
        plt.legend()
        plt.title("Losses")
        plt.xlabel("Epochs")
        plt.savefig(os.path.join(self.output_folder, "training_curve.png"))
        plt.show()
