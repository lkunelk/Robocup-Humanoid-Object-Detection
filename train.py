import os
import time
import numpy as np
import torch
from model import find_batch_bounding_boxes
from my_dataset import initialize_loader, display_image, draw_bounding_boxes
import matplotlib.pyplot as plt


class Trainer:

    def __init__(self, model, learn_rate, batch_size, epochs, output_folder):
        self.model = model
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.output_folder = output_folder
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
        weight = torch.tensor([0.2, 0.6, 0.2])  # weigh importance of the label during training
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight.cuda())

        self.train_losses = []
        self.train_ious = []
        self.train_radius_losses = []

        self.valid_losses = []
        self.valid_ious = []
        self.valid_radius_losses = []

        self.train_loader, self.valid_loader, self.test_loader = initialize_loader(batch_size)
        print('Datasets Loaded! # of batches train:{} valid:{} test:{}'.format(
            len(self.train_loader), len(self.valid_loader), len(self.test_loader)))

    def train_epoch(self, epoch):
        self.model.train()
        start_epoch = time.time()
        batchload_times = []
        losses = []
        t_readimg = time.time()
        for images, masks in self.train_loader:
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

    def validate_epoch(self):
        self.model.eval()
        start_valid = time.time()
        losses = []
        for images, masks in self.valid_loader:
            images = images.cuda()
            masks = masks.cuda()
            outputs, logits = self.model(images.float())
            loss = self.criterion(logits, masks.long())
            losses.append(loss.data.item())

        bbxs = find_batch_bounding_boxes(outputs)
        self.calculate_stats(bbxs, masks)

        for i in range(1):
            print(bbxs[i][1])
            img = draw_bounding_boxes(images[i], bbxs[i][1], (255, 0, 0))

            display_image([
                (img, None, 'Input'),
                (masks[i], None, 'Truth'),
                (outputs[i], None, 'Prediction'),
                (outputs[i][0], 'gray', 'Background'),
                (outputs[i][1], 'gray', 'Ball'),
                (outputs[i][2], 'gray', 'Robot')
            ])

        self.valid_losses.append(np.sum(losses) / len(losses))
        time_elapsed = time.time() - start_valid

        print('{:15}Valid Loss: {: 4.6f}, validation time (s): {: 4.2f}'.format(
            '',
            self.valid_losses[-1],
            time_elapsed))

    def train(self):
        print('Starting Training')
        start_train = time.time()

        self.model.cuda()
        for epoch in range(self.epochs):
            self.train_epoch(epoch)
            self.validate_epoch()

        time_elapsed = time.time() - start_train
        print('Finished training in: {: 4.2f}min'.format(time_elapsed / 60))

        self.plot_losses()

    def calculate_stats(self, batch_bbxs, batch_masks, img_paths):
        '''calculate true/false positive/negative
        the predicted center of bounding box needs to fall on the ground truth prediction'''
        # calculate stats for balls for now
        stats = np.zeros((4))
        for batch_ind, bbxs in enumerate(batch_bbxs):
            masks = batch_masks[batch_ind]
            for pred_class in [1]:
                bbxs = bbxs[pred_class]
                for bbx in bbxs:
                    x_center = int((bbx[0] + bbx[2]) / 2)
                    y_center = int((bbx[1] + bbx[3]) / 2)
                    if masks[y_center][x_center] == pred_class:
                        bbx.append('tp')
                        stats[0] += 1
                    if not masks[y_center][x_center] == pred_class:
                        bbx.append('fp')
                        stats[1] += 1
        return stats
    def plot_losses(self):
        plt.figure()
        plt.plot(self.train_losses, "ro-", label="Train")
        plt.plot(self.valid_losses, "go-", label="Validation")
        plt.legend()
        plt.title("Losses")
        plt.xlabel("Epochs")
        plt.savefig(os.path.join(self.output_folder, "training_curve.png"))
        plt.show()
