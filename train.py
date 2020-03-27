import os
import time
import numpy as np
import torch
from model import find_bounding_boxes
from my_dataset import initialize_loader, display_image, draw_bounding_boxes
import matplotlib.pyplot as plt


def train_epoch():
    # hello
    pass


def train(model,
          learn_rate=0.01,
          batch_size=64,
          epochs=20,
          output_folder='outputs'):
    np.random.seed(1)

    train_losses = []
    train_ious = []
    train_radius_losses = []
    valid_losses = []
    valid_ious = []
    valid_radius_losses = []

    train_loader, valid_loader, test_loader = initialize_loader(batch_size)
    print('# of batches train:{} valid:{} test:{}'.format(len(train_loader), len(valid_loader), len(test_loader)))

    print('Starting Training')
    start_train = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    weight = torch.Tensor([0.2, 0.6, 0.2])
    criterion = torch.nn.CrossEntropyLoss(weight=weight.cuda())

    model.cuda()
    model.train()
    for epoch in range(epochs):
        start_epoch = time.time()
        batchload_times = []
        losses = []
        t_readimg = time.time()
        for images, masks in train_loader:
            batchload_times.append(time.time() - t_readimg)

            images = images.cuda()
            masks = masks.cuda()

            optimizer.zero_grad()
            predictions, _ = model(images.float())
            loss = criterion(predictions, masks.long())
            loss.backward()
            optimizer.step()
            losses.append(loss.data.item())

            t_readimg = time.time()
        train_losses.append(sum(losses) / len(losses))

        time_elapsed = time.time() - start_epoch
        print('Epoch [{:2d}/{:2d}]: Train Loss: {: 4.6f}, Avg. Batch Load (s): {:.4f}, Epoch (s): {: 4.2f}'.format(
            epoch + 1,
            epochs,
            train_losses[-1],
            sum(batchload_times) / len(batchload_times),
            time_elapsed))

        model.eval()
        start_valid = time.time()
        losses = []
        for images, masks in valid_loader:
            images = images.cuda()
            masks = masks.cuda()
            predictions, clipped_pred = model(images.float())
            loss = criterion(predictions, masks.long())
            losses.append(loss.data.item())

        bbxs = find_bounding_boxes(predictions[0][1:2])
        img = draw_bounding_boxes(images[0], bbxs, (255, 0, 0))

        display_image([
            (img, None, 'Input'),
            (masks[0], None, 'Truth'),
            (predictions[0], None, 'Prediction'),
            (clipped_pred[0][0], 'gray', 'Background'),
            (clipped_pred[0][1], 'gray', 'Ball'),
            (clipped_pred[0][2], 'gray', 'Robot')
        ])

        valid_losses.append(np.sum(losses) / len(losses))
        time_elapsed = time.time() - start_valid

        print('{:15}Valid Loss: {: 4.6f}, validation time (s): {: 4.2f}'.format(
            '',
            valid_losses[-1],
            time_elapsed))

    time_elapsed = time.time() - start_train
    print('Finished training in: {: 4.2f}min'.format(
        time_elapsed / 60
    ))

    # Plot training curve
    plt.figure()
    plt.plot(train_losses, "ro-", label="Train")
    plt.plot(valid_losses, "go-", label="Validation")
    plt.legend()
    plt.title("Losses")
    plt.xlabel("Epochs")
    plt.savefig(os.path.join(output_folder, "training_curve.png"))
    plt.show()
