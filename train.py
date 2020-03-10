import os
import time
import numpy as np
import torch
from my_dataset import initialize_loader, display_image


def train_epoch():
    # hello
    pass


def train(model,
          learn_rate=0.01,
          batch_size=64,
          epochs=20):
    np.random.seed(1)

    # Save directory
    save_dir = "outputs/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_loader, valid_loader, test_loader = initialize_loader(batch_size)
    print('# of batches {} {} {}'.format(len(train_loader), len(valid_loader), len(test_loader)))

    model.cuda()

    batchload_times = []

    train_loss = []
    valid_loss = []
    valid_ious = []

    for epoch in range(epochs):
        model.train()
        start_train = time.time()
        losses = []
        t_readimg = time.time()
        for i, (images, masks) in enumerate(train_loader):
            batchload_times.append(time.time() - t_readimg)

            images = images.cuda()
            masks = masks.cuda()

            optimizer.zero_grad()

            predictions = model(images.float())
            loss = criterion(predictions.float(), masks.float())
            loss.backward()
            optimizer.step()
            losses.append(loss.data.item())
            t_readimg = time.time()
        train_loss.append(sum(losses) / len(losses))

        time_elapsed = time.time() - start_train
        print('Epoch [{}/{}], Loss: {}, Avg. Batch Load (s): {}, Epoch (s): {}'.format(
            epoch + 1,
            epochs,
            train_loss[-1],
            sum(batchload_times) / len(batchload_times),
            time_elapsed))

        model.eval()
        start_valid = time.time()
        valid_losses = []
        for images, masks in valid_loader:
            images = images.cuda()
            masks = masks.cuda()
            predictions = model(images.float())
            loss = criterion(predictions, masks.float())
            valid_losses.append(loss.data.item())

        display_image(images.cpu()[0], masks.cpu()[0], predictions.cpu()[0])

        valid_loss.append(np.sum(valid_losses) / len(valid_losses))
        time_elapsed = time.time() - start_valid

        print('valid loss: {}, validation time (s): {}'.format(
            valid_loss[-1],
            sum(batchload_times) / len(batchload_times),
            time_elapsed))
