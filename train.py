import os
import time
import numpy as np
import torch
from my_dataset import initialize_loader


def train_epoch():
    #hi
    pass

def train(model,
          learn_rate=0.001,
          train_batch_size=100,
          val_batch_size=64,
          epochs=50):
    np.random.seed(1)

    # Save directory
    save_dir = "outputs/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    train_loader, valid_loader = initialize_loader(train_batch_size, val_batch_size)
    print('# of batches {}'.format(len(train_loader)))

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
            loss = torch.nn.functional.binary_cross_entropy(predictions, masks.float())
            loss.backward()
            optimizer.step()
            losses.append(loss.data.item())

            t_readimg = time.time()
        time_elapsed = time.time() - start_train
        print('Epoch [{}/{}], Loss: {}, Avg. Batch Load (s): {}, Epoch (s): {}'.format(
            epoch + 1,
            epochs,
            losses[-1],
            sum(batchload_times) / len(batchload_times),
            time_elapsed))
