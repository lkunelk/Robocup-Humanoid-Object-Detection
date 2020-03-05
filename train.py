import os
import time
import numpy as np
import torch
from my_dataset import initialize_loader

def train(model,
        learn_rate=0.01,
        train_batch_size=40,
        val_batch_size=64,
        epochs=10):

    np.random.seed(1)

    # Save directory
    save_dir = "outputs/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    train_loader, valid_loader = initialize_loader(train_batch_size, val_batch_size)
    print(len(train_loader))

    model.cuda()

    start = time.time()

    train_loss = []
    valid_loss = []
    valid_ious = []

    for epoch in range(epochs):
        model.train()

        start_train = time.time()

        losses = []
        for i, (images, masks) in enumerate(train_loader):
            images = images.cuda()
            masks = masks.cuda()

            optimizer.zero_grad()

            predictions = model(images.float())
            loss = torch.nn.functional.binary_cross_entropy(predictions, masks.float())
            loss.backward()
            optimizer.step()
            losses.append(loss.data.item())

            if i == 200:
                print()
                break

        time_elapsed = time.time() - start_train
        print('Epoch [{}/{}], Loss: {}, Time (s): {}'.format(epoch + 1, epochs, losses[-1], time_elapsed), end='')
