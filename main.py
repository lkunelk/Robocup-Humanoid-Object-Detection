import os
import torch
from model import CNN, init_weights
from my_dataset import initialize_loader
from train import Trainer


def train_model():
    experiment = {
        'model_kernel': 3,
        'model_num_features': 16,
        'model_dropout_rate': 0.3,
        'train_class_weight': [1.0, 1.0, 1.0],  # BALL, ROBOT, OTHER
        'train_learn_rate': 0.05,
        'train_batch_size': 64,
        'train_epochs': 20,
        'output_folder': 'outputs',
    }

    print(experiment)

    model = CNN(
        kernel=experiment['model_kernel'],
        num_features=experiment['model_num_features'],
        dropout=experiment['model_dropout_rate'])

    # Save directory
    output_folder = experiment['output_folder']
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model.apply(init_weights)
    # model.load_state_dict(torch.load('outputs/model'))

    trainer = Trainer(model,
                      learn_rate=experiment['train_learn_rate'],
                      batch_size=experiment['train_batch_size'],
                      epochs=experiment['train_epochs'],
                      output_folder=experiment['output_folder'],
                      class_weights=experiment['train_class_weight'])
    trainer.train()

    torch.save(model.state_dict(), 'outputs/model')


def display_dataset():
    [trainl, _, _], [traind, _, testd] = initialize_loader(6, num_workers=1, shuffle=False)
    traind.visualize_images(delay=10)


def test_model():
    model = CNN(kernel=3, num_features=10, dropout=0.2)
    model.cuda()
    model.load_state_dict(torch.load('outputs/model'))
    trainer = Trainer(model, 0.01, 1, 20, 'outputs')
    trainer.test_model('test')


if __name__ == '__main__':
    train_model()
