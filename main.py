import os
import torch
from model import CNN, init_weights
from my_dataset import initialize_loader
from train import Trainer


def train_model():
    experiment = {
        'model_kernel': 3,
        'model_num_features': 16,
        'model_dropout_rate': 0.1,
        'train_class_weight': [.25, .15, .6],  # BALL, ROBOT, OTHER
        'train_learn_rate': 0.01,
        'train_batch_size': 16,
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
    model = CNN(kernel=3, num_features=16)
    model.load_state_dict(torch.load('outputs/model'))
    model.eval()
    [trainl, _, _], [traind, testd] = initialize_loader(6, num_workers=1, shuffle=False)
    testd.visualize_images(delay=100, model=model, start=1500)


def test_model():
    model = CNN(kernel=3, num_features=10, dropout=0.2)
    model.cuda()
    model.load_state_dict(torch.load('outputs/model'))
    trainer = Trainer(model, 0.01, 1, 20, 'outputs')
    trainer.test_model('test')


if __name__ == '__main__':
    display_dataset()
