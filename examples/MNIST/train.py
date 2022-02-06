'''
    Train the MNIST model

    TODO ensure loaded data has a built-in bias, such as removing an arbitrary
    digit.

    Much of this code is copied from:

    https://github.com/pytorch/examples/blob/master/mnist/main.py
'''
import os
import sys
import logging
from pathlib import Path

from rich import print
from rich import pretty
pretty.install()

from rich.traceback import install
install()

from rich.logging import RichHandler

# Create unique logger for this namespace
rich_handler = RichHandler(rich_tracebacks=True,
                           markup=True)
logging.basicConfig(level='INFO', format='%(message)s',
                    datefmt="[%Y/%m/%d %H:%M:%S]",
                    handlers=[rich_handler])
logger = logging.getLogger(__name__)

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from pytorch_net import Net


if __name__ == '__main__':

    data_path = Path('.') / 'data'
    if not data_path.exists():
        data_path.mkdir()

    batch_size = 100
    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    else:
        device = torch.device('cpu')

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    # download data
    train_dataset = MNIST(data_path, transform=ToTensor(),
                          transform=transform,
                          train=True, download=True)
    test_dataset = MNIST(data_path, transform=ToTensor(),
                         transform=transform,
                         train=False, download=True)

    # train a CNN on MNIST
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              **test_kwargs)

    logger.info(f'Started training')

    model = Net().to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=1, gamme=0.7)


    for epoch in tqdm(range(5)):
        pass # TODO add call to train()

    print(f'Saving model to {data_path / "model.pt"}...', end='')
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss}, data_path / 'model.pt')
    print('model saved.')
    model.eval()

    print('Testing on normal test set')
    correct = 0
    total = 0
    for i, (x, target) in enumerate(test_loader):
        x = x.to(device)
        target = target.to(device)
        pred = model(x)
        _, pred_label = torch.max(pred.data, 1)
        total += x.data.size()[0]
        correct += (pred_label == target.data).sum()

    print(f'Normal Test Set Accuracy: {correct / total}')
    print('Testing on altered dataset for multiple epochs')

    correct = 0
    total = 0
    # repeat many times to account for randomness
    # in occlusion patterns
    for e in tqdm(range(1)):
        for i, (x, target) in enumerate(test_loader):
            altered_images = []
            labels = []
            for img, label in zip(x, target):
                altered_img = alter_image(img)
                altered_images.append(altered_img)
                labels.append(label)
            x = torch.stack(altered_images)
            target = torch.stack(labels)

            x = x.to(device)
            target = target.to(device)
            pred = model(x)
            _, pred_label = torch.max(pred.data, 1)
            total += x.data.size()[0]
            correct += (pred_label == target.data).sum()

    print(f'Altered Test Set Accuracy: {correct / total}')
