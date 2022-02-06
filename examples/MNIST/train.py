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
from torch.optim.lr_scheduler import StepLR

from pytorch_net import Net


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target,
                                    reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1,
                                 keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':

    data_path = Path('.') / 'data'
    if not data_path.exists():
        data_path.mkdir()

    batch_size = 64  # from original pytorch MNIST example
    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory' : True,
                       'shuffle'    : True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    else:
        device = torch.device('cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # download data
    train_dataset = MNIST(data_path,
                          transform=transform,
                          train=True, download=True)
    test_dataset = MNIST(data_path,
                         transform=transform,
                         train=False, download=True)

    # train a CNN on MNIST
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              **test_kwargs)

    logger.info(f'Started training')

    model = Net().to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    scheduler = StepLR(optimizer, step_size=1, gamme=0.7)

    for epoch in range(14):  # magic number 14 from original source
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    torch.save(model.state_dict(), data_path / "model.pt")
