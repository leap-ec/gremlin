'''
MNIST_optimize.py

Partially damage MNIST dataset in a controlled way
Train CNN classifier on damaged dataset
Test CNN classifier on normal dataset and on fully damaged dataset
Run Gremlin to find soft spots created by the partial damage
Retrain with emphasis on the soft spots suggested by Gremlin
Test CNN classifier on normal dataset and on fully damaged dataset

Did the model improve after retraining?
'''
import os
import sys

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from pytorch_lenet import Net


# max number of rows/columns to occlude
# or number of pixels to occlude
LENGTH = 10

# exclude these rows/cols from being occluded during training
# this excludes every 4th row/col
EXCLUDE = set(list(range(11, 16)) + list(range(11+28, 16+28)))


def alter_image(img, exclude=[], probabilities=None):
    '''
    Randomly occlude a row or column of an image
    unless in exclude and return a copy
    '''
    copy_img = img.clone()
    changes = 0
    while changes < LENGTH:
        # randomly choose row/col
        if probabilities is None:
            index = np.random.randint(0, 28+28)
        else:
            index = np.random.choice(list(range(0, 28+28)), p=probabilities)
        # do not change these rows/cols so repeat
        if index in exclude:
            continue
        # row
        if index < 28:
            copy_img[:, index, :] = 0.5
        elif index >= 28:
            copy_img[:, :, index - 28] = 0.5
        changes += 1
    return copy_img


if __name__ == '__main__':

    # options are {train, eval} PATH
    data_dir = './data/'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    # enable CUDA
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # download data
    train_dataset = MNIST(data_dir, transform=ToTensor(),
                          train=True, download=True)
    test_dataset = MNIST(data_dir, transform=ToTensor(),
                         train=False, download=True)

    # train a CNN on MNIST
    batch_size = 100
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              shuffle=True)

    print(f'Training CNN model with {EXCLUDE} rows/cols excluded')
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_func = nn.CrossEntropyLoss()

    for epoch in tqdm(range(5)):
        for i, (x, target) in enumerate(train_loader):
            altered_images = []
            labels = []
            for img, label in zip(x, target):
                altered_img = alter_image(img, exclude=EXCLUDE)
                altered_images.append(altered_img)
                labels.append(label)
            altered_images = torch.stack(altered_images)
            labels = torch.stack(labels)

            x = torch.cat((x, altered_images))
            target = torch.cat((target, labels))

            optimizer.zero_grad()
            x = x.to(device)
            target = target.to(device)
            pred = model(x)
            loss = loss_func(pred, target)
            loss.backward()
            optimizer.step()

    print(f'Saving model to {data_dir + "model.pt"}...', end='')
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss}, data_dir + 'model.pt')
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
