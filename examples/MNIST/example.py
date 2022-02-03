'''
example.py

MNIST occlusion problem.

Gremlin will find patterns of row/column occlusion that
cause the model to perform poorly.

This file defines the model, decoder, generator, and analyzer
dynamically imported, instantiated, and used by the Gremlin
interface.

Training the model is separate from Gremlin.
'''
import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from leap_ec.decoder import Decoder


class RowColDecoder(Decoder):
    '''
    Decide which rows and columns are grayed out
    '''
    def decode(self, genome, *args, **kwargs):
        row_indices = genome[genome < 28]
        col_indices = genome[genome >= 28]
        col_indices = col_indices - 28
        return [row_indices, col_indices]


class LeNet(nn.Module):
    def __init__(self, checkpoint_path=None):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.lin1 = nn.Linear(4*4*50, 500)
        self.lin2 = nn.Linear(500, 10)
        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path)
            self.load_state_dict(ckpt['model_state_dict'])
            self.eval()

    def forward(self, xx):
        xx = F.relu(self.conv1(xx))
        xx = F.max_pool2d(xx, 2, 2)
        xx = F.relu(self.conv2(xx))
        xx = F.max_pool2d(xx, 2, 2)
        xx = xx.view(-1, 4*4*50)
        xx = F.relu(self.lin1(xx))
        return self.lin2(xx)


def MNIST_heatmap(population):
    '''
    Generate heatmaps of the population
    genome using Gremlin's output
    '''
    genomes = [ind.genome for ind in population]
    genomes = np.stack(genomes)
    rows = genomes[genomes < 28]
    cols = genomes[genomes >= 28] - 28
    heatmap = np.zeros((28, 28))
    for row in rows:
        heatmap[row, :] += 1
    for col in cols:
        heatmap[:, col] += 1
    plt.imshow(heatmap, cmap='hot')
    plt.title('Population Occlusion Frequency')
    plt.savefig('MNIST_heatmap.png')
    plt.show()


class MNISTRowColOcclusionGenerator:
    '''
    Modifies a dataset of images by "graying-out"
    rows and columns of an image

    The dataset must be of the form (N, C, H, W)
    where N is the number of images, C is the number
    of channels (only supports 1 and 3),
    H is the height of the image,
    and W is the width of the image.

    Attributes
    ----------
    dataset : np.array
        set of images to alter
        required dimensions (N, C, H, W)

    Methods
    -------
    transform(image, rows, columns)
        grey out a row/column of an image
    '''
    def __init__(self, batch_size, **kwargs):
        dataset = MNIST('./data/', transform=ToTensor(),
                        train=False, download=True)
        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=batch_size,
                                             shuffle=True)
        self.images, self.labels = next(iter(loader))

    def transform(self, image, rows, columns):
        # supports grayscale, rgb, rgba
        if image.shape[0] in [1, 3, 4]:
            for c in range(image.shape[0]):
                image[c, rows, :] = 0.5
                image[c, :, columns] = 0.5
        else:
            raise ValueError(
                f'Unsupported image dimensions {image.shape}')
        return image

    def __call__(self, features):
        '''
        Generate a new dataset modifying by features

        Parameters
        ----------
        features : list
           which rows and columns to obfuscate
           features[0] has rows
           features[1] has columns
        '''
        # transform images in the dataset
        images = copy.deepcopy(self.images)
        for i in range(len(images)):
            images[i] = self.transform(images[i],
                                       features[0],
                                       features[1])
        return images
