#!/usr/bin/env python3
"""
    This defines a LEAP Problem subclass for the MNIST example.

    The Problem will decode an individual representing a parameter for the
    model, run the model, and return the accuracy.  This presumes that a
    MNIST model has already been trained via `train.py` that generated a
    `./data/model.pt`.
"""
from pathlib import Path
from numpy import nan
from leap_ec.problem import ScalarProblem

from pytorch_lenet import LeNet as model
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class MNISTProblem(ScalarProblem):
    ''' LEAP Problem subclass that encapsulates a pytorch LeNet style model
        inference used to evaluate how well it can recognize numbers.
    '''

    # Where the trained model is located
    data_path = Path('.') / 'data'
    model_chk_file = data_path / 'model.pt'

    def __init__(self):
        '''

        '''
        # We are _minimizing_ for accuracy; alternatively we could have
        # maximized for loss. I.e., gremlin wants to find where the model
        # performs poorly, not the best.
        super().__init__(maximize=False)

        self.model = model(checkpoint_path=MNISTProblem.model_chk_file)
        self.dataset = MNIST(MNISTProblem.data_path, transform=ToTensor(),
                        train=False, download=True)
        # self.loader = torch.utils.data.DataLoader(dataset=self.dataset,
        #
        #                                      # batch_size=batch_size,
        #                                      shuffle=True)
        # self.images, self.labels = next(iter(self.loader))

        # Set up dict of mapping digits to indices where they are in the data
        count_dict = {i: [] for i in range(10)}
        for i, element in enumerate(self.dataset):
            count_dict[element[1]].append(i)


    def evaluate(self, phenome):
        '''
        Evaluate the phenome in the given model, metric, and
        generator context.

        Parameters
        ----------
        phenome : Iterable
            features to use to generate data or as input
            to the model if not generating data

        Returns
        -------
        float
            performance of the model on the phenome
            determined by the metric
        '''
        # data = self.generator(phenome)
        # out = self.model(data)
        # score = self.metric(out, self.generator.labels)
        # TODO replace with call for running model on data.
        return nan
