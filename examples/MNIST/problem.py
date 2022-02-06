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

from pytorch_lenet import Net as model
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class MNISTProblem(ScalarProblem):
    ''' LEAP Problem subclass that encapsulates a pytorch Net style model
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

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = model(checkpoint_path=MNISTProblem.model_chk_file)
        self.dataset = MNIST(MNISTProblem.data_path, transform=ToTensor(),
                        train=False, download=True)

        # Set up dict of mapping digits to indices where they are in the data
        self.count_dict = {i: [] for i in range(10)}
        for i, element in enumerate(self.dataset):
            self.count_dict[element[1]].append(i)


    def evaluate(self, phenome):
        '''
        Evaluate the phenome with the given model.

        :param phenome: is named tuple where phenome.digit is the current
            number to evaluate against the model
        :returns: score for model performance for this digit
        '''
        # Set up subset loader for the indices for the digit we want
        loader = torch.utils.data.Subset(self.dataset, self.count_dict[phenome.digit])

        predictions = []
        with torch.no_grad():
            for batch in loader:
                prediction = self.model(batch[0])
                prediction = torch.argmax(prediction)
                predictions.append(prediction)

        return nan
