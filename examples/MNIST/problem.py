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

from pytorch_net import Net as model
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class MNISTProblem(ScalarProblem):
    """ LEAP Problem subclass that encapsulates a pytorch Net style model
        inference used to evaluate how well it can recognize numbers.
    """

    def __init__(self):
        """
            We are minimizing for loss.
        """
        super().__init__(maximize=False)


    def evaluate(self, phenome):
        """ Evaluate a pytorch model decoded from an individual

        :param phenome: pytorch model
        :returns: final model error, or NaN if model is invalid
        """
