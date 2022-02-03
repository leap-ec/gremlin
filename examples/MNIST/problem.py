#!/usr/bin/env python3
"""
    This defines a LEAP Problem subclass for the MNIST example.

    The Problem will decode an individual representing a parameter for the
    model, run the model, and return the accuracy.  This presumes that a
    MNIST model has already been trained via `train.py` that generated a
    `./data/model.pt`.
"""
from leap_ec.problem import ScalarProblem
from numpy import nan


class MNISTProblem(ScalarProblem):
    ''' LEAP Problem subclass that encapsulates a pytorch LeNet style model
        inference used to evaluate how well it can recognize numbers.
    '''
    def __init__(self):
        '''

        '''
        # We are maximizing for accuracy; alternatively we could have
        # minimized for loss.
        super().__init__(maximize=True)
        # self.model = model
        # self.metric = metric
        # self.generator = generator

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
