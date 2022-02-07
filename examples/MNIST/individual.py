#!/usr/bin/env python3
"""
    DistributedIndividual subclass to allow for using UUIDs to create sub-
    directories to save CARLA state for each individual.
"""
from leap_ec..individual import RobustIndividual


class MNISTIndividual(RobustIndividual):
    """ We inherit from RobustIndividual so that exceptions are caught and
        processed such that fitness will be NaNs.
    """
    def __init__(self, genome, decoder=None, problem=None):
        super().__init__(genome, decoder, problem)

    def __str__(self):
        phenome = self.decode()
        return f'{self.birth_id}, {phenome}, {self.fitness}'
