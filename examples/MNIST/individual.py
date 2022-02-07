#!/usr/bin/env python3
"""
    DistributedIndividual subclass to allow for using UUIDs to create sub-
    directories to save CARLA state for each individual.
"""
from leap_ec.distrib.individual import DistributedIndividual


class MNISTIndividual(DistributedIndividual):
    """ We inherit from DistributedIndividual so that exceptions are caught and
        processed such that fitness will be NaNs.  We are also setting up
        for additional bookkeeping for eval start and stop times that are
        managed by that class, as well as tracking unique birth IDs when we
        start using leap_ec.distrib EAs.
    """
    def __init__(self, genome, decoder=None, problem=None):
        super().__init__(genome, decoder, problem)

    def __str__(self):
        phenome = self.decode()
        return f'{self.birth_id}, {phenome.digit}, {self.fitness}'
