#!/usr/bin/env python3
"""
    This is bringing over future LEAP functionality in anticipation of a
    new LEAP release.

    Currently adding support for binomial mutatation for integer
    representations.
"""
import numpy as np
from typing import Iterator
from toolz import curry
from leap_ec.ops import iteriter_op

##############################
# Function segmented_mutation()
##############################
@curry
@iteriter_op
def segmented_mutate(next_individual: Iterator, mutator_functions: list):
    """
    A mutation operator that applies a different mutation operator
    to each segment of a segmented genome.
    """
    while True:
        individual = next(next_individual)
        assert (len(individual.genome) == len(
            mutator_functions)), f"Found {len(individual.genome)} segments in this genome, but we've got {len(mutators)} mutators."

        mutated_genome = []
        for segment, m in zip(individual.genome, mutator_functions):
            mutated_genome.append(m(segment))

        individual.genome = mutated_genome

        # invalidate the fitness since we have a modified genome
        individual.fitness = None

        yield individual


##############################
# Function genome_mutate_binomial
##############################
def genome_mutate_binomial(std,
                           bounds: list,
                           expected_num_mutations: float = None,
                           probability: float = None,
                           n: int = 10000):
    """
    Perform additive binomial mutation of a particular genome.

    >>> import numpy as np
    >>> genome = np.array([42, 12])
    >>> bounds = [(0,50), (-10,20)]
    >>> genome_op = genome_mutate_binomial(std=0.5, bounds=bounds,
    ...                                         expected_num_mutations=1)
    >>> new_genome = genome_op(genome)

    """
    assert (bool(expected_num_mutations is not None) ^ bool(
        probability is not None)), f"Got expected_num_mutations={expected_num_mutations} and probability={probability}.  One must be specified, but not both."
    assert ((probability is None) or (probability >= 0))
    assert ((probability is None) or (probability <= 1))

    if isinstance(std, Iterable):
        p = np.array([_binomial_p_from_std(n, s) for s in std])
    else:
        p = _binomial_p_from_std(n, std)

    def mutator(genome):
        """Function to return as a closure."""
        if not isinstance(genome, np.ndarray):
            raise ValueError(("Expected genome to be a numpy array. "
                              f"Got {type(genome)}."))

        datatype = genome.dtype
        if probability is None:
            prob = compute_expected_probability(expected_num_mutations, genome)
        else:
            prob = probability

        selector = np.random.choice([0, 1], size=genome.shape,
                                    p=(1 - prob, prob))
        indices_to_mutate = np.nonzero(selector)[0]

        # Compute binomial parameters for each gene
        selected_p_values = p if not isinstance(p, Iterable) else p[
            indices_to_mutate]
        binom_mean = n * selected_p_values  # this will do elementwise multiplication if p is a vector

        # Apply binomial pertebations
        additive = np.random.binomial(n, selected_p_values,
                                      size=len(indices_to_mutate)) - np.floor(
            binom_mean)
        mutated = genome[indices_to_mutate] + additive
        genome[indices_to_mutate] = mutated

        genome = apply_hard_bounds(genome, bounds).astype(datatype)

        # consistency check on data type
        assert datatype == genome.dtype

        return genome

    return mutator


def _binomial_p_from_std(n, std):
    """Given a number of 'coin flips' n, compute the value of p that is
    needed to achieve a desired standard deviation."""
    if (4 * std ** 2 / n > 1):
        raise ValueError(
            f"The provided value of n ({n}) is too low to support a Binomial distribution with a standard deviation of {std}.  Choose a higher value of n, or reduce the std.")
    # We arrived at this expression by noting that σ^2 = np(1-p)
    # and solving for p via the quadratic formula
    return (1 - np.sqrt(1 - 4 * std ** 2 / n)) / 2
