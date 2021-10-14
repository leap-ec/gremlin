'''
analysis.py

Module for analyzing the results produced by Gremlin.
'''
import pandas as pd


def bsf_summary(population):
    '''
    Computes summary statistics over the best-so-far individuals
    in each generation of an evolutionary algorithm.

    Parameters
    ----------
    population : list
        best performing individual in each generation
    '''
    genomes = [ind.genome for ind in population]
    fitnesses = [ind.fitness for ind in population]
    table = {'genome': genomes, 'fitness': fitnesses}
    df = pd.DataFrame(table)
    print(df.to_string())
    print(df.describe())
