#!/usr/bin/env python3
"""
gremlin.py

usage: gremlin.py [-h] [-d] config_files [config_files ...]

Gremlin finds features sets where a given machine learning model performs
poorly.

positional arguments:
  config_files  path to configuration file(s) which Gremlin uses to set up the
                problem and algorithm

optional arguments:
  -h, --help    show this help message and exit
  -d, --debug   enable debugging output
"""
import sys

# So we can pick up local modules defined in the YAML config file.
sys.path.append('.')

import argparse
import logging
import importlib

from omegaconf import OmegaConf

from rich.logging import RichHandler

# Create unique logger for this namespace
rich_handler = RichHandler(rich_tracebacks=True,
                           markup=True)
logging.basicConfig(level='INFO', format='%(message)s',
                    datefmt="[%Y/%m/%d %H:%M:%S]",
                    handlers=[rich_handler])
logger = logging.getLogger(__name__)

from rich import print
from rich import pretty

pretty.install()

from rich.traceback import install

install()

from leap_ec.algorithm import generational_ea
from leap_ec.probe import AttributesCSVProbe
from leap_ec.global_vars import context
from leap_ec import ops, util
from leap_ec.int_rep.ops import mutate_randint
from leap_ec.real_rep.ops import mutate_gaussian

from toolz import pipe


def read_config_files(config_files):
    """  Read one or more YAML files containing configuration options.

    The notion is that you can have a set of YAML files for controlling the
    configuration, such as having a set of default global settings that are
    overridden or extended by subsequent configuration files.

    E.g.,

    gremlin.py general.yaml this_model.yaml

    :param config_files: command line arguments
    :return: config object of current config
    """
    serial_configs = [OmegaConf.load(x) for x in config_files]
    config = OmegaConf.merge(*serial_configs)

    return config


def parse_config(config):
    """ Extract the population size, maximum generations to run, the Problem
    subclass, and the Representation subclass from the given `config` object.

    :param config: OmegaConf configurations read from YAML files
    :returns: pop_size, max_generations, Problem objects, Representation
        objects, LEAP pipeline operators
    """
    pop_size = int(config.pop_size)
    max_generations = int(config.max_generations)

    # The problem and representations will be something like
    # problem.MNIST_Problem, in the config and we just want to import
    # problem. So we snip out "problem" from that string and import that.
    globals()['problem'] = importlib.import_module(config.problem.split('.')[0])
    globals()['representation'] = importlib.import_module(
        config.representation.split('.')[0])

    if 'imports' in config:
        for extra_module in config.imports:
            globals()[extra_module] = importlib.import_module(extra_module)

    # Now snip out the class that was specified in the config file so that we
    # can properly instantiate that.
    problem_class = config.problem.split('.')[1]
    representation_class = config.representation.split('.')[1]

    problem_obj = eval(f'problem.{problem_class}()')
    representation_obj = eval(f'representation.{representation_class}()')

    # Eval each pipeline function to build the LEAP operator pipeline
    pipeline = [eval(x) for x in config.pipeline]

    return pop_size, max_generations, problem_obj, representation_obj, pipeline


def run_ea(pop_size, max_generations, problem, representation, pipeline,
           pop_file, k_elites=1):
    """ evolve solutions that show worse performing feature sets

    :param pop_size: population size
    :param max_generations: how many generations to run to
    :param problem: LEAP Problem subclass that encapsulates how to
        exercise a given model
    :param representation: how we represent features sets for the model
    :param pipeline: LEAP operator pipeline to be used in EA
    :param pop_file: where to write the population CSV file
    :param k_elites: keep k elites
    :returns: None
    """
    with open(pop_file, 'w') as pop_csv_file:
        # Taken from leap_ec.algorithm.generational_ea and modified pipeline
        # slightly to allow for printing population *after* elites are included
        # in survival selection to get accurate snapshot of parents for next
        # generation.

        # If birth_id is an attribute, print that column, too.
        attributes = ('birth_id',) if hasattr(representation.individual_cls,
                                             'birth_id') else []

        pop_probe = AttributesCSVProbe(stream=pop_csv_file,
                                       attributes=attributes,
                                       do_genome=True,
                                       do_fitness=True)

        # Initialize a population of pop_size individuals of the same type as
        # individual_cls
        parents = representation.create_population(pop_size, problem=problem)

        # Set up a generation counter that records the current generation to
        # context
        generation_counter = util.inc_generation(
            start_generation=0, context=context)

        # Evaluate initial population
        parents = representation.individual_cls.evaluate_population(parents)

        print('Best so far:')
        print('Generation, str(individual), fitness')
        bsf = max(parents)
        print(0, bsf)

        pop_probe(parents)  # print out the parents and increment gen counter
        generation_counter()

        while (generation_counter.generation() < max_generations):
            # Execute the operators to create a new offspring population
            offspring = pipe(parents, *pipeline,
                             ops.elitist_survival(parents=parents,
                                                  k=k_elites),
                             pop_probe
                             )

            if max(offspring) > bsf:  # Update the best-so-far individual
                bsf = max(offspring)

            parents = offspring  # Replace parents with offspring
            generation_counter()  # Increment to the next generation

            # Output the best-so-far individual for each generation
            print(generation_counter.generation(), bsf)


if __name__ == '__main__':
    logger.info('Gremlin started')

    parser = argparse.ArgumentParser(
        description=('Gremlin finds features sets where a given machine '
                     'learning model performs poorly.'))
    parser.add_argument('-d', '--debug',
                        default=False, action='store_true',
                        help=('enable debugging output'))
    parser.add_argument('config_files', type=str, nargs='+',
                        help=('path to configuration file(s) which Gremlin '
                              'uses to set up the problem and algorithm'))
    args = parser.parse_args()

    # set logger to debug if flag is set
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug('Logging set to DEBUG.')

    # combine configuration files into one dictionary
    config = read_config_files(args.config_files)
    logger.debug(f'Configuration: {config}')

    # Import the Problem and Representation classes specified in the
    # config file(s) as well as the pop size and max generations.
    pop_size, max_generations, problem, representation, pipeline = parse_config(
        config)

    # Then run leap_ec.generational_ea() with those classes while writing
    # the output to CSV and other, ancillary files.
    k_elites = int(config.k_elites) if 'k_elites' in config else 1
    run_ea(pop_size, max_generations, problem, representation, pipeline,
           config.pop_file, k_elites)

    logger.info('Gremlin finished.')
