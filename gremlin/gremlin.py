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

from distributed import Client, LocalCluster

from leap_ec.algorithm import generational_ea
from leap_ec.probe import AttributesCSVProbe
from leap_ec.global_vars import context
from leap_ec import ops, util
from leap_ec.int_rep.ops import mutate_randint, mutate_binomial
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.distrib import DistributedIndividual
from leap_ec.distrib import asynchronous
from leap_ec.distrib.probe import log_worker_location, log_pop

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
    :returns: Problem objects, Representation objects, LEAP pipeline operators
    """
    # The problem and representations will be something like
    # problem.MNIST_Problem, in the config and we just want to import
    # problem. So we snip out "problem" from that string and import that.
    globals()['problem'] = importlib.import_module(config.problem.split('.')[0])
    globals()['representation'] = importlib.import_module(
        config.representation.split('.')[0])

    if 'imports' in config:
        for extra_module in config.imports:
            globals()[extra_module] = importlib.import_module(extra_module)

    # Now instantiate the problem and representation objects, including any
    # ctor arguments.
    problem_obj = eval(config.problem)
    representation_obj = eval(config.representation)

    # Eval each pipeline function to build the LEAP operator pipeline
    pipeline = [eval(x) for x in config.pipeline]

    return problem_obj, representation_obj, pipeline


def run_generational_ea(pop_size, max_generations, problem, representation,
                        pipeline,
                        pop_file, k_elites=1):
    """ evolve solutions that show worse performing feature sets using a
    by-generation evolutionary algorithm (as opposed to an asynchronous,
    steady state evolutionary algorithm)

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


def run_async_ea(pop_size, init_pop_size, max_births, problem, representation,
                 pipeline,
                 pop_file,
                 ind_file,
                 scheduler_file=None):
    """ evolve solutions that show worse performing feature sets using an
    asynchronous steady state evolutionary algorithm (as opposed to a by-
    generation EA)

    :param pop_size: population size
    :param init_pop_size: the size of the initial random population, which
        can be different from the constantly updated population size that is
        dictated by `pop_size`; this is generally set to the number of
        available workers, but doesn't have to be
    :param max_births: how many births to run to
    :param problem: LEAP Problem subclass that encapsulates how to
        exercise a given model
    :param representation: how we represent features sets for the model
    :param pipeline: LEAP operator pipeline to be used to create a
        **single offspring**
    :param pop_file: where to write the CSV file of snapshot of population
        given every `pop_size` births
    :param scheduler_file: optional dask scheduler file; will use cores on local
        host if none given
    :returns: None
    """
    if scheduler_file:
        logger.debug('Using cluster for dask')
    else:
        logger.debug('Using all localhost cores for dask')

    track_pop_stream = open(pop_file, 'w')
    track_pop_func = log_pop(pop_size, track_pop_stream)

    track_ind_func = None
    if ind_file is not None:
        track_ind_stream = open(ind_file, 'w')
        track_ind_func = log_worker_location(track_ind_stream)

    with Client(scheduler_file=scheduler_file) as client:
        final_pop = asynchronous.steady_state(client,
                                              births=max_births,
                                              init_pop_size=init_pop_size,
                                              pop_size=pop_size,

                                              representation=representation,

                                              problem=problem,

                                              offspring_pipeline=pipeline,

                                              evaluated_probe=track_ind_func,
                                              pop_probe=track_pop_func)

        print('Final pop: \n%s', final_pop)


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
    # config file(s) as well as the LEAP pipeline of operators
    problem, representation, pipeline = parse_config(config)

    pop_size = int(config.pop_size)

    if config.algorithm == 'async':
        logger.debug('Using async EA')

        scheduler_file = None if 'scheduler_file' not in config['async'] else \
        config['async'].scheduler_file

        run_async_ea(pop_size,
                     int(config['async'].max_births),
                     int(config['async'].init_pop_size),
                     problem, representation, pipeline,
                     config.pop_file, scheduler_file)
    elif config.algorithm == 'bygen':
        # default to by generation approach
        logger.debug('Using by-generation EA')

        # Then run leap_ec.generational_ea() with those classes while writing
        # the output to CSV and other, ancillary files.
        max_generations = int(config.bygen.max_generations)
        k_elites = int(config.bygen.k_elites) if 'k_elites' in config else 1

        run_generational_ea(pop_size, max_generations, problem, representation,
                            pipeline,
                            config.pop_file, k_elites)
    else:
        logger.critical(f'Algorithm type {config.algorithm} not supported')
        sys.exit(1)

    logger.info('Gremlin finished.')
