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
import multiprocessing

# So we can pick up local modules defined in the YAML config file.
sys.path.append('.')

import argparse
import logging
import importlib

from omegaconf import OmegaConf

import rich
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
from rich.pretty import pprint


rich.traceback.install(show_locals=True)

from rich.console import Console
console = Console()




from distributed import Client, LocalCluster

from leap_ec.algorithm import generational_ea
from leap_ec.probe import AttributesCSVProbe
from leap_ec.global_vars import context
from leap_ec import ops, util
from leap_ec.int_rep.ops import mutate_randint, genome_mutate_binomial
from leap_ec.real_rep.ops import mutate_gaussian, genome_mutate_gaussian
from leap_ec.distrib import DistributedIndividual
from leap_ec.segmented_rep.ops import add_segment, remove_segment, apply_mutation
from leap_ec.distrib import asynchronous
from leap_ec.distrib import synchronous
from leap_ec.distrib.probe import log_worker_location, log_pop
from leap_ec.distrib.logger import WorkerLoggerPlugin

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

    As side-effect will instantiate `distribued.client` as Dask client.  Also
    anything in `config.preamble` will be exec()'d into the global space.

    :param config: OmegaConf configurations read from YAML files
    :returns: Problem objects and Representation object and LEAP pipeline
        operators, as well as optional with_client_exec_str
    """

    if 'preamble' in config:
        # This allows for imports and defining functions referred to later in
        # the pipeline
        exec(config.preamble, globals())

    if 'distributed' in config:
        # Then we have some Dask distributed stuff to pull in
        if 'client' in config.distributed:
            globals()['client'] = eval(config.distributed.client)

    # The problem and representations will be something like
    # problem.MNIST_Problem, in the config and we just want to import
    # problem. So we snip out "problem" from that string and import that.
    globals()['problem'] = importlib.import_module(config.problem.split('.')[0])
    globals()['representation'] = importlib.import_module(
        config.representation.split('.')[0])

    # Now instantiate the problem and representation objects, including any
    # ctor arguments.
    problem_obj = eval(config.problem)
    representation_obj = eval(config.representation)

    return problem_obj, representation_obj


def run_generational_ea(pop_size, max_generations, problem, representation,
                        pipeline_list,
                        pop_file, k_elites=1, client=None):
    """ evolve solutions that show worse performing feature sets using a
    by-generation evolutionary algorithm (as opposed to an asynchronous,
    steady state evolutionary algorithm)

    If `client` is set, then we will use parallel fitness evaluations for the
    initial population.  However, since the user has control of the pipeline
    then they have to be mindful to use LEAP `distrib.ops.eval_pool()` for
    parallel evaluations of offspring fitnesses.

    FIXME this puts perhaps an undue burden on the user.

    :param pop_size: population size
    :param max_generations: how many generations to run to
    :param problem: LEAP Problem subclass that encapsulates how to
        exercise a given model
    :param representation: how we represent features sets for the model
    :param pipeline_list: strings of python for LEAP operator pipeline
    :param pop_file: where to write the population CSV file
    :param k_elites: keep k elites
    :param client: optional Dask client
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
        generation_counter = util.inc_generation(context=context)

        # Evaluate initial population
        if client is None:
            parents = representation.individual_cls.evaluate_population(parents)
        else:
            parents = synchronous.eval_population(parents, client=client)

        print('Best so far:')
        print('Generation, str(individual), fitness')
        bsf = max(parents)
        print(0, bsf)

        pop_probe(parents)  # print out the parents and increment gen counter
        generation_counter()

        # Eval each pipeline function to build the LEAP operator pipeline
        pipeline = [eval(x, globals(), {'parents': parents})
                    for x in pipeline_list]

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

            # Output the best-so-far individual for each generation
            print(generation_counter.generation(), bsf)

            generation_counter()  # Increment to the next generation

        if client is not None:
            client.close()


def run_async_ea(pop_size, init_pop_size, max_births, problem, representation,
                 pipeline_list,
                 pop_file,
                 ind_file,
                 ind_file_probe,
                 client):
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
    :param pipeline_list: string of LEAP operator pipeline to be used to create a
        **single offspring**
    :param pop_file: where to write the CSV file of snapshot of population
        given every `pop_size` births
    :param ind_file: where to write the CSV file of each individual just as
        it is evaluated
    :param ind_file_probe: optional function (or functor) for printing out
        individuals to ind_file; if not specified, then
        `leap_ec.distrib.probe.log_worker_location` is used by default
    :param client_str: is the python statement for creating a Dask client
    :param with_client_exec_str: is optional python code for *after* the client
        has been spun up, and which allows for installing Dask plugins.
    :returns: None
    """
    track_pop_stream = open(pop_file, 'w')
    track_pop_func = log_pop(pop_size, track_pop_stream)

    track_ind_func = None
    if ind_file is not None:
        if ind_file_probe is None:
            track_ind_stream = open(ind_file, 'w')
            track_ind_func = log_worker_location(track_ind_stream)
        else:
            track_ind_func = eval(ind_file_probe + '(open(ind_file,"w"))')

    try:
        # Eval each pipeline function to build the LEAP operator pipeline
        pipeline = [eval(x) for x in pipeline_list]

        final_pop = asynchronous.steady_state(client,
                                              max_births=max_births,
                                              init_pop_size=init_pop_size,
                                              pop_size=pop_size,

                                              representation=representation,

                                              problem=problem,

                                              offspring_pipeline=pipeline,

                                              evaluated_probe=track_ind_func,
                                              pop_probe=track_pop_func)

        print('Final pop:')
        print([str(x) for x in final_pop])
    finally:
        client.close()


def get_dask_client(config):
    """ Open and return a Dask client.

        It is expected that you will close it elsewhere.  Also, if
        distributed.with_client exists, all python code there will be executed
        after the client is started to allow for such things as installing
        plugins and waiting for so many workers to come online.

        :param config: Omegaconf configuration
        :returns: Active Dask client
    """
    if 'distributed' in config:
        client = eval(config.distributed.client)

        logger.debug(f'Dask client: {client!s}')

        with_client_exec_str = config.distributed.get('with_client')

        if with_client_exec_str is not None:
            # Execute any user supplied code with the client connected.
            # This allows for tailored plugins, client.upload_file(), and
            # similar invocations to be handled.  These will be found in the
            # optional `with_client` sections in Gremlin YAML config files.
            exec(with_client_exec_str, globals(), locals())

        # Add a logger that is local to each worker
        client.register_worker_plugin(WorkerLoggerPlugin())

        return client
    else:
        logger.warning(f'There is no "distributed" YAML configuration, which '
                       f'may be ok if you are not planning on using Dask for '
                       f'parallel fitness evaluations.')

def main():
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

    # combine configuration files into one dictionary
    config = read_config_files(args.config_files)

    # set logger to debug if flag is set and print out the details of the
    # read in configuration
    if args.debug:
        pretty.pprint(OmegaConf.to_container(config, resolve=True))
        logger.setLevel(logging.DEBUG)
        logger.debug('Logging set to DEBUG.')

    # Import the Problem and Representation classes specified in the
    # config file(s) as well as the LEAP pipeline of operators
    problem, representation = parse_config(config)

    # We explicitly cast to an int because the YAML file may have gotten the
    # value from an environment variable, which means that it'll be treated as
    # a string instead of a number.  So to make sure we cast to int.
    pop_size = int(config.pop_size)

    client = get_dask_client(config)

    try:
        if config.algorithm == 'async':
            logger.debug('Using async EA')

            ind_file = None if 'ind_file' not in config['async'] else \
                config['async'].ind_file

            ind_file_probe = None if 'ind_file_probe' not in config['async'] else \
                config['async'].ind_file_probe


            run_async_ea(pop_size,
                         int(config['async'].init_pop_size),
                         int(config['async'].max_births),
                         problem, representation, config.pipeline,
                         config.pop_file,
                         ind_file,
                         ind_file_probe,
                         client)
        elif config.algorithm == 'bygen':
            # default to by generation approach
            logger.debug('Using by-generation EA')

            # Then run leap_ec.generational_ea() with those classes while writing
            # the output to CSV and other, ancillary files.
            max_generations = int(config.bygen.max_generations)
            k_elites = int(config.bygen.k_elites) if 'k_elites' in config else 1

            # This is for optional code to be executed after the Dask client has
            # been established, but before execution of the EA.  This allows for
            # things like client.wait_for_workers() or client.upload_file() or the
            # registering of dask plugins.  This is a string that will be `exec()`
            # later after a dask client has been connected.
            # TODO LEAP does not (yet) support Dask for by-generation. Soon!
            # with_client_exec_str = None if 'with_client' not in config['bygen'] else \
            #     config['bygen'].with_client

            run_generational_ea(pop_size, max_generations, problem, representation,
                                config.pipeline,
                                config.pop_file, k_elites)
        else:
            logger.critical(f'Algorithm type {config.algorithm} not supported')
            sys.exit(1)
    except Exception as e:
        logger.critical(f'Caught {e!s} during run.  Exiting.')
        console.print_exception()

    logger.info('Gremlin finished.')



if __name__ == '__main__':
    main()
