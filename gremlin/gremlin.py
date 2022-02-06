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

from tqdm import tqdm
from omegaconf import OmegaConf

from rich.logging import RichHandler

# Create unique logger for this namespace
rich_handler = RichHandler(rich_tracebacks=True,
                           markup=True)
logging.basicConfig(level='INFO', format='%(message)s',
                    datefmt="[%Y/%m/%d %H:%M:%S]",
                    handlers=[rich_handler])
logger = logging.getLogger(__name__)

from rich.table import Table
from rich import print
from rich import pretty

pretty.install()

from rich.traceback import install

install()

import analysis

from leap_ec.algorithm import generational_ea
from leap_ec.int_rep.ops import mutate_randint
from leap_ec import ops


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
    problem_module = importlib.import_module(config.problem.split('.')[0])
    representation_module = importlib.import_module(
        config.representation.split('.')[0])

    # Now snip out the class that was specified in the config file so that we
    # can properly instantiate that.
    problem_class = config.problem.split('.')[1]
    representation_class = config.representation.split('.')[1]

    problem = eval(f'problem_module.{problem_class}()')
    representation = eval(f'representation_module.{representation_class}()')

    # Finally snip out the defined pipeline; we need to remove the quotes so
    # that the operators are treated as partial functions and not as strings.
    pipeline_str = str(config.pipeline).replace('\'','')
    pipeline = []
    exec('pipeline = ' + str(pipeline_str))

    return pop_size, max_generations, problem, representation, pipeline


def run_ea(pop_size, max_generations, problem, representation, pipeline):
    """ evolve solutions that show worse performing feature sets

    :param pop_size: population size
    :param max_generations: how many generations to run to
    :param problem: LEAP Problem subclass that encapsulates how to
        exercise a given model
    :param representation: how we represent features sets for the model
    :param pipeline: LEAP operator pipeline to be used in EA
    :returns: None
    """
    generation = generational_ea(max_generations=max_generations,
                                 pop_size=pop_size,
                                 problem=problem,
                                 representation=representation,
                                 pipeline=pipeline)

    print(*list(generation), sep='\\n')


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
    pop_size, max_generations, problem, representation, pipeline = parse_config(config)

    # Then run leap_ec.generational_ea() with those classes while writing
    # the output to CSV and other, ancillary files.
    run_ea(pop_size, max_generations, problem, representation, pipeline)

    logger.info('Gremlin finished.')
