"""
gremlin.py

Command-line client for gremlin.

This will take in various problem-dependent configuration files
and run a default evolutionary algorithm (EA) or a user-defined EA.

Custom classes can be specified in the configuration.
See <insert link to examples directory on code.ornl.gov>
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
    :returns: pop_size, max_generations, Problem objects, and Representation objects
    """
    pop_size = int(config.pop_size)
    max_generations = int(config.max_generations)

    # The problem and representations will be something like
    # problem.MNIST_Problem, in the config and we just want to import
    # problem. So we snip out "problem" from that string and import that.
    problem_module = importlib.import_module(config.problem.split('.')[0])
    representation_module = importlib.import_module(config.representation.split('.')[0])

    # Now snip out the class that was specified in the config file so that we
    # can properly instantiate that.
    problem_class = config.problem.split('.')[1]
    representation_class = config.representation.split('.')[1]

    problem = eval(f'problem_module.{problem_class}()')
    representation = eval(f'representation_module.{representation_class}()')

    return pop_size, max_generations, problem, representation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('Gremlin finds features sets where a given machine '
                     'learning model performs poorly.'))
    parser.add_argument('-d', '--debug',
                        default=False, action='store_true',
                        help=('set debug flag to monitor values during a run'))
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
    pop_size, max_generations, problem, representation = parse_config(config)

    # Then run leap_ec.generational_ea() with those classes while writing
    # the output to CSV and other, ancillary files.
    pass # TODO write this

    logger.info('Run finished.')
