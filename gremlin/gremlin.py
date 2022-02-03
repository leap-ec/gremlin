"""
gremlin.py

Command-line client for gremlin.

This will take in various problem-dependent configuration files
and run a default evolutionary algorithm (EA) or a user-defined EA.

Custom classes can be specified in the configuration.
See <insert link to examples directory on code.ornl.gov>
"""
import sys
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



def check_config(config):
    '''
    Checks if the main configuration contains all of the necessary parts
    for running the algorithm.

    At a bare minimum the confiuration must follow this structure:

    evolution:
        name: custom_generator_function_or_class
        params: {}

    If the problem and representation are not specified, it should be
    handled within a custom evolution class or function.
    Custom classes should be defined in a module that can be imported.

    Parameters
    ----------
    config : dict
        Full configuration to check.

    Returns
    -------
    bool
        True if config meets the minimum requirements.
        False otherwise.
    '''
    if 'evolution' not in config:
        raise ValueError(('Top-level does not contain "evolution".'
                          f'\nFull Config: {config}'))
    if 'name' not in config['evolution']:
        raise ValueError(f'No algorithm specified. Full config: {config}')
    if 'params' not in config['evolution']:
        raise ValueError(('No parameters specified for the algorithm.'
                          f'Full config: {config}'))


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


def dynamic_import(name):
    '''
    Dynamically import a class or function from a string in
    the form "my_package.my_module.MyClass".

    Parameters
    ----------
    name : str
        import module name including class or function to import

    Returns
    -------
    abc.ABCMeta
        imported class or function
    '''
    module, class_or_func = name.rsplit('.', 1)
    module = importlib.import_module(module)
    class_or_func = getattr(module, class_or_func)
    return class_or_func


def load_all(config):
    '''
    Import and instantiate each pre-defined class or function using
    a depth first traversal thorugh the configruation.

    Parameters
    ----------
    config : dict
        Full and valid configuration for the Gremlin algorithm

    Returns
    -------
    dict
        Collapsed configuration into a single function to be called
    '''
    for key, val in config.items():
        # if the value is a dictionary with a "name" key then we
        # try to import it
        if isinstance(val, dict) and 'name' in val:
            class_or_func = dynamic_import(val['name'])
            # if the value has parameters then we instantiate it
            if 'params' in val:
                parameters = load_all(val['params'])
                config[key] = class_or_func(**parameters)
            # otherwise we only need the Callable function
            else:
                config[key] = class_or_func
        # special case of a list as a parameter
        if isinstance(val, list):
            for i, item in enumerate(val):
                if isinstance(item, dict) and 'name' in item:
                    class_or_func = dynamic_import(item['name'])
                    if 'params' in item:
                        parameters = load_all(item['params'])
                        val[i] = class_or_func(**parameters)
                    else:
                        val[i] = class_or_func
    return config


def run(config):
    '''
    Run the full Gremlin algorithm. This dispatches to any setup scripts
    and dynamically loads all modules and classes. It runs the evolutionary
    algorithm and outputs the final population.

    Parameters
    ----------
    config : dict
        Fully combined yaml with a top-level key 'evolution'

    Returns
    -------
    population : list
        final population of the evolutionary algorithm
    '''
    # check for all of the necessary definitions
    logger.debug('Checking for configuration validity...')
    check_config(config)
    logger.debug('Check complete.')

    # dynamically import classes and functions and replace each string
    # with the class (instantiated) or function
    logger.debug(('Attempting to load all classes and functions from the'
                  ' configuration...'))
    config = load_all(config)
    logger.debug('Loading complete.')

    # run the evolutionary algorithm
    logger.debug('Starting the evolutionary algorithm...')
    algorithm = config['evolution']
    best_of_generations = []
    logger.debug('generation, best-individual')
    for i, best in tqdm(algorithm):
        best_of_generations.append(best)
        logger.debug(f'{i}, {best}')
    logger.debug('Evolution complete.')

    # run the analysis
    logger.debug('Analyzing results...')
    if 'analysis' in config:
        analyzer = config['analysis']
        analyzer(best_of_generations)
    else:
        analysis.bsf_summary(best_of_generations)
    logger.debug('Done.')


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
