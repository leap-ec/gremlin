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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('Gremlin finds features sets where a given machine '
                     'learning model performs poorly.'))
    parser.add_argument('-d', '--debug',
                        default=False, action='store_true',
                        help=('set debug flag to monitor values during a run'))
    parser.add_argument('config', type=str, nargs='+',
                        help=('path to configuration file(s) which Gremlin '
                              'uses to set up the problem and algorithm'))
    args = parser.parse_args()

    # set logger to debug if flag is set
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug('Logging set to DEBUG.')

    # combine configuration files into one dictionary
    configurations = []
    for config_path in args.config:
        logger.debug(f'Loading configuration {config_path}')
        cfg = OmegaConf.load(config_path)
        configurations.append(cfg)
    omegaconf_config = OmegaConf.merge(*configurations)
    config = OmegaConf.to_container(omegaconf_config, resolve=True)
    logger.debug(f'Fully merged configuration: {config}')

    # run gremlin algorithm
    run(config)
