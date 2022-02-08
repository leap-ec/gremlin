#!/usr/bin/env python3
''' Define a bespoke LEAP probe for printing individuals to a CSV file.
'''
from pathlib import Path
import sys

from rich import print
from rich import pretty
pretty.install()

from rich.traceback import install
install()

class IndividualProbeCSV():
    """
        Will write out each individual as it receives it in a pipeline and then
        pass it down the pipeline.
    """
    def __init__(self, csv_file):
        super().__init__()
        self.csv_file = Path(csv_file)

    def __call__(self, next_individual):
        """ append the individual to the CSV file
        """
        while True:
            individual = next(next_individual)
            print(f'{individual!s}', file=sys.stderr)

            yield individual
