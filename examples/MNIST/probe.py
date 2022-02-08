#!/usr/bin/env python3
''' Define a bespoke LEAP probe for printing individuals to a CSV file.
'''
import csv
from pathlib import Path

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
        self.csv_writer = csv.DictWriter(open(csv_file, 'w'),
                                         fieldnames=['birth_id','digit','fitness'])
        self.csv_writer.writeheader()

    def __call__(self, next_individual):
        """ append the individual to the CSV file

            This is a LEAP pipeline operator, so we accept an individual from
            the preceding pipeline operator, append it to the CSV file, and
            then pass the individual down the line to the next pipeline
            operator.
        """
        while True:
            individual = next(next_individual)
            phenome = individual.decode()

            self.csv_writer.writerow({'birth_id' : individual.birth_id,
                                      'digit' : phenome.digit,
                                      'fitness' : individual.fitness})

            yield individual
