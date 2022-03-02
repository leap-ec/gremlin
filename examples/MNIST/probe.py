#!/usr/bin/env python3
''' Define a bespoke LEAP probe for printing individuals to a CSV file.
'''
import sys
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
        self.csv_writer = csv.DictWriter(self.csv_file.open('w'),
                                         fieldnames=['birth_id',
                                                     'digit',
                                                     'start_eval_time',
                                                     'stop_eval_time',
                                                     'fitness'])
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

            self.csv_writer.writerow({'birth_id'       : individual.birth_id,
                                      'digit'          : phenome.digit,
                                      'start_eval_time': individual.start_eval_time,
                                      'stop_eval_time' : individual.stop_eval_time,
                                      'fitness'        : individual.fitness})

            yield individual



def log_ind(stream=sys.stdout, header=True):
    """ This is for logging individuals for the asynchronous EA in more detail
    than the optional logging individuals.

    :param stream: to which we want to write the machine details
    :param header: True if we want a header for the CSV file
    :return: a function for recording where individuals are evaluated
    """
    stream = stream
    writer = csv.DictWriter(stream,
                            fieldnames=['hostname', 'pid', 'uuid', 'birth_id',
                                        'digit',
                                        'start_eval_time', 'stop_eval_time',
                                        'fitness'])

    if header:
        writer.writeheader()

    def write_record(individual):
        """ This writes a row to the CSV for the given individual

        evaluate() will tack on the hostname and pid for the individual.  The
        uuid should also be part of the distrib.Individual, too.

        :param individual: to be written to stream
        :return: None
        """
        nonlocal stream
        nonlocal writer

        writer.writerow({'hostname': individual.hostname,
                         'pid': individual.pid,
                         'uuid': individual.uuid,
                         'birth_id': individual.birth_id,
                         'digit': individual.genome[0],
                         'start_eval_time': individual.start_eval_time,
                         'stop_eval_time': individual.stop_eval_time,
                         'fitness': individual.fitness})
        # On some systems, such as Summit, we need to force a flush else there
        # will be no output until the very end of the run.
        stream.flush()

    return write_record
