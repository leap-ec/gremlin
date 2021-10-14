'''
problem.py

Defines a Problem to be solved by an evolutionary algorithm
using LEAP.

A problem must have an `evaluate()` that returns the fitness
of an individual.
'''
from leap_ec.problem import ScalarProblem


class DatasetProblem(ScalarProblem):
    '''
    Defines the context of a typical
    dataset problem:

    Features -> [Generator] -> Model -> Metric

    Features are evolved by an evolutionary algorithm.
    Fitness is determined by the available metric.

    Attributes
    ----------
    model : Callable
        the model to evaluate
    metric : Callable
        the metric used to score the model output
    generator : Callable
        a method of generating data based on given
        features

    Methods
    -------
    evaluate(phenome)
        run input features (phenome) through the
        evaluator pipeline
    '''
    def __init__(self, model, metric, generator, maximize=False):
        '''
        A problem requires a model, metric function, and optionally
        a generator of new data to evaluate individuals

        Parameters
        ----------
        model : Callable
            the model to make predictions with
        metric : Callable
            the metric used to score the model output
        generator : Callable
            a method of generating data based on
            given features, default is None
        maximize : bool, optional
            whether fitness should be maximized or minimized
        '''
        super().__init__(maximize=maximize)
        self.model = model
        self.metric = metric
        self.generator = generator

    def evaluate(self, phenome):
        '''
        Evaluate the phenome in the given model, metric, and
        generator context.

        Parameters
        ----------
        phenome : Iterable
            features to use to generate data or as input
            to the model if not generating data

        Returns
        -------
        float
            performance of the model on the phenome
            determined by the metric
        '''
        data = self.generator(phenome)
        out = self.model(data)
        score = self.metric(out, self.generator.labels)
        return score

    def __str__(self):
        return DatasetProblem.__name__
