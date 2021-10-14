# Gremlin

Gremlin is a machine learning model evaluator. Find out where your model performs poorly.

## Requires
* Python 3.[78]
* LEAP (https://github.com/AureumChaos/LEAP)

## How it works
It utilizes an adversarial evolutionary algorithm (EA) to find features where a model
performs poorly. The intent is for the user to leverage that information to tune training
data for subsequent model retraining to improve performance in those poor performing situations.

## Configuration
At a bare minimum, Gremlin needs an algorithm, a `Problem`, and a `Representation`. The
`Problem` and `Representation` should inherit from `leap_ec.problem.Problem` and
`leap_ec.representation.Representation`, respectively. The model to evaluate should be
handled within the custom `Problem` class.

Example configuration:

```
evolution:
    name: leap_ec.algorithm.generational_ea *or* custom_generator_function
    params:
        max_generations: 50
        pop_size: 25
        problem:
            name: leap_ec.problem.Problem *or* custom_class
            params:
                maximize: False
        representation:
            name: leap_ec.representation.Representation *or* custom_class
            params:
                initialize:
                    name: curried_initializer_function (see leap_ec.int_rep.create_int_vector)
                    params: {}
analysis:
    name: analysis_function
```

The `name:` field specifies the function or class to import. If this field is followed
by `params:` it will attempt to instantiate the function or class with the arguments that
follow prior to running the evolutionary algorithm.

## Example
Example code and configuration for a real problem can be found in `examples/MNIST`.
This problem involves Gremlin evolving patterns of occlusion (graying-out pixels of an
image) in order to cause a convolutional neural network to perform poorly on digit
recognition.

This can be run simply by (must be in `examples/MNIST` directory):

```
$ gremlin MNIST_config.yml
```

## Sub-directories
* `gremlin/` -- main `gremlin` code
* `examples/` -- examples for using gremlin
