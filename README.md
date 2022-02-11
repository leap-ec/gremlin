# Gremlin, an adversarial evolutionary algorithm that discovers biases or weaknesses in machine learners

Gremlin learns where a given machine learner (ML) model performs poorly via an
adversarial evolutionary algorithm (EA).  The EA will find the worst 
performing feature sets such that a practitioner can then, say, tune the 
training data to include more examples of those feature sets.  Then the ML
model can be trained again with the updated training set in the hopes that 
the additional examples will be sufficient for the ML to train models that 
perform better for those sets.

## Requires
* Python 3.[78]
* [LEAP https://github.com/AureumChaos/LEAP](https://github.com/AureumChaos/LEAP)


## Configuration
Gremlin is essentially a thin convenience wrapper around [LEAP]
(https://github.com/AureumChaos/LEAP).  Instead of writing a script in LEAP, 
one would instead point the `gremlin` executable at a YAML file that describes 
what LEAP classes, subclasses, and functions to use, as well as other salient 
run-time characteristics. `gremlin` will parse the YAML file and generate a 
CSV file containing the individuals from the run.  This CSV file should 
contain information that can be exploited to tune training data.

Example Gremlin configuration YAML:

```yaml
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
