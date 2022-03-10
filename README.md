# Gremlin discovers biases or weaknesses in machine learners

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

## Installation

1. Activate your conda or virtual environment
2. cd into top-level gremlin directory
3. `pip install .`

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
pop_size: 25
algorithm: async # or bgen
async: # parameters for asynchronous steady-state EA
  init_pop_size: ${pop_size}
  max_births: 2000
  ind_file: inds.csv # optional file for writing individuals as they are evaluated
  ind_file_probe: probe.log_ind # optional functor or function for writing ind_file

pop_file: pop.csv # where we will write out each generation in CSV format
problem: problem.QLearnerBalanceProblem("${env:GREMLIN_QLEARNER_CARTPOLE_MODEL_FPATH}")
representation: representation.BalanceRepresentation()
imports:
  - probe # need to import our probe.py so that LEAP sees our probe pipeline operator
pipeline: # isotropic means we mutate all genes with the given stds
  - ops.random_selection
  - ops.clone
  - mutate_gaussian(expected_num_mutations='isotropic', std=[0.1, 0.001, 0.01, 0.001], hard_bounds=representation.BalanceRepresentation.genome_bounds)
  - ops.pool(size=1)
```

Essentially, you will have to define the following

* A LEAP `Representation` subclass, as denoted in the above config file
  * This, in turn, will mean defining a LEAP `Decoder` and optionally a 
    bespoke LEAP `Individual` subclass.
    * The latter is mostly to override a `__str__()` member function for 
      pretty printing.
  * We also suggest using a python named tuple to define the genes in the 
    phenotype as a convenience.  (See `examples/MNIST/representation.py`)
* A LEAP `Problem` subclass.
  * This is the meat of defining your problem for Gremlin to tackle.
  * This subclass may be responsible for loading your models and then 
    applying them as denoted by genes values per individual.
  * E.g., the `examples/MNIST/representation.py` dictates that each 
    individual has a single gene that represents a given digit, and the 
    corresponding `problem.py` defines an `evaluate()` function that decodes 
    a given individuals digit as denoted in its gene, and then checks to see 
    how well the MNIST model predicts for that digit across a test dataset.
  * Similarly you will need to define a LEAP `Problem` subclass that is 
    responsible for loading datasets and models and then predicting how 
    effective they are for a given feature represented by a single individual.

## Example
Example code and configuration for a real problem can be found in `examples/MNIST`.
This problem involves Gremlin evolving patterns of occlusion (graying-out pixels of an
image) in order to cause a convolutional neural network to perform poorly on digit
recognition.

This can be run simply by (must be in `examples/MNIST` directory):

```
$ gremlin config.yml
```

## Versions

* `v0.3`
  * Add support for config variable `algorithm` that denotes if using a 
    traditional by-generation EA or an asynchronous steady-state EA
* `v0.2dev`, 2/17/22 
  * revamped config system and heavily 
    refactored/simplified code
* `v0.1dev`, 10/14/21 
  * initial raw release

## Sub-directories
* `gremlin/` -- main `gremlin` code
* `examples/` -- examples for using gremlin; currently only has MNIST example

## Main web site

The `gremlin` github repository is [https://github.com/markcoletti/gremlin]
(https://github.com/markcoletti/gremlin).  `main` is the release branch and 
active work occurs on the `develop` branch.
