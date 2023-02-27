# Gremlin MNIST example
The [Modified National Institute of Standards and Technology (MNIST)]
(https://en.wikipedia.org/wiki/MNIST_database) is a standard benchmark 
machine learning training dataset comprised of a large set of hand drawn digits.

We arbitrarily remove a large fraction of training data for the digit '3' so 
that we can verify that Gremlin identifies that the trained model struggles 
to identify those digits.  Yes, using an evolutionary algorithm (EA) would 
normally be overkill for that simple of a problem.  However, this is 
adequate as a test and example of Gremlin's capabilities.

## To run this example

To apply Gremlin to this training set you will have to:

1. first train the model with the MNIST dataset damaged by removing the 
   digit '3'
2. run Gremlin against the dataset with the given Gremlin YAML configuration 
   file
3. observe that Gremlin found that the ML struggles identifying images for '3'

## Contents


* `individual.py` -- This is an optional file where we just define a way to 
  pretty print individuals and to track eval times.
* `probe.py` -- Another optional file where we define a LEAP pipeline 
  operator for a bespoke probe that will print every new evaluated offspring;
  this probe is referenced in `config.yml`.
* `problem.py` -- Where a LEAP `Problem` subclass is defined that loads the 
  MNIST model and evaluates it for a given digit as dictated by a given 
  individual.
* `pytorch_net.py` -- The pytorch-based model for MNIST classification
* `representation.py` -- LEAP `Representation` subclass that specifies how 
  we represent the MNIST problem for the evolutionary algorithm, Gremlin
* `train.py` - Used to train the MNIST model; intentionally will damage the 
  training data such that we arbitrarily reduce the training instances for 
  '3' such that Gremlin is able to discover that deficiency.
* `config/` -- directory of various YAML configuration scripts

## Anticipated questions

### Isn't an evolutionary algorithm overkill for this problem?

Absolutely.  It'd be trivial to just grind through all ten digits and 
observe the one that's the worst off.  However, we wanted an easy to 
understand example of how to implement Gremlin for a new ML problem, and the 
MNIST dataset is a well-known benchmark dataset.

In reality, you'd be tackling a problem with a rich feature set with a lot 
of unknown interactions.  E.g., Gremlin came from autonomous vehicle (AV) work 
where we wanted to find where a ML AV model performed poorly.  The feature 
set included the virtual town number, a location within that town, the sun 
position, the amount of rain, the amount of wind, the amount of fog, and the 
amount of water on the road.

When we developing Gremlin for this problem, it accidentally discovered a
problem with the fitness criteria used to evaluate autonomous car driving 
instead of with a ML AV model:

Mark A. Coletti, Shang Gao, Spencer Paulissen, Nicholas Quentin Haas, and Robert Patton. 2021. Diagnosing autonomous vehicle driving criteria with an adversarial evolutionary algorithm. Proceedings of the Genetic and Evolutionary Computation Conference Companion. Association for Computing Machinery, New York, NY, USA, 301–302. DOI:https://doi.org/10.1145/3449726.3459573

### I get two CSV files.  Now what?

In this example we generate a snapshot of each generation, `pop.csv`, and we 
capture all the evaluated offspring in `ind.csv`.  You could have something 
similar that would show the unique combination of feature values that yield 
the poorest results from your model.

It's an open question on how to best leverage that information, but naively 
you could add more examples of those pathological feature combos to your 
training data and retrain hour model.  Then run Gremlin again to see if 
there's improvement.
