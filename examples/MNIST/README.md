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

* `config.yml` -- Gremlin configuration file to define how it will run things
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
