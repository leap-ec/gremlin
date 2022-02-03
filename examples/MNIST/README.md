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
