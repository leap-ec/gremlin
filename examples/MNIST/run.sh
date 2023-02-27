#!/usr/bin/env bash
#
# Run a very basic configuration of Gremlin against the MNIST training set
#

# Plain vanilla by-generation mode
../../gremlin/gremlin.py --debug config/common.yml config/bygen.yml
