#!/usr/bin/env bash
#
# Run a very basic configuration of Gremlin against the MNIST training set
#

# Plain vanilla by-generation mode
time ../../gremlin/gremlin.py config/common.yml config/bygen.yml

# Parallel by-generation mode; this actually takes a couple seconds longer
# than the serial evaluation because of the overhead of Dask setup and teardown.
# However, Dask works well for problems with significant evaluation time.
time ../../gremlin/gremlin.py config/common.yml config/distributed.yml config/bygen.yml config/bygen-dist.yml

# This is for running as an asynchronous steady-state EA; the same caveats
# for the previous run for using Dask applies.
time ../../gremlin/gremlin.py config/common.yml config/distributed.yml config/async.yml
