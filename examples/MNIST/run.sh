#!/usr/bin/env bash
#
# Run a very basic configuration of Gremlin against the MNIST training set three
# separate ways to demonstrate use.
#

# 1. Plain vanilla by-generation mode with serial fitness evaluations
time ../../gremlin/gremlin.py config/common.yml config/bygen.yml

# 2. Parallel by-generation mode; this actually takes a couple seconds longer
# than the serial evaluation because of the overhead of Dask setup and teardown.
# However, Dask works well for problems with significant evaluation time. Dask
# warnings about scheduler and closing streams can be ignored as that's nomral
# LocalCluster shutdown.
time ../../gremlin/gremlin.py config/common.yml config/distributed.yml config/bygen.yml config/bygen-dist.yml

# 3. This is for running as an asynchronous steady-state EA; the same caveats
# for the previous run for using Dask applies.
time ../../gremlin/gremlin.py config/common.yml config/distributed.yml config/async.yml
