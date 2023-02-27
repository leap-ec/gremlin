# Gremlin YAML configuration scripts.

These different scripts allow for tailoring Gremlin runs.  E.g., setting 
population sizes, defining how offspring are created, and whether to use Dask
to support distributed fitness evaluations.

* `async.yml` -- how to run Gremlin as an asynchronous steady-state EA
* `bygen.yml` -- run Gremlin in a more traditional by-generation mode
* `bygen-dist.yml` -- modify the previous to allow for distributed fitness 
  evaluations for offspring
* `common.yml` -- common configuration parameters regardless if running by
  generation or asynchronously
* `distributed.yml` -- common Dask configuration for use with `async.yml` or
  `bygen-dist.yml`
* `debug.yml` -- debug configuration that essentially dials down population 
  sizes and maximum generations to small values
