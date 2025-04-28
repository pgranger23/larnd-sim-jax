# larnd-sim-jax

## Presentation

This project is forked from https://github.com/ynashed/larnd-sim.
It aims at rewriting the official DUNE ND simulator (https://github.com/DUNE/larnd-sim) in a differentiable manner by using the JAX framework

## Setup

### For usage only

To install the package, ideally within a virtual/conda environment
```bash
pip install .
```

### Setup for development

To install the package for development, add the `-e` flag (editable mode) to the command above. This will allow you to make changes to the code and have them reflected immediately without needing to reinstall the package. Ideally also add the `[dev]` flag to install the development dependencies.
```bash
pip install -e .[dev]
```

### Executing the code

The code in this repo can be executed at s3df on machines with GPUs using the following docker image https://hub.docker.com/repository/docker/pigranger/larndsim-jax/general (sif file available in `/sdf/group/neutrino/pgranger/larnd-sim-jax.sif`)

## Example

An example script to start a fit is given at `optimize/script/start_fit.sh`