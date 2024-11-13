# larnd-sim-jax

## Presentation

This project is forked from https://github.com/ynashed/larnd-sim.
It aims at rewriting the official DUNE ND simulator (https://github.com/DUNE/larnd-sim) in a differentiable manner by using the JAX framework

## Setup

The code in this repo can be executed at s3df on machines with GPUs using the following docker image https://hub.docker.com/repository/docker/pigranger/larndsim-jax/general (sif file available in `/sdf/group/neutrino/pgranger/larnd-sim-jax.sif`)

## Example

An example script to start a fit is given at `optimize/script/start_fit.sh`