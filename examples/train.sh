#!/bin/bash

# Run the Python script train_swarm_shape_v0.py
python3 $1 -c 0 -n 500_000
python3 $1 -c 1 -n 500_000
python3 $1 -c 2 -n 500_000