#!/bin/bash

# Run the Python script
python $1 -c 0 -n 2_000_000
python $1 -c 1 -n 2_000_000
python $1 -c 2 -n 2_000_000