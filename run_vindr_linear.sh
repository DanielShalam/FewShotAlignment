#!/bin/bash
for shots in 4 8; do
    for seed in 1 2 3; do
        echo "Running for $shots shots and seed $seed..."
        python main_mlp.py --config configs/fhead.yaml --linear_probe dataset VinDrCXR shots $shots seed $seed
    done
done