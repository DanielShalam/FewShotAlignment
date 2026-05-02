#!/bin/bash
for shots in 16 32 64 128; do
    for seed in 1 2 3; do
        echo "Running for $shots shots and seed $seed..."
        python main.py --config configs/vindr.yaml dataset VinDrCXR shots $shots seed $seed
    done
done