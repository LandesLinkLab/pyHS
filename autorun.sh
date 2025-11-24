#!/bin/bash

for cfg in ./config/110525_AuAg_echem/echem/*.py; do
    echo "Running with $cfg"
    python run_analysis.py --config "$cfg"
done

