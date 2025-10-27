#!/bin/bash

for cfg in ./config/0930_AuAg_echem/*.py; do
    echo "Running with $cfg"
    python run_analysis.py --config "$cfg"
done

