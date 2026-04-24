#!/bin/bash

set -e

export PYTHONPATH=$(pwd)

source config.sh

echo "========================"
echo " Converting CSV to trace"
echo "========================"

python3 traces/csv_to_trace.py

echo "========================"
echo " Running congestion control"
echo "========================"

python3 experiments/run_congestion.py

echo "========================"
echo " Running buffer provisioning"
echo "========================"

python3 experiments/run_buffer.py

echo "========================"
echo " Plotting results"
echo "========================"

python3 utils/plot.py

echo "DONE"