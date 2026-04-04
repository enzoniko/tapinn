#!/bin/bash
export PYTHONPATH=.

echo "Cleaning up any floating background experiments..."
pkill -f "python exp_" || true
sleep 1

echo "Starting Exp 1..."
python exp_1_ode_chaos_suite.py "$@"

echo "Starting Exp 2..."
python exp_2_pde_spatiotemporal_suite.py "$@"

echo "Starting Exp 3..."
python exp_3_sota_baselines_and_capacity.py "$@"

echo "Starting Exp 4..."
python exp_4_sensitivity_and_robustness.py "$@"

echo "Starting Exp 5..."
python exp_5_theoretical_optimization_landscape.py "$@"

echo "All experiments finished successfully."
