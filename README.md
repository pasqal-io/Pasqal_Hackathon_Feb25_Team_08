# Quantum Annealing for Electric Charging Optimization

This repository contains code to optimize quantum annealing pulse schedules for solving optimization problems related to electric charging processes. The approach utilizes QUBO (Quadratic Unconstrained Binary Optimization) formulations, pulse shaping techniques, and numerical simulations to enhance performance under realistic noise conditions.

## Features

- **QUBO Formulation:** Constructs the optimization problem as a QUBO matrix.
- **Pulse Optimization:** Uses Bayesian optimization and heuristic methods to optimize annealing schedules.
- **Quantum Simulation:** Implements pulse sequences with Pulser and simulates their execution with QutipEmulator.
- **Noise Analysis:** Evaluates performance under depolarizing noise and SPAM errors.

## Code Structure

### `main.py`
- Loads problem parameters and QUBO matrix.
- Optimizes the pulse schedule using `schedule_optimizer_ev`.
- Simulates the annealing process for different total evolution times.
- Saves results for further analysis.

### `create_parameters.py`
- Generates problem-specific parameters.
- Constructs the QUBO matrix and maps it to the quantum register.
- Saves parameters and mappings for use in the main optimization.

### `noise_analysis.py`
- Loads optimized pulse schedules.
- Simulates execution with depolarizing noise and SPAM errors.
- Computes the average cost under noisy conditions.

## Requirements

Ensure you have Python 3.10 installed. Then, install dependencies using:

```bash
python3.10 -m pip install -r requirements.txt
