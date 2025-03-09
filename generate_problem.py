from algorithms.schedules_optimization import schedule_optimizer, schedule_optimizer_ev, standard_annealing
from algorithms.hamiltonian import get_mapping
from algorithms.ev_costs import ev_cost, average_ev_cost,qubo_ev, find_gs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import optuna
import pickle


import random
from pulser import Pulse, InterpolatedWaveform, DigitalAnalogDevice, Sequence
from pulser_simulation import QutipEmulator

# Define meaningful parameters
N = 2  # Number of entities (e.g., vehicles)
T_ev = 2  # Time evolution steps

# Other parameters
delta_t = 1.0  # Time step
P0 = np.ones(N)  # Charging powers

Lambda = 5  # Regularization parameter
charging_costs = [random.randint(1, 5)]*T_ev  # Random charging costs
deltas = [0.3 *x for x in  charging_costs]  # Delta values based on charging costs
#uncomment this for 4x4 instance
#P = 2
#target_energies = [2,0,4,2]  # Target energies for one instance of the problem

#uncomment this for 3x3 instance
P = 1
target_energies = [1,1,1]  # Target energies for one instance of the problem

#uncomment this for 2x2 instance
# P = 0
# target_energies = [0,0]  # Target energies for one instance of the problem
#target_energies = [random.randint(1, T_ev) for _ in range(N)]  # Target energies for one instance of the problem


# Pack parameters into a dictionary
params = {
    "delta_t": delta_t,
    "P0": P0,
    "deltas": deltas,
    "costs": charging_costs,
    "Lambda": Lambda,
    "N": N,
    "T": T_ev,
    "target_energies": target_energies,
    "P": P
}

# Generate QUBO matrix and get mapping
Q = qubo_ev(params)

reg = get_mapping(Q)

with open(f"results/parameters_N_{str(N)}_T_{str(T_ev)}.pkl","wb") as fp:
    pickle.dump(params,fp)

with open(f"results/reg_N_{str(N)}_T_{str(T_ev)}.pkl", 'wb') as fp:
    pickle.dump(reg, fp)

np.save(f"results/Q_N_{str(N)}_T_{str(T_ev)}",Q)
np.save(f"results/reg_N_{str(N)}_T_{str(T_ev)}",reg)

print("Parameters created")