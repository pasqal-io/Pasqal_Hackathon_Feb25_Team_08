import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import optuna
import pickle
import random
from pulser import Sequence
from algorithms.schedules_optimization import schedule_optimizer_ev, standard_annealing
from algorithms.hamiltonian import get_mapping
from algorithms.ev_costs import ev_cost, find_gs, qubo_ev
from pulser.devices import DigitalAnalogDevice

def load_parameters(N, T_ev):
    """Load problem parameters from saved files."""
    with open(f"results/parameters_N_{N}_T_{T_ev}.pkl", "rb") as fp:
        params = pickle.load(fp)
    with open(f"results/reg_N_{N}_T_{T_ev}.pkl", "rb") as fp:
        reg = pickle.load(fp)
    Q = np.load(f"results/Q_N_{N}_T_{T_ev}.npy", allow_pickle=True)
    return params, reg, Q

def save_parameters(N, T_ev, params, reg, Q):
    """Save parameters to files for reuse."""
    with open(f"results/parameters_N_{N}_T_{T_ev}.pkl", "wb") as fp:
        pickle.dump(params, fp)
    with open(f"results/reg_N_{N}_T_{T_ev}.pkl", "wb") as fp:
        pickle.dump(reg, fp)
    np.save(f"results/Q_N_{N}_T_{T_ev}", Q)

def compute_costs(Q):
    """Compute cost values for all possible bitstrings."""
    bitstrings = [np.binary_repr(i, len(Q)) for i in range(2 ** len(Q))]
    costs = [np.dot(np.array(list(b), dtype=int).T, np.dot(Q, np.array(list(b), dtype=int))) for b in bitstrings]
    return sorted(zip(bitstrings, costs), key=lambda x: x[1])

def run_annealing(N, T_ev, params, reg, Q):
    """Perform standard and optimized annealing."""
    T_values = 5 * np.logspace(2, 3, 8)
    omega_max, delta_max = 4 * np.median(Q[Q > 0]), 4 * np.max(Q[Q > 0])

    costs_standard, costs_optimized, schedules_omega, schedules_delta = [], [], [], []
    annealing_time = [t - t % 4 for t in T_values]

    for t in annealing_time:
        costs_standard.append(standard_annealing(omega_max, delta_max, t, reg, params, Q))
        omega, delta, cost = schedule_optimizer_ev(omega_max, delta_max, 8, 16, t, reg, params, Q)
        costs_optimized.append(cost)
        schedules_omega.append(omega)
        schedules_delta.append(delta)

    return annealing_time, costs_standard, costs_optimized, schedules_omega, schedules_delta

def save_results(N, T_ev, annealing_time, costs_standard, costs_optimized, schedules_omega, schedules_delta):
    """Save results to CSV and NPY files."""
    df = pd.DataFrame({"time": annealing_time, "cost": costs_optimized, "cost_baseline": costs_standard})
    df.to_csv(f"results/output_{N}_{T_ev}.csv", index=False)
    np.save(f'results/omega_schedules_{N}_{T_ev}.npy', np.array(schedules_omega))
    np.save(f'results/delta_schedules_{N}_{T_ev}.npy', np.array(schedules_delta))

def create_parameters(N, T_ev):
    """Generate and save problem parameters."""
    delta_t, Lambda, P0 = 1.0, 5, np.ones(N)
    charging_costs = [random.randint(1, 5)] * T_ev
    deltas = [0.3 * x for x in charging_costs]

    predefined_targets = {
        (4, 4): (2, [2, 0, 4, 2]),
        (3, 3): (1, [1, 1, 1]),
        (3, 2): (1, [0, 0, 1]),
        (2, 2): (0, [0, 0]),
        (6, 2): (0, [0, 0, 0, 0, 0, 0]),
        (4, 3): (1, [1, 1, -1, 1]),
        (5, 2): (1, [0] * 5),
    }
    P, target_energies = predefined_targets.get((N, T_ev), (2, [2] * N))

    params = {"delta_t": delta_t, "P0": P0, "deltas": deltas, "costs": charging_costs, "Lambda": Lambda,
              "N": N, "T": T_ev, "target_energies": target_energies, "P": P}
    Q = qubo_ev(params)
    reg = get_mapping(Q)
    save_parameters(N, T_ev, params, reg, Q)

def plot_results(N, T_ev, annealing_time, costs_standard, costs_optimized, gs):
    """Plot annealing results."""
    plt.figure()
    plt.axhline(y=gs, color='k', linestyle='-')
    plt.plot([t / 1000 for t in annealing_time], costs_optimized, 'ro-', label='Optimized')
    plt.plot([t / 1000 for t in annealing_time], costs_standard, 'bo-', label='Baseline')
    plt.legend()
    plt.xlabel(r'Total time evolution [$\mu$s]')
    plt.xscale('log')
    plt.ylim(min(costs_optimized + costs_standard), max(costs_optimized + costs_standard))
    plt.savefig(f'results/cost_vs_time_ev_N{N}_T{T_ev}.png')

if __name__ == '__main__':
    N, T_ev = 2,2
    create_parameters(N, T_ev)
    params, reg, Q = load_parameters(N, T_ev)
    annealing_time, costs_standard, costs_optimized, schedules_omega, schedules_delta = run_annealing(N, T_ev, params, reg, Q)
    save_results(N, T_ev, annealing_time, costs_standard, costs_optimized, schedules_omega, schedules_delta)
    gs, _, _ = find_gs(params, Q)
    plot_results(N, T_ev, annealing_time, costs_standard, costs_optimized, gs)
