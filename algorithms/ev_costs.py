import numpy as np
import pandas as pd
from itertools import product
from algorithms.hamiltonian import get_mapping

# TODO: Edit charging/discharging prices to be time-dependent

def qubo_ev(params: dict):
    """
    Generates the QUBO matrix for an electric vehicle (EV) charging optimization problem.

    Parameters:
        params (dict): Dictionary containing the problem parameters:
            - N (int): Number of EVs
            - T (int): Number of timesteps
            - delta_t (float): Time step size
            - P0 (list): Charging powers of the vehicles
            - P (float): Power coefficient
            - Lambda (float): Large penalty coefficient for constraints
            - costs (array): NxT matrix of charging costs
            - deltas (array): NxT matrix of buy/sell energy price differences
            - target_energies (list): Target energy levels for each EV

    Returns:
        np.ndarray: The QUBO matrix of size (N*T) x (N*T)
    """
    N, T = params["N"], params["T"]
    delta_t, P0, Lambda = params["delta_t"], params["P0"], params["Lambda"]
    c, deltas, E = params["costs"], params["deltas"], params["target_energies"]
    P = params["P"]

    # Initialize QUBO matrix with random noise (scaled to 0.5 for better mapping)
    Q = np.random.rand(N * T, N * T) * 0.5

    # Primary QUBO cost function and constraints
    for n in range(N):
        for t1 in range(T):
            idx = n * T + t1
            Q[idx, idx] = (2 * delta_t * c[t1] - delta_t * deltas[t1] -
                           4 * Lambda * E[n] * P0[n] - 4 * Lambda * (P0[n] ** 2) -
                           (4 + 2 * P) * Lambda)

            for t2 in range(t1 + 1, T):
                Q[idx, n * T + t2] = Lambda * (P0[n] ** 2) / 2
                Q[n * T + t2, idx] = Q[idx, n * T + t2]

    # Second set of constraints ensuring power allocation feasibility
    for t in range(T):
        for n in range(N):
            for m in range(n + 1, N):
                Q[n * T + t, m * T + t] = 4 * Lambda
                Q[m * T + t, n * T + t] = Q[n * T + t, m * T + t]

    return Q


def ev_cost(string: str, params: dict, Q: np.ndarray):
    """
    Computes the cost of a given EV charging configuration using the QUBO matrix.

    Parameters:
        string (str): Binary string representation of the charging configuration.
        params (dict): Problem parameters (see qubo_ev for details).
        Q (np.ndarray): QUBO matrix.

    Returns:
        float: Computed cost.
    """
    config = np.array(list(string), dtype=int)
    return config.T @ Q @ config


def average_ev_cost(counts: dict, params: dict, Q: np.ndarray):
    """
    Computes the average cost over sampled states.

    Parameters:
        counts (dict): Dictionary mapping bitstrings to observed counts.
        params (dict): Problem parameters (see qubo_ev for details).
        Q (np.ndarray): QUBO matrix.

    Returns:
        float: Average cost over sampled states.
    """
    df = pd.DataFrame(list(counts.items()), columns=['string', 'counts'])
    df['costs'] = df['string'].apply(lambda x: ev_cost(x, params, Q))
    avg_cost = (df["costs"] * df["counts"]).sum() / df["counts"].sum()

    return avg_cost


def constraint_function(string: str, params: dict):
    """
    Checks whether a given state satisfies energy constraints.

    Parameters:
        string (str): Binary string representation of the charging configuration.
        params (dict): Problem parameters (see qubo_ev for details).

    Returns:
        int: 1 if the state is feasible, 0 otherwise.
    """
    P0, N, T, target_energies = params["P0"], params["N"], params["T"], params["target_energies"]
    config = np.array(list(string), dtype=int)

    for n in range(N):
        total_energy = sum(2 * P0[n] * config[n * T + i] for i in range(T))
        if abs(total_energy - target_energies[n]) > 1e-3:
            return 0

    return 1


def constraint_function_total(counts: dict, params: dict):
    """
    Computes the percentage of feasible states in the given sample set.

    Parameters:
        counts (dict): Dictionary mapping bitstrings to observed counts.
        params (dict): Problem parameters (see qubo_ev for details).

    Returns:
        float: Fraction of feasible states.
    """
    df = pd.DataFrame(list(counts.items()), columns=['string', 'counts'])
    df['constraint'] = df['string'].apply(lambda x: constraint_function(x, params))
    return (df["constraint"] * df["counts"]).sum() / df["counts"].sum()


def find_gs(params: dict, Q: np.ndarray):
    """
    Finds the ground state (lowest energy configuration) of the QUBO problem.

    Parameters:
        params (dict): Problem parameters (see qubo_ev for details).
        Q (np.ndarray): QUBO matrix.

    Returns:
        tuple: (ground state energy, optimal string, all energies)
    """
    N, T = params["N"], params["T"]
    all_states = [''.join(map(str, bits)) for bits in product([0, 1], repeat=N * T)]
    energies = [ev_cost(state, params, Q) for state in all_states]

    min_energy = min(energies)
    opt_string = all_states[np.argmin(energies)]

    return min_energy, opt_string, energies


def constrained_perc_approx_ratio(params: dict, counts: dict, Q: np.ndarray):
    """
    Computes the constrained percentage and approximation ratio.

    Parameters:
        params (dict): Problem parameters (see qubo_ev for details).
        counts (dict): Dictionary mapping bitstrings to observed counts.
        Q (np.ndarray): QUBO matrix.

    Returns:
        tuple: (feasibility percentage, approximation ratio)
    """
    gs_energy, _, _ = find_gs(params, Q)
    avg_cost = average_ev_cost(counts, params, Q)
    constraint_percentage = constraint_function_total(counts, params)

    approx_ratio = abs((gs_energy - avg_cost) / max(abs(gs_energy), 1e-3))
    return constraint_percentage, approx_ratio
