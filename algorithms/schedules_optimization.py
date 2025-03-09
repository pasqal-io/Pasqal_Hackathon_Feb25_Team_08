import numpy as np
import matplotlib.pyplot as plt
from pulser import Pulse, Sequence, Register
from pulser_simulation import QutipEmulator
from pulser.devices import DigitalAnalogDevice
from pulser.waveforms import InterpolatedWaveform

from algorithms.ev_costs import ev_cost, average_ev_cost,qubo_ev,constraint_function, constraint_function_total
import cma
import tqdm


def build_omega(params):
    """
    Constructs the amplitude waveform for the pulse.

    Parameters:
        params (list or np.ndarray): Intermediate values for the waveform.

    Returns:
        list: Full waveform starting and ending at 0.
    """
    return [0] + list(params) + [0]


def build_delta(params, threshold=1):
    """
    Constructs the phase waveform for the pulse.

    Parameters:
        params (list or np.ndarray): Intermediate values for the waveform.
        threshold (float): Minimum and maximum bounds for the start and end of the phase.

    Returns:
        list: Full waveform starting below -threshold and ending above +threshold.
    """
    p1 = min(-threshold, -abs(params[0]))  # Ensure start is negative and below -threshold
    p2 = max(threshold, abs(params[-1]))  # Ensure end is positive and above +threshold
    return [p1] + list(params[1:-1]) + [p2]

def objective_ev(omega, delta, T, reg,params,Q):
    """
    Evaluates the cost function for a given pulse schedule.

    Parameters:
        omega (list or np.ndarray): Amplitude schedule parameters.
        delta (list or np.ndarray): Phase schedule parameters.
        T (float): Total evolution time for the pulse.
        reg (Register): Qubit register.
        params (Dictionary): Dictionary of the parameters, needed to create the qubo matrix and cost functions

    Returns:
        float: Computed cost from the emulator results.
    """
    # Build amplitude and phase waveforms
    omega_pulse = build_omega(omega)
    delta_pulse = build_delta(delta)

    # Construct the adiabatic pulse
    adiabatic_pulse = Pulse(
        InterpolatedWaveform(T, omega_pulse),
        InterpolatedWaveform(T, delta_pulse),
        0,
    )

    # Create a sequence with the specified pulse
    seq = Sequence(reg, DigitalAnalogDevice)
    seq.declare_channel("ising", "rydberg_global")
    seq.add(adiabatic_pulse, "ising")

    # Run the adiabatic evolution on the emulator
    simulator = QutipEmulator.from_sequence(seq)
    results = simulator.run()
    final_state = results.get_final_state()

    # Measure and compute the cost
    counts_dict = results.sample_final_state()
    # we only keep the dominant bitstring
    # counts_dict = {list(counts_dict.keys())[np.argmax(counts_dict.values())]:np.max(list(counts_dict.values()))}

    energy = average_ev_cost(counts_dict, params = params, Q = Q)

    #print(counts_dict)

    return energy


def schedule_optimizer_ev(omega_max, delta_max, l1, l2, T, reg, params,Q):
    """
    Optimizes the pulse schedule using CMA-ES.

    Parameters:
        omega_max (float): Maximum amplitude for the pulse.
        delta_max (float): Maximum phase for the pulse.
        l1 (int): Number of intermediate parameters for the amplitude schedule.
        l2 (int): Number of intermediate parameters for the phase schedule.
        T (float): Total evolution time.
        reg (Register): Qubit register.
        params (Dict): parameters of the electric charging problem

    Returns:
        tuple: Optimized omega schedule, delta schedule, and the associated cost.
    """

    # Define bounds for amplitude (omega) and phase (delta)
    bounds_omega = [0, abs(omega_max)/2]  # Amplitude must be non-negative
    bounds_delta = [-abs(delta_max)/2, abs(delta_max)/2]  # Phase can be positive or negative

    # Initialize random parameters within bounds
    initial_params_omega = np.random.uniform(bounds_omega[0], bounds_omega[1], l1)
    initial_params_delta = np.random.uniform(bounds_delta[0], bounds_delta[1], l2)

    # Combine the two parameter sets into a single array
    initial_params = np.concatenate([initial_params_omega, initial_params_delta])

    # Define combined bounds for CMA-ES
    bounds = [
        np.concatenate([np.full(l1, bounds_omega[0]), np.full(l2, bounds_delta[0])]),  # Lower bounds
        np.concatenate([np.full(l1, bounds_omega[1]), np.full(l2, bounds_delta[1])]),  # Upper bounds
    ]

    # Initialize CMA-ES optimizer
    sigma = 1  # Standard deviation for exploration
    es = cma.CMAEvolutionStrategy(initial_params, sigma, {'bounds': bounds})

    # Optimization loop
    max_iterations = 100
    pbar = tqdm.tqdm(range(max_iterations), desc="Optimizing")
    for _ in pbar:
        # Ask the optimizer for a batch of solutions
        solutions = es.ask()
        solutions_omega = [sol[:l1] for sol in solutions]  # First `l1` parameters
        solutions_delta = [sol[l1:] for sol in solutions]  # Last `l2` parameters

        # Evaluate the objective for each solution
        costs = [objective_ev(omega, delta, T, reg,params, Q) for omega, delta in zip(solutions_omega, solutions_delta)]

        # Convert tuples to lists if needed

        es.tell(solutions, costs)  # Update the optimizer with the new costs
        pbar.set_description(f"Best cost: {es.best.f:.6f}")

    # Retrieve the best solution
    best_params = es.best.x
    best_cost = es.best.f

    # Build the optimized schedules
    omega_schedule = build_omega(best_params[:l1])
    delta_schedule = build_delta(best_params[l1:])

    # Plot the optimized pulse
    adiabatic_pulse = Pulse(
        InterpolatedWaveform(T, omega_schedule),
        InterpolatedWaveform(T, delta_schedule),
        0,
    )
    # seq = Sequence(reg, DigitalAnalogDevice)
    # seq.declare_channel("ising", "rydberg_global")
    # seq.add(adiabatic_pulse, "ising")
    # seq.draw()

    return omega_schedule, delta_schedule, best_cost



def standard_annealing(omega_max,delta_max, T, reg, params,Q):
    # Build amplitude and phase waveforms
    time = np.linspace(0,T,200)

    # smooth waveform
    omega_pulse = omega_max*np.sin(np.pi/2*np.sin(np.pi*time/T))**2
    delta_pulse = -delta_max*np.cos(np.pi*time/T) #smooth
    # delta_pulse = 2*delta_max*time/T - delta_max #linear


    # Construct the adiabatic pulse
    adiabatic_pulse = Pulse(
        InterpolatedWaveform(T, omega_pulse),
        InterpolatedWaveform(T, delta_pulse),
        0,
    )

    # Create a sequence with the specified pulse
    seq = Sequence(reg, DigitalAnalogDevice)
    seq.declare_channel("ising", "rydberg_global")
    seq.add(adiabatic_pulse, "ising")
    # seq.draw()

    # Run the adiabatic evolution on the emulator
    simulator = QutipEmulator.from_sequence(seq)
    results = simulator.run()
    final_state = results.get_final_state()

    # Measure and compute the cost
    counts_dict = results.sample_final_state()
    energy = average_ev_cost(counts_dict, params = params, Q =Q)

    return energy
