import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from pulser import Pulse, Sequence, Register
from pulser_simulation import QutipEmulator, SimConfig
from pulser.devices import DigitalAnalogDevice
from pulser.waveforms import InterpolatedWaveform
from algorithms.schedules_optimization import build_delta, build_omega
from algorithms.ev_costs import average_ev_cost

def main(N, T_ev):
    # Load parameters and register
    with open(f"results/parameters_N_{N}_T_{T_ev}.pkl", "rb") as fp:
        params = pickle.load(fp)
    with open(f"results/reg_N_{N}_T_{T_ev}.pkl", "rb") as fp:
        reg = pickle.load(fp)

    # Load QUBO matrix and results summary
    Q = np.load(f"results/Q_N_{N}_T_{T_ev}.npy", allow_pickle=True)
    summary = pd.read_csv(f"results/output_{N}_{T_ev}.csv")

    # Load optimized schedules
    schedules_omega = np.load(f"results/omega_schedules_{N}_{T_ev}.npy")
    schedules_delta = np.load(f"results/delta_schedules_{N}_{T_ev}.npy")

    annealing_time = summary['time']
    lambda_values = np.logspace(-4, -1, 10)
    temperature = 100
    cost_noisy = np.zeros((len(summary['cost']), len(lambda_values)))

    for t, T in enumerate(annealing_time):
        # Construct the adiabatic pulse
        adiabatic_pulse = Pulse(
            InterpolatedWaveform(T, schedules_omega[t, :]),
            InterpolatedWaveform(T, schedules_delta[t, :]),
            0,
        )

        # Create a sequence
        seq = Sequence(reg, DigitalAnalogDevice)
        seq.declare_channel("ising", "rydberg_global")
        seq.add(adiabatic_pulse, "ising")

        for i, l in enumerate(lambda_values):
            simulator = QutipEmulator.from_sequence(
                seq,
                sampling_rate=0.1,
                config=SimConfig(
                    noise=("SPAM", "depolarizing"),
                    depolarizing_rate=l,
                    temperature=temperature,
                    runs=50,
                ),
            )
            results = simulator.run()
            counts_dict = results.sample_final_state()
            cost_noisy[t, i] = average_ev_cost(counts_dict, params, Q)

    # Save results
    np.save(f'results/cost_noisy_{N}_{T_ev}.npy', cost_noisy)
    np.save(f'results/noise_strength_{N}_{T_ev}.npy', lambda_values)

if __name__ == '__main__':
    N = 3
    T_ev = 3
    main(N,T_ev)
