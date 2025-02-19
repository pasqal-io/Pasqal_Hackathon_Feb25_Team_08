from algorithms.schedules_optimization import schedule_optimizer
from algorithms.hamiltonian import get_mapping
import numpy as np
import matplotlib.pyplot as plt


def main():

    Q = np.array(
    [
        [-10.0, 19.7365809, 19.7365809, 5.42015853, 5.42015853],
        [19.7365809, -10.0, 20.67626392, 0.17675796, 0.85604541],
        [19.7365809, 20.67626392, -10.0, 0.85604541, 0.17675796],
        [5.42015853, 0.17675796, 0.85604541, -10.0, 0.32306662],
        [5.42015853, 0.85604541, 0.17675796, 0.32306662, -10.0],
    ]
)

    reg = get_mapping(Q)
    omega_max = np.median(Q[Q > 0].flatten())
    omega_max = 10
    delta_max = 10

    l1 = 3
    l2 = 3

    costs = []
    T = 4*np.logspace(2,3,10)
    for t in T:
        t = t-t%4
        cost_intermediate = []
        for l1 in [2,4]:
            for l2 in [4,6,8]:

                omega_schedules, delta_scedules, cost = schedule_optimizer(omega_max, delta_max, l1, l2, t, reg)
                cost_intermediate.append(cost)
        costs.append(min(cost_intermediate))

    plt.figure()
    plt.plot(T/1000,costs)
    plt.xlabel(r'total time evolution [$\mu$ s]')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('cost_vs_time.png')

if __name__ == '__main__':

    main()
