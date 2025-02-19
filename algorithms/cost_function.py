import numpy as np

def cost(counts):
    # add postselection etc
    shots = np.sum(list(counts.values()))

    energy = counts['01011'] + counts['00111']
    return 1 - energy/shots
