from pulser import Register
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import numpy as np
from pulser.devices import DigitalAnalogDevice

def evaluate_mapping(new_coords, Q):
    """Cost function to minimize. Ideally, the pairwise distances are conserved."""
    new_coords = np.reshape(new_coords, (len(Q), 2))
    # Compute the matrix of distances between all coordinate pairs
    new_Q = squareform(
        DigitalAnalogDevice.interaction_coeff / pdist(new_coords) ** 6
    )
    return np.linalg.norm(new_Q - Q, ord='fro')

def get_mapping(Q):
    """Find an optimal 2D mapping of qubits based on their pairwise distances."""

    # Initialize with random coordinates for optimization
    x0 = np.random.random(len(Q) * 2)

    # Minimize the cost function to find the best mapping
    res = minimize(
        evaluate_mapping,
        x0,
        args=(Q,),
        method="Nelder-Mead",
        tol=1e-6,
        options={"maxiter": 200000, "maxfev": None},
    )

    # Reshape the result into coordinates
    coords = np.reshape(res.x, (len(Q), 2))

    # Compute the maximum distance between any two points
    max_distance = max([(x**2 + y**2)**0.5 for x, y in coords])


    max_length = 49  # Maximum allowed distance for the mapping
    min_distance = 4  # Minimum allowed distance between qubits

    # If the maximum distance exceeds the allowed max length, scale down the coordinates
    if max_distance > max_length:
        scale_factor = max_length / max_distance
        coords = [(x * scale_factor, y * scale_factor) for x, y in coords]
        max_distance = max([(x**2 + y**2)**0.5 for x, y in coords])


        # Calculate pairwise distances and check for minimum distance
        points = np.array(coords)
        distances = [np.linalg.norm(points[i] - points[j]) for i in range(len(points)) for j in range(i + 1, len(points))]


        current_min_distance = min(distances)

        # If the minimum distance is smaller than the allowed minimum, scale the coordinates
        if current_min_distance < min_distance:
            scale_factor = min_distance / current_min_distance
            coords = [(x * scale_factor, y * scale_factor) for x, y in coords]

            # Check if the maximum distance is still within the allowed limit
            max_distance = max([(x**2 + y**2)**0.5 for x, y in coords])
            if max_distance > max_length:
                print("Not possible to map the qubits in the device")
                exit(1)

    # Create the qubit register with the final coordinates
    qubits = {f"q{i}": coord for i, coord in enumerate(coords)}
    reg = Register(qubits)


    return reg
