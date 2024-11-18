import numpy as np
from scipy.linalg import eigh

# Constants
pi = 4 * np.arctan(1.0)

# Reading input values from a file (assuming values are in a specific format)
with open("input.txt", "r") as f:
    lines = f.readlines()
    lattice_const = float(lines[1].strip())
    e_cutoff = float(lines[3].strip())
    V0 = float(lines[5].strip())
    Nplot = int(lines[7].strip())

# Derived parameters
Kmax = pi / lattice_const
Nmax = int(np.sqrt(2 * e_cutoff) / (2 * Kmax))
Ndim = 2 * Nmax + 1
delta_K = Kmax / (Nplot - 1)

# Output file
output_file = open("band.dat", "w")

# Loop over k-points
K = 0.0
for i in range(Nplot):
    # Initialize T and V matrices
    T = np.zeros((Ndim, Ndim))
    V = np.zeros((Ndim, Ndim))
    
    # Populate T matrix
    T[0, 0] = K * K
    for j in range(1, Nmax + 1):
        jj = 2 * j - 1
        T[jj, jj] = (K - j * 2 * Kmax) ** 2
        T[jj + 1, jj + 1] = (K + j * 2 * Kmax) ** 2

    T = T * 0.5  # Multiply by 0.5 for all diagonal elements

    # Populate V matrix
    V[0, 1] = V0
    V[1, 0] = V0
    for j in range(2, Ndim):
        V[j, j - 2] = V0
        V[j - 2, j] = V0

    # Form the Hamiltonian H
    H = T + V

    # Diagonalize H
    eigenvalues= eigh(H, eigvals_only=True)

    # Write K and eigenvalues to output file
    output_file.write(f"{K:.5f} " + " ".join(f"{ev:.5f}" for ev in eigenvalues) + "\n")
    
    # Update K
    K += delta_K

output_file.close()
