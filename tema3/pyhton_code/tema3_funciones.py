import numpy as np
import matplotlib as mtp

def propagation_from_parameters(R, L, G, C, F):
    part1 = R + 1j*np.pi*2*L*F
    part2 = G + 1j*np.pi*2*C*F
    return np.sqrt(part1 * part2)

def impedance_from_parameters(R, L, G, C, F):
    part1 = R + 1j*np.pi*2*L*F
    part2 = G + 1j*np.pi*2*C*F
    return np.sqrt(part1 / part2)
