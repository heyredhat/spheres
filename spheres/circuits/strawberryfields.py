"""
StrawberryFields Circuits
-------------------------

"""

from spheres.stars.pure import *
from spheres.oscillators import *
from spheres.symplectic import *

import strawberryfields as sf
from strawberryfields.ops import *

import scipy as sc

def spin_osc_circ(spin):
    """
    Returns a StrawberryFields circuit that prepares a given spin-j state as a state
    of two photonic oscillator modes. 
    """
    j = (spin.shape[0]-1)/2
    n_modes = 2*int(2*j)
    cutoff_dim = int(2*j+1)

    sph = [np.array([theta_phi[0]/2, theta_phi[1]]) for theta_phi in spin_sph(spin)]
    prog = sf.Program(n_modes)
    with prog.context as q:
        for i in range(0, n_modes-1, 2):
            Fock(1) | q[i]
            theta, phi = sph[int(i/2)]
            BSgate(theta, phi) | (q[i], q[i+1])
        for i in range(2, n_modes-1, 2):
            BSgate() | (q[0], q[i])
            BSgate() | (q[1], q[i+1])
        for i in range(2, n_modes):
            MeasureFock(select=0) | q[i]
    return prog

def test_spin_osc_circ(spin):
    j = (spin.shape[0]-1)/2
    n_modes = 2*int(2*j)
    cutoff_dim = int(2*j+1)

    prog = spin_osc_circ(spin)
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim, "pure": True})
    state = eng.run(prog).state

    correct_state = spin_osc(spin, max_ex=cutoff_dim)
    dirac(correct_state)

    fock_probs = {}
    for i in range(cutoff_dim):
        for j in range(cutoff_dim):
            fock_state = [i, j]+[0]*(n_modes-2) 
            fock_probs[(i, j)] = state.fock_prob(n=fock_state)
            if fock_probs[(i, j)] != 0:
                print("%s: %.3f" % ((i, j), fock_probs[(i, j)]))

def sf_state_xyz(state, n_modes=2, XYZ=None):
    """
    Returns XYZ expectation values of the first two of n_modes oscillator modes. 
    """
    if type(XYZ) == type(None):
        XYZ = symplectic_xyz()
    for o, O in XYZ.items():
        XYZ[o] = upgrade_two_mode_operator(O, 0, 1, n_modes)
    return np.array([state.poly_quad_expectation(XYZ["X"])[0],\
                     state.poly_quad_expectation(XYZ["Y"])[0],\
                     state.poly_quad_expectation(XYZ["Z"])[0]]).real
    