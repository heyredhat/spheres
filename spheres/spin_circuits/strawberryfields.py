"""
StrawberryFields circuits for preparing spin-j states.
"""

from ..stars.pure import *
from ..oscillators import *
from ..symplectic import *

import strawberryfields as sf
from strawberryfields.ops import *

import scipy as sc

def spin_osc_strawberryfields(spin):
    """
    Returns a StrawberryFields circuit that prepares a given spin-j state as a state
    of two photonic oscillator modes. 

    Parameters
    ----------
        spin : qt.Qobj
            Spin-j state.
    
    Returns
    -------
        prog : sf.Program
            StrawberryFields circuit.

    """
    j = (spin.shape[0]-1)/2
    n_modes = 2*int(2*j)

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

def spinj_xyz_strawberryfields(state, on_modes=[0,1], XYZ=None):
    """
    Returns XYZ expectation values of a spin encoded in two oscillator modes.

    Parameters
    ----------
        state : strawberryfields.state

        on_modes : list
            Two modes encoding the spin, e.g., [0,1].
        
        XYZ : dict
            XYZ operators as real symplectic matrices. Constructed if not provided.
    
    Returns
    -------
        xyz : np.array
            XYZ expectation values.
    """
    XYZ = XYZ if XYZ else symplectic_xyz()
    XYZ = dict([(o, upgrade_two_mode_operator(O, on_modes[0], on_modes[1], state.num_modes)) for o, O in XYZ.items()])
    return np.array([state.poly_quad_expectation(XYZ["x"])[0],\
                     state.poly_quad_expectation(XYZ["y"])[0],\
                     state.poly_quad_expectation(XYZ["z"])[0]]).real
    