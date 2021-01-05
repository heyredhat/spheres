import pytest
from spheres import *

def test_spin_tomography_pytket():
    spin = qt.rand_ket(4)
    correct_dm = spin*spin.dag()

    circ_info = spin_sym_pytket(spin)
    tomography_dm = spin_tomography_pytket(circ_info)
    print(np.isclose((correct_dm*tomography_dm).tr(), 1, rtol=1e-01, atol=1e-03))