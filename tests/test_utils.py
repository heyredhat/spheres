import pytest
from spheres import *

def test_pauli_expectations():
    n = 4
    state = qt.rand_ket(2**n)
    state.dims = [[2]*n, [1]*n]
    correct_dm = state*state.dag()

    P = pauli_basis(n)
    exps = to_pauli_basis(state, basis=P)
    dm = from_pauli_basis(exps, basis=P)

    assert np.allclose(dm, correct_dm)