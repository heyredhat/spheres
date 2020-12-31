import pytest
from spheres import *

def test_spin_sym():
    spin = qt.rand_ket(4)
    assert compare_nophase(spin_sym(spin), symmetrize(spin_spinors(spin)))

    dm = qt.rand_dm(4)
    assert np.allclose(sym_spin(spin_sym(dm)), dm)

def test_sym_spin():
    spin = qt.rand_ket(4)
    sym = symmetrize(spin_spinors(spin))
    assert compare_nophase(spin, sym_spin(sym))

def test_symmetrized_basis():
    sym_basis = symmetrized_basis(3, d=2)
    assert sym_basis["map"] == spin_sym_map(3/2)