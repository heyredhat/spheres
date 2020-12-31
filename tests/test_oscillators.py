import pytest
from spheres import *

def test_spin_osc():
    spin = qt.rand_ket(4)
    osc = spin_osc(spin)
    assert np.allclose(osc_spin(osc), spin)
    assert np.allclose(spinj_xyz(spin), osc_xyz(osc))

def test_osc_spinblocks():
    cutoff_dim = 4
    paulis = second_quantized_paulis(cutoff_dim=cutoff_dim)
    for o, O in paulis.items():
        for block in osc_spinblocks(O):
            if block.norm() != 0 and block.shape[0] <= cutoff_dim:
                assert np.allclose(block, qt.jmat((block.shape[0]-1)/2, o))

def test_second_quantize_state():
    state1 = qt.rand_ket(4)
    H1 = qt.rand_herm(4)

    state2 = second_quantize_state(state1, state=True)
    H2 = second_quantize_operator(H1)

    assert np.isclose(qt.expect(H1, state1), qt.expect(H2, state2))

def test_spin_osc_map():
    j = 3/2
    spin = qt.rand_ket(int(2*j + 1))
    assert np.allclose(spin_osc_map(j)*spin, spin_osc(spin))

def test_spins_osc():
    osc = qt.rand_ket(4**2)
    osc.dims = [[4,4],[1,1]]
    assert np.allclose(osc, spins_osc(osc_spins(osc)))