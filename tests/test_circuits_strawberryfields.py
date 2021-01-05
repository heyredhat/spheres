import pytest
from spheres import *

def test_spin_osc_strawberryfields():
    j = 3/2
    spin = qt.rand_ket(int(2*j + 1))
    n_modes = 2*int(2*j)
    cutoff_dim = int(2*j+1)

    prog = spin_osc_strawberryfields(spin)
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
    state = eng.run(prog).state

    correct_state = spin_osc(spin, cutoff_dim=cutoff_dim)

    sf_xyz = spinj_xyz_strawberryfields(state)
    correct_xyz = spinj_xyz_osc(correct_state)
    assert np.allclose(sf_xyz, correct_xyz)

    for i in range(cutoff_dim):
        for j in range(cutoff_dim):
            sf_fock_prob = state.fock_prob(n=[i, j]+[0]*(n_modes-2) )
            correct_fock_prob = abs((qt.tensor(qt.basis(cutoff_dim, i), qt.basis(cutoff_dim, j)).dag()*correct_state)[0][0][0])**2
            assert np.isclose(sf_fock_prob, correct_fock_prob)