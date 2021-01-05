import pytest
from spheres import *

def test_spin_sym_qiskit():
    d = 4
    j = (d-1)/2
    n = int(2*j)
    r = int(n*(n-1)/2)

    spin = qt.rand_ket(d)
    circ_info = spin_sym_qiskit(spin)
    circ_info["circuit"].remove_final_measurements()
    state = execute(circ_info["circuit"], Aer.get_backend('statevector_simulator')).result().get_statevector(circ_info["circuit"])

    qstate = qt.Qobj(state, dims=[[2]*int(n+r), [1]*int(n+r)])
    proj = qt.tensor(*[qt.basis(2,0)*qt.basis(2,0).dag()]*r + [qt.identity(2)]*n)
    qstate = (proj*qstate).unit().ptrace(list(range(r, r+n)))
    correct = spin_sym(spin)
    correct = correct*correct.dag()
    assert np.allclose(correct, qstate)

def test_spin_tomography_qiskit():
    spin = qt.rand_ket(4)
    circ_info = spin_sym_qiskit(spin)

    correct_dm = spin*spin.dag()
    tomography_dm = spin_tomography_qiskit(circ_info)
    assert np.isclose((correct_dm*tomography_dm).tr(), 1, rtol=1e-01, atol=1e-03)