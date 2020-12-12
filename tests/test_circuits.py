import pytest
from spheres import *
from pytket.backends.ibm import AerBackend, AerStateBackend

def test_prepare_qubits():
	d = 4
	spin = qt.rand_ket(d)
	xyzs = spin_xyz(spin)
	circ = prepare_qubits(xyzs)

	state = execute(circ, Aer.get_backend('statevector_simulator')).result().get_statevector(circ)
	qstate = qt.Qobj(state, dims=[[2]*int(d-1), [1]*int(d-1)])
	assert np.allclose(qubits_xyz(qstate)[::-1], xyzs)

def test_spin_circ():
	d = 4
	j = (d-1)/2
	n = int(2*j)
	r = int(n*(n-1)/2)
	spin = qt.rand_ket(d)
	circ = spin_circ(spin, measure_cntrls=False)
	state = execute(circ, Aer.get_backend('statevector_simulator')).result().get_statevector(circ)

	qstate = qt.Qobj(state, dims=[[2]*int(n+r), [1]*int(n+r)])
	proj = qt.tensor(*[qt.basis(2,0)*qt.basis(2,0).dag()]*r + [qt.identity(2)]*n)
	qstate = (proj*qstate).unit().ptrace(list(range(r, r+n)))
	correct = spin_sym(spin)
	correct = correct*correct.dag()
	assert np.allclose(correct, qstate)

def spin_tomography():
	spin = qt.rand_ket(4)
	sym = spin_sym(spin)
	dm = sym*sym.dag()

	circ = spin_circ(spin, measure_cntrls=False)
	dm2 = spin_tomography(circ)

	assert np.isclose((dm*dm2).tr(), 1)

