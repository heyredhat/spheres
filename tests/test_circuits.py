import pytest
from spheres import *
from pytket.backends.ibm import AerBackend, AerStateBackend

def test_prepare_qubits():
    d = 4
    spin = qt.rand_ket(d)
    xyzs = spin_xyz(spin)
    circ = prepare_qubits(xyzs)

    backend = AerStateBackend()
    backend.compile_circuit(circ)
    handle = backend.process_circuit(circ)
    state = backend.get_result(handle).get_state()
    qstate = qt.Qobj(state, dims=[[2]*int(d-1), [1]*int(d-1)])
    assert np.allclose(qubits_xyz(qstate), xyzs)

def test_prepare_spin():
    d = 4
    j = (d-1)/2
    n = int(2*j)
    r = int(n*(n-1)/2)
    spin = qt.rand_ket(d)
    spin_circuit_info = prepare_spin(spin, measure_cntrls=False)

    circ = spin_circuit_info["circuit"]
    backend = AerStateBackend()
    backend.compile_circuit(circ)
    handle = backend.process_circuit(circ)
    state = backend.get_result(handle).get_state()
    
    qstate = qt.Qobj(state, dims=[[2]*int(n+r), [1]*int(n+r)])
    proj = qt.tensor(*[qt.basis(2,0)*qt.basis(2,0).dag()]*r + \
                      [qt.identity(2)]*n)
    qstate = (proj*qstate).unit().ptrace(list(range(r, r+n)))
    correct = symmetrize(spin_spinors(spin))
    correct = correct*correct.dag()
    assert np.allclose(correct, qstate)

def test_tomography_circuits():
	n = 2
	circ = Circuit(2)
	circ.H(0)
	circ.Rx(1/2, 1)
	circ.CX(0,1)

	tomog_circs = tomography_circuits(circ)

	aer_backend = AerBackend()
	[aer_backend.compile_circuit(c["circuit"]) for c in tomog_circs]
	circs = [c["circuit"] for c in tomog_circs]
	aer_handles = aer_backend.process_circuits(circs, n_shots=10000)
	aer_results = aer_backend.get_results(aer_handles)
	tomog_shots = [result.get_shots() for result in aer_results]

	dm = tomography_shots_dm(tomog_circs, tomog_shots)

	aer_state_backend = AerStateBackend()
	aer_state_backend.compile_circuit(circ)
	aer_state_handle = aer_state_backend.process_circuit(circ)
	aer_state = aer_state_backend.get_result(aer_state_handle).get_state()
	qstate = qt.Qobj(aer_state, dims=[[2]*int(n), [1]*int(n)])
	correct_dm = qstate*qstate.dag()

	assert np.isclose((dm*correct_dm).tr(), 1)

def test_prepare_spin_tomography():
	d = 4
	P = pauli_basis(d-1)
	S = spin_sym((d-1)/2)

	spin = qt.rand_ket(d)
	correct_spin_dm = spin*spin.dag()

	spin_info = prepare_spin(spin)
	tomog_circs = tomography_circuits(spin_info["circuit"],\
	                                  on_qubits=spin_info["spin_qubits"])

	backend = AerBackend()
	circs = [tc["circuit"] for tc in tomog_circs]
	[backend.compile_circuit(circ) for circ in circs]
	handles = backend.process_circuits(circs, n_shots=10000)
	results = backend.get_results(handles)
	tomog_shots = [result.get_shots() for result in results]

	postselected_tomog_shots = postselect_shots(tomog_shots, spin_info["postselect_on"])
	dm = tomography_shots_dm(tomog_circs, postselected_tomog_shots)

	spin_dm = (S.dag()*dm*S)
	assert np.isclose((spin_dm*correct_spin_dm).tr(), 1, rtol=1e-02, atol=1e-04)




