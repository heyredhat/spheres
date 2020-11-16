import pytest
from spheres import *
from pytket.backends.ibm import AerBackend, AerStateBackend

def test_spin_sym():
    j = 2
    spin = qt.rand_ket(int(2*j+1))
    spinors = spin_spinors(spin)
    sym = symmetrize(spinors)
    S = spin_sym(j)
    assert compare_nophase(S*spin, sym)

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
    assert np.allclose(qubit_state_xyz(qstate), xyzs)

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

def test_pauli_expectations():
    n = 3
    state = qt.rand_ket(2**n)
    state.dims = [[2]*n, [1]*n]
    correct_dm = state*state.dag()

    P = pauli_basis(n)
    exps = to_pauli_expectations(state, basis=P)
    dm = from_pauli_expectations(exps, basis=P)

    assert np.allclose(dm, correct_dm)

def test_tomography_circuits():
    n = 2
    circ = Circuit(2)
    circ.H(0)
    circ.Rx(1/2, 1)
    circ.CX(0,1)

    tomog_circs = tomography_circuits(circ)

    aer_backend = AerBackend()
    for c in tomog_circs:
        aer_backend.compile_circuit(c["circuit"])
    circs = [c["circuit"] for c in tomog_circs]
    aer_handles = aer_backend.process_circuits(circs, n_shots=10000)
    aer_results = aer_backend.get_results(aer_handles)
    dists = [r.get_distribution() for r in aer_results]

    exps = dists_pauli_expectations(tomog_circs, dists)
    P = pauli_basis(n)
    dm = from_pauli_expectations(exps, basis=P)

    aer_state_backend = AerStateBackend()
    aer_state_backend.compile_circuit(circ)
    aer_state_handle = aer_state_backend.process_circuit(circ)
    aer_state = aer_state_backend.get_result(aer_state_handle).get_state()
    qstate = qt.Qobj(aer_state, dims=[[2]*int(n), [1]*int(n)])
    correct_dm = qstate*qstate.dag()

    assert np.isclose((dm*correct_dm).tr(), 1)

def test_prepare_spin_tomography():
    d = 3
    P = pauli_basis(d-1)
    S = spin_sym((d-1)/2)

    spin = qt.rand_ket(d)
    correct_state = S*spin
    correct_dm = correct_state*correct_state.dag()
    correct_exps = to_pauli_expectations(correct_state, basis=P)

    info = prepare_spin(spin)
    tomog_circs = tomography_circuits(info["circuit"],\
                                         on_qubits=info["spin_qubits"])
    circs = [t["circuit"] for t in tomog_circs]
    cntrl_on = dict([(cbit, 0) for cbit in info["cntrl_bits"].values()])

    backend = AerBackend()
    for circ in circs:
        backend.compile_circuit(circ)
    handles = backend.process_circuits(circs, n_shots=10000)
    results = backend.get_results(handles)

    postselected_dists = postselected_dists(results, cntrl_on)
    exps = dists_pauli_expectations(tomog_circs, postselected_dists)
    dm = from_pauli_expectations(exps, basis=P)

    assert np.isclose((correct_dm, dm).tr(), 1)