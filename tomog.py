from spheres import *
from pytket.backends.ibm import AerBackend, AerStateBackend

d = 5
P = pauli_basis(d-1)
S = spin_sym((d-1)/2)

spin = qt.rand_ket(d)
correct_spin_dm = spin*spin.dag()
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

exps = {}
for i, result in enumerate(results):
    # remove tomography bit
    shots = result.get_shots()
    r = len(info["cntrl_qubits"])
    for j in range(r):
        shots = shots[np.where(shots[:,j] == 0)]
    for j in range(r):
        shots = np.delete(shots, 0, axis=1)
    # remove identity bits
    bad_indices = [j for j, s in enumerate(tomog_circs[i]["pauli"]) if s == "I"]
    shots = np.delete(shots, bad_indices, axis=1)
    if shots.shape[1] > 0:
        # replace 0/1 with eigenvalues
        shots = np.where(shots==1, -1, shots)
        shots = np.where(shots==0, 1, shots)
        shots = np.prod(shots, axis=1)
    else:
        shots = np.array([1])
    # and sum
    exp = sum(shots)/len(shots)
    exps[tomog_circs[i]["pauli"]] = exp

dm = from_pauli_expectations(exps, basis=P)
spin_dm = (S.dag()*dm*S)

print((spin_dm*correct_spin_dm).tr())
print((dm*correct_dm).tr())
