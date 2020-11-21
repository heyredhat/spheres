from spheres import *
from pytket.utils import counts_from_shot_table, probs_from_counts
from pytket.backends.ibm import AerBackend, AerStateBackend

#####################################################################

n_qubits = 2 # number of qubits in circuit to be symmetrized
depth = 5 # depth of circuit to be symmetrized
n_shots = 8000 

n_copies = 4 # copies of original circuit to be employed
every = 2
pairwise = True
reuse_cntrls = False

noise = False # use a noise model
latex = False # output latex representation of circuits
calc_state = False # also calculate the state using state simulator

#####################################################################

tiny_backend = AerBackend()

if noise: # Use a noise model?
    from qiskit.providers.aer.noise import NoiseModel
    from qiskit import IBMQ
    IBMQ.load_account()
    big_backend = AerBackend(NoiseModel.from_backend(\
                    IBMQ.providers()[0].get_backend('ibmq_16_melbourne')))
else:
    big_backend = AerBackend()

#####################################################################

# Generates a random circuit, as well as a latex representation of it.
circ_info = random_circuit(n_qubits=n_qubits, depth=depth)
circ = build_circuit(circ_info)
circ.measure_all()
if latex:
    circ.to_latex_file("circ.tex")

# Generates the symmetrization of the circuit with specified number of copies, as 
# well as the latex representation.
sym_circ = symmetrize_circuit(circ_info,
                              n_copies=n_copies,
                              every=every, 
                              pairwise=pairwise,
                              reuse_cntrls=reuse_cntrls)
if latex:
    sym_circ["circuit"].to_latex_file("sym_circ.tex")

print("%d total qubits" % len(sym_circ["circuit"].qubits))

#####################################################################

# Runs the original random circuit on the backend, and displays the resulting
# measurement distribution.
tiny_backend.compile_circuit(circ)
circ_handle = tiny_backend.process_circuit(circ, n_shots=n_shots)
circ_result = tiny_backend.get_result(circ_handle)
print("original dist:")
for bitstr, prob in circ_result.get_distribution().items():
    print("  %s: %f" % (bitstr, prob))

#####################################################################

# Runs the symmetrized circuit on the backend.
big_backend.compile_circuit(sym_circ["circuit"])
sym_circ_handle = big_backend.process_circuit(sym_circ["circuit"], n_shots=n_shots)
sym_circ_result = big_backend.get_result(sym_circ_handle)

# Does postselection on all the control qubits being in the 0 state.
# There's some funkiness with the indices of the classical registers, hence the flip.
sym_circ_shots = np.flip(sym_circ_result.get_shots(), axis=1)
r = sum([len(group) for group in sym_circ["cntrl_bits"]])
for j in range(r):
    postselected_shots = sym_circ_shots[np.where(sym_circ_shots[:,j] == 0)]
for j in range(r):
    postselected_shots = np.delete(postselected_shots, 0, axis=1)
postselected_shots = np.flip(postselected_shots, axis=1)

# Calculates and displays the measurement distributions for each copy of the original circuit.
n_exp_qubits = n_qubits*n_copies
exp_shots = [postselected_shots.T[i:i+n_qubits].T for i in range(0, n_exp_qubits, n_qubits)]
postselected_dists = [probs_from_counts(counts_from_shot_table(eshots)) for eshots in exp_shots]
print("postselected experiment dists:")
for i, dist in enumerate(postselected_dists):
    print("  experiment %d" % i)
    for bitstr, prob in dist.items():
        print("    %s: %f" % (bitstr, prob))

# Calculates and displays the average measurement distribution across the copies.
averaged = {}
for dist in postselected_dists:
    for bitstr, prob in dist.items():
        if bitstr not in averaged:
            averaged[bitstr] = prob
        else:
            averaged[bitstr] += prob
total = sum(averaged.values())
print("averaged dists:")
for bitstr, prob in averaged.items():
    averaged[bitstr] = prob/total
    print("  %s: %f" % (bitstr, averaged[bitstr]))

#####################################################################

# For testing purposes, calculates the final state of the symmetrized circuit
# and does the projection manually.
if calc_state:
    state_backend = AerStateBackend()
    sym_circ2 = symmetrize_circuit(circ_info, n_copies=n_copies, every=every, measure=False)
    state_backend.compile_circuit(sym_circ2["circuit"])
    state_handle = state_backend.process_circuit(sym_circ2["circuit"])
    state = state_backend.get_result(state_handle).get_state()

    total_qubits = len(sym_circ2["circuit"].qubits)
    qstate = qt.Qobj(state, dims=[[2]*total_qubits, [1]*total_qubits])

    proj = qt.tensor(*[qt.basis(2,0)*qt.basis(2,0).dag()]*len(sym_circ2["cntrl_qubits"]), *[qt.identity(2)]*(total_qubits-len(sym_circ2["cntrl_qubits"])))
    pqstate = (proj*qstate).unit()

#####################################################################