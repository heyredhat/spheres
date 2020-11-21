from spheres import *
from pytket.utils import counts_from_shot_table, probs_from_counts
from pytket.backends.ibm import AerBackend, AerStateBackend

from pytket.qiskit import tk_to_qiskit
from qiskit.ignis.mitigation.measurement import (complete_meas_cal,CompleteMeasFitter)
from qiskit import QuantumCircuit, QuantumRegister, Aer, execute
from qiskit.providers.aer.noise import NoiseModel
from qiskit import IBMQ
IBMQ.load_account()

#####################################################################

n_qubits = 2 # number of qubits in circuit to be symmetrized
depth = 5 # depth of circuit to be symmetrized
n_copies = 2 # copies of original circuit to be employed
n_shots = 8000 

#####################################################################

circ_info = random_circuit(n_qubits=n_qubits, depth=depth)
circ = build_circuit(circ_info)
circ.measure_all()

sym_circ = symmetrize_circuit(circ_info, n_copies=n_copies)

#####################################################################

tiny_backend = AerBackend()
tiny_backend.compile_circuit(circ)
circ_handle = tiny_backend.process_circuit(circ, n_shots=n_shots)
circ_result = tiny_backend.get_result(circ_handle)
print("original dist:")
for bitstr, prob in circ_result.get_distribution().items():
    print("  %s: %f" % (bitstr, prob))

#####################################################################

total_n_qubits = len(sym_circ["circuit"].qubits)
noise_model = NoiseModel.from_backend(IBMQ.providers()[0].get_backend('ibmq_vigo'))
qiskit_backend = Aer.get_backend('qasm_simulator')
meas_calibs, state_labels = complete_meas_cal(qr=QuantumRegister(total_n_qubits), circlabel='mcal')
noise_job = execute(meas_calibs, backend=qiskit_backend, shots=8000, noise_model=noise_model)
cal_results = noise_job.result()
meas_fitter = CompleteMeasFitter(cal_results, state_labels, circlabel='mcal')
big_backend = AerBackend(noise_model)

big_backend.compile_circuit(sym_circ["circuit"])
sym_circ_handle = big_backend.process_circuit(sym_circ["circuit"], n_shots=n_shots)
sym_circ_result = big_backend.get_result(sym_circ_handle)
sym_circ_counts = dict([("".join([str(b) for b in bitstr]), count)\
                        for bitstr, count in sym_circ_result.get_counts().items()])

#mitigated_sym_circ_results = meas_fitter.filter.apply(sym_circ_counts)
#mitigated_sym_circ_counts = mitigated_sym_circ_results.get_counts(0)


