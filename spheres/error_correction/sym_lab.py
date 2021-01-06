import csv
from spheres import *

from pytket.qiskit import tk_to_qiskit
from pytket.backends.ibm import AerBackend, AerStateBackend, IBMQBackend
from qiskit import QuantumCircuit, QuantumRegister, Aer, execute, IBMQ, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
from qiskit.providers.aer.noise import NoiseModel, QuantumError, ReadoutError, pauli_error, depolarizing_error, thermal_relaxation_error
import matplotlib.pyplot as plt

##################################################################################################################

def qiskit_pytket_counts(counts):
    return dict([(tuple([int(b) for b in "".join(bitstr.split())][::-1]), count) for bitstr, count in counts.items()])

def eval_circuit(circuit, noise_model=None, shots=8000, backend=None, analytic=False):
    circ = circuit.copy()
    if analytic:
        backend = AerStateBackend()
        backend.compile_circuit(circ)
        return backend.get_result(backend.process_circuit(circ)).get_distribution()
    circ = circuit.copy()
    AerBackend().compile_circuit(circ)
    qis_circ = tk_to_qiskit(circ)
    counts = execute(qis_circ,\
                     backend=Aer.get_backend('qasm_simulator') if not backend else backend,\
                     noise_model=noise_model,\
                     shots=shots).result().get_counts()
    return qiskit_pytket_counts(counts)

##################################################################################################################

def build_noise_model1(n_qubits=None, on_qubits=None):
    T1s = np.random.normal(50e3, 10e3, len(on_qubits)) 
    T2s = np.random.normal(70e3, 10e3, len(on_qubits)) 
    T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(len(on_qubits))])
    time_u1 = 0
    time_u2 = 50 
    time_u3 = 100 
    time_cx = 300
    time_reset = 1000 
    time_measure = 1000 
    errors_reset = [thermal_relaxation_error(t1, t2, time_reset)
                    for t1, t2 in zip(T1s, T2s)]
    errors_measure = [thermal_relaxation_error(t1, t2, time_measure)
                    for t1, t2 in zip(T1s, T2s)]
    errors_u1  = [thermal_relaxation_error(t1, t2, time_u1)
                for t1, t2 in zip(T1s, T2s)]
    errors_u2  = [thermal_relaxation_error(t1, t2, time_u2)
                for t1, t2 in zip(T1s, T2s)]
    errors_u3  = [thermal_relaxation_error(t1, t2, time_u3)
                for t1, t2 in zip(T1s, T2s)]
    errors_cx = [[thermal_relaxation_error(t1a, t2a, time_cx).expand(
                thermal_relaxation_error(t1b, t2b, time_cx))
                for t1a, t2a in zip(T1s, T2s)]
                for t1b, t2b in zip(T1s, T2s)]
    noise_thermal = NoiseModel()
    for qubit in list(range(n_qubits)):
        if qubit not in on_qubits:
            noise_thermal.add_quantum_error(depolarizing_error(0.000001, 1), ['u1'], [qubit])
    for j, q1 in enumerate(on_qubits):
        noise_thermal.add_quantum_error(errors_reset[j], "reset", [q1])
        noise_thermal.add_quantum_error(errors_measure[j], "measure", [q1])
        noise_thermal.add_quantum_error(errors_u1[j], "u1", [q1])
        noise_thermal.add_quantum_error(errors_u2[j], "u2", [q1])
        noise_thermal.add_quantum_error(errors_u3[j], "u3", [q1])
        for k, q2 in enumerate(on_qubits):
            noise_thermal.add_quantum_error(errors_cx[j][k], "cx", [q1, q2])
    return noise_thermal

def build_noise_model2(n_qubits=None, on_qubits=None):
    p_reset = 0.03
    p_meas = 0.1
    p_gate1 = 0.05
    error_reset = pauli_error([('X', p_reset), ('I', 1 - p_reset)])
    error_meas = pauli_error([('X',p_meas), ('I', 1 - p_meas)])
    error_gate1 = pauli_error([('X',p_gate1), ('I', 1 - p_gate1)])
    error_gate2 = error_gate1.tensor(error_gate1)
    noise_bit_flip = NoiseModel()
    for j, q1 in enumerate(on_qubits):
        noise_bit_flip.add_quantum_error(error_reset, "reset", [q1])
        noise_bit_flip.add_quantum_error(error_meas, "measure", [q1])
        noise_bit_flip.add_quantum_error(error_gate1, ["u1", "u2", "u3"], [q1])
        for k, q2 in enumerate(on_qubits):
            noise_bit_flip.add_quantum_error(error_gate2, "cx", [q1, q2])
    return noise_bit_flip

noise_models = {"thermal_noise": build_noise_model1,\
                "bitflip": build_noise_model2,\
                "ibmq_16_melbourne": lambda n_qubits, on_qubits: NoiseModel.from_backend(provider.get_backend('ibmq_16_melbourne'))}

##################################################################################################################

def run_experiment(n_qubits, depth, n_copies, every, pairwise, noise_model_id, error_on_all, use_remote_simulator, n_shots):
    circ_info = random_circuit(n_qubits=n_qubits, depth=depth)
    sym_circ_info = symmetrize_circuit(circ_info,
                                        n_copies=n_copies,
                                        every=every, 
                                        pairwise=pairwise,
                                        reuse_cntrls=False)
    circ = build_circuit(circ_info)
    sym_circ = sym_circ_info["circuit"]

    if len(sym_circ.qubits) > 14:
        return None

    circ_noise_model = noise_models[noise_model_id](n_qubits=n_qubits, on_qubits=list(range(n_qubits)))
    sym_circ_noise_model = noise_models[noise_model_id](n_qubits=len(sym_circ.qubits), on_qubits=list(range(len(sym_circ.qubits))) if error_on_all else \
                                                                                        list(range(len(sym_circ_info["cntrl_qubits"]), len(sym_circ.qubits))))

    analytic_circ_dist = eval_circuit(circ, analytic=True)
    noisy_circ_counts = eval_circuit(circ.measure_all(), noise_model=circ_noise_model, shots=n_shots)
    noisy_circ_dist = probs_from_counts(noisy_circ_counts)
    noisy_sym_circ_counts = eval_circuit(sym_circ,\
                                        noise_model=sym_circ_noise_model, 
                                        shots=n_shots,\
                                        backend=provider.get_backend("ibmq_qasm_simulator")\
                                            if use_remote_simulator else None)
    noisy_sym_circ_dist = process_sym_counts(sym_circ_info, noisy_sym_circ_counts)

    basis = list(product([0,1], repeat=n_qubits))
    expected_probs, actual_circ_probs, actual_sym_circ_probs = [], [], []
    for b in basis:
        expected_probs.append(analytic_circ_dist[b] if b in analytic_circ_dist else 0)
        actual_circ_probs.append(noisy_circ_dist[b] if b in noisy_circ_dist else 0)
        actual_sym_circ_probs.append(noisy_sym_circ_dist["avg_dist"][b] if b in noisy_sym_circ_dist["avg_dist"] else 0)
    circ_error = np.linalg.norm(np.array(actual_circ_probs) - np.array(expected_probs))
    sym_circ_error = np.linalg.norm(np.array(actual_sym_circ_probs) - np.array(expected_probs))
    
    experiment = {}
    experiment["n_qubits"] = n_qubits
    experiment["depth"] = depth
    experiment["n_copies"] = n_copies
    experiment["every"] = every
    experiment["pairwise"] = pairwise
    experiment["noise_model"] = noise_model_id
    experiment["error_on_all"] = error_on_all
    experiment["n_shots"] = n_shots
    experiment["circ_info"] = circ_info
    experiment["sym_circ_info"] = sym_circ_info
    experiment["analytic_circ_dist"] = analytic_circ_dist
    experiment["noisy_circ_dist"] = noisy_circ_dist
    experiment["noisy_sym_circ_dist"] = noisy_sym_circ_dist
    experiment["circ_error"] = circ_error
    experiment["sym_circ_error"] = sym_circ_error
    experiment["success"] = True if sym_circ_error < circ_error else False
    return experiment

##################################################################################################################

def make_plot(n_qubits, n_copies, every, pairwise, noise_model_id, error_on_all, avg_circ_errors, avg_sym_circ_errors, parameter_name, parameter):
    plt.title("n_qubits: %d, n_copies: %d, every: %d,\npairwise: %s, noise_model_id: %s, error_on_all: %s" % (n_qubits, n_copies, every, pairwise, noise_model_id, error_on_all))
    circ_plot, = plt.plot(parameter, avg_circ_errors, 'r', label="avg circ error")
    sym_plot, = plt.plot(parameter, avg_sym_circ_errors, 'b', label="avg sym circ error")
    plt.xlabel(parameter_name)
    plt.ylabel('error')
    plt.legend(handles=[circ_plot, sym_plot])
    plt.show()

##################################################################################################################

if True:
    provider = IBMQ.load_account()

n_qubits = 2
n_copies = 3
every = 3
pairwise = True
noise_model_id = "ibmq_16_melbourne"
error_on_all = True
use_remote_simulator = False
n_shots = 8000

depths = list(range(1, 12))
avg_circ_errors = []
avg_sym_circ_errors = []
for depth in depths:
    print("depth: %s" % depth)
    circ_errors = []
    sym_circ_errors = []
    for i in range(25):
        experiment = run_experiment(n_qubits, depth, n_copies, every, pairwise, noise_model_id, error_on_all, use_remote_simulator, n_shots)
        circ_errors.append(experiment["circ_error"] if experiment else 0)
        sym_circ_errors.append(experiment["sym_circ_error"] if experiment else 0)
    avg_circ_errors.append(sum(circ_errors)/len(circ_errors))
    avg_sym_circ_errors.append(sum(sym_circ_errors)/len(sym_circ_errors))

make_plot(n_qubits,\
          n_copies,\
          every,\
          pairwise,\
          noise_model_id,\
          error_on_all,\
          avg_circ_errors,\
          avg_sym_circ_errors,\
          "depth", depths)
