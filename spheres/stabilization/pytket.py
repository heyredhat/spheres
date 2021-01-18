"""
Pytket circuits for error stabilization via symmetrization.
"""

from pytket import Circuit, OpType
from pytket.circuit import Unitary1qBox, Unitary2qBox
from pytket.utils import probs_from_counts
from pytket.qiskit import tk_to_qiskit
from pytket.backends.ibm import AerBackend, AerStateBackend, IBMQBackend

from qiskit import QuantumCircuit, QuantumRegister, Aer, execute, IBMQ, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
from qiskit.providers.aer.noise import NoiseModel, QuantumError, ReadoutError, pauli_error, depolarizing_error, thermal_relaxation_error

import matplotlib.pyplot as plt

from ..spin_circuits.pytket import *

def random_pairs(n):
    """
    Generates a random list of pairs of n elements. 

    Parameters
    ----------
        n : int

    Returns
    -------
        list : list
    """
    pairs = list(combinations(list(range(n)), 2))
    np.random.shuffle(pairs) 
    final_pairs = []
    for p in pairs:
        pair = np.array(p)
        np.random.shuffle(pair)
        final_pairs.append(pair)
    return final_pairs 

def random_unique_pairs(n):
    """
    Generates a random list of pairs of n elements with no pair sharing an element.

    Parameters
    ----------
        n : int

    Returns
    -------
        list : list
    """
    pairs = list(combinations(list(range(n)), 2))
    pick = pairs[np.random.choice(len(pairs))]
    good_pairs = [pick]
    used = [*pick]
    def clean_pairs(pairs, used):
        to_remove = []
        for i, pair in enumerate(pairs):
            for u in used:
                if u in pair:
                    to_remove.append(i)
        return [i for j, i in enumerate(pairs) if j not in to_remove]
    pairs = clean_pairs(pairs, used)
    while len(pairs) > 0 and len(pairs) != clean_pairs(pairs, used):
        pick = pairs[np.random.choice(len(pairs))]
        good_pairs.append(pick)
        used.extend(pick)
        pairs = clean_pairs(pairs, used)
    return good_pairs

def random_pytket_circuit(n_qubits=1, depth=1):
    """
    Generates a random circuit specification with a specified number of qubits and depth.
    Returns a dictionary containing a history of gates, divided into layers. 
    We don't return a circuit itself so we can continue to manipulate the circuit.

    Parameters
    ----------
        n_qubits : int

        depth : int

    Returns
    -------
        circ_info : dict
            - "history": a list of circuit layers, each of which contains
                a list of dicts with keys "gate": ('H', 'S', 'T', or 'CX') and "to": list of qubits.
            - "gate_map": a dictionary mapping gate strings to functions that take a circuit
                and qubit indices and add the gate to the circuit
            - "n_qubits"
            - "depth"
    """
    gates_1q = {"U1(pi/2)": lambda c, i: c.add_gate(OpType.U1, 1/2, [i]),\
                "U1(pi/4)": lambda c, i: c.add_gate(OpType.U1, 1/4, [i]),\
                "U1(7pi/4)": lambda c, i: c.add_gate(OpType.U1, 7/4, [i]),\
                "U3(pi/2, 0, pi)": lambda c, i: c.add_gate(OpType.U3, [1/2, 0, 1], [i]),\
                "U3(pi/4, 0, pi)": lambda c, i: c.add_gate(OpType.U3, [1/4, 0, 1], [i])}
    gates_2q = {"CX": lambda c, i, j: c.CX(i, j)}
    circuit_info = {"history": [],\
                    "gate_map": {**gates_1q, **gates_2q},\
                    "n_qubits": n_qubits,\
                    "depth": depth}
    for i in range(depth):
        moment = []
        for j in range(n_qubits):
            rand_gate = np.random.choice(list(gates_1q.keys()))
            moment.append({"gate": rand_gate, "to": [j]})
        if n_qubits > 1:
            k, l = random_pairs(n_qubits)[0]
            rand_gate = np.random.choice(list(gates_2q.keys()))
            moment.append({"gate": rand_gate, "to": [k, l]})
        circuit_info["history"].append(moment)
    return circuit_info

def build_pytket_circuit(circuit_info):
    """
    Given a circuit specification (from `random_circuit()`), constructs the actual pytket circuit.

    Parameters
    ----------
        circuit_info : dict

    Returns
    -------
        circuit : pytket.Circuit
    """
    circ = Circuit(circuit_info["n_qubits"])
    for moment in circuit_info["history"]:
        for gate in moment:
            circuit_info["gate_map"][gate["gate"]](circ, *gate["to"])
    return circ

def symmetrize_pytket_circuit(circuit_info, 
                              n_copies=2, 
                              every=1, 
                              pairwise=False, 
                              reuse_cntrls=False, 
                              measure=True):
    """
    Given a circuit specification, constructs a symmetrized version of the circuit for error correction.

    Parameters
    ----------
        circuit_info : dict
            Dictionary of circuit information.
        
        n_copies : int
            Number of copies of the original circuit to evaluate in parallel.

        every : int
            How often to perform symmetrization, i.e. every `every` layers.
        
        pairwise : bool
            Whether to symmetrize across all circuit copies or across circuit pairs.
        
        reuse_cntrls : bool
            Whether to to reuse control qubits from symmetrization to symmetrization. Only useful
            on quantum computers that allow for intermediate measurements.
        
        measure : bool
            Whether to measure all the non-control qubits in the end.
    
    Returns
    -------
        circuit_info : dict
            - "circuit": pytket.Circuit
            - "n_copies"
            - "every"
            - "pairwise"
            - "reuse_cntrls"
            - "measure"
            - "original": original circ_info dict
            - "qubit_registers": list of qubit registers for each 'experiment'
            - "cbit_registers": list of bit registers for each 'experiment'
            - "cntrl_qubits": list of control qubits for each symmetrization
            - "cntrl_bits": list of control bits for each symmetrization
            
    """
    circ = Circuit()
    qubit_registers = [circ.add_q_register("qcirc%d" % i, circuit_info["n_qubits"]) for i in range(n_copies)]
    cbit_registers = [circ.add_c_register("ccirc%d" % i, circuit_info["n_qubits"]) for i in range(n_copies)]
    
    n_sym_layers = int(circuit_info["depth"]/every)+1

    if pairwise:
        n_pairs = len(random_unique_pairs(n_copies))
        cntrl_qubits = circ.add_q_register("cntrlqubits", n_pairs) if reuse_cntrls \
                       else [circ.add_q_register("cntrlqubits%d" % t, n_pairs) for t in range(n_sym_layers)]
        cntrl_bits = [circ.add_c_register("cntrlbits%d" % t, n_pairs) for t in range(n_sym_layers)]
    else:
        r = int(n_copies*(n_copies-1)/2)
        cntrl_qubits = circ.add_q_register("cntrlqubits", r) if reuse_cntrls \
                       else [circ.add_q_register("cntrlqubits%d" % t, r) for t in range(n_sym_layers)]
        cntrl_bits = [circ.add_c_register("cntrlbits%d" % t, r) for t in range(n_sym_layers)]

    t = 0
    for layer, moment in enumerate(circuit_info["history"]):
        for gate in moment:
            for i in range(n_copies):
                apply_to = [qubit_registers[i][to] for to in gate["to"]]
                circuit_info["gate_map"][gate["gate"]](circ, *apply_to)
        if layer % every == 0:
            if pairwise:
                pairs = random_unique_pairs(n_copies)
                for i, pair in enumerate(pairs):
                    if reuse_cntrls:
                        circ.add_unitary1qbox(Rk_pytket(1), cntrl_qubits[i])
                    else:
                        circ.add_unitary1qbox(Rk_pytket(1), cntrl_qubits[t][i])
                    for j in range(circuit_info["n_qubits"]):
                        if reuse_cntrls:
                            circ.CSWAP(cntrl_qubits[i],\
                                       qubit_registers[pair[0]][j],\
                                       qubit_registers[pair[1]][j])  
                        else:
                            circ.CSWAP(cntrl_qubits[t][i],\
                                       qubit_registers[pair[0]][j],\
                                       qubit_registers[pair[1]][j])
                    if reuse_cntrls:  
                        circ.add_unitary1qbox(Rk_pytket(1, dagger=True), cntrl_qubits[i])
                    else:
                        circ.add_unitary1qbox(Rk_pytket(1, dagger=True), cntrl_qubits[t][i])
                    if measure:
                        if reuse_cntrls:
                            circ.Measure(cntrl_qubits[i], cntrl_bits[t][i])
                        else:
                            circ.Measure(cntrl_qubits[t][i], cntrl_bits[t][i])
            else:
                offset = r
                for k in range(1, n_copies):
                    offset = offset-k
                    if reuse_cntrls:
                        circ.add_unitary1qbox(Rk_pytket(k), cntrl_qubits[offset])
                    else:
                        circ.add_unitary1qbox(Rk_pytket(k), cntrl_qubits[t][offset])
                    for i in range(k-1):
                        if reuse_cntrls:
                            circ.add_unitary2qbox(Tkj_pytket(k, i+1),\
                                              cntrl_qubits[offset+i+1],\
                                              cntrl_qubits[offset+i])
                        else:
                            circ.add_unitary2qbox(Tkj_pytket(k, i+1),\
                                              cntrl_qubits[t][offset+i+1],\
                                              cntrl_qubits[t][offset+i])
                    for i in range(k-1, -1, -1):
                        for j in range(circuit_info["n_qubits"]):
                            if reuse_cntrls:
                                circ.CSWAP(cntrl_qubits[offset+i],\
                                           qubit_registers[k][j],\
                                           qubit_registers[i][j])
                            else:
                                circ.CSWAP(cntrl_qubits[t][offset+i],\
                                           qubit_registers[k][j],\
                                           qubit_registers[i][j])
                    for i in range(k-2, -1, -1):
                        if reuse_cntrls:
                            circ.add_unitary2qbox(Tkj_pytket(k, i+1, dagger=True),\
                                                  cntrl_qubits[offset+i+1],\
                                                  cntrl_qubits[offset+i])
                        else:
                            circ.add_unitary2qbox(Tkj_pytket(k, i+1, dagger=True),\
                                                  cntrl_qubits[t][offset+i+1],\
                                                  cntrl_qubits[t][offset+i])
                    if reuse_cntrls:
                        circ.add_unitary1qbox(Rk_pytket(k, dagger=True), cntrl_qubits[offset])
                    else:
                        circ.add_unitary1qbox(Rk_pytket(k, dagger=True), cntrl_qubits[t][offset])
                if measure:
                    for i in range(r):
                        if reuse_cntrls:
                            circ.Measure(cntrl_qubits[i], cntrl_bits[t][i])
                        else:
                            circ.Measure(cntrl_qubits[t][i], cntrl_bits[t][i])
            t += 1
    if measure:
        for i in range(n_copies):
            for j in range(circuit_info["n_qubits"]):
                circ.Measure(qubit_registers[i][j], cbit_registers[i][j])
    return {"circuit": circ, \
            "n_copies": n_copies,\
            "every": every,\
            "pairwise": pairwise,\
            "reuse_cntrls": reuse_cntrls,\
            "measure": measure,\
            "original": circuit_info,\
            "qubit_registers": qubit_registers,\
            "cbit_registers": cbit_registers,\
            "cntrl_qubits": cntrl_qubits,\
            "cntrl_bits": cntrl_bits}

def process_symmetrized_pytket_counts(sym_circ_info, sym_circ_counts):
    """
    Parameters
    ----------
        sym_circ_info : dict
            Symmetrized circuit info, i.e. from `symmetrize_pytket_circuit`.
        
        sym_circ_counts : dict
            Dictionary of empirical counts.

    Returns
    -------
        results : dict
            - "exp_dists": Distribution of outcomes for each experiment.
            - "avg_dists": Average distribution of outcomes across the experiments.
    """
    n_cntrl_qubits = sum([len(group) for group in sym_circ_info["cntrl_bits"]])
    n_qubits = sym_circ_info["original"]["n_qubits"]
    n_copies = sym_circ_info["n_copies"]
    n_exp_qubits = n_qubits*n_copies

    postselected_sym_circ_counts = dict(filter(lambda e: e[0][-n_cntrl_qubits:].count(0) == n_cntrl_qubits, sym_circ_counts.items()))

    exp_counts = [dict() for i in range(n_copies)]
    for bitstr, count in postselected_sym_circ_counts.items():
        for i in range(0, n_exp_qubits, n_qubits):
            if bitstr[i:i+n_qubits] in exp_counts[int(i/n_qubits)]:
                exp_counts[int(i/n_qubits)][bitstr[i:i+n_qubits]] += count
            else:
                exp_counts[int(i/n_qubits)][bitstr[i:i+n_qubits]] = count
    exp_dists = [probs_from_counts(ec) for ec in exp_counts]

    averaged_dist = {}
    for dist in exp_dists:
        for bitstr, prob in dist.items():
            if bitstr not in averaged_dist:
                averaged_dist[bitstr] = prob
            else:
                averaged_dist[bitstr] += prob
    total = sum(averaged_dist.values())
    averaged_dist = dict([(bitstr, prob/total) for bitstr, prob in averaged_dist.items()])
    return {"exp_dists": exp_dists, "avg_dist": averaged_dist}

def qiskit_pytket_counts(counts):
    """
    Converts qiskit counts into pytket format.

    Parameters
    ----------
        qis_counts : dict
            Qiskit counts.
    
    Returns
    -------
        pyt_counts : dict
            Pytket counts.
    """
    return dict([(tuple([int(b) for b in "".join(bitstr.split())][::-1]), count) for bitstr, count in counts.items()])

def pytket_qiskit_counts(counts):
    """
    Converts pytket counts into qiskit format.

    Parameters
    ----------
        pyt_counts : dict
            Pytket counts.
    
    Returns
    -------
        qis_counts : dict
            Qiskit counts.
    """
    return dict([("".join([str(b) for b in bitstr[::-1]]), count) for bitstr, count in counts.items()])

def eval_pytket_circuit_ibm(circuit, noise_model=None, shots=8000, backend=None, analytic=False):
    """
    Evaluates pytket circuit on IBM backend. 

    Parameters
    ----------
        circuit : pytket.Circuit

        noise_mode: qiskit.providers.aer.noise.NoiseModel

        shots : int

        backend : qiskit backend

        analytic : bool
            Whether to process the circuit analytically.

    Returns
    -------
        counts : dict
            Dictionary of counts (or distribution if analytic).
    """
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

def eval_symmetrized_pytket_circuit(circ_info,\
                                    n_copies=2,\
                                    every=1,\
                                    pairwise=False,\
                                    noise_model=None,\
                                    backend=None,\
                                    shots=8000):
    """
    Evaluates a symmetrized version of the provided circuit in the form of a dictionary (cf. `random_pytket_circuit`).

    Parameters
    ----------
        circ_info : dict

        n_copies : int

        every : int

        pairwise : bool

        noise_model : qiskit.providers.aer.noise.NoiseModel

        backend : qiskit backend

        shots : int

    Returns
    -------
        dists : dict
        
    """
    sym_circ_info = symmetrize_pytket_circuit(circ_info,
                                              n_copies=n_copies,
                                              every=every, 
                                              pairwise=pairwise,
                                              reuse_cntrls=False)
    counts = eval_pytket_circuit_ibm(sym_circ_info["circuit"], noise_model=noise_model, shots=shots, backend=backend)
    return process_symmetrized_pytket_counts(sym_circ_info, counts)

def eval_pytket_symmetrization_performance(n_qubits=2,\
                                           depth=3,\
                                           n_copies=2,\
                                           every=1,\
                                           pairwise=True,\
                                           noise_model=None,\
                                           noise_model_name="",\
                                           error_on_all=True,\
                                           backend=None,\
                                           shots=8000):
    """
    Compares the performance of a random circuit with and without symmetrization. 

    Parameters
    ----------
        n_qubits : int
            Number of qubits in the original circuit.
        
        depth : int
            Number of layers in the original circuit.
        
        n_copies : int
            Number of copies of original circuit to symmetrize over.

        every : int
            How often to symmetrize.

        pairwise : bool
            Whether to symmetrize across all circuit copies or just pairwise.
        
        noise_model : func
            A function which takes (n_qubits, on_qubits) and returns a qiskit error model.

        noise_model_name : str
            Name for the noise model.
        
        error_on_all : bool
            Whether to apply noise to all qubits, including controls, or whether to exclude the controls.

        backend : qiskit backend
        
        shots : int
            Number of shots.

    Returns
    -------
        experiment : dict
            Dictionary of information about the experiment run.
    """
    circ_info = random_pytket_circuit(n_qubits=n_qubits, depth=depth)
    sym_circ_info = symmetrize_pytket_circuit(circ_info,
                                              n_copies=n_copies,
                                              every=every, 
                                              pairwise=pairwise,
                                              reuse_cntrls=False)
    circ = build_pytket_circuit(circ_info)
    sym_circ = sym_circ_info["circuit"]

    circ_noise_model = noise_model(n_qubits=n_qubits, on_qubits=list(range(n_qubits))) if noise_model else None
    sym_circ_noise_model = (noise_model(n_qubits=len(sym_circ.qubits), on_qubits=list(range(len(sym_circ.qubits))) if error_on_all else \
                                                                                list(range(len(sym_circ_info["cntrl_qubits"]), len(sym_circ.qubits)))))\
                                                                                    if noise_model else None

    analytic_circ_dist = eval_pytket_circuit_ibm(circ, analytic=True)
    noisy_circ_dist = probs_from_counts(eval_pytket_circuit_ibm(circ.measure_all(), noise_model=circ_noise_model, backend=backend, shots=shots))

    noisy_sym_circ_dist = process_symmetrized_pytket_counts(sym_circ_info,\
                                eval_pytket_circuit_ibm(sym_circ,\
                                        noise_model=sym_circ_noise_model, 
                                        shots=shots,\
                                        backend=backend))

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
    experiment["noise_model"] = noise_model_name
    experiment["error_on_all"] = error_on_all
    experiment["shots"] = shots
    experiment["circ_info"] = circ_info
    experiment["sym_circ_info"] = sym_circ_info
    experiment["analytic_circ_dist"] = analytic_circ_dist
    experiment["noisy_circ_dist"] = noisy_circ_dist
    experiment["noisy_sym_circ_dist"] = noisy_sym_circ_dist
    experiment["circ_error"] = circ_error
    experiment["sym_circ_error"] = sym_circ_error
    experiment["success"] = True if sym_circ_error < circ_error else False
    return experiment

def plot_symmetrization_performance(parameter, experiments):
    """
    Given a parameter (e.g. "depth") and a list of symmetrization experiments, plots the error varying the parameter.

    Parameters
    ----------
        parameter : str
            Could be "n_qubits", "depth", "n_copies", "every", "pairwise", "noise_model_name", "error_on_all".
        
        experiments : list
            List of experiments returned by `eval_pytket_symmetrization_performance`.
    """
    fixed_parameters = [p for p in ["n_qubits", "depth", "n_copies", "every", "pairwise", "noise_model_name", "error_on_all"] if p != parameter]
    plt.title(", ".join(["%s: %s" % (fixed_parameter, str(experiments[0][fixed_parameter])) for fixed_parameter in fixed_parameters]))
    varying_parameter = [experiment[parameter] for experiment in experiments]
    circ_errors =  [experiment["circ_error"] for experiment in experiments]
    sym_circ_errors =  [experiment["sym_circ_error"] for experiment in experiments]
    circ_plot, = plt.plot(varying_parameter, circ_errors, 'r', label="avg circ error")
    sym_plot, = plt.plot(varying_parameter, sym_circ_errors, 'b', label="avg sym circ error")
    plt.xlabel(parameter)
    plt.ylabel("error")
    plt.legend(handles=[circ_plot, sym_plot])
    plt.show()

def thermal_noise_model(n_qubits=None, on_qubits=None):
    """
    Thermal noise model.

    Parameters
    ----------
        n_qubits : int

        on_qubits : list

    Returns
    -------
        noise_model : qiskit noise model
    """
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

def bitflip_noise_model(n_qubits=None, on_qubits=None):
    """
    Bitflip noise model.

    Parameters
    ----------
        n_qubits : int

        on_qubits : list

    Returns
    -------
        noise_model : qiskit noise model
    """
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

def ibmq_16_melbourne_noise_model(n_qubits=None, on_qubits=None):
    """
    ibmq_16_melbourne_noise_model noise model.

    Parameters
    ----------
        n_qubits : int

        on_qubits : list

    Returns
    -------
        noise_model : qiskit noise model
    """
    provider = IBMQ.load_account()
    return NoiseModel.from_backend(provider.get_backend('ibmq_16_melbourne'))

def qiskit_error_calibration(n_qubits, noise_model, use_remote_simulator=False, shots=8000):
    """
    Initializes Qiskit error calibration. 

    Parameters
    ----------
        n_qubits : int

        noise_model : qiskit noise model

        use_remote_simulator : bool

    Returns
    -------
        filter : qiskit noise filter
    """
    meas_calibs, state_labels = complete_meas_cal(qr=QuantumRegister(n_qubits), circlabel='mcal')
    if not use_remote_simulator:
        noise_job = execute(meas_calibs, backend=Aer.get_backend("qasm_simulator"), shots=shots, noise_model=noise_model)
        cal_results = noise_job.result()
    else:
        noise_job = job_manager.run(meas_calibs, backend=provider.get_backend("ibmq_qasm_simulator"), shots=shots, noise_model=noise_model)
        cal_results = noise_job.results().combine_results()                                  
    meas_fitter = CompleteMeasFitter(cal_results, state_labels, circlabel='mcal')
    return meas_fitter.filter