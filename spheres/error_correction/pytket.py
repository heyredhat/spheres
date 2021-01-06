"""
Pytket circuits for error correction via symmetrization.
"""
from pytket import Circuit
from pytket.circuit import Unitary1qBox, Unitary2qBox
from pytket.utils import probs_from_counts

from ..spin_circuits.pytket import *

def random_circuit(n_qubits=1, depth=1):
    """
    Generates a random circuit specification with a specified number of qubits and depth.
    Returns a dictionary containing a history of gates, divided into layers. 
    We don't return a circuit itself so we can continue to manipulate the circuit.
    """
    gates_1q = {"H": lambda c, i: c.H(i),\
                "S": lambda c, i: c.S(i),\
                "T": lambda c, i: c.T(i)}
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

def build_circuit(circuit_info):
    """
    Given a circuit specification (from `random_circuit()`), constructs the actual circuit.
    """
    circ = Circuit(circuit_info["n_qubits"])
    for moment in circuit_info["history"]:
        for gate in moment:
            circuit_info["gate_map"][gate["gate"]](circ, *gate["to"])
    return circ

def symmetrize_circuit(circuit_info, 
                       n_copies=2, 
                       every=1, 
                       pairwise=False, 
                       reuse_cntrls=False, 
                       measure=True):
    """
    Given a circuit specification, constructs a circuit with `n_copies` of the original circuit
    running in parallel, with symmetrization being performed across the copies after each layer.
    """
    circ = Circuit()
    qubit_registers = [circ.add_q_register("qexp%d" % i, circuit_info["n_qubits"]) for i in range(n_copies)]
    cbit_registers = [circ.add_c_register("cexp%d" % i, circuit_info["n_qubits"]) for i in range(n_copies)]
    
    n_sym_layers = int(circuit_info["depth"]/every)+1

    if pairwise:
        n_pairs = len(random_unique_pairs(n_copies))
        if reuse_cntrls:
            cntrl_qubits = circ.add_q_register("cntrlqubits", n_pairs)
        else:
            cntrl_qubits =  [circ.add_q_register("cntrlqubits%d" % t, n_pairs) for t in range(n_sym_layers)]
        cntrl_bits = [circ.add_c_register("cntrlbits%d" % t, n_pairs) for t in range(n_sym_layers)]
    else:
        r = int(n_copies*(n_copies-1)/2)
        if reuse_cntrls:
            cntrl_qubits = circ.add_q_register("cntrlqubits", r)
        else:
            cntrl_qubits =  [circ.add_q_register("cntrlqubits%d" % t, r) for t in range(n_sym_layers)]
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
                            print("*** %d" % t)
                            circ.Measure(cntrl_qubits[i], cntrl_bits[t][i])
                        else:
                            circ.Measure(cntrl_qubits[t][i], cntrl_bits[t][i])
            t += 1
    if measure:
        for i in range(n_copies):
            for j in range(circuit_info["n_qubits"]):
                circ.Measure(qubit_registers[i][j], cbit_registers[i][j])
    return {"circuit": circ, \
            "qubit_registers": qubit_registers,\
            "cbit_registers": cbit_registers,\
            "cntrl_qubits": cntrl_qubits,\
            "cntrl_bits": cntrl_bits,\
            "n_copies": n_copies,\
            "every": every,\
            "pairwise": False,\
            "reuse_cntrls": False,\
            "measure":True,\
            "n_qubits": circuit_info["n_qubits"]}

def process_sym_counts(sym_circ_info, sym_circ_counts):
    n_cntrl_qubits = sum([len(group) for group in sym_circ_info["cntrl_bits"]])
    n_qubits = sym_circ_info["n_qubits"]
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