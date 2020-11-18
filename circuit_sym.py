import numpy as np
from itertools import combinations

from spheres import *

from pytket import Circuit
from pytket.utils import counts_from_shot_table, probs_from_counts

#import boto3
#from pytket.backends.braket import BraketBackend

from pytket.backends.ibm import AerBackend, AerStateBackend
from pytket.backends.ibm import IBMQBackend
from qiskit.providers.aer.noise import NoiseModel
from qiskit import IBMQ

def random_pairs(of):
    pairs = list(combinations(of, 2)) 
    np.random.shuffle(pairs) 
    final_pairs = []
    for p in pairs:
        pair = np.array(p)
        np.random.shuffle(pair)
        final_pairs.append(pair)
    return final_pairs 

def random_circuit(n_qubits=1, depth=1):
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
            k, l = random_pairs(list(range(n_qubits)))[0]
            rand_gate = np.random.choice(list(gates_2q.keys()))
            moment.append({"gate": rand_gate, "to": [k, l]})
        circuit_info["history"].append(moment)
    return circuit_info

def fake_random_circuit(history, n_qubits=1, depth=1):
    gates_1q = {"H": lambda c, i: c.H(i),\
                "S": lambda c, i: c.S(i),\
                "T": lambda c, i: c.T(i)}
    gates_2q = {"CX": lambda c, i, j: c.CX(i, j)}
    circuit_info = {"history": history,\
                    "gate_map": {**gates_1q, **gates_2q},\
                    "n_qubits": n_qubits,\
                    "depth": depth}
    return circuit_info

def build_circuit(circuit_info):
    circ = Circuit(circuit_info["n_qubits"])
    for moment in circuit_info["history"]:
        for gate in moment:
            circuit_info["gate_map"][gate["gate"]](circ, *gate["to"])
    return circ

def symmetrize_circuit(circuit_info, n_copies=2, every=1, measure=True):
    circ = Circuit()
    qubit_registers = [circ.add_q_register("qexp%d" % i, circuit_info["n_qubits"]) for i in range(n_copies)]
    cbit_registers = [circ.add_c_register("cexp%d" % i, circuit_info["n_qubits"]) for i in range(n_copies)]
    
    r = int(n_copies*(n_copies-1)/2)
    cntrl_qubits = circ.add_q_register("cntrlqubits", r)
    cntrl_bits = [circ.add_c_register("cntrlbits%d" % i, r) for i in range(circuit_info["depth"])]

    for t, moment in enumerate(circuit_info["history"]):
        for gate in moment:
            for i in range(n_copies):
                apply_to = [qubit_registers[i][to] for to in gate["to"]]
                circuit_info["gate_map"][gate["gate"]](circ, *apply_to)
        
        offset = r
        for k in range(1, n_copies):
            offset = offset-k
            circ.add_unitary1qbox(Rk(k),\
                                  cntrl_qubits[offset])
            for i in range(k-1):
                circ.add_unitary2qbox(Tkj(k, i+1),\
                                      cntrl_qubits[offset+i+1],\
                                      cntrl_qubits[offset+i])
            for i in range(k-1, -1, -1):
                for j in range(circuit_info["n_qubits"]):
                    circ.CSWAP(cntrl_qubits[offset+i],\
                               qubit_registers[k][j],\
                               qubit_registers[i][j])
            for i in range(k-2, -1, -1):
                circ.add_unitary2qbox(Tkj(k, i+1, dagger=True),\
                                      cntrl_qubits[offset+i+1],\
                                      cntrl_qubits[offset+i])    
            circ.add_unitary1qbox(Rk(k, dagger=True),\
                                  cntrl_qubits[offset])
        if measure:
            for i in range(r):
                circ.Measure(cntrl_qubits[i], cntrl_bits[t][i])
    if measure:
        for i in range(n_copies):
            for j in range(circuit_info["n_qubits"]):
                circ.Measure(qubit_registers[i][j], cbit_registers[i][j])
    return {"circuit": circ, \
            "qubit_registers": qubit_registers,\
            "cbit_registers": cbit_registers,\
            "cntrl_qubits": cntrl_qubits,\
            "cntrl_bits": cntrl_bits}

#####################################################################

n_qubits = 3
depth = 10
n_copies = 3 
every = 1
n_shots = 8000
calc_state = False

backend = AerBackend()
#provider = IBMQ.load_account()
#noise_model = NoiseModel.from_backend(provider.get_backend('ibmq_16_melbourne'))
#backend = AerBackend(noise_model)
#backend = IBMQBackend("ibmq_qasm_simulator")

#aws_account_id = boto3.client("sts").get_caller_identity()["Account"]
#backend = BraketBackend(\
#    s3_bucket="amazon-braket-4462aa97a5a2",\
#    s3_folder="Output",\
#    device_type="quantum-simulator",\
#    provider="amazon")

#####################################################################

circ_info = random_circuit(n_qubits=n_qubits, depth=depth)
circ = build_circuit(circ_info)
circ.measure_all()
circ.to_latex_file("circ.tex")

sym_circ = symmetrize_circuit(circ_info, n_copies=n_copies, every=every)
sym_circ["circuit"].to_latex_file("sym_circ.tex")

print("%d total qubits" % len(sym_circ["circuit"].qubits))

#####################################################################

backend.compile_circuit(circ)
circ_handle = backend.process_circuit(circ, n_shots=n_shots)
circ_result = backend.get_result(circ_handle)
print("original dist:")
for bitstr, prob in circ_result.get_distribution().items():
    print("  %s: %f" % (bitstr, prob))

#####################################################################

backend.compile_circuit(sym_circ["circuit"])
sym_circ_handle = backend.process_circuit(sym_circ["circuit"], n_shots=n_shots)
sym_circ_result = backend.get_result(sym_circ_handle)
sym_circ_shots = np.flip(sym_circ_result.get_shots(), axis=1)

r = sum([len(group) for group in sym_circ["cntrl_bits"]])
for j in range(r):
    postselected_shots = sym_circ_shots[np.where(sym_circ_shots[:,j] == 0)]
for j in range(r):
    postselected_shots = np.delete(postselected_shots, 0, axis=1)
postselected_shots = np.flip(postselected_shots, axis=1)

n_exp_qubits = n_qubits*n_copies
exp_shots = [postselected_shots.T[i:i+n_qubits].T for i in range(0, n_exp_qubits, n_qubits)]
postselected_dists = [probs_from_counts(counts_from_shot_table(eshots)) for eshots in exp_shots]
print("postselected experiment dists:")
for i, dist in enumerate(postselected_dists):
    print("  experiment %d" % i)
    for bitstr, prob in dist.items():
        print("    %s: %f" % (bitstr, prob))

#####################################################################

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