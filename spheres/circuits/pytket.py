"""
"""
from ..stars.pure import *

from pytket import Circuit
from pytket.circuit import Unitary1qBox, Unitary2qBox
from pytket.utils import probs_from_counts

def prepare_qubits(xyzs):
    """
    Given n cartesian points on the sphere, returns a circuit which prepares
    n qubits with spins pointed in those directions.
    """
    circ = Circuit()
    spin_qubits = circ.add_q_register("spinqubits", len(xyzs))
    for i, xyz in enumerate(xyzs):
        theta, phi = xyz_sph(xyz)
        circ.Ry(theta/np.pi, spin_qubits[i])
        circ.Rz(phi/np.pi, spin_qubits[i])
    return circ

def Rk(k, dagger=False):
    """
    Single qubit operator employed in symmetrization circuit to prepare control qubits.
    """
    M = (1/np.sqrt(k+1))*np.array([[1, -np.sqrt(k)],\
                                     [np.sqrt(k), 1]])
    return Unitary1qBox(M if not dagger else M.T)

def Tkj(k, j, dagger=False):
    """
    Two qubit operator employed in symmetrization circuit to prepare control qubits.
    """
    M = (1/np.sqrt(k-j+1))*np.array([[np.sqrt(k-j+1), 0, 0, 0],\
                                       [0, 1, np.sqrt(k-j), 0],\
                                       [0, -np.sqrt(k-j), 1, 0],\
                                       [0, 0, 0, np.sqrt(k-j+1)]])
    return Unitary2qBox(M if not dagger else M.T)

def prepare_spin(spin, measure_cntrls=True):
    """
    Given a spin-j state, constructs a circuit which prepares that state as a 
    permutation symmetric state of 2j qubits. Returns a dictionary containing
    the circuit, the spin qubits, the control qubits, the control bits, and
    a postselection map. The circuit is probabalistic and depends on the control qubits
    being postselected all on the up state (measurements included if `measure_cntrls=True`).
    """
    j = (spin.shape[0]-1)/2
    n = int(2*j)
    r = int(n*(n-1)/2)

    circ = prepare_qubits(spin_xyz(spin))
    spin_qubits = circ.qubits
    cntrl_qubits = circ.add_q_register("cntrlqubits", r)
    cntrl_bits = circ.add_c_register("cntrlbits", r)
    qubits = circ.qubits

    offset = r
    for k in range(1, n):
        offset = offset-k
        circ.add_unitary1qbox(Rk(k),\
                              qubits[offset])
        for i in range(k-1):
            circ.add_unitary2qbox(Tkj(k, i+1),\
                                  qubits[offset+i+1],\
                                  qubits[offset+i])
        for i in range(k-1, -1, -1):
            circ.CSWAP(qubits[offset+i],\
                       qubits[r+k],\
                       qubits[r+i])
        for i in range(k-2, -1, -1):
            circ.add_unitary2qbox(Tkj(k, i+1, dagger=True),\
                                  qubits[offset+i+1],\
                                  qubits[offset+i])    
        circ.add_unitary1qbox(Rk(k, dagger=True),\
                              qubits[offset])
    if measure_cntrls:
        for i in range(r):
            circ.Measure(cntrl_qubits[i], cntrl_bits[i])
    return {"spin_qubits": spin_qubits,\
            "cntrl_qubits": cntrl_qubits,\
            "cntrl_bits": cntrl_bits,\
            "circuit": circ,\
            "postselect_on": dict([(cbit, 0) for cbit in cntrl_bits.values()])}

def postselect_shots(original_shots, postselect_on):
    """
    Given an array of shots data, postselects on certain qubits being in a certain state,
    specified by a dictionary.
    """
    r = len(postselect_on)
    postselected_shots = []
    for i, shots in enumerate(original_shots):
        for j in range(r):
            shots = shots[np.where(shots[:,j] == 0)]
        for j in range(r):
            shots = np.delete(shots, 0, axis=1)
        postselected_shots.append(shots)
    return postselected_shots

def tomography_circuits(circuit, on_qubits=None):
    """
    Given a circuit and a list of qubits, constructs a set of circuits that
    implement tomography on those qubits.
    """
    on_qubits = circuit.qubits if on_qubits == None else on_qubits
    n_qubits = len(on_qubits)
    IXYZ = ["I", "X", "Y", "Z"]
    circuits = []
    for pauli_str in product(IXYZ, repeat=n_qubits):
        circ = circuit.copy()
        tomog_bits = circ.add_c_register("tomogbits", n_qubits)
        for i, o in enumerate(pauli_str):
            if o == "Y":
                circ.Rx(1/2, on_qubits[i])
                circ.Measure(on_qubits[i], tomog_bits[i])
            elif o == "X":
                circ.Ry(-1/2, on_qubits[i])
                circ.Measure(on_qubits[i], tomog_bits[i])
            elif o == "Z":
                circ.Measure(on_qubits[i], tomog_bits[i])
        circuits.append({"pauli": "".join(pauli_str),\
                         "circuit": circ,\
                         "tomog_bits": tomog_bits})
    return circuits

def tomography_shots_dm(tomog_circs, tomog_shots):
    """
    Given a set of tomography circuits and the results of measurements (shots),
    reconstructs the density matrix of the quantum state.
    """
    exps = {}
    for i, shots in enumerate(tomog_shots):
        bad_indices = [j for j, s in enumerate(tomog_circs[i]["pauli"]) if s == "I"]
        shots = np.delete(shots, bad_indices, axis=1)
        if shots.shape[1] > 0:
            shots = np.where(shots==1, -1, shots)
            shots = np.where(shots==0, 1, shots)
            shots = np.prod(shots, axis=1)
        else:
            shots = np.array([1])
        exp = sum(shots)/len(shots)
        exps[tomog_circs[i]["pauli"]] = exp
    return from_pauli_basis(exps)

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
                        circ.add_unitary1qbox(Rk(1), cntrl_qubits[i])
                    else:
                        circ.add_unitary1qbox(Rk(1), cntrl_qubits[t][i])
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
                        circ.add_unitary1qbox(Rk(1, dagger=True), cntrl_qubits[i])
                    else:
                        circ.add_unitary1qbox(Rk(1, dagger=True), cntrl_qubits[t][i])
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
                        circ.add_unitary1qbox(Rk(k), cntrl_qubits[offset])
                    else:
                        circ.add_unitary1qbox(Rk(k), cntrl_qubits[t][offset])
                    for i in range(k-1):
                        if reuse_cntrls:
                            circ.add_unitary2qbox(Tkj(k, i+1),\
                                              cntrl_qubits[offset+i+1],\
                                              cntrl_qubits[offset+i])
                        else:
                            circ.add_unitary2qbox(Tkj(k, i+1),\
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
                            circ.add_unitary2qbox(Tkj(k, i+1, dagger=True),\
                                                  cntrl_qubits[offset+i+1],\
                                                  cntrl_qubits[offset+i])
                        else:
                            circ.add_unitary2qbox(Tkj(k, i+1, dagger=True),\
                                                  cntrl_qubits[t][offset+i+1],\
                                                  cntrl_qubits[t][offset+i])
                    if reuse_cntrls:
                        circ.add_unitary1qbox(Rk(k, dagger=True), cntrl_qubits[offset])
                    else:
                        circ.add_unitary1qbox(Rk(k, dagger=True), cntrl_qubits[t][offset])
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

