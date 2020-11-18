"""
Circuits
--------

"""

from spheres import *

from pytket import Circuit
from pytket.circuit import Unitary1qBox, Unitary2qBox

def prepare_qubits(xyzs):
    circ = Circuit()
    spin_qubits = circ.add_q_register("spinqubits", len(xyzs))
    for i, xyz in enumerate(xyzs):
        r, phi, theta = xyz_sph(xyz)
        circ.Ry(theta/np.pi, spin_qubits[i])
        circ.Rz(phi/np.pi, spin_qubits[i])
    return circ

def Rk(k, dagger=False):
    M = (1/np.sqrt(k+1))*np.array([[1, -np.sqrt(k)],\
                                     [np.sqrt(k), 1]])
    return Unitary1qBox(M if not dagger else M.T)

def Tkj(k, j, dagger=False):
    M = (1/np.sqrt(k-j+1))*np.array([[np.sqrt(k-j+1), 0, 0, 0],\
                                       [0, 1, np.sqrt(k-j), 0],\
                                       [0, -np.sqrt(k-j), 1, 0],\
                                       [0, 0, 0, np.sqrt(k-j+1)]])
    return Unitary2qBox(M if not dagger else M.T)

def prepare_spin(spin, measure_cntrls=True):
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
	exps = {}
	for i, shots in enumerate(tomog_shots):
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
	return from_pauli_basis(exps)
