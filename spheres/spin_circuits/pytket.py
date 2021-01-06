"""
Pytket circuits for preparing spin-j states as permutation symmetric multiqubit states.
"""
from ..stars.pure import *
from ..symmetrization import *

from pytket.backends.ibm import AerBackend
from pytket import Circuit
from pytket.circuit import Unitary1qBox, Unitary2qBox
from pytket.utils import probs_from_counts

def Rk_pytket(k, dagger=False):
    """
    Single qubit operator employed in symmetrization circuit to prepare control qubits.

    Parameters
    ----------
        k : int

        dagger : bool
            Whether to return the adjoint.
    
    Returns
    -------
        O : pytket.circuit.Unitary1qBox
    """
    M = (1/np.sqrt(k+1))*np.array([[1, -np.sqrt(k)],\
                                     [np.sqrt(k), 1]])
    return Unitary1qBox(M if not dagger else M.T)

def Tkj_pytket(k, j, dagger=False):
    """
    Two qubit operator employed in symmetrization circuit to prepare control qubits.

    Parameters
    ----------
        k : int

        j : int

        dagger : bool
            Whether to return the adjoint.

    Returns
    -------
        O : pytket.circuit.Unitary2qBox
    """
    M = (1/np.sqrt(k-j+1))*np.array([[np.sqrt(k-j+1), 0, 0, 0],\
                                       [0, 1, np.sqrt(k-j), 0],\
                                       [0, -np.sqrt(k-j), 1, 0],\
                                       [0, 0, 0, np.sqrt(k-j+1)]])
    return Unitary2qBox(M if not dagger else M.T)

def spin_sym_pytket(spin):
    """
    Given a spin-j state, constructs a qiskit circuit which prepares that state as a 
    permutation symmetric state of 2j qubits. The circuit is probabalistic and depends on the control qubits
    being postselected on the up/0 state.
    
    Returns a dictionary whose elements are:
        - "circuit": Qiskit circuit
        - "spin_qubits": Qiskit quantum register for the qubits encoding the spin
        - "cntrl_qubits": Qiskit quantum register for the control qubits
        - "cntrl_bits": Qiskit classical register for the control measurements
        - "postselect_on": "cntrl_bits" (which classical register to control on)
        - "postselection": [0] x len(cntrl_bits) (postselection state to impose)

    Parameters
    ----------
        spin : qt.Qobj
            Spin-j state.

    Returns
    -------
        circuit_info : dict
    """
    j = (spin.shape[0]-1)/2
    n = int(2*j)
    r = int(n*(n-1)/2)
    stars = spin_sph(spin)

    circ = Circuit()
    spin_qubits = circ.add_q_register("spin_qubits", len(stars))
    for i, star in enumerate(stars):
        theta, phi = star
        circ.Ry(theta/np.pi, spin_qubits[i])
        circ.Rz(phi/np.pi, spin_qubits[i])

    cntrl_qubits = circ.add_q_register("cntrl_qubits", r)
    cntrl_bits = circ.add_c_register("cntrl_bits", r)
    qubits = circ.qubits

    offset = r
    for k in range(1, n):
        offset = offset-k
        circ.add_unitary1qbox(Rk_pytket(k), qubits[offset])
        for i in range(k-1):
            circ.add_unitary2qbox(Tkj_pytket(k, i+1),\
                                  qubits[offset+i+1],\
                                  qubits[offset+i])
        for i in range(k-1, -1, -1):
            circ.CSWAP(qubits[offset+i],\
                       qubits[r+k],\
                       qubits[r+i])
        for i in range(k-2, -1, -1):
            circ.add_unitary2qbox(Tkj_pytket(k, i+1, dagger=True), qubits[offset+i+1], qubits[offset+i])    
        circ.add_unitary1qbox(Rk_pytket(k, dagger=True), qubits[offset])
    for i in range(r):
        circ.Measure(cntrl_qubits[i], cntrl_bits[i])

    return {"circuit": circ,\
            "spin_qubits": spin_qubits,\
            "cntrl_qubits": cntrl_qubits,\
            "cntrl_bits": cntrl_bits,\
            "postselect_on": "cntrl_bits",\
            "postselection": [0]*r}

def postselect_shots(postselection_indices, postselection_values, original_shots):
    """
    Given an array of shots data, postselects on certain bits having certain values.

    Parameters
    ----------
        postselection_indices : list
            Indices to postselect on.
        
        postselection_values : list
            Desired values for each index.

        original_shots : list
            Shots data.
    
    Returns
    -------
        postselected_shots : list
    """
    postselected_shots = []
    for i, shots in enumerate(original_shots):
        for j, index in enumerate(postselection_indices):
            shots = shots[np.where(shots[:,index] == postselection_values[j])]
        shots = np.delete(shots, postselection_indices, axis=1)
        postselected_shots.append(shots)
    return postselected_shots

def tomography_circuits_pytket(circuit, on_qubits=None):
    """
    Given a pytket circuit and a list of qubits, constructs a set of circuits that
    implement tomography on those qubits.

    Parameters
    ----------
        circuit : pytket.Circuit

        on_qubits : list
            If not provided, tomography performed on all qubits.

    Returns
    -------
        tomography_circuit_info : list
            List of tomography circuits. Each element is a dictionary:
                - "circuit": Pytket circuit
                - "pauli": Pauli string corresponding to circuit
                - "tomog_bits": Pytket register.
    """
    on_qubits = circuit.qubits if on_qubits == None else on_qubits
    n_qubits = len(on_qubits)
    IXYZ = ["I", "X", "Y", "Z"]
    circuits = []
    for pauli_str in product(IXYZ, repeat=n_qubits):
        circ = circuit.copy()
        tomog_bits = circ.add_c_register("tomog_bits", n_qubits)
        for i, o in enumerate(pauli_str):
            if o == "Y":
                circ.Rx(1/2, on_qubits[i])
                circ.Measure(on_qubits[i], tomog_bits[i])
            elif o == "X":
                circ.Ry(-1/2, on_qubits[i])
                circ.Measure(on_qubits[i], tomog_bits[i])
            elif o == "Z":
                circ.Measure(on_qubits[i], tomog_bits[i])
        circuits.append({"circuit": circ,\
                         "pauli": "".join(pauli_str),\
                         "tomog_bits": tomog_bits})
    return circuits

def tomography_dm_pytket(tomog_circs_info, tomog_shots):
    """
    Given a set of Pytket tomography circuits and the results of measurements (shots),
    reconstructs the density matrix of the quantum state.

    Parameters
    ----------
        tomog_circs_info : dict
        
        tomog_shots : list

    Returns
    -------
        dm : qt.Qobj
    """
    exps = {}
    for i, shots in enumerate(tomog_shots):
        bad_indices = [j for j, s in enumerate(tomog_circs_info[i]["pauli"]) if s == "I"]
        shots = np.delete(shots, bad_indices, axis=1)
        if shots.shape[1] > 0:
            shots = np.where(shots==1, -1, shots)
            shots = np.where(shots==0, 1, shots)
            shots = np.prod(shots, axis=1)
        else:
            shots = np.array([1])
        exp = sum(shots)/len(shots)
        exps[tomog_circs_info[i]["pauli"]] = exp
    return from_pauli_basis(exps)

def spin_tomography_pytket(circ_info, backend=None, shots=8000):
    """
    Given a Pytket circuit preparing a spin state, runs tomography to reconstruct the density matrix.

    Parameters
    ----------
        circ_info : dict
            Information about the Pytket circuit.

        backend : pytket.backend.Backend

        shots : int

    Returns
    -------
        dm : qt.Qobj
    """
    backend = backend if backend else AerBackend()

    tomog_circs_info = tomography_circuits_pytket(circ_info["circuit"], on_qubits=circ_info["spin_qubits"])
    tomog_circs = [tomog_circ_info["circuit"] for tomog_circ_info in tomog_circs_info]
    [backend.compile_circuit(tomog_circ) for tomog_circ in tomog_circs]

    handles = backend.process_circuits(tomog_circs, n_shots=shots)
    results = backend.get_results(handles)
    tomog_shots = [result.get_shots() for result in results]

    postselection_indices = [i for i, bit in enumerate(tomog_circs_info[0]["circuit"].bits) if bit.reg_name == circ_info["postselect_on"]]
    return sym_spin(tomography_dm_pytket(tomog_circs_info, postselect_shots(postselection_indices, circ_info["postselection"], tomog_shots)))
