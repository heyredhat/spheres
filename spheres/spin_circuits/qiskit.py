"""
Qiskit circuits for preparing spin-j states as permutation symmetric multiqubit states.
"""

from ..stars.pure import *
from ..symmetrization import *
from copy import deepcopy

from qiskit import QuantumCircuit, execute, ClassicalRegister, QuantumRegister
from qiskit import Aer, IBMQ, transpile
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit.quantum_info.operators import Operator
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter

def Rk_qiskit(k):
    """
    Single qubit operator employed in symmetrization circuit to prepare control qubits.

    Parameters
    ----------
        k : int
    
    Returns
    -------
        O : qiskit.quantum_info.operators.Operator
    """
    return Operator((1/np.sqrt(k+1))*\
                    np.array([[1, -np.sqrt(k)],\
                              [np.sqrt(k), 1]]))

def Tkj_qiskit(k, j):
    """
    Two qubit operator employed in symmetrization circuit to prepare control qubits.

    Parameters
    ----------
        k : int

        j : int

    Returns
    -------
        O : qiskit.quantum_info.operators.Operator
    """
    return Operator((1/np.sqrt(k-j+1))*\
                    np.array([[np.sqrt(k-j+1), 0, 0, 0],\
                              [0, 1, np.sqrt(k-j), 0],\
                              [0, -np.sqrt(k-j), 1, 0],\
                              [0, 0, 0, np.sqrt(k-j+1)]]))

def spin_sym_qiskit(spin):
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
        - "postselection": "0" x len(cntrl_bits) (postselection state to impose)

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

    circ = QuantumCircuit()
    spin_qubits = QuantumRegister(len(stars), "spin_qubits")
    circ.add_register(spin_qubits)

    for i, star in enumerate(stars):
        theta, phi = star
        circ.ry(theta, spin_qubits[i])
        circ.rz(phi, spin_qubits[i])

    cntrl_qubits = QuantumRegister(r, "cntrl_qubits")
    circ.add_register(cntrl_qubits)

    qubits = list(cntrl_qubits) + list(spin_qubits)
    offset = r
    for k in range(1, n):
        offset = offset-k
        circ.append(Rk_qiskit(k), [qubits[offset]])
        for i in range(k-1):
            circ.append(Tkj_qiskit(k, i+1), [qubits[offset+i+1], qubits[offset+i]])
        for i in range(k-1, -1, -1):
            circ.fredkin(qubits[offset+i], qubits[r+k], qubits[r+i])
        for i in range(k-2, -1, -1):
            circ.append(Tkj_qiskit(k, i+1).adjoint(), [qubits[offset+i+1], qubits[offset+i]])    
        circ.append(Rk_qiskit(k).adjoint(), [qubits[offset]])

    cntrl_bits = ClassicalRegister(r, "cntrl_bits")
    circ.add_register(cntrl_bits)
    for i in range(r):
        circ.measure(cntrl_qubits[i], cntrl_bits[i])

    return {"circuit": circ,\
            "spin_qubits": spin_qubits,\
            "cntrl_qubits": cntrl_qubits,\
            "cntrl_bits": cntrl_bits,\
            "postselect_on": "cntrl_bits",\
            "postselection": "0"*r}

def postselect_results_qiskit(circ_info, raw_results):
    """
    Performs postselection on Qiskit results given information in the provided dictionary. 

    Removes classical registers corresponding to circ_info["postselect_on"], leaving only those results with satisfy circ_info["postselection"].

    Parameters
    ----------
        circ_info : dict
    
        raw_results : qiskit.Result

    Returns
    -------
        postselected_results : qiskit.Result
    """
    n_not_cntrl = [len(creg) for creg in circ_info["circuit"].cregs if creg.name == circ_info["postselect_on"]][0]
    new_result = deepcopy(raw_results)
    for resultidx, _ in enumerate(raw_results.results):
        old_counts, new_counts = raw_results.get_counts(resultidx), {}
        postselection_index = next(i for i, v in enumerate(new_result.results[resultidx].header.creg_sizes[::-1]) if v[0] == circ_info["postselect_on"])
        new_result.results[resultidx].header.clbit_labels = list(filter(lambda clbit_label: clbit_label[0] != circ_info["postselect_on"], new_result.results[resultidx].header.clbit_labels))
        new_result.results[resultidx].header.creg_sizes = list(filter(lambda creg_size: creg_size[0] != circ_info["postselect_on"], new_result.results[resultidx].header.creg_sizes))
        new_result.results[resultidx].header.memory_slots = circ_info["circuit"].num_qubits - n_not_cntrl
        for reg_key in old_counts:
            reg_bits = reg_key.split(" ")
            if reg_bits[postselection_index] == circ_info["postselection"]:
                new_counts[" ".join([reg_bit for i, reg_bit in enumerate(reg_bits) if i != postselection_index])] = old_counts[reg_key]
            new_result.results[resultidx].data.counts = new_counts
    return new_result

def spin_tomography_qiskit(circ_info, backend_name="qasm_simulator", shots=8000):
    """
    Performs tomography on the provided symmetric multiqubit circuit, given postselection on the control qubits.

    Parameters
    ----------
        circ_info : dict

        backend_name : str
            Qiskit backend.
        
        shots : int
            Number of shots.
    
    Returns
    -------
        dm : qt.Qobj
            Reconstructed spin-j density matrix.
    """
    circ_sans_aux = circ_info["circuit"].remove_final_measurements(inplace=False)
    tomog_circs = state_tomography_circuits(circ_info["circuit"], circ_info["spin_qubits"])
    tomog_circs_sans_aux = state_tomography_circuits(circ_sans_aux, circ_info["spin_qubits"])

    if backend_name == "qasm_simulator":
        backend = Aer.get_backend("qasm_simulator")
        job = execute(tomog_circs, backend, shots=shots)
        raw_results = job.result()
    else:
        provider = IBMQ.load_account()
        job_manager = IBMQJobManager()
        backend = provider.get_backend(backend_name)
        job = job_manager.run(transpile(tomog_circs, backend=backend), backend=backend, name="spin_sym", shots=shots)
        raw_results = job.results().combine_results()

    postselected_results = postselect_results_qiskit(circ_info, raw_results)
    tomog_fit = StateTomographyFitter(postselected_results, tomog_circs_sans_aux)
    dm = qt.Qobj(tomog_fit.fit())
    dm.dims = [[2]*len(circ_info["spin_qubits"]), [2]*len(circ_info["spin_qubits"])]
    return sym_spin(dm)