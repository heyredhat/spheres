"""
"""

from ..stars.pure import *
from copy import deepcopy

from qiskit import QuantumCircuit, execute, ClassicalRegister, QuantumRegister
from qiskit import Aer, IBMQ, transpile
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit.quantum_info.operators import Operator
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter

def xyz_circ(xyzs):
    """
    Given n cartesian points on the sphere, returns a circuit which prepares
    n qubits with spins pointed in those directions.
    """
    circ = QuantumCircuit()
    spin_qubits = QuantumRegister(len(xyzs), "spin_qubits")
    circ.add_register(spin_qubits)
    for i, xyz in enumerate(xyzs):
        theta, phi = xyz_sph(xyz)
        circ.ry(theta, spin_qubits[i])
        circ.rz(phi, spin_qubits[i])
    return circ

def Rk(k):
    """
    Single qubit operator employed in symmetrization circuit to prepare control qubits.
    """
    return Operator((1/np.sqrt(k+1))*\
                    np.array([[1, -np.sqrt(k)],\
                              [np.sqrt(k), 1]]))

def Tkj(k, j):
    """
    Two qubit operator employed in symmetrization circuit to prepare control qubits.
    """
    return Operator((1/np.sqrt(k-j+1))*\
                    np.array([[np.sqrt(k-j+1), 0, 0, 0],\
                              [0, 1, np.sqrt(k-j), 0],\
                              [0, -np.sqrt(k-j), 1, 0],\
                              [0, 0, 0, np.sqrt(k-j+1)]]))

def spin_sym_circ(spin, measure_cntrls=False):
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

    circ = xyz_circ(spin_xyz(spin))
    spin_qubits = circ.qubits[:]
    cntrl_qubits = QuantumRegister(r, "cntrl_qubits")
    circ.add_register(cntrl_qubits)

    qubits = list(cntrl_qubits) + spin_qubits
    offset = r
    for k in range(1, n):
        offset = offset-k
        circ.append(Rk(k), [qubits[offset]])
        for i in range(k-1):
            circ.append(Tkj(k, i+1), [qubits[offset+i+1], qubits[offset+i]])
        for i in range(k-1, -1, -1):
            circ.fredkin(qubits[offset+i], qubits[r+k], qubits[r+i])
        for i in range(k-2, -1, -1):
            circ.append(Tkj(k, i+1).adjoint(), [qubits[offset+i+1], qubits[offset+i]])    
        circ.append(Rk(k).adjoint(), [qubits[offset]])
    if measure_cntrls:
        cntrl_bits = ClassicalRegister(r, "cntrl_bits")
        circ.add_register(cntrl_bits)
        for i in range(r):
            circ.measure(cntrl_qubits[i], cntrl_bits[i])
    return circ

def get_cntrl_qubits(circ):
    return [qreg for qreg in circ.qregs if qreg.name == "cntrl_qubits"][0]

def get_cntrl_bits(circ):
    return [creg for creg in circ.cregs if creg.name == "cntrl_bits"][0]

def get_spin_qubits(circ):
    return [qreg for qreg in circ.qregs if qreg.name == "spin_qubits"][0]

def postselect_spin_results(circ, raw_results):
    r = get_cntrl_qubits(circ).size
    n = get_spin_qubits(circ).size
    cntrl_index = circ.cregs.index(get_cntrl_bits(circ))
    new_result = deepcopy(raw_results)
    for resultidx, _ in enumerate(raw_results.results):
        old_counts = raw_results.get_counts(resultidx)
        new_counts = {}
        if cntrl_index == 0:
            new_result.results[resultidx].header.creg_sizes = [new_result.results[resultidx].header.creg_sizes[-1]]
            new_result.results[resultidx].header.clbit_labels = new_result.results[resultidx].header.clbit_labels[r:]
        else:
            new_result.results[resultidx].header.creg_sizes = [new_result.results[resultidx].header.creg_sizes[0]]
            new_result.results[resultidx].header.clbit_labels = new_result.results[resultidx].header.clbit_labels[0:-r]
        new_result.results[resultidx].header.memory_slots = n
        for reg_key in old_counts:
            reg_bits = reg_key.split(" ")
            if cntrl_index == 0:
                if reg_bits[-1] == "0"*r:
                    new_counts[reg_bits[0]] = old_counts[reg_key]
            else:
                if reg_bits[0] == "0"*r:
                    new_counts[reg_bits[1]] = old_counts[reg_key]
            new_result.results[resultidx].data.counts = new_counts
    return new_result

def spin_tomography(circ, backend_name="qasm_simulator", shots=8000):
    cnrl_qubits = get_cntrl_qubits(circ)
    r = len(cnrl_qubits)
    n = len(circ.qubits)-r

    tomog_circs = state_tomography_circuits(circ, circ.qregs[0])
    tomog_circs_sans_aux = deepcopy(tomog_circs)

    ca = ClassicalRegister(r)
    for tomog_circ in tomog_circs:
        tomog_circ.add_register(ca)
        for i in range(r):
            tomog_circ.measure(n+i, ca[i])

    if backend_name == "qasm_simulator":
        backend = Aer.get_backend("qasm_simulator")
        job = execute(tomog_circs, backend, shots=shots)
        raw_results = job.result()
    else:
        provider = IBMQ.load_account()
        job_manager = IBMQJobManager()
        backend = provider.get_backend(backend_name)
        job = job_manager.run(transpile(tomog_circs, backend=backend),\
                        backend=backend, name="spin_sym", shots=shots)
        raw_results = job.results().combine_results()

    postselected_results = postselect_spin_results(circ, raw_results)
    tomog_fit = StateTomographyFitter(postselected_results, tomog_circs_sans_aux)
    dm = qt.Qobj(tomog_fit.fit())
    dm.dims = [[2]*n, [2]*n]
    return dm