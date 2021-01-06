import vpython as vp

from ..oscillators import *
from ..stars import *
from ..utils import *
from .majorana_sphere import *
from .operator_sphere import *

class OscillatorPlane:
    """
    Container for a 3D representation of a 2D oscillator state in the plane. Amplitudes at discretized positions
    are represented by arrows, and the expected position is a yellow sphere.
    """
    def __init__(self, state, pos):
        self.state = state
        self.pos = pos
        self.cutoff_dim = self.state.dims[0][0]

        self.Q = [qt.tensor(qt.position(self.cutoff_dim), qt.identity(self.cutoff_dim)),\
                  qt.tensor(qt.identity(self.cutoff_dim), qt.position(self.cutoff_dim))]
        QL, QV = qt.position(self.cutoff_dim).eigenstates()
        self.Qstates = [[qt.tensor(QV[i], QV[j]) for j in range(self.cutoff_dim)] for i in range(self.cutoff_dim)]

        self.plane_origin = vp.vector(0,-4,0)           
        pos_amps = [[self.state.overlap(self.Qstates[i][j]) for j in range(self.cutoff_dim)] for i in range(self.cutoff_dim)]
        self.vplane = vp.box(pos=self.pos+self.plane_origin, length=QL[-1]*2, height=QL[-1]*2, width=0.01)
        self.vpositions = [[vp.arrow(pos=self.pos+self.plane_origin+vp.vector(QL[i], QL[j], 0),\
                                     color=vp.color.black,\
                                     axis=2*vp.vector(pos_amps[i][j].real, pos_amps[i][j].imag, 0))\
                                        for j in range(self.cutoff_dim)] for i in range(self.cutoff_dim)]
        self.vexpected_pos = vp.sphere(pos=self.pos+self.plane_origin+vp.vector(qt.expect(self.Q[0], self.state).real,\
                                                                                qt.expect(self.Q[1], self.state).real, 0),\
                                        color=vp.color.yellow, radius=0.1)

    def refresh(self, state):
        self.state = state
        for i in range(self.cutoff_dim):
            for j in range(self.cutoff_dim):
                amp = self.state.overlap(self.Qstates[i][j]) 
                self.vpositions[i][j].axis = 2*vp.vector(amp.real, amp.imag, 0)
        self.vexpected_pos.pos = self.pos+self.plane_origin+vp.vector(qt.expect(self.Q[0], self.state).real,\
                                                                      qt.expect(self.Q[1], self.state).real, 0)

    def destroy(self):
        self.vplane.visible = False
        del self.vplane
        self.vexpected_pos.visible = False
        del self.vexpected_pos
        for i in range(self.cutoff_dim):
            for j in range(self.cutoff_dim):
                self.vpositions[i][j].visible = False
        self.vposition = None

class SchwingerSpheres:
    """
    Visualization for two oscillators as a tower of spin-j states. If `show_plane=True`, 
    displays a representation of the 2D oscillator position states.
    """
    def __init__(self, state=None, scene=None, pos=vp.vector(0,0,0), show_plane=False):
        super().__setattr__("state", state if state else vacuum())
        super().__setattr__("show_plane", show_plane)
        self.scene = scene if scene else vp.scene
        self.pos = pos

        self.cutoff_dim = self.state.dims[0][0]
        self.a = annihilators(n=2, cutoff_dim=self.cutoff_dim)
        self.map = osc_spintower_map(self.cutoff_dim)

        self.spins = osc_spins(self.state, map=self.map)
        self.n_spins = len(self.spins)
        positions = [0]
        for i in range(1, self.n_spins):
            next_position = positions[-1] + 2.3
            positions.append(next_position)
        positions = np.array(positions)-positions[-1]/2

        self.vspheres = [MajoranaSphere(self.spins[i],\
                                        scene=scene,\
                                        pos=self.pos+vp.vector(positions[i], 0, 0),\
                                        radius=1,\
                                        show_phase=True,\
                                        show_norm=True)\
                        for i in range(self.n_spins)]

        self.paulis = second_quantized_paulis(cutoff_dim=self.cutoff_dim)
        self.pauli_projectors = {}
        for s, o in self.paulis.items():
            L, V = o.eigenstates()
            P = [v*v.dag() for v in V]
            outcomes = {}
            for i, l in enumerate(L):
                if l not in outcomes:
                    outcomes[l] = P[i]
                else:
                    outcomes[l] += P[i]
            self.pauli_projectors[s] = outcomes

        if self.show_plane:
            self.vplane = OscillatorPlane(self.state, self.pos)
        
    def __setattr__(self, name, value):
        if name == "show_plane":
            if self.show_plane == False:
                self.vplane = OscillatorPlane(self.state, self.pos)
            else:
                self.vplane.destroy()
        super().__setattr__(name, value)
        if name == "state":
            self.refresh()

    def vacuum(self):
        """
        Load in the vacuum state.
        """
        self.state = vacuum(cutoff_dim=self.cutoff_dim)

    def random(self):
        """
        Load in a random state.
        """
        new_state = qt.rand_ket(self.state.shape[0])
        new_state.dims = self.state.dims
        self.state = new_state

    def raise_spin(self, spin, replace=False):
        """
        Raises a spin-j state.

        Parameters
        ----------
            spin : qt.Qobj

            replace : bool
                If True, raises the spin state from the vacuum.
        """
        if replace:
            self.vacuum()
        new_state = (second_quantize_spin_state(spin, self.a)*self.state)
        new_state = new_state.unit() if new_state.norm() != 0 else new_state
        self.state = new_state

    def lower_spin(self, spin):
        """
        Lowers a spin-j state.

        Parameters
        ----------
            spin : qt.Qobj
        """
        new_state = (second_quantize_spin_state(spin, self.a).dag()*self.state)
        new_state = new_state.unit() if new_state.norm() != 0 else new_state
        self.state = new_state

    def random_hamiltonian(self):
        """
        Returns
        -------
            H : qt.Qobj
                Random Hamiltonian of the right dimensions.
        """
        H = qt.rand_herm(self.state.shape[0])
        H.dims = [self.state.dims[0], self.state.dims[0]]
        return H

    def evolve(self, H=None, dt=0.05, T=2*np.pi):
        """
        Evolves the state, updating the visual in real time.

        Parameters
        ----------
            H : qt.Qobj
                Hamiltonian. If provided with a first quantized Hamiltonian, automatically second quantizes.

            dt : float
                Time step.

            T : float
                Time interval.
        """
        H = H if H else self.random_hamiltonian()
        if H.dims[0] == self.state.dims[0]:
            U = (-1j*H*dt).expm()
        else:
            U = (-1j*second_quantize_operator(H, self.a)*dt).expm()
        for t in np.linspace(0, T, int(T/dt)):
            self.state = U*self.state
            vp.rate(50)

    def measure(self, direction):
        """
        Applies a projective measurement (with random outcomes).

        Parameters
        ----------
            direction : str
                "x", "y", "z", or "q" (2D position).
        """
        if self.show_plane and direction == 'q':
            probs, indices = [], []
            for i in range(self.cutoff_dim):
                for j in range(self.cutoff_dim):
                    probs.append(abs(self.state.overlap(self.vplane.Qstates[i][j]))**2)
                    indices.append((i, j))
            probs = np.array(probs)
            choice = np.random.choice(list(range(len(probs))), p=abs(probs/sum(probs)))
            i, j = indices[choice]
            self.state = (self.vplane.Qstates[i][j]*self.vplane.Qstates[i][j].dag()*self.state).unit()
        else:        
            projectors = self.pauli_projectors[direction]
            eigenvalues = projectors.keys()
            P = [projectors[eig] for eig in eigenvalues]
            choice = measure(self.state, P)
            self.state = (P[choice]*self.state).unit()

    def refresh(self):
        self.spins = osc_spins(self.state, map=self.map)
        for i, vsphere in enumerate(self.vspheres):
            vsphere.spin = self.spins[i]
        if self.show_plane:
            self.vplane.refresh(self.state)

    def destroy(self):
        """
        Destroys the Schwinger spheres.
        """
        for vsphere in self.vspheres:
            vsphere.destroy()
        if self.show_plane:
            self.vplane.destroy()