import vpython as vp

from ..oscillators import *
from ..stars import *
from ..utils import *
from .majorana_sphere import *
from .operator_sphere import *

class SchwingerSpheres:
    """
    Visualization for two oscillators as a tower of spin-j states.
    """
    def __init__(self, scene=None, pos=vp.vector(0,0,0), state=None, cutoff_dim=3, show_plane=False):
        self.pos = pos
        self.n_osc = 2
        self.cutoff_dim = state.dims[0][0] if state else cutoff_dim
        self.state = state if state else vacuum(n=2, cutoff_dim=cutoff_dim)
        self.a = annihilators(n=self.n_osc, cutoff_dim=self.cutoff_dim)
        self.P = osc_spintower_map(self.cutoff_dim)

        if self.state.type == "oper":
            self.spins = osc_spinblocks(self.state, map=self.P)
        else:
            self.spins = osc_spins(self.state, map=self.P)
        self.n_spins = len(self.spins)
        positions = [0]
        for i in range(1, self.n_spins):
            next_position = positions[-1] + 2.3
            positions.append(next_position)
        positions = np.array(positions)-positions[-1]/2
        if self.state.type == "oper":
            self.vspheres = [OperatorSphere(self.spins[i],\
                                            scene=scene,\
                                            pos=self.pos+vp.vector(positions[i], 0, 0)) \
                            for i in range(self.n_spins)]
        else:
            self.vspheres = [MajoranaSphere(self.spins[i],\
                                        scene=scene,\
                                        position=self.pos+vp.vector(positions[i], 0, 0),\
                                        radius=1,\
                                        show_phase=True,\
                                        show_norm=True) \
                            for i in range(self.n_spins)]

        self.show_plane = show_plane
        if self.show_plane:
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

            self.vexpected_pos = vp.sphere(pos=self.pos+self.plane_origin+vp.vector(qt.expect(self.Q[0], self.state).real, qt.expect(self.Q[1], self.state).real, 0),\
                                           color=vp.color.yellow, radius=0.1)

        self.paulis = {"x": second_quantize_operator(qt.sigmax(), self.a),\
                       "y": second_quantize_operator(qt.sigmay(), self.a),\
                       "z": second_quantize_operator(qt.sigmaz(), self.a)}
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
        
    def raise_spin(self, spin, replace=False):
        if replace:
            self.vacuum()
        new_state = (second_quantize_spin_state(spin, self.a)*self.state)
        new_state = new_state.unit() if new_state.norm() != 0 else new_state
        self.state = new_state
        self.refresh()

    def lower_spin(self, spin):
        new_state = (second_quantize_spin_state(spin, self.a).dag()*self.state)
        new_state = new_state.unit() if new_state.norm() != 0 else new_state
        self.state = new_state
        self.refresh()
    
    def random(self):
        new_state = qt.rand_ket(self.state.shape[0])
        new_state.dims = self.state.dims
        self.state = new_state
        self.refresh()

    def vacuum(self):
        self.state = vacuum(n=self.n_osc, cutoff_dim=self.cutoff_dim)
        self.refresh()

    def randomH(self):
        H = qt.rand_herm(self.state.shape[0])
        H.dims = [self.state.dims[0], self.state.dims[0]]
        return H

    def evolve(self, H, dt=0.05, T=2*np.pi):
        if H.dims[0] == self.state.dims[0]:
            U = (-1j*H*dt).expm()
        else:
            U = (-1j*second_quantize_operator(H, self.a)*dt).expm()
        for t in np.linspace(0, T, int(T/dt)):
            self.state = U*self.state
            self.refresh()
            vp.rate(50)

    def measure(self, direction):
        if self.show_plane and direction == 'q':
            probs = []
            indices = []
            for i in range(self.cutoff_dim):
                for j in range(self.cutoff_dim):
                    amp = self.state.overlap(self.Qstates[i][j])
                    probs.append(abs(amp)**2)
                    indices.append((i, j))
            probs = np.array(probs)
            choice = np.random.choice(list(range(len(probs))), p=abs(probs/sum(probs)))
            i, j = indices[choice]
            self.state = (self.Qstates[i][j]*self.Qstates[i][j].dag()*self.state).unit()
        else:        
            projectors = self.pauli_projectors[direction]
            eigenvalues = projectors.keys()
            P = [projectors[eig] for eig in eigenvalues]
            probs = np.array([qt.expect(P[i], self.state) for i in range(len(P))])
            choice = np.random.choice(list(range(len(probs))), p=abs(probs/sum(probs)))
            self.state = (P[choice]*self.state).unit()
        self.refresh()

    def set(self, state):
        self.state = state
        self.refresh()

    def refresh(self):
        if self.state.type == "oper":
            self.spins = osc_spinblocks(self.state, map=self.P)
            for i, vsphere in enumerate(self.vspheres):
                vsphere.set(self.spins[i])
        else:
            self.spins = osc_spins(self.state, map=self.P)
            for i, vsphere in enumerate(self.vspheres):
                vsphere.spin = self.spins[i]

        if self.show_plane:
            for i in range(self.cutoff_dim):
                for j in range(self.cutoff_dim):
                    amp = self.state.overlap(self.Qstates[i][j]) 
                    self.vpositions[i][j].axis = 2*vp.vector(amp.real, amp.imag, 0)
            self.vexpected_pos.pos = self.pos+self.plane_origin+vp.vector(qt.expect(self.Q[0], self.state).real, qt.expect(self.Q[1], self.state).real, 0)

    def destroy(self):
        for vsphere in self.vspheres:
            vsphere.destroy()
        if self.show_plane:
            self.vexpected_pos.visible = False
            self.vexpected_pos = None
            for i in range(self.cutoff_dim):
                for j in range(self.cutoff_dim):
                    self.vpositions[i][j].visible = False
            self.vpositions = None