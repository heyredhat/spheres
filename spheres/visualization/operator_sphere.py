import vpython as vp
import numpy as np
import qutip as qt

from .majorana_sphere import *
from ..stars import *

class OperatorSphere:
    """
    Visualization for density matrices and operators. Using the spherical tensor decomposition, the operator
    or density matrix is represented by a series of concentric spheres with their own constellations. The
    operator/density matrix is represented by a list of integer valued spin states. The norms of these states
    become the radii of the spheres, and the phases of the states become the colors. The spin-0 sector is
    represented by the label at the bottom. For hermitian matrices, the constellations all have antipodal symmetry.
    The lower spin states can be interpreted as the partial states in the permutation symmetric qubit representation.
    """
    def __init__(self, dm,\
                       scene=None,\
                       pos=vp.vector(0,0,0)):
        self.scene = scene if scene else vp.scene
        self.pos = pos

        super().__setattr__("dm", dm)
        self.j = (dm.shape[0]-1)/2
        self.T_basis = spherical_tensor_basis(self.j)

        self.spins = operator_spins(self.dm, T_basis=self.T_basis)
        self.radii = [spin.norm() for spin in self.spins[1:]]
        self.colors = [vp.color.hsv_to_rgb(vp.vector(phase_angle(spin)/(2*np.pi), 1, 1)) for spin in self.spins[1:]]
        self.vspin0 = vp.label(pos=pos-vp.vector(0,0.15+max(self.radii) if len(self.radii) > 0 else 0, 0), height=10, text="%.1f+i%.1f" % (self.spins[0][0][0][0].real, self.spins[0][0][0][0].imag))
        self.vspheres = [vp.sphere(pos=pos,\
                                   radius=self.radii[i],\
                                   color=self.colors[i],\
                                   opacity=0.2)
                            for i, spin in enumerate(self.spins[1:])]
        self.vstars = [[vp.sphere(pos=pos+self.radii[i]*vp.vector(*xyz),\
                                  radius=0.2*self.radii[i],\
                                  color=self.colors[i],\
                                  emissive=True)
                                    for xyz in spin_xyz(spin)]
                                        for i, spin in enumerate(self.spins[1:])]
        axis = vp.vector(qt.expect(qt.jmat(self.j, 'x'), self.dm),\
                         qt.expect(qt.jmat(self.j, 'y'), self.dm),\
                         qt.expect(qt.jmat(self.j, 'z'), self.dm)) if self.j != 0 and self.dm.isherm else vp.vector(0,0,0)
        self.vspin_axis = vp.arrow(pos=pos, axis=axis,\
                                   visible=False if self.j == 0 or axis == vp.vector(0,0,0) or self.dm.isherm == False else True,\
                                   color=vp.color.yellow)
        self.dragging = False
        self.refreshing = False
        self.scene.bind("mousedown", self.mousedown)
        self.scene.bind("mousemove", self.mousemove)
        self.scene.bind("mouseup", self.mouseup)

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == "dm":
            self.refresh()
    
    def mousedown(self):
        pick  = self.scene.mouse.pick 
        if pick in self.vspheres or pick in self.vstars or pick == self.vspin0:
            self.dragging = True           
    
    def mousemove(self):
        if self.dragging:
            self.pos = self.scene.mouse.pos
            if not self.refreshing:
                self.refresh()     

    def mouseup(self):
        self.dragging = False
        
    def refresh(self):
        self.refreshing = True
        self.spins = operator_spins(self.dm, T_basis=self.T_basis)
        self.radii = [spin.norm() for spin in self.spins[1:]]
        self.colors = [vp.color.hsv_to_rgb(vp.vector(phase_angle(spin)/(2*np.pi), 1, 1)) for spin in self.spins[1:]]
        self.vspin0.pos = self.pos-vp.vector(0,0.15+max(self.radii) if len(self.radii) > 0 else 0, 0)
        self.vspin0.text = "%.1f+i%.1f" % (self.spins[0][0][0][0].real, self.spins[0][0][0][0].imag)
        for i, spin in enumerate(self.spins[1:]):
            self.vspheres[i].pos = self.pos
            self.vspheres[i].radius = self.radii[i]
            self.vspheres[i].color = self.colors[i]
            for j, xyz in enumerate(spin_xyz(spin)):
                self.vstars[i][j].visible = False if np.linalg.norm(xyz) == 0 else True
                self.vstars[i][j].pos = self.pos + self.radii[i]*vp.vector(*xyz)
                self.vstars[i][j].color = self.colors[i]
        self.vspin_axis.pos = self.pos
        self.vspin_axis.axis = vp.vector(*spinj_xyz(self.dm)) if self.dm.isherm else vp.vector(0,0,0)
        self.vspin_axis.visible = False if self.j == 0 or self.vspin_axis.axis == vp.vector(0,0,0) or self.dm.norm() == 0 or self.dm.isherm == False else True
        self.refreshing = False

    def evolve(self, H, dt=0.05, T=2*np.pi):
        """
        Evolves the mixed state/operator, updating the visual in real time.

        Parameters
        ----------
            H : qt.Qobj
                Hamiltonian.

            dt : float
                Time step.

            T : float
                Time interval.
        """
        U = (-1j*H*dt).expm()
        for t in np.linspace(0, T, int(T/dt)):
            self.dm = U*self.dm*U.dag()
            vp.rate(50)

    def destroy(self):
        """
        Destroys the Operator sphere.
        """
        self.vspin0.visible = False
        del self.vspin0
        for i, vsphere in enumerate(self.vspheres):
            vsphere.visible = False
            del vsphere
            for j, vstar in enumerate(self.vstars[i]):
                vstar.visible = False
                del vstar
        self.vspin_axis.visible = False
        del self.vspin_axis
        self.scene.unbind("mousedown", self.mousedown)
        self.scene.unbind("mousemove", self.mousemove)
        self.scene.unbind("mouseup", self.mouseup)