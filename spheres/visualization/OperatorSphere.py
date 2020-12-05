
from spheres.visualization import *

import vpython as vp
import numpy as np
import qutip as qt

class OperatorSphere:
    def __init__(self, dm, pos=vp.vector(0,0,0), scene=None):
        if scene != None:
            self.scene = scene
        else:
            if spheres.visualization.global_scene == None:
                spheres.visualization.global_scene = vp.canvas(background=vp.color.white,\
                                         align="center", 
                                         width=800, 
                                         height=600)
            self.scene = spheres.visualization.global_scene 
        self.pos = pos

        self.dm = dm
        self.dm.dims = [[self.dm.shape[0]], [self.dm.shape[0]]]
        self.j = (dm.shape[0]-1)/2
        self.T_basis = spherical_tensor_basis(self.j)

        self.spins = operator_spins(self.dm, T_basis=self.T_basis)
        self.radii = [spin.norm() for spin in self.spins[1:]]
        self.colors = [vp.color.hsv_to_rgb(vp.vector(phase_angle(spin)/(2*np.pi), 1, 1)) for spin in self.spins[1:]]
        self.vspin0 = vp.label(pos=pos-vp.vector(0,0.15+max(self.radii), 0), height=10, text="%.1f+i%.1f" % (self.spins[0][0][0][0].real, self.spins[0][0][0][0].imag))
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
        self.vspin_axis = vp.arrow(pos=pos, axis=vp.vector(qt.expect(qt.jmat(self.j, 'x'), self.dm),\
                                                          qt.expect(qt.jmat(self.j, 'y'), self.dm),\
                                                          qt.expect(qt.jmat(self.j, 'z'), self.dm)),\
                                    color=vp.color.yellow)
        self.dragging = False
        self.refreshing = False
        self.scene.bind('click', self.mouseclick)
        self.scene.bind('mousemove', self.mousemove)

    def set(self, dm):
        self.dm = dm
        self.dm.dims = [[self.dm.shape[0]], [self.dm.shape[0]]]
        self.refresh()
    
    def mouseclick(self):
        pick  = self.scene.mouse.pick 
        if pick in self.vspheres or pick in self.vstars or pick == self.vspin0:
            self.dragging = True if not self.dragging else False
        else:
            self.dragging = False
    
    def mousemove(self, event):
        if self.dragging:
            self.pos = self.scene.mouse.pos
            if not self.refreshing:
                self.refresh()     
        
    def refresh(self):
        self.refreshing = True
        self.spins = operator_spins(self.dm, T_basis=self.T_basis)
        self.radii = [spin.norm() for spin in self.spins[1:]]
        self.colors = [vp.color.hsv_to_rgb(vp.vector(phase_angle(spin)/(2*np.pi), 1, 1)) for spin in self.spins[1:]]
        self.vspin0.pos = self.pos-vp.vector(0,0.15+max(self.radii), 0)
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
        self.vspin_axis.axis = vp.vector(qt.expect(qt.jmat(self.j, 'x'), self.dm),\
                                                   qt.expect(qt.jmat(self.j, 'y'), self.dm),\
                                                   qt.expect(qt.jmat(self.j, 'z'), self.dm))
        self.refreshing = False

    def evolve(self, H, dt=0.05, T=2*np.pi):
        U = (-1j*H*dt).expm()
        for t in np.linspace(0, T, int(T/dt)):
            self.dm = U*self.dm*U.dag()
            self.refresh()
            vp.rate(50)

    def cmp_spin(self):
        for spin in self.spins[1:]:
            j = (spin.shape[0]-1)/2


    def destroy(self):
        self.vspin0.visible = False
        for i, vsphere in enumerate(self.vspheres):
            vsphere.visible = False
            for j, vstar in enumerate(self.vstars[i]):
                vstar.visible = False
        self.vspin0 = None
        self.vspheres = None
        self.vstars = None

        self.scene.unbind('click', self.mouseclick)
        self.scene.unbind('mousemove', self.mousemove)