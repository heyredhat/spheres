"""
Visualization Tools
-------------------

"""

from spheres.stars import *
from spheres.utils import *

import vpython as vp

class MajoranaSphere:
    def __init__(self, spin,\
                       scene=None,\
                       position=[0,0,0],\
                       sphere_color=vp.color.blue,\
                       sphere_opacity=0.3,\
                       star_colors="random",\
                       make_trails=False,\
                       sphere_draggable=True,\
                       stars_draggable=True,\
                       show_rotation_axis=True,\
                       show_phase=True,\
                       #phase_draggable=True,\
                       show_axes=True,\
                       show_wavefunction=None,\
                       wavefunction_samples=15):
        super().__setattr__("spin", spin)
        self.j = (self.spin.shape[0]-1)/2
        self.xyz = spin_xyz(self.spin)
        self.phase = phase(self.spin)
        
        self.scene = scene if scene != None else \
                        vp.canvas(background=vp.color.white,\
                                    align="center", 
                                    width=600, 
                                    height=600)
        self.vsphere = vp.sphere(pos=vp.vector(*position),\
                                 radius=self.j,\
                                 color=sphere_color,\
                                 opacity=sphere_opacity)

        self.fix_stars = True if make_trails == True else False
        if star_colors == None:
            self.star_colors = [vp.color.white]*len(self.xyz)
        elif type(star_colors) == vp.vector:
            self.star_colors = [star_colors]*len(self.xyz)
        elif type(star_colors) == list:
            self.star_colors = star_colors
            self.fix_stars = True
        elif star_colors == "random":
            self.star_colors = [vp.vector(*np.random.random(3)) for i in range(len(self.xyz))]
            self.fix_stars = True

        self.vstars = [vp.sphere(pos=self.vsphere.pos+self.vsphere.radius*vp.vector(*xyz),\
                                 color=self.star_colors[i],\
                                 radius=0.2*self.vsphere.radius,\
                                 make_trail=True,
                                 emissive=True) for i, xyz in enumerate(self.xyz)]

        self.toggle_rotation_axis(show_rotation_axis)
        self.toggle_phase(show_phase)
        self.toggle_axes(show_axes)
        self.toggle_sphere_draggable(sphere_draggable)
        self.toggle_stars_draggable(stars_draggable)
        self.toggle_trails(make_trails)

        self.wavefunction_samples = wavefunction_samples
        self.toggle_wavefunction(show_wavefunction)

        self.evolving = False
        self.saved_make_trails = None
        self.selected = None

    def __exists__(self, attr):
        return hasattr(self, attr) and getattr(self, attr) != None

    def __setattr__(self, name, value):
        if name == "show_rotation_axis":
            self.toggle_rotation_axis(value)
        elif name == "show_phase":
            self.toggle_phase(value)
        elif name == "show_axes":
            self.toggle_axes(value)
        elif name == "sphere_draggable":
            self.toggle_sphere_draggable(value)
        elif name == "stars_draggable":
            self.toggle_stars_draggable(value)
        elif name == "make_trails":
            self.toggle_trails(value)
        elif name == "show_wavefunction":
            self.toggle_wavefunction(value)
        super().__setattr__(name, value)
        if name == "spin":
            self.refresh()

    def toggle_rotation_axis(self, toggle=None):
        super().__setattr__("show_rotation_axis", \
            (True if not self.show_rotation_axis else False) if toggle == None else toggle)
        if self.show_rotation_axis and not self.__exists__("vrotation_axis"):
            self.vrotation_axis = vp.arrow(pos=self.vsphere.pos,\
                                           color=vp.color.yellow,\
                                           axis=vp.vector(*sum([0.5*xyz for xyz in self.xyz])))
        if not self.show_rotation_axis and self.__exists__("vrotation_axis"):
            self.vrotation_axis.visible = False
            self.vrotation_axis = None    

    def toggle_phase(self, toggle=None):
        super().__setattr__("show_phase", \
            (True if not self.show_phase else False) if toggle == None else toggle)
        if self.show_phase and not self.__exists__("vphase"):
            self.vphase_ring = vp.ring(pos=self.vsphere.pos+vp.vector(0,self.j+0.1,0),\
                                       radius=self.j,\
                                       thickness=0.02,\
                                       axis=vp.vector(0,1,0),\
                                       color=vp.color.black,\
                                       opacity=0.4)
            self.vphase = vp.arrow(pos=self.vphase_ring.pos,\
                                   axis=self.j*vp.vector(self.phase.real,0,self.phase.imag),\
                                   shaftwidth=0.05,\
                                   opacity=0.3,\
                                   color=vp.color.green)
            self.phase_dragging = False
        if not self.show_phase and self.__exists__("vphase"):
            self.vphase_ring.visible = False
            self.vphase.visible = False
            self.vphase_ring = None
            self.vphase = None

    def toggle_axes(self, toggle=None):
        super().__setattr__("show_axes", \
            (True if not self.show_axes else False) if toggle == None else toggle)
        if self.show_axes and not self.__exists__("vaxes"):
            self.vaxes = [vp.arrow(pos=self.vsphere.pos,\
                                   axis=1.5*self.j*axis,\
                                   color=vp.color.red,\
                                   shaftwidth=0.01,\
                                   opacity=0.4) for axis in [vp.vector(1,0,0),\
                                                             vp.vector(0,1,0),\
                                                             vp.vector(0,0,1)]]
            self.vaxis_labels = [vp.text(text=label,\
                                         color=vp.color.red,\
                                         height=0.1,\
                                         pos=self.vsphere.pos+self.vaxes[i].axis)\
                                            for i, label in enumerate(["X", "Y", "Z"])]
        if not self.show_axes and self.__exists__("vaxes"):
            for axis in self.vaxes:
                axis.visible = False
            for label in self.vaxis_labels:
                label.visible = False
            self.vaxes = None
            self.vaxis_labels = None

    def toggle_sphere_draggable(self, toggle=None):
        super().__setattr__("sphere_draggable", \
            (True if not self.sphere_draggable else False) if toggle == None else toggle)
        if self.sphere_draggable:
            self.sphere_dragging = False
            if not self.__exists__("clicks_bound"):
                self.clicks_bound = True 
                self.scene.bind('click', self.mouseclick)
                self.scene.bind('mousemove', self.mousemove)
        if not self.sphere_draggable:
            self.sphere_dragging = None
            self.vsphere.color = vp.color.blue
            if self.__exists__("clicks_bound") and not self.stars_draggable:
                self.clicks_bound = None
                self.scene.unbind('click', self.mouseclick)
                self.scene.unbind('mousemove', self.mousemove)

    def toggle_stars_draggable(self, toggle=None):
        super().__setattr__("stars_draggable", \
            (True if not self.stars_draggable else False) if toggle == None else toggle)
        if self.stars_draggable:
            self.star_dragging = -1
            if not self.__exists__("clicks_bound"):
                self.clicks_bound = True 
                self.scene.bind('click', self.mouseclick)
                self.scene.bind('mousemove', self.mousemove)
        if not self.stars_draggable:
            self.star_dragging = -1
            for i, star in enumerate(self.vstars):
                star.color = self.star_colors[i]
            if self.__exists__("clicks_bound") and not self.sphere_draggable:
                self.clicks_bound = None
                self.scene.unbind('click', self.mouseclick)
                self.scene.unbind('mousemove', self.mousemove)

    def toggle_trails(self, toggle=None):
        super().__setattr__("make_trails", \
            (True if not self.make_trails else False) if toggle == None else toggle)
        for star in self.vstars:
            star.clear_trail()
            star.make_trail = self.make_trails

    def clear_trails(self):
        for star in self.vstars:
            star.clear_trail()

    def toggle_wavefunction(self, toggle=None):
        if toggle != None and self.__exists__("show_wavefunction"):
            self.toggle_wavefunction()
        super().__setattr__("show_wavefunction", toggle)
        if self.show_wavefunction == None or self.show_wavefunction == False:
            if self.__exists__("vwavefunction"):
                for i in range(self.wavefunction_samples):
                    for j in range(self.wavefunction_samples):
                        self.vwavefunction[i][j].visible = False
                self.vwavefunction = None
        else:
            self.phi, self.theta = \
                np.meshgrid(
                    np.linspace(0, 2*np.pi, self.wavefunction_samples),\
                    np.linspace(0, np.pi, self.wavefunction_samples))
            self.tangent_plane_rotations = \
                    [[tangent_plane_rotation(self.phi[i][j], self.theta[i][j])\
                        for j in range(self.wavefunction_samples)]
                            for i in range(self.wavefunction_samples)]
            if self.show_wavefunction == "coherent_state":
                self.coherent_states = \
                    [[qt.spin_coherent(self.j, self.theta[i][j], self.phi[i][j])\
                            for j in range(self.wavefunction_samples)]\
                                for i in range(self.wavefunction_samples)]
                amps = [[(self.coherent_states[i][j].dag()*self.spin)[0][0][0]\
                                for j in range(self.wavefunction_samples)]\
                                    for i in range(self.wavefunction_samples)]
            elif self.show_wavefunction == "majorana":
                self.poly = spin_poly(self.spin, spherical=True, normalized=True)
                amps = [[self.poly(np.array([1, self.phi[i][j], self.theta[i][j]]))\
                                for j in range(self.wavefunction_samples)]\
                                    for i in range(self.wavefunction_samples)]
            self.vwavefunction = [[\
                            vp.arrow(pos=self.vsphere.pos+self.vsphere.radius*vp.vector(*sph_xyz([1, self.phi[i][j], self.theta[i][j]])),\
                                     axis=self.j*0.5*vp.vector(*(self.tangent_plane_rotations[i][j] @ \
                                                np.array([amps[i][j].real, amps[i][j].imag,0]))),\
                                     opacity=0.4)
                                        for j in range(self.wavefunction_samples)]\
                                            for i in range(self.wavefunction_samples)]
    def mouseclick(self):
        if self.show_phase:
            if self.scene.mouse.pick == self.vphase:
                self.phase_dragging = True if not self.phase_dragging else False
            else:
                self.phase_dragging = False
            if self.phase_dragging:
                self.vphase.color = vp.color.magenta
            else:
                self.vphase.color = vp.color.green

        if self.sphere_draggable:
            if self.scene.mouse.pick == self.vsphere:
                self.sphere_dragging = True if not self.sphere_dragging else False
            else:
                self.sphere_dragging = False

            if self.sphere_dragging:
                self.vsphere.color = vp.color.purple
                self.saved_make_trails = self.make_trails
                self.toggle_trails(False)
            else:
                self.vsphere.color = vp.color.blue
                if self.saved_make_trails != None:
                    self.toggle_trails(self.saved_make_trails)
                    self.saved_make_trails = None

        if self.stars_draggable and not self.evolving:
            pick = self.scene.mouse.pick 
            if pick in self.vstars:
                index = self.vstars.index(pick)
                self.star_dragging = index if self.star_dragging != index else -1
            else:
                self.star_dragging = -1

            if self.star_dragging != -1:
                for i, vstar in enumerate(self.vstars):
                    if i == self.star_dragging:
                        vstar.color = vp.color.magenta
                    else:
                        vstar.color = self.star_colors[i]
            else:
                for i, vstar in enumerate(self.vstars):
                    vstar.color = self.star_colors[i]

    def mousemove(self):
        if self.show_phase:
            if self.phase_dragging:
                x, y, z = self.scene.mouse.pos.value
                x, z = normalize(np.array([x, z]))
                phase = x + 1j*z
                new_spin = phase*normalize_phase(self.spin)
                super().__setattr__("spin", new_spin)
                if not self.evolving:
                    self.refresh()
                return

        if self.stars_draggable and not self.evolving:
            if self.star_dragging != -1:
                xyz = normalize(np.array((self.scene.mouse.pos-self.vsphere.pos).value))
                self.xyz[self.star_dragging] = xyz
                new_spin = self.phase*xyz_spin(self.xyz)
                super().__setattr__("spin", new_spin)
                if not self.evolving:
                    self.refresh()
                return

        if self.sphere_draggable:
            if self.sphere_dragging:
                self.vsphere.pos = self.scene.mouse.pos
                if not self.evolving:
                    self.refresh()

    def refresh(self):
        self.xyz = spin_xyz(self.spin) if not self.fix_stars else fix_stars(self.xyz, spin_xyz(self.spin))
        self.phase = phase(self.spin)

        for i, xyz in enumerate(self.xyz):
            self.vstars[i].pos = self.vsphere.pos + self.vsphere.radius*vp.vector(*xyz)

        if self.show_rotation_axis:
            self.vrotation_axis.pos = self.vsphere.pos
            self.vrotation_axis.axis = vp.vector(*sum([0.5*xyz for xyz in self.xyz]))

        if self.show_phase:
            self.vphase_ring.pos = self.vsphere.pos + vp.vector(0, self.j+0.1, 0)
            self.vphase.pos = self.vphase_ring.pos
            self.vphase.axis = self.j*vp.vector(self.phase.real, 0, self.phase.imag)

        if self.show_axes:
            for i, axis in enumerate(self.vaxes):
                axis.pos = self.vsphere.pos
                self.vaxis_labels[i].pos = self.vsphere.pos + axis.axis

        if self.show_wavefunction:
            if self.show_wavefunction == "coherent_state":
                amps = [[(self.coherent_states[i][j].dag()*self.spin)[0][0][0]\
                                for j in range(self.wavefunction_samples)]\
                                    for i in range(self.wavefunction_samples)]
            elif self.show_wavefunction == "majorana":
                self.poly = spin_poly(self.spin, spherical=True, normalized=True)
                amps = [[self.poly(np.array([1, self.phi[i][j], self.theta[i][j]]))\
                                for j in range(self.wavefunction_samples)]\
                                    for i in range(self.wavefunction_samples)]                   
            for i in range(self.wavefunction_samples):
                for j in range(self.wavefunction_samples):
                    self.vwavefunction[i][j].axis = \
                        self.j*0.5*vp.vector(*(self.tangent_plane_rotations[i][j] @ \
                                    np.array([amps[i][j].real, amps[i][j].imag, 0])))
                    if self.sphere_dragging:
                        self.vwavefunction[i][j].pos = self.vsphere.pos+self.vsphere.radius*vp.vector(*sph_xyz([1, self.phi[i][j], self.theta[i][j]]))

    def evolve(self, H, dt=0.01, T=1000):
        self.evolving = True
        U = (-1j*H*dt).expm()
        for t in range(T):
            self.spin = U*self.spin
            self.refresh()
            vp.rate(50)
        self.evolving = False

    def destroy(self):
        self.vsphere.visible = False
        self.vsphere = None
        for star in self.vstars:
            star.clear_trail()
            star.visible = False
        self.vstars = []
        if self.show_rotation_axis:
            self.vrotation_axis.visible = False
            self.vrotation_axis = None
        if self.show_phase:
            self.vphase_ring.visible = False
            self.vphase_ring = None
            self.vphase.visible = False
            self.vphase = None
        if self.show_axes:
            for i, axis in enumerate(self.vaxes):
                axis.visible = False
                self.vaxis_labels[i].visible = False
            self.vaxes = None
            self.vaxis_labels = None
        if self.show_wavefunction:
            for i in range(self.wavefunction_samples):
                for j in range(self.wavefunction_samples):
                    self.vwavefunction[i][j].visible = False
            self.vwavefunction = None


