"""
Visualization Tools
-------------------

"""

from spheres.stars import *
from spheres.utils import *

import vpython as vp
global_scene = None

def tangent_plane_rotation(phi, theta):
    """
    Constructs rotation into the tangent plane to the sphere 
    at the given point specified in spherical coordinates. Returns
    a 3x3 np.ndarray representing the linear transformation corresponding
    to the rotation.

    """
    normal = sph_xyz(np.array([1, phi, theta]))
    tangent = sph_xyz(np.array([1, phi, theta+np.pi/2]))
    return np.linalg.inv(\
                np.array([tangent,\
                          normalize(np.cross(tangent, normal)),\
                          normal]))

class MajoranaSphere:
    """
    `MajoranaSphere` provides a nice way to visualize (pure) spin-j states using
    vpython for graphics, whether in a jupyter notebook or in a standalone 
    environment.

    Create a `MajoranaSphere` by specifying a spin state. The radius of the sphere is
    determined by the j value of the spin. One can
    specify a vpython scene in which to place the MajoranaSphere (if none is provided,
    one is created by default) the position at which to place the sphere, 
    the color and opacity of the sphere, the colors of the stars, whether the stars 
    leave trails, whether the sphere and the stars
    are draggable by mouse, whether to show the expected rotation axis of the
    spin as a yellow arrow, whether to show reference cartesian axes in red, whether
    to show the complex phase of the spin as a green arrow hovering atop the sphere, 
    and whether to additionally visualize the spin as a wavefunction on the sphere:
    both as a coherent state wave function or in terms of the Majorana function. 
    One can specify the number of sample points to evaluate at, and the wavefunction
    amplitudes are visualized as arrows tangent to the sphere at these points.

    Attributes
    ----------
    spin : qt.Qobj
        The spin-j state represented. If this attribute is set, the visualization
        is automatically updated.
    j : float
        Its j value.
    xyz : np.ndarray
        Majorana points in cartesian coordinates.
    phase : complex
        Complex phase of the spin state.
    show_rotation_axis : bool
        Whether to show the expected rotation axis. If this attribute is set, the visualization
        is automatically updated. Can also use :py:meth:`toggle_rotation_axis`.
    show_phase : bool
        Whether to show the phase. If this attribute is set, the visualization
        is automatically updated. Can also use :py:meth:`toggle_phase`.
    show_axes : bool
        Whether to show reference cartesian axes. If this attribute is set, the visualization
        is automatically updated. Can also use :py:meth:`toggle_axes`.
    sphere_draggable : bool
        Whether one can drag the sphere with the mouse. If this attribute is set, the visualization
        is automatically updated. Can also use :py:meth:`toggle_sphere_draggable`.
    stars_draggable : bool
        Whether one can drag the individual stars with the mouse. If this attribute is set, the visualization
        is automatically updated. Can also use :py:meth:`toggle_stars_draggable`.
    phase_draggable : bool
        Whether one can drag the phase around with the mouse. If this attribute is set, the visualization
        is automatically updated. Can also use :py:meth:`toggle_phase_draggable`.
    make_trails : bool
        Whether the stars leave trails. If this attribute is set, the visualization
        is automatically updated. Can also use :py:meth:`toggle_trails`.
    show_wavefunction : bool or str
        Can specify "majorana", "coherent_state", or `None/False`. If this attribute is set, the visualization
        is automatically updated. Can also use :py:meth:`toggle_wavefunction`.

    """
    def __init__(self, spin,\
                       scene=None,\
                       position=vp.vector(0,0,0),\
                       sphere_color=vp.color.blue,\
                       sphere_opacity=0.3,\
                       star_colors="random",\
                       make_trails=True,\
                       sphere_draggable=True,\
                       stars_draggable=True,\
                       show_rotation_axis=True,\
                       show_phase=True,\
                       phase_draggable=True,\
                       show_axes=True,\
                       show_wavefunction=None,\
                       wavefunction_samples=15):
        global global_scene
        """
        Parameters
        ----------
        spin : qt.Qobj
            The spin state to visualize.
        scene : vp.canvas, optional
            `vpython` canvas in which to place the sphere. If none is provided,
            a new canvas is created.
        position : vp.vector, optional
            3D `vpython` vector coordinates at which to place the sphere.
        sphere_color : vp.vector, optional
            `vpython` vector representing the RGB color values for the sphere. 
        sphere_opacity : float, optional
            Specifies the opacity of the sphere. 
        star_colors : vp.vector or list or str, optional
            Specifies the color of the stars. If given a `vpython` vector representing
            RGB color values, all the stars are colored with that color. If given a list
            of such vectors, the stars are colored with those colors. If given "random",
            random colors are generated for each star. Note that if the stars are
            different colors, there is some performance overhead as a little extra work
            has to be done to keep track of which star is which: in other words, 
            :py:meth:`spin_xyz`  doesn't nail down the ordering of the stars and so they
            may "switch places." This is undetectable if the stars are all colored the same,
            but if they aren't, we use :py:meth:`fix_ordering` to try to keep continuity.
        make_trails : bool, optional
            If `True`, the stars leave (colored) trails behind them. Similarly to the above,
            we must use :py:meth:`fix_ordering` to keep continuity.
        sphere_draggable : bool, optional
            If `True`, one can click the sphere to (un)select it (it turns purple), and
            then drag it around the scene.
        stars_draggable : bool, optional
            If `True`, one can click the stars to un(select) them (they turn magenta),
            and then drag them around the surface of the sphere.
        show_rotation_axis : bool, optional
            If `True`, shows the expected spin axis in yellow.
        show_phase : bool, optional
            If `True`, shows the complex phase of the wavefunction as a green arrow
            within a black ring hovering above the sphere like a halo.
        phase_draggable : bool, optional
            If `True`, allows the complex phase to be dragged via the mouse.
        show_axes : bool, optional
            If `True`, shows reference cartesian axes in red.
        show_wavefunction : None or str, optional
            If "majorana", visualizes the normalized Majorana function on the sphere.
            If "coherent_state", visualizes the coherent state wavefunction on the sphere.
            The two are antipodal to each other. 
            The complex amplitudes appear as arrows tangent to the surface of the sphere
            at each sample point.
        wavefunction_samples : int, optional
            Number of sample points :math:`n` at which to evaluate the wavefunction. The total
            number of points is :math:`n^2`.

        """
        super().__setattr__("spin", spin)
        self.j = (self.spin.shape[0]-1)/2
        self.xyz = spin_xyz(self.spin)
        self.phase = phase(self.spin)
        
        if scene != None:
            self.scene = None
        else:
            if global_scene == None:
                global_scene = vp.canvas(background=vp.color.white,\
                                         align="center", 
                                         width=600, 
                                         height=600)
            self.scene = global_scene 
                        
        self.vsphere = vp.sphere(pos=position,\
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
        self.toggle_phase_draggable(phase_draggable)
        self.toggle_trails(make_trails)

        self.wavefunction_samples = wavefunction_samples
        self.toggle_wavefunction(show_wavefunction)

        self.evolving = False
        self.saved_make_trails = None
        self.refreshing = False
        self.snapshots = []

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
        elif name == "phase_draggable":
            self.toggle_phase_draggable(value)
        elif name == "make_trails":
            self.toggle_trails(value)
        elif name == "show_wavefunction":
            self.toggle_wavefunction(value)
        super().__setattr__(name, value)
        if name == "spin":
            self.clear_trails()
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
                                       thickness=0.03*self.j,\
                                       axis=vp.vector(0,1,0),\
                                       color=vp.color.black,\
                                       opacity=0.4)
            self.vphase = vp.arrow(pos=self.vphase_ring.pos,\
                                   axis=self.j*vp.vector(self.phase.real,0,self.phase.imag),\
                                   shaftwidth=self.j*0.05,\
                                   opacity=0.3,\
                                   color=vp.color.green)
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
                                   shaftwidth=0.05*self.j,\
                                   opacity=0.4) for axis in [vp.vector(1,0,0),\
                                                             vp.vector(0,1,0),\
                                                             vp.vector(0,0,1)]]
            self.vaxis_labels = [vp.text(text=label,\
                                         color=vp.color.red,\
                                         height=self.j*0.2,\
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
            if self.__exists__("clicks_bound") and not self.stars_draggable and not self.phase_draggable:
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
            if self.__exists__("clicks_bound") and not self.sphere_draggable and not self.phase_draggable:
                self.clicks_bound = None
                self.scene.unbind('click', self.mouseclick)
                self.scene.unbind('mousemove', self.mousemove)

    def toggle_phase_draggable(self, toggle=None):
        super().__setattr__("phase_draggable", \
            (True if not self.phase_draggable else False) if toggle == None else toggle)
        if self.phase_draggable:
            self.phase_dragging = False
            if not self.__exists__("clicks_bound"):
                self.clicks_bound = True 
                self.scene.bind('click', self.mouseclick)
                self.scene.bind('mousemove', self.mousemove)
        if not self.phase_draggable:
            self.phase_dragging = None
            if self.__exists__("clicks_bound") and not self.sphere_draggable and not self.stars_draggable:
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
        """
        Clears the star trails.
        """
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
            self.vwavefunction = []
            for i in range(self.wavefunction_samples):
                row = []
                for j in range(self.wavefunction_samples):
                    arrow = vp.arrow(pos=self.vsphere.pos+self.vsphere.radius*vp.vector(*sph_xyz([1, self.phi[i][j], self.theta[i][j]])),\
                                     opacity=0.4)
                    axis = self.j*0.5*(self.tangent_plane_rotations[i][j] @ np.array([amps[i][j].real, amps[i][j].imag,0]))
                    if np.isclose(np.linalg.norm(axis), 0):
                        arrow.visible = False
                    arrow.axis = vp.vector(*axis)
                    row.append(arrow)
                self.vwavefunction.append(row)

    def mouseclick(self):
        if self.phase_draggable:
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
                self.saved_make_trails = self.make_trails
                self.toggle_trails(False)
                for i, vstar in enumerate(self.vstars):
                    if i == self.star_dragging:
                        vstar.color = vp.color.magenta
                    else:
                        vstar.color = self.star_colors[i]
            else:
                for i, vstar in enumerate(self.vstars):
                    vstar.color = self.star_colors[i]
                if self.saved_make_trails != None:
                    self.toggle_trails(self.saved_make_trails)
                    self.saved_make_trails = None

    def mousemove(self):
        if self.show_phase:
            if self.phase_dragging:
                x, y, z = self.scene.mouse.pos.value
                x, z = normalize(np.array([x, z]))
                phase = x + 1j*z
                new_spin = phase*normalize_phase(self.spin)
                super().__setattr__("spin", new_spin)
                if not self.evolving and not self.refreshing:
                    self.refresh()
                return

        if self.stars_draggable and not self.evolving:
            if self.star_dragging != -1:
                xyz = normalize(np.array((self.scene.mouse.pos-self.vsphere.pos).value))
                self.xyz[self.star_dragging] = xyz
                new_spin = self.phase*xyz_spin(self.xyz)
                super().__setattr__("spin", new_spin)
                if not self.evolving and not self.refreshing:
                    self.refresh()
                return

        if self.sphere_draggable:
            if self.sphere_dragging:
                self.vsphere.pos = self.scene.mouse.pos
                if not self.evolving and not self.refreshing:
                    self.refresh()

    def snapshot(self):
        """
        Takes a snapshot of the current locations of the stars
        and, if applicable, the phase, rotation axis, and
        wavefunction amplitudes. In other words, clones them.
        Can take multiple snapshots, which are added to a list.
        """
        clone = {}
        clone["vstars"] = [star.clone() for star in self.vstars]
        if self.show_phase:
            clone["vphase"] = self.vphase.clone()
        if self.show_rotation_axis:
            clone["vrotation_axis"] = self.vrotation_axis.clone()
        if self.show_wavefunction:
            clone["vwavefunction"] = [[self.vwavefunction[i][j].clone()
                        for j in range(self.wavefunction_samples)]\
                            for i in range(self.wavefunction_samples)]
        self.snapshots.append(clone)

    def clear_snapshot(self):
        """
        Clears a snapshot. If there's more than one snapshot,
        we clear them first in, first out.
        """
        clone = self.snapshots.pop(0)
        for star in clone["vstars"]:
            star.visible = False
        if "vphase" in clone:
            clone["vphase"].visible = False
        if "vrotation_axis" in clone:
            clone["vrotation_axis"].visible = False
        if "vwavefunction" in clone:
            for i in range(self.wavefunction_samples):
                for j in range(self.wavefunction_samples):
                    clone["vwavefunction"][i][j].visible = False
        clone = {}

    def refresh(self):
        """
        Refreshes the visualization from the value of `self.spin`.
        """
        self.refreshing = True
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
                    axis = self.j*0.5*(self.tangent_plane_rotations[i][j] @ np.array([amps[i][j].real, amps[i][j].imag, 0]))
                    self.vwavefunction[i][j].visible = False if np.isclose(np.linalg.norm(axis),0) else True
                    self.vwavefunction[i][j].axis = vp.vector(*axis)
                    if self.sphere_dragging:
                        self.vwavefunction[i][j].pos = self.vsphere.pos+self.vsphere.radius*vp.vector(*sph_xyz([1, self.phi[i][j], self.theta[i][j]]))
        self.refreshing = False

    def evolve(self, H, dt=0.05, T=2*np.pi):
        """
        Visualizes the evolution of the spin state under the specified Hamiltonian.

        Parameters
        ----------
        H : qt.Qobj
            The Hamiltonian. 
        dt : float
            The time step for each sample of the evolution after which the 
            visualization is updated.
        T : int
            The total number of time steps to evolve for.
        """
        self.evolving = True
        U = (-1j*H*dt).expm()
        for t in np.linspace(0, T, int(T/dt)):
            super().__setattr__("spin", U*self.spin)
            self.refresh()
            vp.rate(50)
        self.evolving = False

    def destroy(self):
        """
        Destroys the MajoranaSphere.
        """
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
        while len(self.snapshots) > 0:
            self.clear_snapshot()

