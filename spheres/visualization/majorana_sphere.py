from ..stars import *
from ..utils import *
from .vp_object import *

import vpython as vp

def tangent_plane_rotation(theta, phi):
    """
    Constructs rotation into the tangent plane to the sphere 
    at the given point specified in spherical coordinates. 
    
    Parameters
    ----------
        theta : float

        phi : float

    Returns
    -------
        T : np.array
        A 3x3 matrix representing the linear transformation corresponding to the rotation.
    """
    normal = sph_xyz(np.array([theta, phi]))
    tangent = sph_xyz(np.array([theta+np.pi/2, phi]))
    return np.linalg.inv(np.array([tangent,\
                                   normalize(np.cross(tangent, normal)),\
                                   normal]))

class SphericalWavefunction:
    """
    Container for a 3D representation of a spin coherent wavefunction.
    """
    def __init__(self, spin, pos=vp.vector(0,0,0), radius=1, wavefunction_type="coherent", wavefunction_samples=15):
        """
        Parameters
        ----------
            spin : qt.Qobj
                Spin-j state.
            
            pos : vp.vector
                Position.

            radius : float
                Radius.

            wavefunction_type : str
                "coherent" or "majorana". The former evaluates the amplitude on a spin coherent state at sample
                points on the sphere. The latter evaluates the normalized Majorana function. The two should be
                antipodal to each other.

            wavefunction_samples : int
                Number of sample points.
        """
        self.spin = spin
        self.j = (self.spin.shape[0]-1)/2
        self.pos = pos
        self.radius = radius
        self.wavefunction_type = wavefunction_type
        self.wavefunction_samples = wavefunction_samples

        self.theta, self.phi = np.meshgrid(np.linspace(0, np.pi, self.wavefunction_samples),\
                                           np.linspace(0, 2*np.pi, self.wavefunction_samples))
        
        self.tangent_plane_rotations = [[tangent_plane_rotation(self.theta[i][j], self.phi[i][j])\
                                            for j in range(self.wavefunction_samples)]
                                                for i in range(self.wavefunction_samples)]

    def create_wavefunction(self):
        if self.wavefunction_type == "coherent":
            self.coherent_states = [[qt.spin_coherent(self.j, self.theta[i][j], self.phi[i][j])\
                                        for j in range(self.wavefunction_samples)]\
                                            for i in range(self.wavefunction_samples)]
            amps = [[(self.coherent_states[i][j].dag()*self.spin)[0][0][0]\
                            for j in range(self.wavefunction_samples)]\
                                for i in range(self.wavefunction_samples)]
        elif self.wavefunction_type == "majorana":
            self.poly = spin_poly(self.spin, spherical=True, normalized=True)
            amps = [[self.poly([self.theta[i][j], self.phi[i][j]])\
                            for j in range(self.wavefunction_samples)]\
                                for i in range(self.wavefunction_samples)]

        self.vwavefunction = []
        for i in range(self.wavefunction_samples):
            row = []
            for j in range(self.wavefunction_samples):
                arrow = vp.arrow(pos=self.pos+self.radius*vp.vector(*sph_xyz([self.theta[i][j], self.phi[i][j]])), opacity=0.4)
                axis = self.radius*0.5*(self.tangent_plane_rotations[i][j] @ np.array([amps[i][j].real, amps[i][j].imag,0]))
                if np.isclose(np.linalg.norm(axis), 0):
                    arrow.visible = False
                arrow.axis = vp.vector(*axis)
                row.append(arrow)
            self.vwavefunction.append(row)
        return flatten(self.vwavefunction)
    
    def refresh_wavefunction(self, update_position=True, update_amplitudes=True):
        if update_amplitudes:
            if self.wavefunction_type == "coherent":
                amps = [[(self.coherent_states[i][j].dag()*self.spin)[0][0][0]\
                                for j in range(self.wavefunction_samples)]\
                                    for i in range(self.wavefunction_samples)]
            elif self.wavefunction_type == "majorana":
                self.poly = spin_poly(self.spin, spherical=True, normalized=True)
                amps = [[self.poly(np.array([self.theta[i][j], self.phi[i][j]]))\
                                for j in range(self.wavefunction_samples)]\
                                    for i in range(self.wavefunction_samples)]                   
        for i in range(self.wavefunction_samples):
            for j in range(self.wavefunction_samples):
                if update_amplitudes:
                    axis = self.radius*0.5*(self.tangent_plane_rotations[i][j] @ np.array([amps[i][j].real, amps[i][j].imag, 0]))
                    self.vwavefunction[i][j].visible = False if np.isclose(np.linalg.norm(axis),0) else True
                    self.vwavefunction[i][j].axis = vp.vector(*axis)
                if update_position: 
                    self.vwavefunction[i][j].pos = self.pos+self.radius*vp.vector(*sph_xyz([self.theta[i][j], self.phi[i][j]]))

class MajoranaSphere(VObject):
    """
    `MajoranaSphere` provides a nice way to visualize (pure) spin-j states using
    vpython for graphics, whether in a jupyter notebook or in a standalone 
    environment.

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

    scene : vp.canvas
        Scene in which to place the Majorana sphere. Defaults to a global scene.

    show_rotation_axis : bool
        Whether to show the expected rotation axis. If this attribute is set, the visualization
        is automatically updated.

    show_phase : bool
        Whether to show the phase. If this attribute is set, the visualization
        is automatically updated.

    show_reference_axes : bool
        Whether to show reference cartesian axes. If this attribute is set, the visualization
        is automatically updated. 

    show_norm : bool
        Whether to show the norm of the state as a label. If this attribute is set, the visualization
        is automatically updated. 

    sphere_draggable : bool
        Whether one can drag the sphere with the mouse. If this attribute is set, the visualization
        is automatically updated.

    make_trails : bool
        Whether the stars leave trails. If this attribute is set, the visualization
        is automatically updated.

    show_wavefunction : bool 
        Whether to show spin coherent wavefunction. If this attribute is set, the visualization
        is automatically updated.

    wavefunction_type : str
        "majorana" or "coherent". The former evaluates the amplitude on a spin coherent state at sample
        points on the sphere. The latter evaluates the normalized Majorana function. The two should be
        antipodal to each other.If this attribute is set, the visualization is automatically updated.

    wavefunction_samples : int
        Number of sample points.
    """
    def __init__(self, 
                 spin,\
                 scene=None,\
                 pos=vp.vector(0,0,0),\
                 radius=None,
                 sphere_color=vp.color.blue,\
                 sphere_opacity=0.3,\
                 sphere_draggable=True,\
                 star_colors=None,\
                 make_trails=False,\
                 show_rotation_axis=True,\
                 show_phase=False,\
                 show_reference_axes=False,\
                 show_norm=False,\
                 show_wavefunction=False,\
                 wavefunction_type="coherent",\
                 wavefunction_samples=15):
        super().__init__(scene=scene)
        super().__setattr__("spin", spin)
        self.auto_refresh_attrs["spin"] = self.refresh_spin 

        self.j = (self.spin.shape[0]-1)/2
        self.xyz = spin_xyz(self.spin)
        self.phase = phase(self.spin)

        if self.j == 0:
            show_rotation_axis = False
            show_phase = True
            show_norm = True

        self.radius = self.j if not radius else radius  
        self.vsphere = vp.sphere(pos=pos,\
                                 radius=self.radius,\
                                 color=sphere_color,\
                                 opacity=sphere_opacity,\
                                 visible=False if self.j == 0 else True)
        self.vchildren.append(self.vsphere)

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
                                 visible=False if sum(xyz) == 0 else True,
                                 emissive=True) for i, xyz in enumerate(self.xyz)]
        self.vchildren.extend(self.vstars)

        self.add_toggle("show_rotation_axis", self.create_rotation_axis)
        self.add_toggle("show_reference_axes", self.create_reference_axes)
        self.add_toggle("show_phase", self.create_phase)
        self.add_toggle("show_norm", self.create_norm)
        self.add_toggle("show_wavefunction", self.create_wavefunction)

        self.auto_refresh_attrs["make_trails"] = self.refresh_trails

        self.sphere_draggable = sphere_draggable
        self.show_rotation_axis = show_rotation_axis
        self.show_reference_axes = show_reference_axes
        self.show_phase = show_phase
        self.show_norm = show_norm
        self.make_trails = make_trails
        self.wavefunction_samples = wavefunction_samples
        self.wavefunction_type = wavefunction_type
        self.show_wavefunction = show_wavefunction
        self.auto_refresh_attrs["wavefunction_type"] = self.change_wavefunction_type

        self.refreshments.extend([self.refresh_stars,\
                                  self.refresh_rotation_axis,\
                                  self.refresh_reference_axes,\
                                  self.refresh_phase,\
                                  self.refresh_norm,\
                                  self.refresh_wavefunction])

        self.mousedown_callbacks[self.vsphere] = self.start_sphere_dragging
        self.mousemove_callbacks.append(self.sphere_drag)
        self.mouseup_callbacks.append(self.stop_sphere_dragging)

        self.sphere_dragging = False
        self.evolving = False
        self.snapshots = []

    ############################################################

    def create_rotation_axis(self):
        axis = spinj_xyz(self.spin)
        self.vrotation_axis = vp.arrow(pos=self.vsphere.pos,\
                                       color=vp.color.yellow,\
                                       axis=vp.vector(*axis),\
                                       visible=not np.isclose(np.linalg.norm(axis), 0))
        return [self.vrotation_axis]

    def refresh_rotation_axis(self):
        if self.toggles["show_rotation_axis"]["exists"]:
            self.vrotation_axis.pos = self.vsphere.pos
            axis = spinj_xyz(self.spin)
            self.vrotation_axis.axis = vp.vector(*axis)
            self.vrotation_axis.visible = not np.isclose(np.linalg.norm(axis), 0)

    ############################################################

    def create_reference_axes(self):
        self.vaxes = [vp.arrow(pos=self.vsphere.pos,\
                               axis=1.5*self.radius*axis,\
                               color=vp.color.red,\
                               shaftwidth=0.05*self.radius,\
                               opacity=0.4) for axis in [vp.vector(1,0,0),\
                                                         vp.vector(0,1,0),\
                                                         vp.vector(0,0,1)]]
        self.vaxis_labels = [vp.text(text=label,\
                                     color=vp.color.red,\
                                     height=self.radius*0.2,\
                                     pos=self.vsphere.pos+self.vaxes[i].axis)\
                                            for i, label in enumerate(["X", "Y", "Z"])]
        return self.vaxes + self.vaxis_labels

    def refresh_reference_axes(self):
        if self.toggles["show_reference_axes"]["exists"]:
            for i, axis in enumerate(self.vaxes):
                axis.pos = self.vsphere.pos
                self.vaxis_labels[i].pos = self.vsphere.pos + axis.axis
            
    ############################################################

    def create_phase(self):
        self.vphase_ring = vp.ring(pos=self.vsphere.pos+vp.vector(0,self.radius+0.1 if self.j!=0 else 0,0),\
                                   radius=(self.radius if self.radius != 0 else 1),\
                                   thickness=0.03*(self.radius if self.radius != 0 else 1),\
                                   axis=vp.vector(0,1,0) if self.j != 0 else vp.vector(0,0,1),\
                                   color=vp.color.black,\
                                   opacity=0.4)
        self.vphase = vp.arrow(pos=self.vphase_ring.pos,\
                                   axis=self.radius*vp.vector(self.phase.real,0,self.phase.imag) if self.j != 0 else self.radius*vp.vector(self.phase.real,self.phase.imag,0),\
                                   shaftwidth=(self.radius if self.radius != 0 else 1)*0.05,\
                                   opacity=0.3,\
                                   color=vp.color.green)
        return [self.vphase_ring, self.vphase]

    def refresh_phase(self):
        if self.show_phase:
            self.vphase_ring.pos = self.vsphere.pos + vp.vector(0, self.radius+0.1 if self.j!=0 else 0, 0)
            self.vphase_ring.radius = self.radius
            self.vphase.pos = self.vphase_ring.pos
            self.vphase.axis = self.radius*vp.vector(self.phase.real, 0, self.phase.imag) if self.j != 0 else vp.vector(self.phase.real, self.phase.imag, 0)

    ############################################################

    def create_norm(self):
        self.vnorm = vp.label(pos=self.vsphere.pos-vp.vector(0, 0.25+self.radius if self.radius !=0 else 1.25, 0),\
                              text='%.3f' % self.spin.norm())
        return [self.vnorm]

    def refresh_norm(self):
        if self.show_norm:
            self.vnorm.pos = self.vsphere.pos-vp.vector(0, 0.25+self.radius if self.radius !=0 else 1.25, 0)
            self.vnorm.text = '%.3f' % self.spin.norm()

    ############################################################

    def create_wavefunction(self):
        self.vwavefunction = SphericalWavefunction(self.spin, pos=self.vsphere.pos, radius=self.radius, wavefunction_type=self.wavefunction_type, wavefunction_samples=self.wavefunction_samples)
        return self.vwavefunction.create_wavefunction()

    def refresh_wavefunction(self):
        if self.show_wavefunction:
            self.vwavefunction.pos = self.vsphere.pos
            self.vwavefunction.spin = self.spin
            self.vwavefunction.refresh_wavefunction(update_position=self.sphere_dragging, update_amplitudes=self.evolving)

    def change_wavefunction_type(self):
        if self.show_wavefunction:
            self.show_wavefunction = False
            self.show_wavefunction = True

    ############################################################

    def clear_trails(self):
        """
        Clear star trails.
        """
        for star in self.vstars:
            star.clear_trail()

    def refresh_trails(self):
        self.clear_trails()
        if self.make_trails:
            self.fix_stars = True
        for star in self.vstars:
            star.make_trail = self.make_trails

    ############################################################

    def refresh_spin(self):
        self.xyz = spin_xyz(self.spin) if not self.fix_stars else fix_stars(self.xyz, spin_xyz(self.spin))
        self.phase = phase(self.spin)
        if not self.evolving:
            self.clear_trails()
        self.refresh()

    def refresh_stars(self):
        for i, xyz in enumerate(self.xyz):
            self.vstars[i].visible = False if sum(xyz) == 0 else True
            self.vstars[i].pos = self.vsphere.pos + self.radius*vp.vector(*xyz)

    ############################################################

    def start_sphere_dragging(self):
        if self.sphere_draggable:
            self.sphere_dragging = True

    def sphere_drag(self):
        if self.sphere_dragging:
            self.vsphere.pos = self.scene.mouse.pos
            if not self.evolving and not self.refreshing:
                self.refresh()
            self.clear_trails()
    
    def stop_sphere_dragging(self):
        self.sphere_dragging = False

    ############################################################

    def snapshot(self):
        """
        Takes a snaphot of the stars, phase, and rotation axis. In other words, makes a copy of them.
        """
        clone = {}
        clone["vstars"] = [star.clone() for star in self.vstars]
        if self.show_phase:
            clone["vphase"] = self.vphase.clone()
        if self.show_rotation_axis:
            clone["vrotation_axis"] = self.vrotation_axis.clone()
        self.snapshots.append(clone)

    def clear_snapshot(self):
        """
        Clears the last snapshot taken.
        """
        clone = self.snapshots.pop(0)
        for star in clone["vstars"]:
            star.visible = False
        if "vphase" in clone:
            clone["vphase"].visible = False
        if "vrotation_axis" in clone:
            clone["vrotation_axis"].visible = False
        clone = {}

    ############################################################

    def evolve(self, H, dt=0.05, T=2*np.pi):
        """
        Evolves the state, updating the visual in real time.

        Parameters
        ----------
            H : qt.Qobj
                Hamiltonian.

            dt : float
                Time step.

            T : float
                Time interval.
        """
        self.evolving = True
        U = (-1j*H*dt).expm()
        for t in np.linspace(0, T, int(T/dt)):
            self.spin = U*self.spin
            vp.rate(50)
        self.evolving = False

    ############################################################

    def destroy(self):
        """
        Destroys the Majorana sphere.
        """
        super().destroy()
        while len(self.snapshots) > 0:
            self.clear_snapshot()