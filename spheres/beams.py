"""
Majorana formalism for structured Gaussian beams.
"""

import numpy as np
from numpy import sqrt, pi, exp, abs, angle
from math import factorial
from scipy.special import eval_genlaguerre

from .stars.pure import *

def laguerre_gauss_mode(N, l, coordinates="cartesian"):
    r"""
    Returns a function evaluating a Laguerre-Gauss mode, which may take cartesian/cylindrical coordinates or vectors thereof.

    .. math::

        LG(r, \phi, z) = \frac{i^{|l|-N}}{w}\sqrt{\frac{2^{|l|+1}[\frac{N-l}{2}]!}{\pi[\frac{N+|l|}{2}]!}}e^{-\frac{r^2}{w^2}}(\frac{r}{w})^{|l|}e^{il\phi}L_{\frac{N-|l|}{2}}^{|l|}(\frac{2r^2}{w^2})

    Where :math:`w=\sqrt{1+(\frac{z}{\pi})^2}` and :math:`L_{a}^{b}` is a generalized Laguerre polynomial.

    Parameters
    ----------
        N : int
            An integer specifying the Laguerre-Gauss mode (N, l).
        l : int
            An integer specifying the Laguerre-Gauss mode (N, l).
        coodinates : str
            Whether to return a function of "cartesian" or "cylindrical" coordinates.

    Returns
    -------
        lg : func
            (Vectorized) function of cartesian or cylindrical coordinates.
    """
    def mode(r, phi, z): # cylindrical coordinates
        w0 = 1 # waist radius
        n = 1 # index of refraction
        lmbda = 1 # wavelength
        zR = (pi*n*(w0**2))/lmbda # rayleigh range
        w = w0*sqrt(1+(z/zR)**2) # spot size parameter
        return \
            ((1j**(abs(l)-N))/w)*\
            sqrt(((2**(abs(l)+1))*factorial((N-abs(l))/2))/\
                    (pi*factorial((N+abs(l))/2)))*\
            exp(-(r**2)/w**2)*\
            ((r/w)**abs(l))*\
            exp(1j*l*phi)*\
            eval_genlaguerre((N-abs(l))/2, abs(l), (2*r**2)/w**2)
    if coordinates == "cylindrical":
        return mode
    elif coordinates == "cartesian":
        def cartesian(x, y, z):
            c = x+1j*y
            r, phi = abs(c), angle(c)
            return mode(r, phi, z)
        return cartesian

def spin_beam(spin, coordinates="cartesian"):
    r"""
    Converts a spin state into a structured Gaussian beam, the latter being function 
    of cartesian or cylindrical coordinates, expressing the intensity and phase of the classical light beam
    in the paraxial approximation. A spin :math:`\mid j, m \rangle` state is identified with LG mode (2j, 2m).

    Parameters
    ----------
        spin : qt.Qobj
            Spin-j state

        coordinates : str
            "cartesian" or "cylindrical

    Returns
    -------
        sgb : func
            (Vectorized) function of cartesian or cylindrical coordinates.
    """
    j = (spin.shape[0]-1)/2
    v = components(spin)
    lg_basis = [laguerre_gauss_mode(int(2*j), int(2*m), coordinates=coordinates) for m in np.arange(-j, j+1)]
    if coordinates == "cartesian":
        def beam(x, y, z):
            return sum([v[int(m+j)]*lg_basis[int(m+j)](x, y, z) for m in np.arange(-j, j+1)])
        return beam
    elif coordinates == "cylindrical":
        def beam(r, phi, z):
            return sum([v[int(m+j)]*lg_basis[int(m+j)](r, phi, z) for m in np.arange(-j, j+1)])
        return beam

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from colorsys import hls_to_rgb

def colorize(z):
    """
    Converts complex values into colors: hue represents phase and brightness magnitude.
    
    Adapted from https://stackoverflow.com/questions/17044052/mathplotlib-imshow-complex-2d-array.

    Parameters
    ----------
        z : np.array
            Complex values.

    Returns
    -------
        c : np.array
            Color values.
    """
    n,m = z.shape
    c = np.zeros((n,m,3))
    c[np.isinf(z)] = (1.0, 1.0, 1.0)
    c[np.isnan(z)] = (0.5, 0.5, 0.5)
    idx = ~(np.isinf(z) + np.isnan(z))
    A = (np.angle(z[idx]) + np.pi) / (2*np.pi)
    A = (A + 0.5) % 1.0
    B = 1.0 - 1.0/(1.0+abs(z[idx]))
    c[idx] = [hls_to_rgb(a, b, 1) for a,b in zip(A,B)]
    return c

def viz_beam(beam, size=3.5, n_samples=200):
    """
    Visualizes a structured Gaussian beam with matplotlib.

    Parameters
    ----------
        beam : func
            Beam function.

        size : float
            Size of plot.
        
        n_samples : int
            Number of samples of the beam function.
    """
    x = np.linspace(-size, size, n_samples)
    y = np.linspace(-size, size, n_samples)
    X, Y = np.meshgrid(x, y)
    plt.imshow(colorize(beam(X, Y, np.zeros(X.shape))), interpolation="none", extent=(-size,size,-size,size))
    plt.show()

def viz_spin_beam(spin, size=3.5, n_samples=200):
    """
    Visualizes a spin state and its corresponding structured Gaussian beam side by side with matplotlib.

    Parameters
    ----------
        spin : qt.Qobj
            Spin-j state.

        size : float
            Size of plot.
        
        n_samples : int
            Number of samples of the beam function.
    """
    stars = spin_xyz(spin)
    beam = spin_beam(spin)
    fig = plt.figure(figsize=plt.figaspect(0.5))

    bloch_ax = fig.add_subplot(1, 2, 1, projection='3d')
    sphere = qt.Bloch(fig=fig, axes=bloch_ax)
    if spin.shape[0] != 1:
        sphere.point_size=[300]*(spin.shape[0]-1)
        sphere.add_points(stars.T)
        sphere.add_vectors(stars)
    sphere.make_sphere()

    beam_ax = fig.add_subplot(1, 2, 2)
    x = np.linspace(-size, size, n_samples)
    y = np.linspace(-size, size, n_samples)
    X, Y = np.meshgrid(x, y)
    beam_ax.imshow(colorize(beam(X, Y, np.zeros(X.shape))), interpolation="none", extent=(-size,size,-size,size))

    plt.show()

def animate_spin_beam(spin, H, dt=0.1, T=2*np.pi, size=3.5, n_samples=200, filename=None, fps=20):
    """
    Animates a spin state and its corresponding structured Gaussian beam side by side with matplotlib.

    Parameters
    ----------
        spin : qt.Qobj
            Spin-j state.

        H : qt.Qobj
            Hamiltonian.

        dt : float
            Time step.

        T : float
            How long to evolve for.

        size : float
            Size of plot.
        
        n_samples : int
            Number of samples of the beam function.

        filename : str
            Filename at which to save movie.

        fps : int
            Frames per second.
    """
    fig = plt.figure(figsize=plt.figaspect(0.5))

    bloch_ax = fig.add_subplot(1, 2, 1, projection='3d')
    sphere = qt.Bloch(fig=fig, axes=bloch_ax)
    sphere.point_size=[300]*(spin.shape[0]-1)
    sphere.make_sphere()

    beam_ax = fig.add_subplot(1, 2, 2)
    x = np.linspace(-size, size, n_samples)
    y = np.linspace(-size, size, n_samples)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)

    U = (-1j*H*dt).expm()
    sphere_history = []
    beam_history = []
    steps = int(T/dt)
    for t in range(steps):
        if spin.shape[0] != 1:
            sphere_history.append(spin_xyz(spin))
        beam = spin_beam(spin)
        beam_history.append(beam(X, Y, Z))
        spin = U*spin

    sphere.make_sphere()
    im = beam_ax.imshow(colorize(beam_history[0]), interpolation="none", extent=(-size,size,-size,size))
    
    def animate(t):
        if spin.shape[0] != 1:
            sphere.clear()
            sphere.add_points(sphere_history[t].T)
            sphere.add_vectors(sphere_history[t])
            sphere.make_sphere()

        im.set_array(colorize(beam_history[t])) 
        return [bloch_ax, im]

    ani = animation.FuncAnimation(fig, animate, range(steps), repeat=False)
    if filename:
        ani.save(filename, fps=fps)
    plt.close()
    #return ani
