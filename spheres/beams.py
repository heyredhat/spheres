from spheres import *

import numpy as np
from numpy import sqrt, pi, exp
from math import factorial
from cmath import phase
from scipy.special import eval_genlaguerre

#import mpld3
#mpld3.enable_notebook()
import matplotlib.pyplot as plt

from colorsys import hls_to_rgb
def colorize(z):
    n,m = z.shape
    c = np.zeros((n,m,3))
    c[np.isinf(z)] = (1.0, 1.0, 1.0)
    c[np.isnan(z)] = (0.5, 0.5, 0.5)

    idx = ~(np.isinf(z) + np.isnan(z))
    A = (np.angle(z[idx]) + np.pi) / (2*np.pi)
    A = (A + 0.5) % 1.0
    B = 1.0 - 1.0/(1.0+abs(z[idx])**0.3)
    c[idx] = [hls_to_rgb(a, b, 0.8) for a,b in zip(A,B)]
    return c

# N is an integer.
# l runs from -N to N in steps of 2. 
def laguerre_gauss_beam(N, l, coordinates="cartesian"):
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
            r = abs(c)
            phi = phase(c)
            return mode(r, phi, z)
        return cartesian

def spin_beam(spin, coordinates="cartesian"):
    j = (spin.shape[0]-1)/2
    v = components(spin)
    lg_basis = [laguerre_gauss_beam(int(2*j), int(2*m), coordinates=coordinates) for m in np.arange(-j, j+1)]
    if coordinates == "cartesian":
        def beam(x, y, z):
            return sum([v[int(m+j)]*lg_basis[int(m+j)](x, y, z) for m in np.arange(-j, j+1)])
        return beam
    elif coordinates == "cylindrical":
        def beam(r, phi, z):
            return sum([v[int(m+j)]*lg_basis[int(m+j)](r, phi, z) for m in np.arange(-j, j+1)])
        return beam

def viz_beam(beam, size=5, n_samples=100):
    x = np.linspace(-size, size, n_samples)
    y = np.linspace(-size, size, n_samples)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[beam(x_, y_, 0) for y_ in y] for x_ in x])
    plt.imshow(colorize(Z), interpolation="none", extent=(-size,size,-size,size))
    plt.show()