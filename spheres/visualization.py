"""
Visualization Tools
-------------------

"""

import pylab
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

from spheres.stars import *

def animate(state, H, dt=0.1, T=300, filename=None):
	U = (-1j*H*dt).expm()

	fig = pylab.figure()
	ax = Axes3D(fig)
	sphere = qt.Bloch(fig=fig, axes=ax)

	def _animate_(i):
		nonlocal state, ax
		stars = spin_xyz(state)
		state = U*state

		sphere.clear()
		sphere.add_points(list(stars.T))
		sphere.add_vectors(stars)
		sphere.make_sphere()
		return ax

	ani = animation.FuncAnimation(fig,\
								  _animate_,\
								  frames=range(T),\
                            	  repeat=False)
	if filename:
		ani.save(filename, fps=20)
	pylab.show()

