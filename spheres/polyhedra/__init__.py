"""
Polyhedra related functions.
"""

import sys
import random
from math import pi, asin, atan2, cos, sin, sqrt

def equidistribute_points(n, verbose=False):
    """
    Returns n more or less equidistributed points on the sphere.

    Thanks to https://www.chiark.greenend.org.uk/~sgtatham/polyhedra/

    Parameters
    ----------
        n : int
            Number of points.

        verbose : bool
    
    Returns
    -------
        points : list
    """
    points = []
    for i in range(n):
        # Invent a randomly distributed point.
        #
        # To distribute points uniformly over a spherical surface, the
        # easiest thing is to invent its location in polar coordinates.
        # Obviously theta (longitude) must be chosen uniformly from
        # [0,2*pi]; phi (latitude) must be chosen in such a way that
        # the probability of falling within any given band of latitudes
        # must be proportional to the total surface area within that
        # band. In other words, the probability _density_ function at
        # any value of phi must be proportional to the circumference of
        # the circle around the sphere at that latitude. This in turn
        # is proportional to the radius out from the sphere at that
        # latitude, i.e. cos(phi). Hence the cumulative probability
        # should be proportional to the integral of that, i.e. sin(phi)
        # - and since we know the cumulative probability needs to be
        # zero at -pi/2 and 1 at +pi/2, this tells us it has to be
        # (1+sin(phi))/2.
        #
        # Given an arbitrary cumulative probability function, we can
        # select a number from the represented probability distribution
        # by taking a uniform number in [0,1] and applying the inverse
        # of the function. In this case, this means we take a number X
        # in [0,1], scale and translate it to obtain 2X-1, and take the
        # inverse sine. Conveniently, asin() does the Right Thing in
        # that it maps [-1,+1] into [-pi/2,pi/2].

        theta = random.random() * 2*pi
        phi = asin(random.random() * 2 - 1)
        points.append((cos(theta)*cos(phi), sin(theta)*cos(phi), sin(phi)))
    while 1:
        # Determine the total force acting on each point.
        forces = []
        for i in range(len(points)):
            p = points[i]
            f = (0,0,0)
            ftotal = 0
            for j in range(len(points)):
                if j == i: continue
                q = points[j]

                # Find the distance vector, and its length.
                dv = (p[0]-q[0], p[1]-q[1], p[2]-q[2])
                dl = sqrt(dv[0]**2 + dv[1]**2 + dv[2]**2)

                # The force vector is dv divided by dl^3. (We divide by
                # dl once to make dv a unit vector, then by dl^2 to
                # make its length correspond to the force.)
                dl3 = dl ** 3
                fv = (dv[0]/dl3, dv[1]/dl3, dv[2]/dl3)

                # Add to the total force on the point p.
                f = (f[0]+fv[0], f[1]+fv[1], f[2]+fv[2])

            # Stick this in the forces array.
            forces.append(f)

            # Add to the running sum of the total forces/distances.
            ftotal = ftotal + sqrt(f[0]**2 + f[1]**2 + f[2]**2)

        # Scale the forces to ensure the points do not move too far in
        # one go. Otherwise there will be chaotic jumping around and
        # never any convergence.
        if ftotal > 0.25:
            fscale = 0.25 / ftotal
        else:
            fscale = 1

        # Move each point, and normalise. While we do this, also track
        # the distance each point ends up moving.
        dist = 0
        for i in range(len(points)):
            p = points[i]
            f = forces[i]
            p2 = (p[0] + f[0]*fscale, p[1] + f[1]*fscale, p[2] + f[2]*fscale)
            pl = sqrt(p2[0]**2 + p2[1]**2 + p2[2]**2)
            p2 = (p2[0] / pl, p2[1] / pl, p2[2] / pl)
            dv = (p[0]-p2[0], p[1]-p2[1], p[2]-p2[2])
            dl = sqrt(dv[0]**2 + dv[1]**2 + dv[2]**2)
            dist = dist + dl
            points[i] = p2

        # Done. Check for convergence and finish.
        if verbose:
            sys.stderr.write(str(dist) + "\n")
        if dist < 1e-6:
            break
    return points