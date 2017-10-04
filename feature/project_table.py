'''
Created on Oct 4, 2017

@author: Tuan
'''


'''
The whole purpose of this code
is to project blocks into the surface
that has been marked as the edges of the table
'''
from scipy.optimize import leastsq

from util import Geometry3D
from ..simulator.util import Cube2D

'''
Parameters
----------
points: np array of size (3, n)

Returns
----------
plane: np array of size 4: (a, b, c, d) where norm([a,b,c]) == 1
'''
def estimate_plane ( points ):
	# Inital guess of the plane
	p0 = [1, 1, 1, 0]

	# distance from a point X to a plane p
	def distance(p,X):
		plane_xyz = p[0:3]
		distance = (plane_xyz*X.T).sum(axis=1) + p[3]
		return distance / np.linalg.norm(plane_xyz)

    sol = leastsq(distance, p0, args=(points,))[0]

	sol = sol / norm(sol[:3])

	return sol


'''
Infer the position of top markers on the 2D surface of the table
given the position of side markers.


Parameters
----------
block_side_markers: np.array of size 12 ( 4 x 3D markers )
table: Geometry3D
table_markers_mapping: dict from table.markers index to (2D) positions on planar table surface
block_size: unit (typically it is 0.18 meter)

Returns
----------
top_side_markers: Cube2D
'''
def cube_2d_inference( block_side_markers, table, table_markers_mapping, block_size = 0.18):
	# Calculate the plane from the table surface
	# Using scipy optimize to estimate the plane of the table

	# (n,3)
	markers = table.get_markers()


	# An estimation of Cube2D
	top_side_markers = Cube2D()

	return top_side_markers

	
