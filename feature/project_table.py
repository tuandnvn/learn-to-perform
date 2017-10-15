'''
Created on Oct 4, 2017

@author: Tuan
'''


'''
The whole purpose of this code
is to project blocks into the surface
that has been marked as the edges of the table
'''
import numpy as np
from scipy.linalg import norm
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

	p = sol / norm(sol[:3])

	return p

'''

'''
def get_projected_point (p, plane):
	t = (np.dot(p, plane[:3]) + plane[3])/norm(plane[:3])
	return p - plane[:3] * t

'''
Infer the position of top markers on the 2D surface of the table
given the position of side markers.


Parameters
----------
block_side_markers: np.array of size 12 ( 4 x 3D markers )
table: np.array of size (n,3)
table_center: np.array of size 3 (just pick a point as a center on the table)
block_size: unit (typically it is 0.18 meter)

Returns
----------
top_side_markers: Cube2D
'''
def cube_2d_inference( block_side_markers, table_markers, block_size = 0.18):
	# Calculate the plane from the table surface
	# Using scipy optimize to estimate the plane of the table


	# plane estimation of the table
	plane = estimate_plane(table_markers.T)

	block_side_markers_reshape = np.copy(block_side_markers)
	block_side_markers_reshape.shape = (4,3)

	# plane estimation of the block surface
	block_markers_plane = estimate_plane(block_side_markers_reshape.T)
	block_markers_centroid = np.mean(block_side_markers_reshape, axis = 0)

	# Get the cross products of two normal vectors of plane and block_markers_plane
	# give vector for one edge of the resulted polygon
	edge_vector = np.cross( plane[:3], block_markers_plane[:3] )

	# so now we get the edge_vector that is on the block_markers_plane and parallel to plane
	# we also get the vector perpedicular to block_markers_plane
	# could easy to calculate a 3d polygon that can be projected to table plan
	# given an offset

	v1 = edge_vector / norm(edge_vector) * size/2
	v2 = block_markers_plane[:3] / norm(block_markers_plane[:3]) * size/2

	rectangle = [block_markers_centroid + v1 + v2,  block_markers_centroid + v1 - v2, 
		block_markers_centroid - v1 - v2, block_markers_centroid - v1 + v2 ]


	

	rectangle_projected = [ get_projected_point(x, plane) for x in rectangle]

	# An estimation of Cube2D
	top_side_markers = Cube2D()



	return top_side_markers

	
