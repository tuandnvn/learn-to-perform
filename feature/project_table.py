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
import math
from scipy.linalg import norm
from scipy.optimize import leastsq

from util import Geometry3D
from ..simulator.util import Cube2D, Transform2D

'''
Parameters
----------
points: np array of size (3, n)

Returns
----------
plane: np array of size 4: (a, b, c, d) where norm([a,b,c]) == 1
'''
def estimate_plane ( points ):
    # Remove any point if value is not-finite:
    filtered_points = [p for p in points if np.all(np.isfinite(p))]
    if len(filtered_points) <= 2:
        raise Exception("You need at least 3 non-finite points to find a plane")

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
To estimate the transform from the projected coordination

Parameters
----------
rectangle_projected: [ np.array size 3 ] 3-d points lying on the surface of projection
first_point: size 2 -> projected onto the plane as the origin point
second_point: size 2 -> the vector from first_point to second_point is the same as Ox

Returns
----------
bottom_side_markers: a Cube2D that estimate locations of rectangle_projected on the 2d coordination
'''
def estimate_cube_2d ( rectangle_projected, first_point, second_point ):
    plane = estimate_plane(rectangle_projected)
    # An estimation of Cube2D
    bottom_side_markers = Cube2D(transform = Transform2D([0,0], 0, 1))

    return bottom_side_markers

'''
Infer the position of bottom markers on the 2D surface of the table
given the position of top/side markers.
Firstly, we check to see if the given marker should be considered top
or side marker by checking the angle between two planes.

I also need to estimate for the case when there are a missing corner (three corner lefts)
- The case when there are three corner left is quite simple (just need to infer its position from the other three)
We don't consider the case where there is only < 3 corner left because it makes it too hard

Parameters
----------
block_markers: np.array of size 12 ( 4 x 3D markers )
table_markers: np.array of size (n,3)
block_size: unit (typically it is 0.18 meter)

Returns
----------
rectangle_projected: projected 3d points on the surface
'''
def project_markers( block_markers, table_markers, block_size = 0.18):
    # Calculate the plane from the table surface
    # Using scipy optimize to estimate the plane of the table

    # plane estimation of the table
    plane = estimate_plane(table_markers.T)

    block_markers_reshape = np.reshape(block_markers, (4,3))

    # plane estimation of the block surface
    try:
        block_markers_plane = estimate_plane(block_markers_reshape.T)
    except:
        return []

    # Cosin between two norm vectors
    # If the marker is on the side, this value would be closer to 0, say < 0.5
    # If the marker is on the top, this value would be closer to 1, say > 0.5
    cosin_angle = np.abs(np.dot(plane[:3], block_markers_plane[:3]))

    block_markers_centroid = np.mean(block_markers_reshape, axis = 0)
    
    if cosin_angle < 0.5: # side marker
        # Get the cross products of two normal vectors of plane and block_markers_plane
        # give vector for one edge of the resulted polygon
        edge_vector = np.cross( plane[:3], block_markers_plane[:3] )

        # so now we get the edge_vector that is on the block_markers_plane and parallel to plane
        # we also get the vector perpedicular to block_markers_plane
        # could easy to calculate a 3d polygon that can be projected to table plan
        # given an offset

        # v1 parallel
        v1 = edge_vector / norm(edge_vector) * block_size/2
        # v2 perpendicular
        v2 = block_markers_plane[:3] / norm(block_markers_plane[:3]) * block_size/2
        # v2 going into the screen
        if v2[2] < 0:
            v2 = -v2

        rectangle = [block_markers_centroid + v1 + 2 * v2,  block_markers_centroid + v1, 
            block_markers_centroid - v1, block_markers_centroid - v1 + 2 * v2 ]

        rectangle_projected = [ get_projected_point(x, plane) for x in rectangle]
    else:  # top marker
        marker_projected = [ get_projected_point(x, plane) for x in block_markers]

        block_markers_centroid = np.mean(marker_projected, axis = 0)

        # centroid to corner vectors
        cs = [m - block_markers_centroid for m in marker_projected]

        qs = [c * math.sqrt(0.5) * block_size / norm(c)  for c in cs] 

        rectangle_projected = [q + block_markers_centroid for q in qs]


    return rectangle_projected


    

    
