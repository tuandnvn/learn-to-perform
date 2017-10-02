import numpy as np

'''
Author: Tuan Do

In this code, I will create a simplified simulator that 
can help to emulate actions of robots without physical engine

This simulator should be able to calculate the path of a moving object so that
it doesn't cross with a static object.

This simulator should be able to allow users to create objects in the scene, at least
in the form of cube blocks, and then perform a movement of object:

- Move to (x,y)
'''
class Transform2D (object):
	'''
	position: 2D position (x,y) (meter)
	rotation: 1D rotation counterclockwise angle theta (radiant)
	scale: 1D scale (unit)
	'''
	def __init__(self, position, rotation, scale):
		self.position = position
		self.rotation = rotation
		self.scale = scale

class Geometry2D (object):
	'''
	markers: a set of points on the geometry object that allow tracking
	'''
	markers = np.zeros((0, 2), dtype = np.float32)

	#-------------------------------------------------------------------
	'''
	transform: Transform2D
	'''
	def __init__(self, transform):
		self.transform = transform
	
	'''
	Get the markers after transforming with transform

	This calculation is not standard

	Normally we would use Rotation x A with Rotation being rotation matrix with theta 
	angle counterclockwise with A of size (2,n)
	We instead use A_transpose x Rotation_transpose  of size (n, 2)
	'''
	def get_markers(self):
		rotate_matrix = np.array([[ np.cos(-self.rotation), -np.sin(-self.rotation) ], 
			[np.sin(-self.rotation), np.cos(-self.rotation)]])
		return np.dot(markers, rotate_matrix)


class Cube2D (Geometry2D):
	'''
	markers: corners points of a standard square with size 1 centering at (0,0)
	of size (4, 2)
	'''
	markers = np.array([[1,1], [1,-1], [-1, 1], [-1,-1]], dtype = np.float32)

	#-------------------------------------------------------------------
	'''
	It is not exactly a cube, but just a square, with markers 
	being the corners of the square 
	'''
	def __init__(self, transform):
		Geometry2D.__init__(self, transform)


class Environment (object):
	def __init__(self):
		self.objects = []
		
		self.__speed = 1

	# Movement of object per frame meter/frame
	# Default value is 1
	@property
	def speed(self):
		return self.__speed

	@speed.setter
	def speed(self, s):
		if x <= 0:
			raise ValueError('speed could not be <= 0') 
		else:
			self.__speed = s

	'''
	Check if two segments are intersected or not

	Parameters
	----------
	p: First segment np.array [[p1x, p1y], [p2x, p2y]]
	q: Second segment np.array [[q1x, q1y], [q2x, q2y]]

	Return
	----------
	value: = 1 intersected, = 0 not intersected, =2 if touching but not intersected
	'''
	@classmethod
	def check_overlap_segment( p, q ):
		

	'''
	A very simple checking condition is to check if every segments
	made from markers of two objects cut each other.

	It's just an estimation, works for objects of similar size, but
	doesn't work if an object is significantly smaller than the other
	'''
	@classmethod
	def check_overlap( o1, o2 ):


	'''
	A simple check for overlapping constraint
	
	Just need to check if the argument object o 
	overlaps with other objects already in the environment
	but doesn't check overlapping conditions of objects already
	in environment. 


	'''
	def check_overlap_consistency(self, o):
		pass

	'''
	Add more object into the environment
	'''
	def add_object(self, o):
		self.objects.append(o)


