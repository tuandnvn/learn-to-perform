import numpy as np
from numpy.linalg import norm
import math

'''
Author: Tuan Do

In this code, I will create a simplified simulator that 
can help to emulate actions of robots without physical engine

This simulator should be able to calculate the path of a moving object so that
it doesn't cross with a static object.

This simulator should be able to allow users to create objects in the scene, at least
in the form of cube blocks, and then perform a movement of object:

- Translocate object with a rotation: (x, y, theta)
'''
class Transform2D (object):
	'''
	position: 2D position [x,y] (meter)
	rotation: 1D rotation counterclockwise angle theta (radiant)
	scale: 1D scale (unit)
	'''
	def __init__(self, position, rotation, scale):
		self.position = np.array(position)
		self.position.shape = (1,2)
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
		rotate_matrix = np.array([[ np.cos(-self.transform.rotation), -np.sin(-self.transform.rotation) ], 
			[np.sin(-self.transform.rotation), np.cos(-self.transform.rotation)]])
		return self.transform.position + np.dot(self.markers, rotate_matrix) * self.transform.scale


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

	def __str__(self):
		return 'Cube :' + ', '.join(str(th) for th in self.get_markers())

class Polygon2D (Geometry2D):

	#-------------------------------------------------------------------
	'''
	Polygon is basically a convex hull polygon with markers on the 

	We would not check the condition of being convex hull, it is the responsibility
	of users to input a correct convex polygon
	'''
	def __init__(self, markers, transform):
		self.markers = markers
		Geometry2D.__init__(self, transform)

	def __str__(self):
		return 'Polygon :' + ', '.join(str(th) for th in self.get_markers())


class Environment (object):
	'''
	Parameters
	----------
	boundary: should be of type Polygon2D
	speed: movement of object per frame meter/frame
	'''
	def __init__(self, boundary = None, speed = 1):
		self.objects = []
		self.boundary = boundary
		self.__speed = speed

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
	make sure values are np.float32
	p1, p2: First segment np.array [[p1x, p1y], [p2x, p2y]]
	q1, q2: Second segment np.array [[q1x, q1y], [q2x, q2y]]

	Return
	----------
	value: = 1 intersected, = 0 not intersected, = 2 if touching but not intersected
	'''
	@staticmethod
	def check_intersect(p1, p2, q1, q2 ):
		# print ('---CHECK---')
		[p1x, p1y] = np.cast['f'](p1)
		[p2x, p2y] = np.cast['f'](p2)
		[q1x, q1y] = np.cast['f'](q1)
		[q2x, q2y] = np.cast['f'](q2)

		# Range of x that the interception should be in between
		IntervalX = (np.max([np.min([p1x, p2x]), np.min([q1x, q2x])]), 
			np.min([np.max([p1x, p2x]), np.max([q1x, q2x])]))

		IntervalY = (np.max([np.min([p1y, p2y]), np.min([q1y, q2y])]), 
			np.min([np.max([p1y, p2y]), np.max([q1y, q2y])]))

		# print('IntervalX ' + str(IntervalX))
		# print('IntervalY ' + str(IntervalY))

		if IntervalX[0] > IntervalX[1] or IntervalY[0] > IntervalY[1]:
			return 0

		# Calculate two slope values;
		with np.errstate(divide='ignore'):
			a1 = (p1y - p2y) / (p1x - p2x)
			a2 = (q1y - q2y) / (q1x - q2x)

			# Two points
			if np.isnan(a1) and np.isnan(a2):
				if p1x == q1x:
					return 2
				else:
					return 0

			# One point and a segment
			# Just not intersected
			if np.isnan(a1) or np.isnan(a2):
				return 2

			# Two vertical lines (parallel to Oy):
			if np.isinf(a1) and np.isinf(a2):
				return 2

			b1 = p1y - a1 * p1x
			b2 = q1y - a2 * q1x

			# (Jx, Jy) is the intersection of two lines
			# One vertical line:
			if np.isinf(a1) or np.isinf(a2):
				if np.isinf(a1):
					Jx = p1x
					Jy = a2 * p1x + b2
					# print ('Jy = %.2f' % Jy)

				if np.isinf(a2):
					Jx = q1x
					Jy = a1 * q1x + b1
					# print ('Jy = %.2f' % Jy)

				# Special case 
				# # One vertical, one horizontal
				if np.isinf(a1) and a2 == 0:
					if min(q1x, q2x) < Jx < max(q1x, q2x) and min(p1y, p2y) < Jy < max(p1y, p2y):
						return 1
					return 2
				elif np.isinf(a2) and a1 == 0:
					if min(p1x, p2x) < Jx < max(p1x, p2x) and min(q1y, q2y) < Jy < max(q1y, q2y):
						return 1
					return 2
				elif IntervalY[0] < Jy < IntervalY[1]:
					return 1
				elif Jy == IntervalY[0] or Jy == IntervalY[1]:
					return 2

				return 0


			# print ('a1 = %.2f a2 = %.2f b1 = %.2f b2 = %.2f' % (a1, a2, b1, b2) )

			if a1 == a2:
				if b1 == b2:
					return 2
				return 0

			
			Jx = (b2 - b1) / (a1 - a2)

			if IntervalX[0] < Jx < IntervalX[1]:
				return 1

			if Jx == IntervalX[0] or Jx == IntervalX[1]:
				return 2

			return 0

	'''
	A very simple checking to see if p is included in o

	Parameters
	----------
	p: A point [x,y]
	o: Should be a Polygon2D object

	Return
	----------
	value: = True if p is bounded by o (or p is on o boundary), = False otherwise
	'''
	@staticmethod
	def is_point_bounded( p, o ):
		ms = o.get_markers()
		for i in range(1, len(ms)):
			# three indices on a row
			first, second, third = i - 1, i, (i + 1) % len(ms)
			# three corners on a row
			first, second, third = ms[first], ms[second], ms[third]

			if all(np.equal(first, p)) or all(np.equal(second, p)) or all(np.equal(third, p)):
				return True

			l1 = first - second
			l2 = p - second
			l3 = third - second

			# check if l2 is between l1 and l3
			first_second_third = np.arccos(np.clip( np.dot(l1, l3)/norm(l1)/norm(l3), -1, 1))
			first_second_p = np.arccos(np.clip( np.dot(l1, l2)/norm(l1)/norm(l2), -1, 1))
			third_second_p = np.arccos(np.clip( np.dot(l2, l3)/norm(l2)/norm(l3), -1, 1))

			if math.isclose(first_second_third, first_second_p + third_second_p, rel_tol=1e-3, abs_tol=1e-5):
				continue

			return False

		return True


	'''
	A very simple checking to see if o1 is included in o2

	Parameters
	----------
	o1: A geometric object
	o2: Should be a Polygon2D object

	Return
	----------
	value: = True if o1 is bounded by o2, = False otherwise
	'''
	@staticmethod
	def is_bounded( o1, o2 ):
		for marker in o1.get_markers():
			if not Environment.is_point_bounded(marker, o2):
				return False
		return True


	'''
	A very simple checking condition is to check if every segments
	made from markers of two objects cut each other.

	It's just an estimation, works for objects of similar size, but
	doesn't work if an object is significantly smaller than the other

	Parameters
	----------
	o1, o2: Two geometric objects

	Return
	----------
	value: = True overlapped, = False not overlapped (but could be touching)
	'''
	@staticmethod
	def is_overlap( o1, o2 ):
		m1 = o1.get_markers() 
		m2 = o2.get_markers()

		# print (m1)
		# print (m2)

		# Just multiple all (check + 1)
		result = 1

		for i in range(m1.shape[0]):
			for j in range(i+1, m1.shape[0]):
				for k in range(m2.shape[0]):
					for l in range(k+1, m2.shape[0]):
						r = Environment.check_intersect(m1[i], m1[j], m2[k], m2[l])
						# print ('%s %s %s %s %d' % (m1[i], m1[j], m2[k], m2[l], r))

						result *= r + 1

		if result % 2 == 0:
			return True

		return False

	'''
	A simple check for overlapping constraint
	
	Just need to check if the argument object o 
	overlaps with other objects already in the environment
	but doesn't check overlapping conditions of objects already
	in environment. 

	Parameters
	----------
	o: A geometric object

	Return
	----------
	value: = True consistent, = False not consistent
	'''
	def is_overlap_consistency(self, o):
		for other in self.objects:
			if Environment.is_overlap(o, other):
				return False

		return True

	'''
	Add more object into the environment

	Parameters
	----------
	o: A geometric object

	Return
	----------
	value: = True can add object, = False couldn't add object
	'''
	def add_object(self, o):
		if (self.is_overlap_consistency(o)):
			self.objects.append(o)
			return True
		return False

	'''
	Just print out the objects
	'''
	def __str__(self):
		return 'Environment: \n' + '\n'.join([ str(o) for o in self.objects])