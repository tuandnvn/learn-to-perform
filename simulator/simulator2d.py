'''
Created on Oct 4, 2017

@author: Tuan
'''
import numpy as np
from numpy.linalg import norm
import math
from scipy.spatial import ConvexHull

from .utils import Transform2D, Geometry2D, Cube2D, Polygon2D, Command

class Environment (object):
	'''
	In Environment, I will create a simplified simulator that 
	can help to emulate actions of robots without physical engine

	This simulator should be able to calculate the path of a moving object so that
	it doesn't cross with a static object.

	This simulator should be able to allow users to create objects in the scene, at least
	in the form of cube blocks, and then perform a movement of object:
	'''
	def __init__(self, boundary = None, speed = 1):
		'''
		Parameters
		----------
		boundary: should be of type Polygon2D
		speed: movement of object per frame meter/frame
		'''
		self.objects = []
		self.boundary = boundary
		self.__speed = speed

		"""
		The environment records all the actions it has taken
		"""
		self.action_records = []

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

	
	@staticmethod
	def check_intersect(p1, p2, q1, q2 ):
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

	
	@staticmethod
	def is_point_bounded( p, o ):
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


	
	@staticmethod
	def is_bounded( o1, o2 ):
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
		for marker in o1.get_markers():
			if not Environment.is_point_bounded(marker, o2):
				return False
		return True


	
	@staticmethod
	def is_overlap( o1, o2 ):
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

	
	def is_overlap_consistency(self, o, exclude_indices = []):
		'''
		A simple check for overlapping constraint
		
		Just need to check if the argument object o 
		overlaps with other objects already in the environment
		but doesn't check overlapping conditions of objects already
		in environment. 

		Parameters
		----------
		o: A geometric object
		exclude_indices: indices of object in self.objects that should not be checked for overlapping

		Return
		----------
		value: = True consistent, = False not consistent
		'''
		for i in range(len(self.objects)):
			if not i in exclude_indices:
				other = self.objects[i]
				if Environment.is_overlap(o, other):
					return False

		return True

	
	def add_object(self, o):
		'''
		Add more object into the environment

		Parameters
		----------
		o: A geometric object

		Return
		----------
		value: = True can add object, = False couldn't add object
		'''
		if (self.boundary == None or Environment.is_bounded(o, self.boundary)) and\
		self.is_overlap_consistency(o):
			self.objects.append(o)
			return True
		return False

	
	@staticmethod
	def get_convex_hull( points ):
		'''
		Get convex hull of an object

		Parameters
		----------
		points: n points, np.array of size (n, 2)

		Return
		----------
		o: convex hull, a Polygon2D
		'''
		# My own implementation
		# if len(points) < 1:
		# 	return
		# # Get the point that has smallest y, if there are multiple, get the one that has lowest x
		# # This point must lies on the convex hull
		# min_y = np.min(points[:,1])
		# # all points that has x == min_x
		# q = points[points[:,1] == min_y]

		# min_x = np.min(q[:,0])
		# # start_point = [min_x, min_y]
		# # order the remaining points by ranking slope of vector made by start_point and other points
		
		# # return a tuple 
		# # ( angle with (0,0)->(1,0) , distance from min_x, min_y) 
		# def rank(point):
		# 	vector = point - [min_x, min_y]

		# 	return (np.arctan2(vector[1], vector[0]) , norm(vector))


		# sorted_points = sorted(points, key = lambda point : rank(point) )

		# # This is not convex yet
		# boundary = []
		# boundary.append(sorted_points[0])

		# for i in range(1, len(sorted_points)):
		# 	# Check if sorted_points[i] is included in the ray to the next sorted_points
		# 	if i < len(sorted_points) - 1:
		# 		vector_i = sorted_points[i] - [min_x, min_y]
		# 		vector_i_1 = sorted_points[i+1] - [min_x, min_y]
		# 		if np.arctan2(vector_i[1], vector_i[0]) == np.arctan2(vector_i_1[1], vector_i_1[0]):
		# 			continue
		# 	boundary.append(sorted_points[i])

		# convex_hull = []
		convex_hull = ConvexHull([t for t in points if all(~np.isnan(t))])
		return Polygon2D(markers = points[convex_hull.vertices])

	def act(self, obj_index, command):
		'''
		Act on one object with just one command

		Parameters
		----------
		obj_index: To get an object from the set of objects in the environments

		Returns
		----------
		success: whether the command is succesfully executed or not
		'''
		if 0 <= obj_index < len(self.objects):
			obj = self.objects[obj_index]
		else:
			raise ValueError('Object index %d is not in the range' % obj_index)

		# Project the movement of the object according to each command
		# Make sure there is clearance of the path between beginning and end points
		# check clearance by creating a path object
		# a simple assumption is that the path is made from the object at its
		# original rotation 
		# however we would extrapolate object rotation along the path
		new_obj = obj.clone()

		markers_before = new_obj.get_markers()
		new_obj.transform.position = command.position
		# Only change position to check path
		markers_after = new_obj.get_markers()

		# Create path as the flow of the object from original position to new position without rotating
		path = Environment.get_convex_hull ( np.concatenate([markers_before, markers_after]))

		# Now change the rotation as well
		new_obj.transform.rotation = command.rotation

		# Check to see if the path is inside the playfield
		# and path doesn't overlap with other objects
		# and final position doesn't overlap with other objects
		if (self.boundary == None or Environment.is_bounded(path, self.boundary)) and\
			self.is_overlap_consistency(path, exclude_indices = [obj_index]) and\
			self.is_overlap_consistency(new_obj, exclude_indices = [obj_index]):
			# command satisfy
			# set obj to new_obj
			
			self.objects[obj_index] = new_obj
			return True
		else:
			return False



	'''
	Just print out the objects
	'''
	def __str__(self):
		tr = 'Environment: \n'

		if self.boundary != None:
			tr += 'Boundary: ' + str(self.boundary) + '\n'

		tr += '\n'.join([ str(o) for o in self.objects])

		return tr