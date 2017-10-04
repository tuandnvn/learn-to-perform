import numpy as np

'''
- Tranform of an object: (x, y, theta)
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

	def clone(self):
		return Transform2D(np.copy(self.position), self.rotation, self.scale)

class Geometry2D (object):
	'''
	markers: a set of points on the geometry object that allow tracking
	'''
	markers = np.zeros((0, 2), dtype = np.float32)

	#-------------------------------------------------------------------
	'''
	transform: Transform2D
	'''
	def __init__(self, transform, markers = markers):
		self.transform = transform
		self.markers = markers
	
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

	def clone(self):
		return Geometry2D(transform = self.transform.clone(), markers = np.copy(self.markers))


class Cube2D (Geometry2D):
	'''
	markers: corners points of a standard square with size 1 centering at (0,0)
	of size (4, 2)
	'''
	markers = np.array([[1,1], [1,-1], [-1,-1], [-1, 1]], dtype = np.float32)

	#-------------------------------------------------------------------
	'''
	It is not exactly a cube, but just a square, with markers 
	being the corners of the square 
	'''
	def __init__(self, transform):
		Geometry2D.__init__(self, transform, markers = Cube2D.markers)

	def __str__(self):
		return 'Cube :' + ', '.join(str(th) for th in self.get_markers())

class Polygon2D (Geometry2D):

	#-------------------------------------------------------------------
	'''
	Polygon is basically a convex hull polygon with markers on the 

	We would not check the condition of being convex hull, it is the responsibility
	of users to input a correct convex polygon
	'''
	def __init__(self, transform = Transform2D([0,0], 0, 1), markers = Cube2D.markers):
		Geometry2D.__init__(self, transform)
		self.markers = markers

	def __str__(self):
		return 'Polygon :' + ', '.join(str(th) for th in self.get_markers())

class Command ( object ):
	#-------------------------------------------------------------------
	'''
	Just set the tranform of the object to a new transform (but not new scale)
	'''
	def __init__( self, position, rotation ):
		self.position = np.array(position)
		self.position.shape = (1,2)
		self.rotation = rotation