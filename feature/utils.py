class Transform3D (object):
	'''
	position: 3D position [x,y] (meter)
	rotation_axis: axis = [x,y,z]
	rotation_angle: counterclockwise angle = theta (radiant)
	scale: 1D scale (unit)
	'''
	def __init__(self, position, rotation_axis, rotation_angle, scale):
		self.position = np.array(position)
		self.position.shape = (1,3)
		self.rotation = object()
		self.rotation.axis = np.asarray(rotation_axis)
		self.rotation.angle = rotation_angle 
		self.scale = scale

	def clone(self):
		return Transform3D(np.copy(self.position), np.copy(self.rotation.axis), 
			self.rotation.angle, self.scale)


class Geometry3D (object):
	'''
	markers: a set of points on the geometry object that allow tracking
	'''
	markers = np.zeros((0, 3), dtype = np.float32)

	#-------------------------------------------------------------------
	'''
	transform: Transform3D
	'''
	def __init__(self, transform, markers = markers):
		self.transform = transform
		self.markers = markers
	
	@staticmethod
	def rotation_matrix(axis, theta):
	    """
	    Return the rotation matrix associated with counterclockwise rotation about
	    the given axis by theta radians.
	    """
	    axis = np.asarray(axis)
	    axis = axis/math.sqrt(np.dot(axis, axis))
	    a = math.cos(theta/2.0)
	    b, c, d = -axis*math.sin(theta/2.0)
	    aa, bb, cc, dd = a*a, b*b, c*c, d*d
	    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
	    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
	                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
	                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

	'''
	Get the markers after transforming with transform

	This calculation is not standard

	Normally we would use Rotation x A with Rotation being rotation matrix with theta 
	angle counterclockwise with A of size (3,n)
	'''
	def get_markers(self):
		rotate_matrix = Geometry3D.rotation_matrix(self.transform.rotation.axis, self.transform.rotation.angle)
		return self.transform.position + np.transpose(np.dot(rotate_matrix, np.transpose(self.markers))) * self.transform.scale

	def clone(self):
		return Geometry3D(transform = self.transform.clone(), markers = np.copy(self.markers))