import unittest
import numpy as np
from simulator2d import Environment, Cube2D, Transform2D, Polygon2D, Command

class TestSimulator2D ( unittest.TestCase ):
	def setUp(self):
		pass

	def test_intersected_segments(self):
		# Typical
		p1, p2, q1, q2 = [0,0], [1,1], [0,1],[1,0]
		self.assertEqual( Environment.check_intersect(p1, p2, q1, q2), 1 )
		self.assertEqual( Environment.check_intersect(q1, q2, p1, p2), 1 )

		# Jointed parallel lines
		p1, p2, q1, q2 = [0,0], [1,1], [1,1],[2,2]
		self.assertEqual( Environment.check_intersect(p1, p2, q1, q2), 2 )
		self.assertEqual( Environment.check_intersect(q1, q2, p1, p2), 2 )

		# One segment is included point
		p1, p2, q1, q2 = [0,0], [1,1], [1,1], [1,1]
		self.assertEqual( Environment.check_intersect(p1, p2, q1, q2), 2 )
		self.assertEqual( Environment.check_intersect(q1, q2, p1, p2), 2 )

		# One segment is separated point
		p1, p2, q1, q2 = [0,0], [1,1], [2,2], [2,2]
		self.assertEqual( Environment.check_intersect(p1, p2, q1, q2), 0 )
		self.assertEqual( Environment.check_intersect(q1, q2, p1, p2), 0 )

		# Same segment
		p1, p2, q1, q2 = [0,0], [0,1], [0,0], [1,0]
		self.assertEqual( Environment.check_intersect(p1, p2, q1, q2), 2 )
		self.assertEqual( Environment.check_intersect(q1, q2, p1, p2), 2 )

		# 
		p1, p2, q1, q2 = [0,0], [0,1], [-1,0], [1,0]
		self.assertEqual( Environment.check_intersect(p1, p2, q1, q2), 2 )
		self.assertEqual( Environment.check_intersect(q1, q2, p1, p2), 2 )

		# One vertical line cut One horizontal line
		p1, p2, q1, q2 = [0,-1], [0,1], [-1,0], [1,0]
		self.assertEqual( Environment.check_intersect(p1, p2, q1, q2), 1 )
		self.assertEqual( Environment.check_intersect(q1, q2, p1, p2), 1 )

		# One vertical line joint One horizontal line
		p1, p2, q1, q2 = [0,-1], [0,1], [-1,1], [1,1]
		self.assertEqual( Environment.check_intersect(p1, p2, q1, q2), 2 )
		self.assertEqual( Environment.check_intersect(q1, q2, p1, p2), 2 )

		# One vertical line separate One horizontal line
		p1, p2, q1, q2 = [0,-1], [0,1], [-1,2], [1,2]
		self.assertEqual( Environment.check_intersect(p1, p2, q1, q2), 0 )
		self.assertEqual( Environment.check_intersect(q1, q2, p1, p2), 0 )

		# Two vertical lines
		p1, p2, q1, q2 = [0,-1], [0,1], [1,-1], [1,1]
		self.assertEqual( Environment.check_intersect(p1, p2, q1, q2), 0 )
		self.assertEqual( Environment.check_intersect(q1, q2, p1, p2), 0 )

		# Overllaping vertical lines
		p1, p2, q1, q2 = [0,-1], [0,1], [0,0], [0,2]
		self.assertEqual( Environment.check_intersect(p1, p2, q1, q2), 2 )
		self.assertEqual( Environment.check_intersect(q1, q2, p1, p2), 2 )

		# Overllaping normal lines
		p1, p2, q1, q2 = [0,0], [2,2], [1,1], [3,3]
		self.assertEqual( Environment.check_intersect(p1, p2, q1, q2), 2 )
		self.assertEqual( Environment.check_intersect(q1, q2, p1, p2), 2 )

	def test_is_overlap(self):
		o1 = Cube2D(Transform2D( [0.0,0.0], 0.0, 1.0) )
		o2 = Cube2D(Transform2D( [1.0,1.0], 0.0, 1.0))

		self.assertTrue( Environment.is_overlap(o1, o2) )

		o1 = Cube2D(Transform2D( [0.0,0.0], 0.0, 1.0) )
		o2 = Cube2D(Transform2D( [2.0,2.0], 0.0, 1.0))

		self.assertFalse( Environment.is_overlap(o1, o2) )

		o1 = Cube2D(Transform2D( [0.0,0.0], 0.0, 1.0) )
		o2 = Cube2D(Transform2D( [1.9,1.9], np.pi/4, 1.0))

		self.assertFalse( Environment.is_overlap(o1, o2) )

		o1 = Cube2D(Transform2D( [0.0,0.0], 0.0, 1.0) )
		o2 = Cube2D(Transform2D( [1.7,1.7], np.pi/4, 1.0))

		self.assertTrue( Environment.is_overlap(o1, o2) )

	def test_is_point_bounded(self):
		o = Polygon2D ( markers = np.array([[0,1], [1,-1], [-1, -1]], dtype = np.float32), 
			transform = Transform2D( [0.0,0.0], 0.0, 1.0))
		p = [0,0]
		self.assertTrue(Environment.is_point_bounded(p, o))

		p = [0,1]
		self.assertTrue(Environment.is_point_bounded(p, o))

		p = [1,0]
		self.assertFalse(Environment.is_point_bounded(p, o))

		o = Polygon2D ( markers = np.array([[1,2], [-1,2], [-1, -2], [1,-2]], dtype = np.float32), 
			transform = Transform2D( [0.0,0.0], 0.0, 1.0))
		p = [2,0]
		self.assertFalse(Environment.is_point_bounded(p, o))

		p = [1,0]
		self.assertTrue(Environment.is_point_bounded(p, o))

	def test_is_bounded(self):
		o2 = Polygon2D ( markers = np.array([[1,1], [-1,1], [-1, -1], [1,-1]], dtype = np.float32), 
			transform = Transform2D( [0.0,0.0], 0.0, 1.0))
		o1 = Cube2D (transform = Transform2D( [-0.5,0.0], np.pi/4, 0.5))

		self.assertFalse(Environment.is_bounded(o1, o2))

		o1 = Cube2D (transform = Transform2D( [-0.5,0.0], np.pi/2, 0.5))

		self.assertTrue(Environment.is_bounded(o1, o2))

	def test_environment(self):
		o1 = Cube2D(Transform2D( [0.0,0.0], 0.0, 1.0))
		o2 = Cube2D(Transform2D( [1.0,1.0], 0.0, 1.0))
		o3 = Cube2D(Transform2D( [2.0,2.0], 0.0, 1.0))

		e = Environment()

		e.add_object(o1)
		e.add_object(o2)
		e.add_object(o3)

		self.assertIn(o1, e.objects)
		self.assertNotIn(o2, e.objects)
		self.assertIn(o3, e.objects)

		# True scale
		o1 = Cube2D(Transform2D( [0.0,0.0], 0.0, 0.1))
		o2 = Cube2D(Transform2D( [-0.9,0.0], np.pi/4, 0.1))
		o3 = Cube2D(Transform2D( [0.9,0.0], np.pi/2, 0.1))

		e = Environment(boundary = Polygon2D ( markers = np.array([[1,1], [-1,1], [-1, -1], [1,-1]], dtype = np.float32), 
			transform = Transform2D( [0.0,0.0], 0.0, 1.0)),  speed = 0.2 )

		e.add_object(o1)
		e.add_object(o2)
		e.add_object(o3)

		self.assertIn(o1, e.objects)
		self.assertNotIn(o2, e.objects)
		self.assertIn(o3, e.objects)

		# Actions on a certain object
		# Let move o3 around o1
		commands = [ Command([0.0, 0.8], 0.0), Command([-0.9,0.0], -np.pi/2), Command([0.8, 0], 0.0) ]

		print (e.act(1, commands))

		print (e)


if __name__ == '__main__':
	unittest.main()