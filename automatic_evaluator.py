"""
This file includes some
automatic methods to evaluate my set of actions
- Slide Close
- Slide Next To
- Slide Away
- Slide Past
- Slide Around
"""

import numpy as np
from config import block_size

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def determinant(v,w):
    return v[0]*w[1]-v[1]*w[0]

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((0,1), (1,0))
    -1.5707963267948966
    >>> angle_between( (1,0), (0,1))
    1.5707963267948966
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.copysign(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)), determinant(v1_u, v2_u) ) 

def test_slide_around ( env, alpha_1 = np.pi / 2, alpha_2 = 3 * np.pi / 4 ):
	"""
	This test uses a very simple interpretation for Slide Around

	My interpretation for "Slide Around" is quite simple: just calculate the covering
	angle of the moving object around the static object from the start point to the
	end point. The angle for each moving step will be positive if
	the angle is counter-clockwise, negative if the angle is clockwise. The final
	result is the absolute of the sum for all steps. Note that we do not take into
	account the change of the distance between two objects over time, because emperically
	my method of generating action does not have smooth trajectory.

	I add two hyperparameters $\alpha_1$ and $\alpha_2$. To soften the definition,
	the output of the evaluator will be either 0, 1 or 0.5. If the calculated covering
	angle $\alpha$ < $\alpha_1$, outputs 0, $\alpha_1$ <= $\alpha$ <= $\alpha_2$, outputs 0.5,
	otherwise outputs 1.

	Parameters:
	==============
	env: BlockMovementEnv
	"""
	moving_object = None
	static_pos = None

	accumulated_angle = 0

	for object_index, prev_transform, next_transform, _, _, success, _, _ in env.action_storage:
		if not success:
			continue

		if moving_object is None:
			moving_object = object_index
			static_pos = self.e.objects[1 - moving_object].transform.position

		v1 = prev_transform.position - static_pos
		v2 = next_transform.position - static_pos

		angle = angle_between(v1, v2)
		accumulated_angle += angle

	accumulated_angle = abs(accumulated_angle)

	if accumulated_angle < alpha_1:
		return 0

	if alpha_1 <= accumulated_angle < alpha_2:
		return 0.5

	return 1

def test_slide_close ( env, threshold = 2 * block_size):
	"""
	This test uses a simple interpretation for Slide Close

	Slide closer means that for all the steps taken in the environment,
	the moving object needs to get closer and closer to the target object.

	The algorithm is as followings:
	- For each action step of the moving object, check the distance toward the static object,
	we should always have each move getting the two objects closer.
	- If one action step doesn't satisfy this condition, the whole action sequence is
	considered wrong.
	- The distance at the end of the action needs to be small enough.

	Parameters:
	==============
	env: BlockMovementEnv
	"""
	for object_index, prev_transform, next_transform, _, _, success, _, _ in env.action_storage:
		if not success:
			continue

		if moving_object is None:
			moving_object = object_index
			static_pos = self.e.objects[1 - moving_object].transform.position

		v1 = prev_transform.position - static_pos
		v2 = next_transform.position - static_pos

		if (np.linalg.norm(v1) < np.linalg.norm(v2)):
			return 0

	if np.linalg.norm(v2) > threshold:
		return 0

	return 1

def test_slide_nextto ():
	"""
	This test uses a very simple interpretation for Slide Next To
	
	This action is a special action in our corpus of actions, because 
	we only care about the final position of the moving block. A perfect position
	for "slide next to" is when the two blocks are aligned, and very close.

	We would set two hyper-parameters for this method. One is the threshold for angle 
	difference between two blocks (two squares). For example, difference of 0 to 15 degree seems
	quite aligned, but anything larger than that is quite bad. The distance between 
	two square centers is normalized with block size. A threshold of 1.3 seems a 
	reasonable for this hyper parameter.

	Parameters:
	==============
	env: BlockMovementEnv 
	"""
	pass

def test_slide_past ():
	"""
	This is a very simple interpretation for Slide Past

	My interpretation for Slide Past is that the distance between two blocks needs to be widest 
	at the beginning and end states. If the total movement involves multiples action steps,
	that distance at a intermediate step < max(distance beginning, distance end). 
	Moreover, we also want that the total path that the moving block has traversed
	is long enough, so from certain viewpoint, we can observe some occlusion
	happens during a period of time. Here I will just constrain that the largest edge of the triangle 
	made from the beginning position, the end position and the static object position
	is the one between beginning postion and end positon. Also the angles next to
	this edge need to be smaller than a threshold angle (a hyper-parameter). 

	Parameters:
	==============
	env: BlockMovementEnv
	"""
	pass

def test_slide_away ():
	"""
	This is a very simple test for Slide Away

	Slide away means that for all the steps taken in the environment,
	the moving object needs to get further and further to the target object.

	The algorithm is as followings:
	- For each action step of the moving object, check the distance toward the static object,
	we should always have each move getting the two objects further.
	- If one action step doesn't satisfy this condition, the whole action sequence is
	considered wrong.
	- The ratio of the distance at the end and at the beginning needs to be
	higher than a threshold.

	Parameters:
	==============
	env: BlockMovementEnv
	"""
	pass