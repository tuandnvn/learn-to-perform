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
from config import Config

c = Config()

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def determinant(v,w):
    return v[0]*w[1]-v[1]*w[0]

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2':

    >>> angle_between((0,1), (1,0))
    1.5707963267948966
    >>> angle_between( (1,0), (0,1))
    1.5707963267948966
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def p_angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2':

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
	static_object = None

	accumulated_angle = 0

	for object_index, prev_transform, next_transform, _, _, success, _, _ in env.action_storage:
		if not success:
			continue

		if static_object is None:
			static_object = env.e.objects[1 - object_index].transform

		v1 = prev_transform.position - static_object.position
		v2 = next_transform.position - static_object.position

		angle = p_angle_between(v1.flatten(), v2.flatten())
		accumulated_angle += angle
		#print (angle)

	accumulated_angle = abs(accumulated_angle)
	#print (accumulated_angle)

	if accumulated_angle < alpha_1:
		return 0

	if alpha_1 <= accumulated_angle < alpha_2:
		return 0.5

	return 1

def test_slide_close ( env, threshold = 2 * c.block_size):
	"""
	This test uses a simple interpretation for Slide Close

	Slide closer means that for all the steps taken in the env.eironment,
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
	static_object = None

	for object_index, prev_transform, next_transform, _, _, success, _, _ in env.action_storage:
		if not success:
			continue

		if static_object is None:
			static_object = env.e.objects[1 - object_index].transform

		v1 = prev_transform.position - static_object.position
		v2 = next_transform.position - static_object.position

		if (np.linalg.norm(v1) < np.linalg.norm(v2)):
			return 0

	if np.linalg.norm(v2) > threshold:
		return 0

	return 1

def test_slide_nextto ( env, angle_diff = np.pi/9, threshold = 1.3 * c.block_size ):
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
	static_object = None

	# Run reverse to find the last successful action step
	for object_index, prev_transform, next_transform, _, _, success, _, _ in env.action_storage[::-1]:
		if not success:
			continue

		if static_object is None:
			static_object = env.e.objects[1 - object_index].transform

		v_end = next_transform.position - static_object.position

		if v_end > threshold:
			return 0

		# 0 <= next_transform.rotation, static_object.rotation <= pi/2
		# -pi/2 <= rot <= pi/2
		rot = next_transform.rotation - static_object.rotation
		rot = abs(rot)
		if rot > np.pi/4:
			rot = np.pi/2 - rot

		if rot > angle_diff:
			return 0

		return 1

def test_slide_past ( env, angle_threshold = .4 * np.pi ):
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

	We also need to avoid the case that a Slide Around might be recognized as Slide Past,
	so we calculate test_slide_around, if it is Slide Around, return 0

	Parameters:
	==============
	env: BlockMovementEnv
	"""
	t = test_slide_around ( env )
	if t != 0:
		return 0

	static_object = None
	v_start = None

	# Run reverse to find the last successful action step
	for object_index, prev_transform, next_transform, _, _, success, _, _ in env.action_storage[::-1]:
		if not success:
			continue

		if static_object is None:
			static_object = env.e.objects[1 - object_index].transform

		v1 = prev_transform.position - static_object.position
		v2 = next_transform.position - static_object.position

		if v_start is None:
			v_start = v1
			l_max = np.linalg.norm(v_start)

		if np.linalg.norm(v2) > l_max:
			l_max = np.linalg.norm(v2)

	# Condition for longest distance
	if l_max > max ( np.linalg.norm(v2), np.linalg.norm(v_start) ):
		return 0

	# Check condition for the triangle
	v_other = v2 - v_start
	if np.linalg.norm(v2) > np.linalg.norm(v_other):
		return 0

	if np.linalg.norm(v_start) > np.linalg.norm(v_other):
		return 0

	if angle_between(v_other, -v_start) >= angle_threshold:
		return 0

	if angle_between(v_other, v2) >= angle_threshold:
		return 0

	return 1

def test_slide_away ( env, ratio_threshold = 1.5):
	"""
	This is a very simple test for Slide Away

	Slide away means that for all the steps taken in the env.eironment,
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
	static_object = None
	v_start = None
	for object_index, prev_transform, next_transform, _, _, success, _, _ in env.action_storage:
		if not success:
			continue

		if static_object is None:
			static_object = env.e.objects[1 - object_index].transform

		v1 = prev_transform.position - static_object.position
		v2 = next_transform.position - static_object.position

		if v_start is None:
			v_start = v1

		if (np.linalg.norm(v1) > np.linalg.norm(v2)):
			return 0

	if np.linalg.norm(v2)/np.linalg.norm(v_start) < ratio_threshold:
		return 0

	return 1