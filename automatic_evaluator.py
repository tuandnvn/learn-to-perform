"""
This file includes some
automatic methods to evaluate my set of actions
- Slide Close
- Slide Next To
- Slide Away
- Slide Past
- Slide Around
"""


def test_slide_around ( env ):
	"""
	This test uses a very simple interpretation for Slide Around

	
	
	Parameters:
	==============
	env: BlockMovementEnv
	"""
	pass


def test_slide_close ():
	"""
	This test uses a simple interpretation for Slide Close

	Slide closer means that for all the steps taken in the environment,
	the moving object needs to get closer and closer to the target object.

	The algorithm is as followings:
	- For each action step of the moving object, check the distance toward the static object,
	we should always have each move getting the two objects closer.
	- If one action step doesn't satisfy this condition, the whole action sequence is
	considered wrong.

	Parameters:
	==============
	env: BlockMovementEnv
	"""
	pass

def test_slide_nextto ():
	"""
	This test uses a very simple interpretation for Slide Next To
	
	This action is a special action in our corpus of actions, because 
	we only care about the final position of the moving block. A perfect position
	for "slide next to" is when the two blocks are aligned, and very close.

	We would set two hyper-parameters for this method. One is the threshold for angle 
	difference between two blocks (two squares). For example, difference of 0 to 15 degree seems
	quite aligned, but anything larger than that is quite bad. The distance between 
	two square centers is normalized with block size. A threshold of 1.5 seems a 
	reasonable for this hyper parameter.

	Parameters:
	==============
	env: BlockMovementEnv 
	"""
	pass

def test_slide_past ():
	"""
	This is a very simple interpretation for Slide Past

	My interpretation for Slide Past is that the action steps between the beginning 
	and the end position of the moving 

	Parameters:
	==============
	env: BlockMovementEnv
	"""
	pass

def test_slide_away ():
	"""
	This is a very simple test for Slide Away

	Parameters:
	==============
	env: BlockMovementEnv
	"""
	pass