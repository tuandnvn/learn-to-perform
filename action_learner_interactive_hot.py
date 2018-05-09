from action_learner_interactive import InteractiveLearner


class InteractiveLearnerHot ( InteractiveLearner ):
	"""
	An interactive learner that support fixing an action in a demonstration by choosing another action in the searching space.
	
	This interactive learner should only use Greedy search on Continuous space, because the new actions are taken on Continuous space.
	"""
	def __init__(self, c = None, action_type = "SlideAround", online = True, project_path = None, progress_model_path = None):
		super.__init__(self, c = c, action_type = action_type, discrete = False, online = online, 
			project_path = project_path, progress_model_path = progress_model_path)

	def 