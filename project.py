import session_util
import generate_util
from util import DATA_DIR, SESSION_LEN, SESSION_OBJ_2D
from progress_learner import EventProgressEstimator

class Project(object):
	"""
	Just a simple object to store session and other statistics
	for one action type.

	This class encapsulates the data logic
	"""
	def __init__(self, name, session_names, config):
		self.session_names = session_names
		self.name = name
		self.sessions = []
		self.config = config

	def load_data(self):
		for session_name in self.session_names:
			session = load_one_param_file(os.path.join( DATA_DIR, self.name, session_name, 'files.param'))
			self.sessions.append(session)

	def preprocess(self):
		session_util.project_to2d(session, from_frame = 0, to_frame = session[SESSION_LEN])

	def __iter__(self):
		return iter(self.sessions)


	def standardize(self):
		session_util.interpolate_multi_object_data(session, object_names = session[SESSION_OBJ_2D].keys())

		self.down_sample_quotient = session_util.get_down_sample_quotient(project)
		print('down_sample_quotient = %d' % down_sample_quotient)

		self.speed = session_util.get_action_speed(project, down_sample_quotient)
		print (speed)

		self.sessions = session_util.down_sample(project, down_sample_quotient)

	def generate_data(self):
		# First step is to generate data with hop_step interpolation
		# rearranged_data = (samples, num_steps, data_point_size)
		# rearranged_lbls = (samples, num_steps)
		rearranged_data, rearranged_lbls = generate_util.turn_to_intermediate_data(project, 
			data_point_size, self.config.num_steps, self.config.hop_step)

		# Generate training and testing data 
		self.training_data, self.testing_data = generate_data(rearranged_data, rearranged_lbls, config)

	def save(self, file_path):
		with open(file_path, 'wb') as f:
            pickle.dump(self, 
                        f, pickle.HIGHEST_PROTOCOL)

        print('----Done saving project---')

    @staticmethod
	def load(file_path):
		with open(file_path, 'rb') as f: 
			return pickle.load(f)
		print('----Done loading project---') 

	

