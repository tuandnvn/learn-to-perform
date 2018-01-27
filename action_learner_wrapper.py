from rl import action_learner_search 

class Learner(object):
    """
    An interface for all kinds of leaner
    """
    def receive_state (self, *args):
        pass

    def produce_action (self, *args):
        pass

def continuous_learner_generator (wrapper, env):
	"""
	Wrapper is Action_Learner_Using_Search
	"""
	return action_learner_search.ActionLearner_Search(wrapper.config, wrapper.project, wrapper.progress_estimator, wrapper.limit_step, wrapper.session, env)

class Action_Learner_Using_Search (Learner):
    """
    This is just a wrapper around 
    ActionLearner_Search

    """
    def __init__(self, config, project, progress_estimator, learner_generator, limit_step = 10, session = None):
        """
        learner_generator :  (self, env) -> learner that supports .learn_one_setup
        """
        self.config = config

        # All of these components should be 
        # This should belong to class Project
        # We assume that the data put in the project here has been preprocessed
        self.project = project

        # This should be a kind of class EventProgressEstimator
        # We assume that the progress_estimator put in the project has been learned
        self.progress_estimator = progress_estimator

        self.limit_step = limit_step
        self.session = session

        self.learner_generator = learner_generator

        # Without receiving some initial states, we don't 
        self.wrapped_learner = None

    def receive_state (self, states):
        """
        states includes:

        states = (table, block1, block2)

        table : Center  (0,  0.7, 0.5)
                Extends (0.9, 0, 0.7)       
        
        block1 : Center (-0.3, 0.8, 1.0)
        block2 : Center (-0.3, 0.8, 1.0)

        Just ignore values in 0y
        """

        table, block1, block2 = states

        self.config.playground_x = [table[0][0] - table[1][0], table[0][2] - table[1][2], 0]
        self.config.playground_dim = [2 * table[1][0], 2 * table[1][2], 0]

        env = bme.BlockMovementEnv(self.config, self.project.speed, self.project.name, 
                progress_estimator = self.progress_estimator, session = self.session)

        env._reset_env()

        scale = self.config.block_size / 2

        o1 = Cube2D(transform = Transform2D([block1[0], block1[2]], 0, scale))
        env.add_object(o1)

        o2 = Cube2D(transform = Transform2D([block2[0], block2[2]], 0, scale))
        env.add_object(o2)

        self.wrapped_learner = self.learner_generator(self, env)

    def produce_action (self, select_object):
        explorations = self.wrapped_learner.learn_one_setup(select_object, verbose = True)

        action_coordinates = []

        # Produce the best exploration
        for _, _, transform, _, _, success, _, _ in explorations[0].action_storage:
            if success:
                _2d_coordinates = transform.position
                _3d_coordinates = [_2d_coordinates[0][0], 0.8, _2d_coordinates[0][1]]

                action_coordinates.append(_3d_coordinates)

        return (explorations[0], action_coordinates)
