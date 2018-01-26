from simulator.utils import Cube2D, Transform2D, Command
from rl import action_learner_search 
from rl import block_movement_env as bme
from importlib import reload

reload(action_learner_search)

class Learner(object):
    """
    An interface for all kinds of leaner
    """
    def receive_state (self, *args):
        pass

    def produce_action (self, *args):
        pass

class Action_Learner_Using_Search (Learner):
    """
    This is just a wrapper around 
    ActionLearner_Search

    """
    def __init__(self, config, project, progress_estimator, 
            policy_estimator, action_policy, limit_step = 10, session = None):
        self.config = config

        # All of these components should be 
        # This should belong to class Project
        # We assume that the data put in the project here has been preprocessed
        self.project = project

        # This should be a kind of class EventProgressEstimator
        # We assume that the progress_estimator put in the project has been learned
        self.progress_estimator = progress_estimator

        # This should belong to class PolicyEstimator
        # This is actually not really related,
        # but let's just it be here
        self.policy_estimator = policy_estimator

        self.action_policy = action_policy

        self.limit_step = limit_step
        self.session = session

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

        self.wrapped_learner = action_learner_search.ActionLearner_Search(self.config, self.project, self.progress_estimator, 
            self.policy_estimator, self.limit_step, self.session, env)

    def produce_action (self, select_object):
        explorations = self.wrapped_learner.learn_one_setup(self.action_policy, select_object, verbose = True)

        action_coordinates = []

        # Produce the best exploration
        for _, _, transform, _, _, success, _, _ in explorations[0].action_storage:
            if success:
                _2d_coordinates = transform.position
                _3d_coordinates = [_2d_coordinates[0][0], 0.8, _2d_coordinates[0][1]]

                action_coordinates.append(_3d_coordinates)

        return (explorations[0], action_coordinates)

class Ensemble_Learner(object):
    def __init__(self, config, action_types, projects, progress_estimators, 
            policy_estimator, action_policy, limit_step = 10, session = None):
        """
        projects : Dictionary from action type to processed Project
        progress_estimators: Dictionary from action type to learned progress estimator
        policy_estimator, action_policy: could use the same one for all action types
        """
        self.all_learners = {}
        self.action_types = action_types
        for action_type in self.action_types:
            self.all_learners[action_type] = Action_Learner_Using_Search(config, projects[action_type], progress_estimators[action_type], 
                policy_estimator, action_policy, limit_step, session)

    def receive_state (self, states):
        for action_type in self.action_types:
            self.all_learners[action_type].receive_state(states)

    def produce_action (self, action_type, select_object):
        exploration, action_coordinates = self.all_learners[action_type].produce_action(select_object)

        def make_command_str( action ):
            command_str = "slide(block%d,<%.2f; %.2f; %.2f>)"

            cstr = command_str % ( select_object + 1, action[0], action[1], action[2] )

            return cstr

        cstrs = []
        for action in action_coordinates:
            cstrs.append( make_command_str(action) )

        return (exploration, ';;'.join(cstrs))