from simulator.utils import Cube2D, Transform2D, Command
from rl import block_movement_env as bme
from importlib import reload



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