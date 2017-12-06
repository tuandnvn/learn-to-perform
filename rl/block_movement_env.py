import numpy as np
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from simulator.utils import Cube2D, Transform2D, Command

class BlockMovementEnv(gym.Env):
    """
    This class encapsulate an environment and allow 
    """

    reward_range = (0, 1)
    metadata = {'render.modes': ['human']}
    """
    """
    def __init__(self, target, playground_x = [-1,-1, 0],
                 playground_dim = [2, 2, np.pi/2], name=None, n_objects = 2,
                block_size = 0.15, 
                 progress_threshold = 0.9, progress_estimator = None):
        """
        Parameters:
        - name: the name of the event action to be learned
        - target: is a function that produces
        a value between 0 and 1 (event progress function)
        - playground: a rectangle (x, y, rot, width, height, rot_range)
        where (x,y) is a corner of the rectangle
        - block_size: the default size for a block
        - n_objects: number of objects to be randomized
        - progress_threshold: condition for an episode to end
        
        **Note**
        target_function:
        event progress function will be defined based on the event type 
        currently learned
        
        target_function would be an LSTM
        
        
        """
        # This env is just a wrapper around an environment that 
        # I have created before
        self.e = Environment()
        self.progress_estimator = progress_estimator
        self.target = target
        self.n_objects = n_objects
        self.playground_x = playground_x
        self.playground_dim = playground_dim
        self.name = name
        self.block_size = block_size
        self.progress_threshold = progress_threshold
        
        # Action space is dynamically created
        # The action space would be a combination of a 
        self.action_space = None
        # observation space is a subset of multiple object spaces
        self.observation_space = None 
        
        self._seed()
        
        self.object_space = Uniform(p = playground_x, 
                                         dimension = playground_dim, 
                                         randomizer = self.np_random)
        
        # This list would store 
        self.features = []

        self._reset()
        
    def _step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the environment
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        # action is generated from the action_policy (external to the environment)
        object_index, new_location = action
        
        position = new_location[:2]
        rotation = new_location[2]
        scale = self.inner_state[object_index].transform.scale
        
        # self.inner_state is a map of index to object
        if self.e.act(object_index, Command(position, rotation)):
            self.lastaction = action
        
        observation = self._get_observation()
        current_progress = self.target.predict()

        (observation, reward, done, info)

        return (self.inner_state)

    def capture(self):
        # This is copied from simulator2d
        # TODO: change this
        # frame need to be subtracted from previous segment of movement
        # should < self.speed
        left_over_distance = 0.0

        path_distance = norm(obj.transform.position - original_transform.position)

        print ('path_distance = %.2f' % path_distance)
        pos = self.speed - left_over_distance
        while pos < path_distance:
            print ('pos = %.2f' % pos)
            new_obj = obj.clone()

            interpolated_position = (pos / path_distance) * original_transform.position +\
                (1 - pos/path_distance) * obj.transform.position
            interpolated_rotation = (pos / path_distance) * original_transform.rotation +\
                (1 - pos/path_distance) * obj.transform.rotation

            new_obj.transform.position = interpolated_position
            new_obj.transform.rotation = interpolated_rotation

            captures.append(new_obj.get_markers())

            # increase step
            pos += self.speed

        left_over_distance = self.speed + path_distance - pos
        print ('After %s' % obj)

    def _get_observation(self):
        """
        Observation is calculated from the inner state

        TODO: implement this observation
        """
        return None
    
    def _reset(self):
        # states would be a list of location/orientation for block
        # sampled from the observation space
        for i in range(self.n_objects):
            # Retry randomize a few times
            retry = 0
            
            while True:
                obj_params = self.object_space.sample()

                position = obj_params[:2]
                rotation = obj_params[2]
                scale = self.block_size / 2

                o = Cube2D(transform = Transform2D(position, rotation, scale))

                if self.e.add_object(o):
                    break
                    
                if retry > 3:
                    break
                
                retry += 1
            
        self.lastaction=None
        self.inner_state = self.e.objects

        # Set the first observation
        observation = self._get_observation

        return observation

    def _render(self, mode='human', close=False):
        if close:
            return
        
        fig, ax = plt.subplots()
        ax.set_xticks(np.arange(self.playground_x[0], 
                                self.playground_x[0] + self.playground_dim[0], 0.1))
        ax.set_yticks(np.arange(self.playground_x[1], 
                                self.playground_x[1] + self.playground_dim[1], 0.1))
        ax.set_xlim([self.playground_x[0], 
                     self.playground_x[0] + self.playground_dim[0]])
        ax.set_ylim([self.playground_x[1], 
                     self.playground_x[1] + self.playground_dim[1]])
        fig.set_size_inches(20, 12)
        
        for i in range(self.n_objects):
            # Obj is action position and rotation of object
            obj = self.e.objects[i]
            
            shape = obj.get_markers()
            
            lines = make_lines(shape)
            lc = mc.LineCollection(lines, colors=colors[i], linewidths=2)
            ax.add_collection(lc)
        
        ax.autoscale()
        ax.margins(0.1)

        plt.show()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]