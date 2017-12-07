import numpy as np
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from simulator.utils import Cube2D, Transform2D, Command
from ..feature_utils import qsr_feature_extractor, get_location_objects_most_active, get_most_active_objects_interval
from ..utils import SESSION_LEN, SESSION_OBJ_2D, SESSION_FEAT

class BlockMovementEnv(gym.Env):
    """
    This class encapsulate an environment and allow 
    """

    reward_range = (0, 1)
    metadata = {'render.modes': ['human']}
    """
    """
    def __init__(self, config, speed, name=None, progress_estimator = None):
        """
        Parameters:
        - name: the name of the event action to be learned
        - progress_estimator: is a function that produces
        a value between 0 and 1 (event progress function)
        
        
        **Note**
        progress_estimator:
        event progress function will be defined based on the event type 
        currently learned
        
        progress_estimator would be an LSTM
        
        
        """
        # This env is just a wrapper around an environment that 
        # I have created before


        """
        These values are from config
        - playground: a rectangle (x, y, rot, width, height, rot_range)
        where (x,y) is a corner of the rectangle
        - block_size: the default size for a block
        - n_objects: number of objects to be randomized
        - progress_threshold: condition for an episode to end
        """
        self.e = Environment()
        self.progress_estimator = progress_estimator
        self.n_objects = config.n_objects
        self.playground_x = config.playground_x
        self.playground_dim = config.playground_dim
        self.name = name
        self.block_size = config.block_size
        self.progress_threshold = config.progress_threshold
        self.num_steps = config.num_steps

        # Action space is dynamically created
        # The action space would be a combination of a 
        self.action_space = None
        # observation space is a subset of multiple object spaces
        self.observation_space = None 
        
        self._seed()
        
        self.object_space = Uniform(p = playground_x, 
                                         dimension = playground_dim, 
                                         randomizer = self.np_random)
        
        # frame need to be subtracted from previous segment of movement
        # should < self.speed

        # Store all succesful actions has been made
        # each action = (object_index, prev_transform, cur_transform)
        self.action_storage = []

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

        prev_transform = self.e.objects[object_index].transform

        if self.e.act(object_index, Command(position, rotation)):
            self.lastaction = action
            cur_transform = self.e.objects[object_index].transform
            self.action_storage.append( [(object_index, prev_transform, cur_transform)] )
        
        # captures the last self.num_steps + 1 frames
        # last_num_steps_frames is a SESSION
        last_num_steps_frames = self.capture_last(self.num_steps + 1)

        observation = self._get_observation(last_num_steps_frames)

        # Extract features from the last frames
        # inputs: np.array (self.num_steps, n_input)
        # because we can make last_num_steps_frames a little bit longer
        # so we just clip down a little bit
        inputs = self._get_features(last_num_steps_frames)[-self.num_steps:]

        # inputs: np.array (1, self.num_steps, n_input)
        inputs = np.expand_dims(inputs, axis = 0)
        current_progress = self.progress_estimator.predict(inputs)

        if current_progress > self.progress_threshold:
            # Finish action
            done = True
            reward = current_progress
            info = {}
        else:
            done = False
            reward = 0
            info = {}

        return (observation, reward, done, info)

    def capture_last(self, frames):
        """
        Capture movement of objects during the last number of frames
        frames should be num_step + 1

        In case there are not enough frames, we interpolate with the same object location

        Args:
        -----
            frames (int): # of frames

        Returns:
        --------
            a session structure to pass into feature_util.qsr_feature_extractor
            session[SESSION_OBJ_2D] 
            session[SESSION_LEN] = frames
        """
        session = {}
        session[SESSION_LEN] = frames
        session[SESSION_OBJ_2D] = {}

        for object_index in range(self.e.objects):
            session[SESSION_OBJ_2D][object_index] = []

        left_over_distance = 0

        captures = {}
        for i in range(self.n_objects):
            captures[i] = []

        frame_counter = 1

        # For the first frame, all objects are at the current positions
        for i in range(self.n_objects):
            captures[i].append(self.e.objects[i])

        for object_index, prev_transform, next_transform in self.action_storage[::-1]:
            obj = self.e.objects[object_index]

            path_distance = np.linalg.norm(prev_transform.position - next_transform.position)

            pos = self.speed - left_over_distance
            while pos < path_distance and frame_counter < frames:
                new_obj = obj.clone()

                interpolated_position = (pos / path_distance) * next_transform.position +\
                    (1 - pos/path_distance) * prev_transform.position
                interpolated_rotation = (pos / path_distance) * next_transform.rotation +\
                    (1 - pos/path_distance) * prev_transform.rotation

                new_obj.transform.position = interpolated_position
                new_obj.transform.rotation = interpolated_rotation

                captures[object_index].append(new_obj)

                # For static objects, just add the last frames
                for i in range(self.n_objects):
                    if i != object_index:
                        captures[i].append(captures[i][-1])

                frame_counter += 1

        # back to the beginning, just interpolate the last frame
        if frame_counter < frames:
            while frame_counter < frames:
                for i in range(self.n_objects):
                    captures[i].append(captures[i][-1])

                frame_counter += 1

        for object_index in range(self.n_objects):
            session[SESSION_OBJ_2D][object_index] = captures[object_index][::-1]

        return session

    def _get_observation(self, session):
        """
        Observation is calculated from the last positions (last two frames) of most salient objects

        """
        object_data = session[SESSION_OBJ_2D]
        object_1_name, object_2_name = get_most_active_objects_interval(object_data, object_data.keys(), 0, session[SESSION_LEN])

        features = []

        for name in [object_1_name, object_2_name]:
            for frame in [-2, -1]:
                features.append(object_data[name][frame].get_feat())

        return np.concatenate( features )

    def _get_features(self, session):
        """
        Features are calculated from session

        Args:
        -----
            session: Session type (dictionary of SESSION keys)

        Returns:
        --------
        """
        qsr_feature_extractor( session, get_location_objects = get_location_objects_most_active )

        return session[SESSION_FEAT]
    
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

        # Set the first observation
        observation = self._get_observation()

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