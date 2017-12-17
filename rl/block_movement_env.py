import numpy as np
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import pylab as pl
from matplotlib import collections as mc

from . import uniform_env_space
from simulator import simulator2d
from simulator.utils import Cube2D, Transform2D, Command
import feature_utils
from utils import SESSION_LEN, SESSION_OBJ_2D, SESSION_FEAT

def make_lines(shape):
    lines = []
    for i in range(len(shape)):
        j = (i + 1) % len(shape)
        lines.append( [ shape[i], shape[j] ] )
    
    return lines

colors = [ (1, 0, 0, 1), (0,1,0,1), (0,0,1,1), 
          (0.5, 0.5, 0, 1), (0,0.5, 0.5,1), (0.5, 0, 0.5,1),
         (0.7, 0.3, 0, 1), (0,0.7, 0.3,1), (0.7, 0, 0.3,1),
         (0.3, 0.7, 0, 1), (0,0.3, 0.7,1), (0.3, 0, 0.7,1)]

from importlib import reload
reload (simulator2d)
# reload(feature_utils)

SPEED = 'SPEED'
WHOLE = 'WHOLE'

class BlockMovementEnv(gym.Env):
    """
    This class encapsulate an environment, allowing checking constraints of the environment
    
    """

    reward_range = (0, 1)
    metadata = {'render.modes': ['human']}
    """
    """
    def __init__(self, config, speed, name=None, 
        progress_estimator = None, graph_size = 8, session = None):
        """
        Parameters:
        - name: the name of the event action to be learned
        - progress_estimator: is a function that produces
        a value between 0 and 1 (event progress function)
        - config: A config object 
        - graph_size: How big you want to render the environment
        - session: (Optional) a tensorflow session
        
        **Note**
        progress_estimator:
        event progress function will be defined based on the event type 
        currently learned. progress_estimator would be an LSTM
        
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
        self.default_boundary = Cube2D(
            transform = Transform2D(position=[0.0, 0.0], rotation=0.0, scale = 1.0))
        self.e = simulator2d.Environment( boundary = self.default_boundary )
        self.config = config
        self.progress_estimator = progress_estimator
        self.n_objects = config.n_objects
        self.playground_x = config.playground_x
        self.playground_dim = config.playground_dim
        self.name = name
        self.block_size = config.block_size
        self.progress_threshold = config.progress_threshold
        self.num_steps = config.num_steps
        self.graph_size = graph_size
        self.speed = speed
        self.session = session

        # Action space is dynamically created
        self.action_space = None
        # observation space is a subset of multiple object spaces
        self.observation_space = None 
        
        self._seed()
        
        # Just hard code
        playground_x = [self.block_size-1,self.block_size-1, 0]
        playground_dim = [2-2*self.block_size, 2-2*self.block_size, np.pi/2]
        self.object_space = uniform_env_space.Uniform(p = playground_x, 
                                         dimension = playground_dim, 
                                         randomizer = self.np_random)
        
        # frame need to be subtracted from previous segment of movement
        # should < self.speed

        # Store all succesful actions has been made
        # each action = (object_index, prev_transform, cur_transform, resulted_observation, resulted_progress)
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
        object_index, new_location, action_means, action_stds = action
        
        position = new_location[:2]
        rotation = new_location[2]

        prev_transform = self.e.objects[object_index].transform

        if len(self.action_storage) > 0:
            last_progress = self.action_storage[-1][4]
        else:
            last_progress = 0

        if self.e.act(object_index, Command(position, rotation)):
            # print ('Action accepted')
            cur_transform = self.e.objects[object_index].transform
            # I need to call self.action_storage.append before get_observation_and_progress
            self.action_storage.append( [object_index, prev_transform, cur_transform, None, None, True, action_means, action_stds] )
            observation, progress = self.get_observation_and_progress()
            self.action_storage[-1][3:5] = [observation, progress]
        else:
            if len(self.action_storage) > 0:
                # Just return observation and progress of last action
                _, _, _, observation, progress, _, _, _ = self.action_storage[-1]
            else:
                # First action failed
                observation, _ = self.get_observation_and_progress()
                progress = 0
            
            self.action_storage.append( [object_index, prev_transform, prev_transform, observation, progress, False, action_means, action_stds] )

        info = {}

        # Typical threshold approach
        if progress > self.progress_threshold:
            # Finish action
            done = True
        else:
            done = False
        
        reward = progress - last_progress
        #print ('Progress = %.2f ; reward = %.2f' % (progress, reward))

        return (observation, reward, done, info)

    def back(self):
        """
        Back one step

        This allows multiple tries of an action at the same state
        """
        if len(self.action_storage) == 0:
            """No memory"""
            return False

        object_index, prev_transform, cur_transform, _, _, success, _, _ = self.action_storage[-1]

        "Only if the last action succeeded, we back up, otherwise just delete it"
        if success:
            # Assumption is that we can always do this
            act_back = self.e.act(object_index, 
                self.command_from_transform(prev_transform), check_condition = False )

        del self.action_storage[-1]

        return True

    def command_from_transform(self, transform ):
        position = transform.position.flatten()
        rotation = transform.rotation

        return Command(position, rotation)

    def get_observation_and_progress(self, verbose = False):
        # captures the last self.num_steps + 1 frames
        # last_num_steps_frames is a SESSION
        last_num_steps_frames = self.capture_last(self.num_steps + 1)

        observation = self._get_observation(last_num_steps_frames)

        # Extract features from the last frames
        # inputs: np.array (self.num_steps, n_input)
        # because we can make last_num_steps_frames a little bit longer
        # so we just clip down a little bit
        inputs = self._get_features(last_num_steps_frames)[-self.num_steps:]

        if verbose:
            print (inputs)

        # inputs: np.array (config.batch_size, self.num_steps, n_input)
        inputs = np.repeat(np.expand_dims(inputs, axis = 0), self.config.batch_size, axis = 0)
        current_progress = self.progress_estimator.predict(inputs, sess = self.session)

        progress = current_progress[0]
        # print ('progress = %.2f' % progress)

        return (observation, progress)

    def capture_last(self, frames, mode = WHOLE):
        """
        Capture movement of objects during the last number of frames
        frames should be num_step + 1

        In case there are not enough frames, we interpolate with the same object location

        Args:
        -----
            frames (int): # of frames
            mode: important flag, if mode == 'SPEED' -> Always interpolate depend on project.speed (It would not work when scale increase)
                                                    because it seems like most of the time it just give the progress score for very last frames
                                  if mode == 'WHOLE' -> Always interpolate from begining -> should be better because it explains the whole process 
                                                    that the user see.

        Returns:
        --------
            a session structure to pass into feature_util.qsr_feature_extractor
            session[SESSION_OBJ_2D] 
            session[SESSION_LEN] = frames
        """
        session = {}
        session[SESSION_LEN] = frames
        session[SESSION_OBJ_2D] = {}

        for object_index in range(len(self.e.objects)):
            session[SESSION_OBJ_2D][object_index] = []

        # must < self.speed
        left_over_distance = 0

        captures = {}
        for i in range(self.n_objects):
            captures[i] = []

        frame_counter = 1

        # For the first frame, all objects are at the current positions
        for i in range(self.n_objects):
            captures[i].append(self.e.objects[i])

        if mode == SPEED:
            for object_index, prev_transform, next_transform, _, _, success, _, _ in self.action_storage[::-1]:
                if not success:
                    continue
                obj = self.e.objects[object_index]

                path_distance = np.linalg.norm(prev_transform.position - next_transform.position)

                pos = self.speed - left_over_distance
                while pos < path_distance and frame_counter < frames:
                    new_obj = obj.clone()

                    interpolated_position = (pos / path_distance) * prev_transform.position +\
                        (1 - pos/path_distance) * next_transform.position
                    interpolated_rotation = (pos / path_distance) * prev_transform.rotation +\
                        (1 - pos/path_distance) * next_transform.rotation

                    new_obj.transform.position = interpolated_position
                    new_obj.transform.rotation = interpolated_rotation

                    captures[object_index].append(new_obj)

                    # For static objects, just add the last frames
                    for i in range(self.n_objects):
                        if i != object_index:
                            captures[i].append(captures[i][-1])

                    pos += self.speed
                    frame_counter += 1

                # We have enough frames, don't need to trace back anymore
                if frame_counter >= frames:
                    break
                else:
                    # pos >= path_distance
                    # recalculate left_over_distance for next action
                    left_over_distance = pos - path_distance
        elif mode == WHOLE:
            # First pass:
            # Calculate the total travelling distance
            total_path_distance = 0.0
            for object_index, prev_transform, next_transform, _, _, success, _, _ in self.action_storage[::-1]:
                if not success:
                    continue
                obj = self.e.objects[object_index]

                path_distance = np.linalg.norm(prev_transform.position - next_transform.position)

                total_path_distance += path_distance

            # Actual speed
            frame_distance = total_path_distance / (frames - 1)

            for object_index, prev_transform, next_transform, _, _, success, _, _ in self.action_storage[::-1]:
                # After each loop, left_over_distance < frame_distance
                if not success:
                    continue
                obj = self.e.objects[object_index]

                path_distance = np.linalg.norm(prev_transform.position - next_transform.position)

                pos = frame_distance - left_over_distance

                while pos < path_distance:
                    new_obj = obj.clone()

                    interpolated_position = (pos / path_distance) * prev_transform.position +\
                        (1 - pos/path_distance) * next_transform.position
                    interpolated_rotation = (pos / path_distance) * prev_transform.rotation +\
                        (1 - pos/path_distance) * next_transform.rotation

                    new_obj.transform.position = interpolated_position
                    new_obj.transform.rotation = interpolated_rotation

                    captures[object_index].append(new_obj)

                    # For static objects, just add the last frames
                    for i in range(self.n_objects):
                        if i != object_index:
                            captures[i].append(captures[i][-1])

                    pos += frame_distance
                    frame_counter += 1

                # 0 <= left_over_distance < frame_distance
                left_over_distance = pos - path_distance

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
        
        # just return position and rotation
        """
        object_data = session[SESSION_OBJ_2D]
        sess_len = session[SESSION_LEN]

        object_1_name, object_2_name = feature_utils.get_most_active_objects_interval(object_data, object_data.keys(), 0, sess_len)

        features = []

        for name in [object_1_name, object_2_name]:
            for frame in [-2, -1]:
                object_data[name][frame].transform.position
                features.append(object_data[name][frame].transform.get_feat())

        return np.concatenate( features ).flatten()

    def _get_features(self, session):
        """
        Features are calculated from session

        Args:
        -----
            session: Session type (dictionary of SESSION keys)

        Returns:
        --------
            session[SESSION_FEAT] : np.array (# frames, # features)
        """
        feature_utils.qsr_feature_extractor( session, get_location_objects = feature_utils.get_location_objects_most_active )

        return session[SESSION_FEAT]
    
    def _reset(self):
        self.e = simulator2d.Environment(boundary = self.default_boundary )
        self.action_storage = []

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
                    
                if retry > 10:
                    break
                
                retry += 1
            

        last_frames = self.capture_last(frames = 2, mode = SPEED)

        # Set the first observation
        observation = self._get_observation(last_frames)

        return observation

    def default(self):
        """
        This reset the environment to a default testing state where
        locations of objects are predefined
        """
        self.e = simulator2d.Environment(boundary = self.default_boundary )
        self.action_storage = []
        scale = self.block_size / 2

        o = Cube2D(transform = Transform2D([-0.71322928, -0.68750558], 0.50, scale))
        self.e.add_object(o)
        o = Cube2D(transform = Transform2D([-0.5, 0.3], 0.60, scale))
        # o = Cube2D(transform = Transform2D([-0.2344808, -0.16797299], 0.60, scale))
        self.e.add_object(o)

        last_frames = self.capture_last(frames = 2, mode = SPEED)

        # Set the first observation
        observation = self._get_observation(last_frames)

        return observation

    def default_action(self):
        """
        Use for default setup, for debugging purpose
        """
        self._step((0, [ -0.75, 0.5,  0.5], None, None))
        self._step((0, [ -0.3, 0.55,  0.5], None, None))
        self._step((0, [ -0.2, 0.15,  0.5], None, None))
        self._step((0, [ -0.5, 0.0,  0.5], None, None))

    def replay(self, verbose = True):
        """
        For debugging purpose, we want to replay the session (just showing all the steps has been made from the beginning and progress values)
        """
        action_storage_clone = self.action_storage[:]
        while self.back():
            continue

        prev_graph_size = self.graph_size

        # Resize to make it smaller
        self.graph_size = self.graph_size / 2

        self._render()
        for object_index, _, next_transform, _, _, success, action_means, action_stds in action_storage_clone:
            if not success:
                continue

            print ((action_means, action_stds))
            print (next_transform)
            self.step((object_index, next_transform.get_feat(), action_means, action_stds))

            _, progress = self.get_observation_and_progress(verbose = verbose)

            print ("Progress = %.2f" % progress)
            self._render(action_means = action_means, action_stds = action_stds)

        self.graph_size = prev_graph_size

    def animate(self):
        """
        For debugging purpose, we want to replay the session
        """
        pass

    def _render(self, mode='human', close=False, action_means = None, action_stds = None):
        if close:
            return
        
        fig, ax = plt.subplots()
        fig.set_size_inches(self.graph_size, self.graph_size)
        x_range = np.arange(self.playground_x[0], 
                                self.playground_x[0] + self.playground_dim[0], 0.1)
        y_range = np.arange(self.playground_x[1], 
                                self.playground_x[1] + self.playground_dim[1], 0.1)
        ax.set_xticks(x_range)
        ax.set_yticks(y_range)
        ax.set_xlim(self.playground_x[0], 
                                self.playground_x[0] + self.playground_dim[0])
        ax.set_ylim(self.playground_x[1], 
                                self.playground_x[1] + self.playground_dim[1])
        
        lc = mc.PolyCollection([self.e.objects[i].get_markers() for i in range(self.n_objects)], 
                               edgecolors=[colors[i] for i in range(self.n_objects)], 
                               facecolors=[colors[i] for i in range(self.n_objects)], linewidths=[2,2])

        ax.add_collection(lc)
        
        # ax.autoscale()
        ax.margins(0.1)

        if not action_means is None  and not action_stds is None:
            X, Y = np.meshgrid(x_range, y_range)
            z = mlab.bivariate_normal(X, Y, action_stds[0], action_stds[1],
                 action_means[0], action_means[1])
            plt.contour(x_range, y_range, z, 10, alpha = 0.3)

        plt.show()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]