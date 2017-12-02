import numpy as np
import gym

class Uniform(gym.Space):
    """
    A uniform distributioin in a bounded 
    N-dimensional cube
    
    A sample usage:
    Create a square (-1,-1), (-1,1), (1,1), (1,-1)
    self.state_space = Uniform(p = [-1,-1], dimension = [2,2])
    """
    def __init__(self, p, dimension, randomizer = np.random):
        self.p = np.array(p)
        self.dimension = np.array(dimension)
        self.p_opposite = self.p + self.dimension
        self.randomizer = randomizer
        
    def sample(self):
        return self.randomizer.uniform(self.p, self.p_opposite)
    
    def __repr__(self):
        return "Uniform (p= " + str(self.p) + "; dimension = " + str(self.dimension) +")"
    
    def __eq__(self, other):
        return np.allclose(self.p, other.p) and \
                np.allclose(self.dimension, other.dimension)