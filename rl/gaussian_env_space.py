import numpy as np
import gym

class Gaussian(gym.Space):
    """
    A Gaussian space randomizes an action as a datapoint
    using a location and a covariance.
    
    This is actually a multivariate normal distribution (MVN),
    but with non-correlated variables 
    (the covariance matrix is diagonal and positive)
    
    A sample usage:
    self.action_space = Gaussian(location = [-1,2], diagonal_cov = [1,1])
    """
    def __init__(self, location, diagonal_cov, n_objects = 2, shape=None):
        """
        Two kinds of valid inputs
        
        - location and diagonal_cov are scalar -> Gaussian distribution
        - location and diagonal_cov are np array of same size
        """
        self.n_objects = n_objects
        
        if np.isscalar(location) and np.isscalar(diagonal_cov):
            """Gaussian distribution"""
            self.location = np.array([location])
            self.diagonal_cov = np.array([diagonal_cov])
            self.shape = (1,)
        elif isinstance(location, list) and isinstance(diagonal_cov, list):
            assert len(location) == len(diagonal_cov)
            
            self.location = np.array(location)
            self.diagonal_cov = np.diag(diagonal_cov)
            
            self.shape = self.location.shape
        else:
            assert isinstance(location, np.ndarray)
            assert isinstance(diagonal_cov, np.ndarray)
            assert location.shape == diagonal_cov.shape
        
            self.shape = location.shape
            
            self.location = np.flatten(location)
            self.diagonal_cov = np.diag(np.flatten(diagonal_cov))
            
    def sample(self, object_index = None):
        """
        sample an action to take:
        
        if object_index == None:
            sample both object_index and location of final point
        else:
            sample jus the location of final point
        """
        s = np.random.multivariate_normal(self.location, self.diagonal_cov)
        
        # Reshape to original 
        s.shape = self.shape
        
        if object_index:
            return (object_index, s)
        else:
            object_index = np.random.choice(self.n_objects)
            return (object_index, s)
            
    def __repr__(self):
        return "MVN (location= " + str(self.location) + "; variances = " + str(self.diagonal_cov) +")"
    
    def __eq__(self, other):
        return np.allclose(self.location, other.location) and \
                np.allclose(self.diagonal_cov, other.diagonal_cov)