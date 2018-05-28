import numpy as np
import matplotlib.pyplot as plt


# import cartpool_discretized
# wrapped_environment = cartpool_discretized(environment)
# wrapped_environment.reset()

class Wrapper(object):
    def __init__(self, env):
        self.env = env
        self.apply_to = ['reset', 'step']
        
    def __getattr__(self, attr):
        env_attr = getattr(self.env, attr)
        if attr in self.apply_to and callable(res):
            return (lambda *a, **kw: self.__transformer__(env_attr(*a, **kw)))
        return env_attr

    def __transformer__(self, s):
        if s.shape != (4,):
            raise ValueError("Expected array of shape (4,). Instead got: %s"%(str(x.shape)))
        
        out = np.zeros((4,))

        # cart position range [-2.4 2.4]
        out[0] = np.round( 10 * (s[0]+2.4)/4.8, 0) # normalized to [0 1]
        
        # cart velocity [-inf +inf]
        out[1] = np.round( 10 * np.sqrt(np.abs(s[1])) * np.sign(s[1]), 0)
        # pole angle : [-41.8deg 41.8deg]
        out[2] = np.round( 10 * (s[2]+42)/84, 0) # normalized to [0 1] then multiplied by 10
        # pole velocity at tip
        out[3] = np.round( 10 * np.sqrt(np.abs(s[3])) * np.sign(s[3]), 0)
        return tuple(out)
