import numpy as np


class QNetwork:

    def Qs(self, state, task):
        pass  # TODO

    def Q(self, state, action, task):
        pass  # TODO

    def Q_array(self, states, task):
        pass # TODO: Should get a batch of states and return all live Qsa values for these states

    def Qp_array(self, states, task):
        ret = np.zeros((states.shape[0], 3))
        ret[:,0] = 1
        return ret
        pass  # TODO: Should get a batch of states and return all fixed Qsa values for these states

    def train(self):
        pass  # TODO
