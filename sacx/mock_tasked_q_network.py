import numpy as np
import sacx.generic_tasked_q_network as genq


class QNetwork(genq.QNetwork):
    def __init__(self):
        pass

    def Qs(self, state, task):
        pass  # TODO

    def Q(self, state, action, task):
        pass  # TODO

    def Q_array(self, states, task):
        pass  # TODO: Should get a batch of states and return all live Qsa values for these states

    def Qp_array(self, states, task):
        ret = np.zeros((states.shape[0], 3))
        ret[:,task] = 1
        return ret
        pass  # TODO: Should get a batch of states and return all fixed Qsa values for these states

    def train(self, trajectories):
        pass  # TODO
