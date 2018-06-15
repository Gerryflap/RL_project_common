import numpy as np

from agent import Agent
from core import FiniteActionEnvironment
from policy import EpsilonGreedyPolicy
from q_estimator import QEstimator


class SarsaLambda(Agent):
    """
        Sarsa Lambda agent using linear function approximation
    """

    def __init__(self,
                 env: FiniteActionEnvironment,
                 features: callable,
                 weights,
                 lam: float = 0.2,
                 eta: float = 0.01,
                 gamma: float = 1.0
                 ):
        """
        Create a new SarsaLambda Agent
        :param env: The environment the agent will learn from
        :param features: A function that, when given an input state-action pair, computes a feature vector
        :param weights: An array of weights corresponding to each feature
        :param lam: The lambda parameter
        :param eta: Step-size parameter
        :param gamma: Reward discount factor
        """
        super().__init__(env)
        assert 0 <= lam <= 1
        self.env = env
        self.q = LinearWeights(x=features, w=weights)
        self.policy = self.q.derive_policy(EpsilonGreedyPolicy,
                                           env.valid_actions_from,
                                           epsilon=lambda s: 0.05
                                           )
        self.lam = lam
        self.eta = eta
        self.x = features
        self.w = weights
        self.gamma = gamma

    def learn(self, num_iter=100000) -> EpsilonGreedyPolicy:
        """
        Learn a policy from the environment
        :param num_iter: The number of iterations the algorithm should run
        :return: the learned policy
        """
        q, w, x, pi = self.q.Q, self.w, self.x, self.policy
        for _ in range(num_iter):
            s = self.env.reset()
            a = self.env.sample()
            while not s.is_terminal():
                s_p, r = self.env.step(a)
                a_p = pi.sample(s_p)
                delta = r + self.gamma * q(s_p, a_p) - q(s, a)
                for i, f in enumerate(x(s, a)):
                    w[i] += self.eta * delta * f
                s, a = s_p, a_p
        return pi


class LinearWeights(QEstimator):
    """
        Lists of weights and features that are used to estimate a Q value for a state-action pair
    """

    def __init__(self, w, x):
        """
        Create a new LinearWeights QEstimator
        :param w: The initial weights
        :param x: A function that computes a feature vector when given a state-action pair
        """
        self.w, self.x = w, x

    def Q(self, state, action) -> float:
        """
        Estimate a Q-value for the state-action pair by multiplying the weights with the corresponding features
        :param state: The state for which the Q-value should be estimated
        :param action: The action for which the Q-value should be estimated
        :return: the estimated Q-value
        """
        q = np.dot(self.x(state, action), self.w)
        if isinstance(q, float):
            return q
        else:
            raise Exception('Feature vector and weights dimensions do not match!')

    def Qs(self, state, actions) -> dict:
        """
        Estimate Q-values for all state-action pairs
        :param state: The state for which the Q-value should be estimated
        :param actions: A list of actions for which the Q-values should be estimated
        :return: a dictionary mapping all actions to their estimated Q-value
        """
        return {action: self.Q(state, action) for action in actions}


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from environments.easy21 import Easy21, Easy21State

    def easy21_feature_vector(s, a):
        """
        Easy21 feature vector as proposed in the Easy21 exercise
        :param s: The state for which features should be computed
        :param a: The action for which features should be computed
        :return: a feature vector corresponding to the state-action pair
        """
        i = 0
        fs = np.zeros(36)
        for d_interval in [range(1, 5), range(4, 8), range(7, 11)]:
            for p_interval in [range(1, 7), range(4, 10), range(7, 13), range(10, 16), range(13, 19), range(16, 22)]:
                for _a in Easy21.ACTIONS:
                    fs[i] = 1 if s.p_sum in p_interval and s.d_sum in d_interval and a == _a else 0
                    i += 1
        return fs.T

    _env = Easy21()

    procedure = SarsaLambda(_env, features=easy21_feature_vector, weights=np.zeros(36), lam=0.2, eta=0.01)

    _q = procedure.learn()

    vs = np.zeros(shape=(21, 10))

    for p_sum in range(1, 22):
        for d_sum in range(1, 11):
            _s = Easy21State(p_sum, d_sum, False)
            vs[p_sum - 1, d_sum - 1] = max([procedure.q.Q(_s, _a) for _a in _env.valid_actions()])

    plt.imshow(vs)
    plt.show()
