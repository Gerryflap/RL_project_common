# import random
# import numpy as np
#
# from version4.agent import Agent
# from version4.core import DiscreteEnvironment
# from version4.policy import EpsilonGreedy
# from version4.q_estimator import QEstimator
#
#
# class SarsaLambda(Agent):  TODO
#
#     def __init__(self, env: DiscreteEnvironment, features: callable, weights, lam: float = 0.2, eta: float = 0.01,
#                  gamma: float = 1.0):
#
#         super().__init__(env)
#         assert 0 <= lam <= 1
#         self.env = env
#         self.q = LinearWeights(x=features, w=weights)
#         self.policy = EpsilonGreedy(env.observation_space,
#                                     env.valid_actions,
#                                     self.q,
#                                     epsilon=lambda x: 0.05
#                                     )
#
#         self.lam = lam
#         self.eta = eta
#         self.x = features
#         self.w = weights
#         self.gamma = gamma
#
#     def learn(self, num_iter=100000) -> EpsilonGreedy:
#         q, w, x, pi = self.q.Q, self.w, self.x, self.policy
#         for _ in range(num_iter):
#             s = self.env.reset()
#             a = self.env.sample()
#
#             while not s.is_terminal():
#                 s_p, r = self.env.step(a)
#
#                 a_p = pi(s_p)
#
#                 delta = r + self.gamma * q(s_p, a_p) - q(s, a)
#
#                 for i, f in enumerate(x(s, a)):
#                     w[i] += self.eta * delta * f
#
#                 s, a = s_p, a_p
#         return pi
#
#
# class LinearWeights(QEstimator):
#
#     def __init__(self, w, x):
#         self.w, self.x = w, x
#
#     def Q(self, observation, action) -> float:
#         a = np.dot(self.x(observation, action), self.w)
#         print(a)
#         return a
#
#     def Qs(self, observation, actions) -> dict:
#         return {action: self.Q(observation, action) for action in actions}
#
#
# if __name__ == '__main__':
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from version4.environments.easy21 import Easy21, Easy21Observation
#
#     def easy21_feature_vector(s, a):
#         i = 0
#         fs = np.zeros(36)
#         for d_interval in [range(1, 5), range(4, 8), range(7, 11)]:
#             for p_interval in [range(1, 7), range(4, 10), range(7, 13), range(10, 16), range(13, 19), range(16, 22)]:
#                 for _a in [True, False]:
#                     fs[i] = 1 if s.p_sum in p_interval and s.d_sum in d_interval and a == _a else 0
#                     i += 1
#         return fs.T
#
#     _env = Easy21()
#
#     procedure = SarsaLambda(_env, features=easy21_feature_vector, weights=np.zeros(36), lam=0.2, eta=0.01)
#
#     _q = procedure.learn()
#
#     vs = np.zeros(shape=(21, 10))
#
#     for p_sum in range(1, 22):
#         for d_sum in range(1, 11):
#             _s = Easy21Observation(p_sum, d_sum, False)
#             vs[p_sum - 1, d_sum - 1] = max([procedure.q.Q(_s, _a) for _a in _env.valid_actions()])
#
#     plt.imshow(vs)
#     plt.show()
