from collections import defaultdict

import q_table as q_table

from agent import Agent
from core import DiscreteEnvironment
from policy import EpsilonGreedy


class SarsaLambda(Agent):
    """
        Sarsa-lambda Agent implementation
    """

    def __init__(self, env: DiscreteEnvironment, lam: float = 0.2, gamma: float = 1.0):
        """
        Create a new SarsaLambda Agent
        :param env: The environment the agent will learn from
        :param lam: The lambda parameter
        :param gamma: Reward discount factor
        """
        super().__init__(env)
        assert 0 <= lam <= 1

        self.q_table = q_table.for_env(env)
        self.visit_count = defaultdict(int)
        self.eligibility_trace = defaultdict(int)
        self.policy = EpsilonGreedy(env.observation_space,
                                    env.valid_actions,
                                    self.q_table,
                                    self.epsilon
                                    )
        self.env = env
        self.lam = lam
        self.gamma = gamma

    def learn(self, num_iter=100000) -> EpsilonGreedy:
        """
        Learn a policy from the environment
        :param num_iter: The number of iterations the algorithm should run
        :return: the derived policy
        """
        N, Q, E, pi = self.visit_count, self.q_table, self.eligibility_trace, self.policy
        for _ in range(num_iter):
            E.clear()
            s = self.env.reset()
            a = self.env.sample()

            N[s] += 1
            N[s, a] += 1

            while not s.is_terminal():
                s_p, r = self.env.step(a)
                N[s_p] += 1

                a_p = pi(s)

                E[s, a] += 1
                N[s_p, a_p] += 1

                delta = r + self.gamma * Q[s_p, a_p] - Q[s, a]
                for k in E.keys():
                    Q[k] += (1 / N[k]) * delta * E[k]
                    E[k] *= self.gamma * self.lam

                s, a = s_p, a_p
        return pi

    def epsilon(self, s):
        N_0, N = 100, self.visit_count
        return N_0 / (N_0 + N[s])


class MonteCarlo(SarsaLambda):
    """
        SarsaLambda with lambda=1 is equivalent to MonteCarlo
    """

    def __init__(self, env: DiscreteEnvironment):
        super().__init__(env, lam=1)


class TD0(SarsaLambda):
    """
        SarsaLambda with lambda=0 is equivalent to TD(0)
    """

    def __init__(self, env: DiscreteEnvironment):
        super().__init__(env, lam=0)


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from environments.easy21 import Easy21

    _env = Easy21()

    procedure = SarsaLambda(_env, lam=0.2)

    q = procedure.learn(num_iter=1000000)

    table = procedure.q_table

    print(table)

    vs = np.zeros(shape=(21, 10))

    for (state, action), value in table.items():
        vs[state.p_sum - 1, state.d_sum - 1] = max([table[state, a] for a in _env.valid_actions()])

    plt.imshow(vs)
    plt.show()
