from collections import defaultdict

from agent import Agent
from core import FiniteActionEnvironment
from policy import EpsilonGreedyPolicy
from q_table import QTable


class SarsaLambda(Agent):
    """
        Sarsa-lambda Agent implementation
    """

    def __init__(self,
                 env: FiniteActionEnvironment,
                 lam: float = 0.2,
                 gamma: float = 1.0,
                 epsilon=0.05,
                 epsilon_step_factor=1.0,
                 epsilon_min=0.0,
                 fex: callable=lambda x: x
                 ):
        """
        Create a new SarsaLambda Agent
        :param env: The environment the agent will learn from
        :param lam: The lambda parameter
        :param gamma: Reward discount factor
        :param fex: Optional feature extraction from observation states
        """
        super().__init__(env)
        assert 0 <= lam <= 1

        self.q_table = QTable()
        self.visit_count = defaultdict(int)
        self.eligibility_trace = defaultdict(int)
        self.policy = self.q_table.derive_policy(EpsilonGreedyPolicy,
                                                 env.valid_actions_from,
                                                 epsilon=self.epsilon)
        self.lam = lam
        self.gamma = gamma

        self.epsilon_step_factor = epsilon_step_factor
        self.epsilon_min = epsilon_min
        self.epsilon_v = epsilon

        
        self.fex = fex

    def learn(self, num_iter=100000, result_handler=None) -> EpsilonGreedyPolicy:
        """
        Learn a policy from the environment
        :param num_iter: The number of iterations the algorithm should run
        :return: the derived policy
        """
        N, Q, E, pi = self.visit_count, self.q_table, self.eligibility_trace, self.policy
        for _ in range(num_iter):
            E.clear()
            s, terminal = self.env_reset()
            a = self.env.sample()
            sum_reward = 0
            N[s] += 1
            N[s, a] += 1

            while not terminal:
                s_p, r, terminal = self.env_step(a)
                N[s_p] += 1
                sum_reward += r
                a_p = pi.sample(s)

                E[s, a] += 1
                N[s_p, a_p] += 1

                delta = r + self.gamma * Q[s_p, a_p] - Q[s, a]
                for k in E.keys():
                    # learning rate decays due to 1/N[k]
                    Q[k] += (1 / N[k]) * delta * E[k]
                    #Q[k] += 0.1 * delta * E[k]
                    E[k] *= self.gamma * self.lam

                self.epsilon_decay()
                s, a = s_p, a_p
            if result_handler is not None:
                result_handler(sum_reward)
        return pi

    def epsilon_decay(self):
        if self.epsilon_v > self.epsilon_min:
            self.epsilon_v *= self.epsilon_step_factor
        else:
            self.epsilon_v = self.epsilon_min
            
        
    def epsilon(self, s):
        #N_0, N = 100, self.visit_count
        #   return N_0 / (N_0 + N[s])
        return self.epsilon_v
    
    def env_reset(self):
        s = self.env.reset()
        return self.fex(s), s.is_terminal()

    def env_step(self, a):
        s, r = self.env.step(a)
        return self.fex(s), r, s.is_terminal()


class MonteCarlo(SarsaLambda):
    """
        SarsaLambda with lambda=1 is equivalent to MonteCarlo
    """

    def __init__(self, env: FiniteActionEnvironment):
        super().__init__(env, lam=1)


class TD0(SarsaLambda):
    """
        SarsaLambda with lambda=0 is equivalent to TD(0)
    """

    def __init__(self, env: FiniteActionEnvironment):
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
