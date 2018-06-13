from collections import deque
import numpy as np

from agent import Agent
from core import FiniteActionEnvironment
from policy import EpsilonGreedyPolicy
from q_network import QNetwork


class DeepSarsa(Agent):

    def __init__(self,
                 env: FiniteActionEnvironment,
                 model: QNetwork,
                 lam: float = 0.2,
                 gamma: float=1.0,
                 replay_memory_size: int=3000,
                 minibatch_size: int=32,
                 ):
        super().__init__(env)
        self.qnetwork = model
        self.lam = lam
        self.gamma = gamma
        self.policy = model.derive_policy(EpsilonGreedyPolicy,
                                          env.valid_actions_from,
                                          epsilon=self.epsilon)
        self.replay_memory = deque(maxlen=replay_memory_size)
        self.minibatch_size = minibatch_size

    def learn(self, num_episodes: int=1000000) -> EpsilonGreedyPolicy:
        Q, pi = self.qnetwork, self.policy
        for _ in range(num_episodes):
            s = self.env.reset()
            a = pi.sample(s)
            trajectory = []
            while not s.is_terminal():
                s_p, r = self.env.step(a)
                a_p = pi.sample(s_p)
                trajectory += [(s, a, r, s_p, a_p)]
                # TODO -- train network
                s, a = s_p, a_p

            self.store_in_replay_memory(trajectory)
            pass

        return pi

    def epsilon(self, state):
        pass  # TODO

    def sample_minibatch(self) -> list:
        """
        Get a random minibatch of samples from current replay memory
        :return: A list of samples
        """
        ixs = np.random.choice(range(len(self.replay_memory)),  # Sample random indices to take samples from
                               size=self.minibatch_size)
        return [self.replay_memory[i] for i in ixs]             # TODO -- faster by sorting ixs

    def store_in_replay_memory(self, trajectory):
        self.replay_memory.append(trajectory)
