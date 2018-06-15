from collections import deque
import numpy as np

from agent import Agent
from core import FiniteActionEnvironment
from policy import EpsilonGreedyPolicy
from q_network_sarsa_lambda import QNetworkSL


class DeepSarsa(Agent):

    def __init__(self,
                 env: FiniteActionEnvironment,
                 model: QNetworkSL,
                 replay_memory_size: int=3000,
                 minibatch_size: int=32,
                 epsilon=0.05,
                 epsilon_step_factor=1.0,
                 epsilon_min=0.0
                 ):
        super().__init__(env)
        self.epsilon_step_factor = epsilon_step_factor
        self.epsilon_min = epsilon_min
        self.qnetwork = model
        self.epsilon_v = epsilon
        self.policy = model.derive_policy(EpsilonGreedyPolicy,
                                          env.valid_actions_from,
                                          epsilon=self.epsilon)
        self.replay_memory = deque(maxlen=replay_memory_size)
        self.minibatch_size = minibatch_size

    def learn(self, num_episodes: int=1000000, result_handler=None) -> EpsilonGreedyPolicy:
        Q, pi = self.qnetwork, self.policy
        for i in range(num_episodes):
            s = self.env.reset()
            r_sum = 0
            trajectory = []
            
            while not s.is_terminal():
                a = pi.sample(s)
                s_p, r = self.env.step(a)
                trajectory += [(s, a, r)]
                r_sum += r

                # Train Q-network
                if len(self.replay_memory) > 0:
                    mini_batch = self.sample_minibatch()
                    self.qnetwork.fit_on_trajectories(mini_batch)
                if self.epsilon_v > self.epsilon_min:
                    self.epsilon_v *= self.epsilon_step_factor
                else:
                    self.epsilon_v = self.epsilon_min
                s = s_p

            self.store_in_replay_memory(trajectory)

            print("[%i/%i]: %f" % (i, num_episodes, r_sum))
            if callable(result_handler):
                result_handler(r_sum)
            pass

        return pi

    def epsilon(self, state):
        return self.epsilon_v

    def sample_minibatch(self) -> list:
        """
        Get a random minibatch of samples from current replay memory
        :return: A list of samples
        """
        batch = []
        for i in range(self.minibatch_size):
            trajectory = self.replay_memory[np.random.randint(0, len(self.replay_memory))]
            trajectory = trajectory[np.random.randint(0, len(trajectory)):]
            batch.append(trajectory)
        return batch


    def store_in_replay_memory(self, trajectory):
        self.replay_memory.append(trajectory)
