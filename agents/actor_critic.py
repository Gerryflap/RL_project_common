"""
    An implementation of an Actor Critic Agent

    Q returns calculated using: Sarsa-λ
    Policy is updated by maximizing π(s) * ( Q(s, θ') - α*log(π(s) ) or using Advantage Actor Critic

"""
from collections import deque

from agent import Agent
from core import FiniteActionEnvironment
from p_network import PNetwork
from policy import Policy
from q_network_sarsa_lambda import QNetworkSL
import numpy as np


class ActorCriticAgent(Agent):
    def __init__(self,
                 env: FiniteActionEnvironment,
                 value_model: QNetworkSL,
                 policy_model: PNetwork,
                 replay_memory_size: int=3000,
                 minibatch_size: int=32
                 ):
        """
        Initializes the Actor Critic Agent
        :param env: The environment to operate on
        :param value_model: A QNetworkSL that predicts the value function
        :param policy_model: A PNetwork used to train and estimate a policy
        :param replay_memory_size: The size of the replay memory in #Trajectories
        :param minibatch_size: The size of minibatches used for training
        """
        super().__init__(env)
        self.value_model = value_model
        self.policy_model = policy_model
        policy_model.q_network = value_model
        self.replay_memory = deque(maxlen=replay_memory_size)
        self.minibatch_size = minibatch_size

    def learn(self, num_episodes: int = 1000000, result_handler=None) -> Policy:
        """
        Trains the agent for a specified number of episodes
        :param num_episodes: The number of episodes to train
        :param result_handler: A result handler that will process the results
        :return: The policy model that has been trained
        """
        Q, pi = self.value_model, self.policy_model
        for i in range(num_episodes):
            s = self.env.reset()
            r_sum = 0
            trajectory = []
            initial_state = s

            while not s.is_terminal():
                a = pi.sample(s)
                s_p, r = self.env.step(a)
                trajectory += [(s, a, r)]
                r_sum += r

                # Train Networks
                if len(self.replay_memory) > 0:
                    mini_batch = self.sample_minibatch()
                    self.value_model.fit_on_trajectories(mini_batch)
                    self.policy_model.train(mini_batch)
                s = s_p

            self.store_in_replay_memory(trajectory)

            print("[%i/%i]: %f" % (i, num_episodes, r_sum))
            print("Initial state policy, q-values: ",
                  self.policy_model.distribution(initial_state),
                  self.value_model.Qs(initial_state),
                  )
            if callable(result_handler):
                result_handler(r_sum)
            pass

        return pi

    def epsilon(self, state):
        # This Agent doesn't have an epsilon
        return 0

    def sample_minibatch(self) -> list:
        """
        Get a random minibatch of samples from current replay memory
        :return: A list of samples (#minibatch_size trajectories)
        """
        batch = []
        for i in range(self.minibatch_size):
            trajectory = self.replay_memory[np.random.randint(0, len(self.replay_memory))]
            trajectory = trajectory[np.random.randint(0, len(trajectory)):]
            batch.append(trajectory)
        return batch

    def store_in_replay_memory(self, trajectory):
        self.replay_memory.append(trajectory)