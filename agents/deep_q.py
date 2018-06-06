from collections import deque

import numpy as np

from agent import Agent
from core import DiscreteEnvironment
from policy import EpsilonGreedy
from q_network import QNetwork


class DeepQLearning(Agent):
    """
        Deep Q Learning algorithm implementation
    """

    def __init__(self, env: DiscreteEnvironment, q_network: QNetwork, gamma: float = 1.0, minibatch_size: int = 32):
        """
        Create a new Deep Q Learning agent
        :param env: The environment the algorithm is subjected to
        :param q_network: The neural network used in this procedure
        :param gamma: Reward discount factor
        :param minibatch_size: Size of batches used to train the network
        """
        super().__init__(env)
        self.env = env
        self.q_network = q_network
        self.replay_memory = deque(maxlen=3000)
        self.policy = EpsilonGreedy(env.observation_space,
                                    env.valid_actions,
                                    q_source=self.q_network,  # Use the neural network to estimate Q
                                    epsilon=lambda x: 0.05    # Keep epsilon constant
                                    )
        self.minibatch_size = minibatch_size
        self.gamma = gamma

    def learn(self, num_episodes=100000) -> EpsilonGreedy:
        """
        Train the Q-Network
        :param num_episodes: Number of episodes that should be run
        :return: A policy derived from the trained Q network
        """
        Q, pi = self.q_network, self.policy
        for e in range(num_episodes):
            s = self.env.reset()                            # Initialize the environment
            while not s.is_terminal():                      # Repeat until environment is terminal:
                a = pi(s)                                   # - Epsilon-greedily pick an action
                s_p, r = self.env.step(a)                   # - Perform the action, obtain feedback
                self.add_to_replay_memory(s, a, r, s_p)     # - Store result as sample to be trained on
                Q.fit_on_samples(self.sample_minibatch())   # - Train the model on a random batch of samples
                s = s_p                                     # - Continue to next state
        return pi

    def sample_minibatch(self) -> list:
        """
        Get a random minibatch of samples from current replay memory
        :return: A list of samples
        """
        ixs = np.random.choice(range(len(self.replay_memory)),  # Sample random indices to take samples from
                               size=self.minibatch_size)
        return [self.replay_memory[i] for i in ixs]             # TODO -- faster by sorting ixs

    def add_to_replay_memory(self, s, a, r, sp):
        """
        Add one sample to the replay memory
        :param s: State
        :param a: Action performed on that state
        :param r: Reward obtained from performing the action
        :param sp: Resulting state
        """
        self.replay_memory.append((s, a, r, sp))


if __name__ == '__main__':
    import keras as ks
    from environments.cartpole import CartPole

    nn = ks.models.Sequential()
    nn.add(ks.layers.Dense(32, activation='sigmoid', input_shape=(4,)))
    nn.add(ks.layers.Dense(32, activation='sigmoid'))
    nn.add(ks.layers.Dense(2, activation='linear'))

    nn.compile(optimizer=ks.optimizers.Adam(lr=0.001),
               loss='mse')

    _e = CartPole(render=True)
    _out_map = _e.valid_actions()

    dqn = QNetwork(nn, _out_map, lambda x: np.reshape(x.observation, newshape=(1, 4)))

    dql = DeepQLearning(_e, dqn)

    q = dql.learn()
