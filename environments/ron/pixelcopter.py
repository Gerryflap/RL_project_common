import random

import numpy as np
import pygame
from ple import PLE
from ple.games import Pixelcopter

from environments.ron.core import State, DiscreteActionEnvironment

pygame.init()


class PixelCopterState(State):
    """
        A PixelCopter environment state
    """

    def __init__(self, observation):
        """
        Create a new PixelCopter state
        :param observation: Environment observation to be stored in this state
        """
        super().__init__()
        self.observation = observation

    def __str__(self) -> str:
        """
        :return: A string representation of this state
        """
        return str(self.observation)

    def copy(self):
        """
        :return: A copy of this state
        """
        c = PixelCopterState(self.observation)
        c.terminal = self.terminal
        return c

    def set_observation(self, observation):
        self.observation = observation


class PixelCopter(DiscreteActionEnvironment):
    """
        PixelCopter environment class
    """

    def __init__(self, size: tuple = (48, 48)):
        self.width, self.height = size
        self.game = Pixelcopter(width=self.width, height=self.height)
        self.game.screen = pygame.display.set_mode(self.game.getScreenDims(), 0, 32)
        self.game.clock = pygame.time.Clock()
        self.game.rng = np.random.RandomState(24)

        self.game.rewards['loss'] = -1
        self.game.rewards['win'] = 1

        self.ple = PLE(self.game)
        # self.game.init()
        self.ple.init()

        self.i = 0

        self.state = self.reset()

    def sample_action(self):
        """
        :return: A random sample from the action space
        """
        return bool(random.getrandbits(1))

    def step(self, action, update=True) -> tuple:
        """
        Perform an action on the current environment state
        :param action: The action to be performed
        :param update: A boolean indicating whether the change in the environment should be stored
        :return: A two-tuple of (observation, reward)
        """
        if self.state.is_terminal():
            raise Exception('Cannot perform action on terminal state!')
        s = self.state if update else self.state.copy()

        if action:
            reward = self.ple.act(pygame.K_w)
        else:
            reward = self.ple.act(self.ple.NOOP)

        s.set_observation(self.game.getGameState())
        s.terminal = self.ple.game_over()

        # if self.i % 10 == 0:
        pygame.display.update()

        return s.copy() if update else s, reward

    def reset(self):
        """
        Reset the environment state
        :return: A state containing the initial observation
        """
        self.ple.reset_game()
        self.state = PixelCopterState(self.game.getGameState())

        self.i += 1

        return self.state.copy()

    def action_space(self, state) -> set:
        """
        Get the actions that can be performed on the specified state
        :param state: The state on which an action should be performed
        :return: A set of actions
        """
        return {False, True}  # Actions independent of state


class VisualPixelCopter(DiscreteActionEnvironment):
    """
        PixelCopter environment class giving screen captures as observation
    """

    def __init__(self, size: tuple = (48, 48)):
        self.width, self.height = size
        self.game = Pixelcopter(width=self.width, height=self.height)
        self.game.screen = pygame.display.set_mode(self.game.getScreenDims(), 0, 32)
        self.game.clock = pygame.time.Clock()
        self.game.rng = np.random.RandomState(24)

        self.game.rewards['loss'] = -1
        self.game.rewards['win'] = 1

        self.ple = PLE(self.game)
        self.ple.init()

        self.i = 0

        self.state = self.reset()

    def sample_action(self):
        """
        :return: A random sample from the action space
        """
        return bool(random.getrandbits(1))

    def step(self, action, update=True) -> tuple:
        """
        Perform an action on the current environment state
        :param action: The action to be performed
        :param update: A boolean indicating whether the change in the environment should be stored
        :return: A two-tuple of (observation, reward)
        """
        if self.state.is_terminal():
            raise Exception('Cannot perform action on terminal state!')
        s = self.state if update else self.state.copy()

        if action:
            reward = self.ple.act(pygame.K_w)
        else:
            reward = self.ple.act(self.ple.NOOP)

        s.set_observation(self.ple.getScreenGrayscale())
        s.terminal = self.ple.game_over()

        # if self.i % 10 == 0:
        pygame.display.update()

        return s.copy() if update else s, reward

    def reset(self):
        """
        Reset the environment state
        :return: A state containing the initial observation
        """
        self.ple.reset_game()
        self.state = PixelCopterState(self.ple.getScreenGrayscale())

        self.i += 1

        return self.state.copy()

    def action_space(self, state) -> set:
        """
        Get the actions that can be performed on the specified state
        :param state: The state on which an action should be performed
        :return: A set of actions
        """
        return {False, True}  # Actions independent of state


if __name__ == '__main__':
    import numpy as np
    import time
    import algorithms.deep_sarsa_lambda_learning as dsl
    from environments.ron.core import EnvironmentWrapper
    import tensorflow as tf

    def normalize_state(s):
        o = np.zeros(shape=(7,))
        o[0] = s.observation['player_y'] / height
        o[1] = s.observation['player_dist_to_ceil'] / (height / 2)
        o[2] = s.observation['player_dist_to_floor'] / (height / 2)
        o[3] = s.observation['player_vel']
        o[4] = s.observation['next_gate_dist_to_player'] / width
        o[5] = s.observation['next_gate_block_top'] / height
        o[6] = s.observation['next_gate_block_bottom'] / height
        return o


    width, height = size = 256, 256
    env = EnvironmentWrapper(PixelCopter(size), state_transformer=normalize_state)

    def network(x):
        ks = tf.keras
        x = ks.layers.Dense(150, activation='relu')(x)
        x = ks.layers.Dense(150, activation='relu')(x)
        x = ks.layers.Dense(150, activation='relu')(x)
        x = ks.layers.Dense(50, activation='relu')(x)
        return ks.layers.Dense(2, activation='linear')(x)

    agent = dsl.DeepSARSALambdaAgent(0.9, env.action_space, network, alpha=0.001, state_shape=(7,), epsilon=0.1, epsilon_step_factor=0.9999, epsilon_min=0.005, gamma=0.9, fixed_steps=100, reward_scale=0.1, replay_mem_size=10000, sarsa=True)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        s = env.reset()
        while True:
            scores = []
            for i in range(4):
                score = agent.run_episode(env, sess)
                scores.append(score)

            print("Score: ", sum(scores)/len(scores))
            print("Eps: ", agent.epsilon)
            print("Q: ", agent.Q(s, sess))
            print("Q fixed: ", agent.Q_fixed(s, sess))


