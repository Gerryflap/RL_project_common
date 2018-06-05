import numpy as np
import pygame
from ple import PLE
import ple.games

from proposed_standardization.core import DiscreteEnvironment, Observation, Action

"""
    Flappy Bird Environment wrapper for PyGame Learning Environment's Flappy Bird
    https://github.com/ntasfi/PyGame-Learning-Environment
"""

pygame.init()


class FlappyBirdObservation(Observation):
    """
        Flappy Bird Observation
    """

    def __init__(self, observation: dict, terminal: bool):
        """
        Create a new Flappy Bird Observation
        :param observation: a dictionary containing an observation
        :param terminal: a boolean indicating whether the environment state is terminal
        """
        super().__init__(terminal)
        self.observation = observation

    def __str__(self) -> str:
        return str(self.observation)


class FlappyBirdAction(Action):
    """
        Flappy Bird Action that can be performed on the environment state
    """

    def __init__(self, key):
        """
        Create a new Flappy Bird Action
        :param key: PyGame key that corresponds to performing the action
        """
        self.key = key


class FlappyBird(DiscreteEnvironment):
    """
        FlappyBird Environment class
    """

    def __init__(self, size: tuple = (48, 48)):
        """
        Create a new Flappy Bird Environment
        :param size: Game window dimensions
        """
        super().__init__(FlappyBirdObservation, FlappyBirdAction)
        self.width, self.height = size
        self.game = ple.games.FlappyBird(width=self.width, height=self.height)
        self.game.screen = pygame.display.set_mode(self.game.getScreenDims(), 0, 32)
        self.game.clock = pygame.time.Clock()
        self.game.rng = np.random.RandomState(24)

        self.game.rewards['loss'] = -1
        self.game.rewards['win'] = 1

        self.ple = PLE(self.game)
        self.ple.init()

        flap = FlappyBirdAction(pygame.K_w)
        rest = FlappyBirdAction(self.ple.NOOP)
        self._actions = [rest, flap]

        self.terminal = False
        self.reset()

    def step(self, action: FlappyBirdAction) -> tuple:
        """
        Perform an action on the current environment state
        :param action: The action to be performed
        :return: A two-tuple of (observation, reward)
        """
        if self.terminal:
            raise Exception('Cannot perform action on terminal state!')

        reward = self.ple.act(action.key)
        observation = self.game.getGameState()
        self.terminal = self.ple.game_over()
        pygame.display.update()
        return FlappyBirdObservation(observation, self.terminal), reward

    def reset(self):
        """
        Reset the environment state
        :return: A state containing the initial observation
        """
        self.ple.reset_game()
        observation = self.game.getGameState()
        self.terminal = self.ple.game_over()
        return FlappyBirdObservation(observation, self.terminal)

    def valid_actions(self) -> list:
        """
        :return: A list of actions that can be performed on the current environment state
        """
        return list(self._actions)  # Actions independent of state


if __name__ == '__main__':
    import numpy as np
    import time

    _width, _height = _size = 288, 512
    _e = FlappyBird(_size)

    _s = _e.reset()
    while not _s.is_terminal():
        _s, _r = _e.step(np.random.choice(_e.valid_actions(), p=[0.9, 0.1]))
        print(_r)
        if _s.is_terminal():
            _s = _e.reset()
        time.sleep(0.1)
