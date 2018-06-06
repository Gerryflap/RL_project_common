import numpy as np
import pygame
from ple import PLE
from ple.games import Pixelcopter

from core import Observation, Action, DiscreteEnvironment

"""
    PixelCopter Environment wrapper for PyGame Learning Environment's PixelCopter
    https://github.com/ntasfi/PyGame-Learning-Environment
"""

pygame.init()


class PixelCopterObservation(Observation):
    """
        A PixelCopter environment Observation
    """

    def __init__(self, observation, terminal: bool):
        """
        Create a new PixelCopter Observation
        :param observation: Environment observation to be stored in this state
        :param terminal: A boolean indicating whether the environment state is terminal
        """
        super().__init__(terminal)
        self.observation = observation

    def __str__(self) -> str:
        return str(self.observation)


class PixelCopterAction(Action):
    """
        A PixelCopter Environment Action
    """

    def __init__(self, key):
        """
        Create a new PixelCopter Action
        :param key: The PyGame key corresponding to this action
        """
        self.key = key


class PixelCopter(DiscreteEnvironment):
    """
        PixelCopter environment class
    """

    def __init__(self, size: tuple = (48, 48)):
        """
        Create a new PixelCopter Environment
        :param size: Game window dimensions
        """
        super().__init__(PixelCopterObservation, PixelCopterAction)
        self.width, self.height = size
        self.game = Pixelcopter(width=self.width, height=self.height)
        self.game.screen = pygame.display.set_mode(self.game.getScreenDims(), 0, 32)
        self.game.clock = pygame.time.Clock()
        self.game.rng = np.random.RandomState(24)

        self.game.rewards['loss'] = -1
        self.game.rewards['win'] = 1

        self.ple = PLE(self.game)
        self.ple.init()

        ascend = PixelCopterAction(pygame.K_w)
        descend = PixelCopterAction(self.ple.NOOP)

        self._actions = [descend, ascend]
        self.terminal = False
        self.reset()

    def valid_actions(self) -> list:
        """
        :return: A list of actions that can be performed on the current environment state
        """
        return list(self._actions)

    def step(self, action: PixelCopterAction) -> tuple:
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
        return PixelCopterObservation(observation, self.terminal), reward

    def reset(self):
        """
        Reset the environment state
        :return: A state containing the initial observation
        """
        self.ple.reset_game()
        self.terminal = self.ple.game_over()
        return PixelCopterObservation(self.game.getGameState(), self.terminal)


class VisualPixelCopter(PixelCopter):
    """
        PixelCopter environment class giving screen captures as observation
    """

    def step(self, action) -> tuple:
        """
        Perform an action on the current environment state
        :param action: The action to be performed
        :return: A two-tuple of (observation, reward)
        """
        if self.terminal:
            raise Exception('Cannot perform action on terminal state!')

        reward = self.ple.act(action.key)
        observation = self.ple.getScreenGrayscale()
        self.terminal = self.ple.game_over()
        pygame.display.update()
        return PixelCopterObservation(observation, self.terminal), reward

    def reset(self):
        """
        Reset the environment state
        :return: A state containing the initial observation
        """
        self.ple.reset_game()
        self.terminal = self.ple.game_over()
        return PixelCopterObservation(self.ple.getScreenGrayscale(), self.terminal)


if __name__ == '__main__':
    import numpy as np
    import time

    _width, _height = _size = 256, 256
    _e = PixelCopter(_size)

    _s = _e.reset()
    while not _s.is_terminal():
        _s, _r = _e.step(np.random.choice(_e.valid_actions(), p=[0.9, 0.1]))
        print(_r)
        if _s.is_terminal():
            _s = _e.reset()
        time.sleep(0.1)
