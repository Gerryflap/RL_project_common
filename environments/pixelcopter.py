import numpy as np
import pygame
from ple import PLE
from ple.games import Pixelcopter

from core import State, Action, FiniteActionEnvironment

"""
    PixelCopter Environment wrapper for PyGame Learning Environment's PixelCopter
    https://github.com/ntasfi/PyGame-Learning-Environment
"""

pygame.init()


class PixelCopterState(State):
    """
        A PixelCopter environment State
    """

    def __init__(self, state, terminal: bool):
        """
        Create a new PixelCopter State
        :param state: Environment state to be stored in this state
        :param terminal: A boolean indicating whether the environment state is terminal
        """
        super().__init__(terminal)
        self.state = state

    def __str__(self) -> str:
        return str(self.state)


class PixelCopterAction(Action):
    """
        A PixelCopter Environment Action
    """

    def __init__(self, ascend: bool):
        self._ascend = ascend

    def ascend(self):
        return self._ascend


class PixelCopter(FiniteActionEnvironment):
    """
        PixelCopter environment class
    """

    ASCEND = PixelCopterAction(True)
    DESCEND = PixelCopterAction(False)
    ACTIONS = [DESCEND, ASCEND]

    def __init__(self, size: tuple = (48, 48)):
        """
        Create a new PixelCopter Environment
        :param size: Game window dimensions
        """
        super().__init__()
        self.width, self.height = size
        self.game = Pixelcopter(width=self.width, height=self.height)
        self.game.screen = pygame.display.set_mode(self.game.getScreenDims(), 0, 32)
        self.game.clock = pygame.time.Clock()
        self.game.rng = np.random.RandomState(24)

        self.game.rewards['loss'] = -1
        self.game.rewards['win'] = 1

        self.ple = PLE(self.game)
        self.ple.init()

        self.terminal = False
        self.reset()

    @staticmethod
    def action_space() -> list:
        return list(PixelCopter.ACTIONS)

    @staticmethod
    def valid_actions_from(state) -> list:
        return PixelCopter.action_space()

    def valid_actions(self) -> list:
        """
        :return: A list of actions that can be performed on the current environment state
        """
        return self.action_space()

    def step(self, action: PixelCopterAction) -> tuple:
        """
        Perform an action on the current environment state
        :param action: The action to be performed
        :return: A two-tuple of (state, reward)
        """
        if self.terminal:
            raise Exception('Cannot perform action on terminal state!')

        if action.ascend():
            key = pygame.K_w
        else:
            key = self.ple.NOOP
        reward = self.ple.act(key)
        state = self.game.getGameState()
        self.terminal = self.ple.game_over()
        pygame.display.update()
        return PixelCopterState(state, self.terminal), reward

    def reset(self):
        """
        Reset the environment state
        :return: A state containing the initial state
        """
        self.ple.reset_game()
        self.terminal = self.ple.game_over()
        return PixelCopterState(self.game.getGameState(), self.terminal)


class VisualPixelCopter(PixelCopter):
    """
        PixelCopter environment class giving screen captures as state
    """

    def step(self, action) -> tuple:
        """
        Perform an action on the current environment state
        :param action: The action to be performed
        :return: A two-tuple of (state, reward)
        """
        if self.terminal:
            raise Exception('Cannot perform action on terminal state!')

        if action.ascend():
            key = pygame.K_w
        else:
            key = self.ple.NOOP
        reward = self.ple.act(key)
        state = self.ple.getScreenGrayscale()
        self.terminal = self.ple.game_over()
        pygame.display.update()
        return PixelCopterState(state, self.terminal), reward

    def reset(self):
        """
        Reset the environment state
        :return: A state containing the initial state
        """
        self.ple.reset_game()
        self.terminal = self.ple.game_over()
        return PixelCopterState(self.ple.getScreenGrayscale(), self.terminal)


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
