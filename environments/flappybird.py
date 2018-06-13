import numpy as np
import pygame
from ple import PLE
import ple.games

from core import State, Action, FiniteActionEnvironment

"""
    Flappy Bird Environment wrapper for PyGame Learning Environment's Flappy Bird
    https://github.com/ntasfi/PyGame-Learning-Environment
"""

pygame.init()


class FlappyBirdState(State):
    """
        Flappy Bird State
    """

    def __init__(self, state: dict, terminal: bool):
        """
        Create a new Flappy Bird State
        :param state: a dictionary containing an state
        :param terminal: a boolean indicating whether the environment state is terminal
        """
        super().__init__(terminal)
        self.state = state

    def __str__(self) -> str:
        return str(self.state)


class FlappyBirdAction(Action):
    """
        Flappy Bird Action that can be performed on the environment state
    """

    def __init__(self, flap: bool):
        self._flap = flap

    def flap(self):
        return self._flap


class FlappyBird(FiniteActionEnvironment):
    """
        FlappyBird Environment class
    """

    FLAP = FlappyBirdAction(True)
    REST = FlappyBirdAction(False)
    ACTIONS = [REST, FLAP]

    def __init__(self, size: tuple = (48, 48)):
        """
        Create a new Flappy Bird Environment
        :param size: Game window dimensions
        """
        super().__init__()
        self.width, self.height = size
        self.game = ple.games.FlappyBird(width=self.width, height=self.height)
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
        return list(FlappyBird.ACTIONS)

    @staticmethod
    def valid_actions_from(state) -> list:
        return FlappyBird.action_space()

    def step(self, action: FlappyBirdAction) -> tuple:
        """
        Perform an action on the current environment state
        :param action: The action to be performed
        :return: A two-tuple of (state, reward)
        """
        if self.terminal:
            raise Exception('Cannot perform action on terminal state!')

        if action.flap():
            key = pygame.K_w
        else:
            key = self.ple.NOOP
        reward = self.ple.act(key)
        state = self.game.getGameState()
        self.terminal = self.ple.game_over()
        pygame.display.update()
        return FlappyBirdState(state, self.terminal), reward

    def reset(self):
        """
        Reset the environment state
        :return: A state containing the initial state
        """
        self.ple.reset_game()
        state = self.game.getGameState()
        self.terminal = self.ple.game_over()
        return FlappyBirdState(state, self.terminal)

    def valid_actions(self) -> list:
        """
        :return: A list of actions that can be performed on the current environment state
        """
        return self.action_space()


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
