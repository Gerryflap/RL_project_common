import math
import random

import numpy as np
import pygame
from ple import PLE
import ple.games
from ple.games.primitives import Player, Creep
from ple.games.puckworld import PuckCreep

from core import State, Action, FiniteActionEnvironment
from sacx.extcore import TaskEnvironment, Task

"""
    PuckWorld Environment wrapper for PyGame Learning Environment's PuckWorld
    https://github.com/ntasfi/PyGame-Learning-Environment
"""

pygame.init()


class PuckWorldState(State):
    """
        PuckWorld State
    """

    def __init__(self, state: dict, terminal: bool):
        """
        Create a new PuckWorld State
        :param state: a dictionary containing an state
        :param terminal: a boolean indicating whether the environment state is terminal
        """
        super().__init__(terminal)
        self.state = state

    def __str__(self) -> str:
        return str(self.state)


class PuckWorldAction(Action):
    """
        PuckWorld Action that can be performed on the environment state
    """

    def __init__(self, direction):
        self.direction = direction


class PuckWorld(TaskEnvironment, FiniteActionEnvironment):
    """
        PuckWorld Environment class
    """

    UP = PuckWorldAction('up')
    DOWN = PuckWorldAction('down')
    RIGHT = PuckWorldAction('right')
    LEFT = PuckWorldAction('left')

    ACTIONS = [UP, DOWN, RIGHT, LEFT]

    KEYS = {UP: pygame.K_w, LEFT: pygame.K_a, RIGHT: pygame.K_d, DOWN: pygame.K_s}

    NE_TASK = Task('ne corner')
    NW_TASK = Task('nw corner')
    SE_TASK = Task('se corner')
    SW_TASK = Task('sw corner')

    AUX_TASKS = [NE_TASK, NW_TASK, SE_TASK, SW_TASK]

    DELTA_SG = 1

    def __init__(self, duration, size: tuple = (48, 48)):
        """
        Create a new PuckWorld Environment
        :param size: Game window dimensions
        """
        super().__init__()
        self.width, self.height = size

        self.game = ExtPuckWorld(width=self.width, height=self.height, duration=duration, r_m=self._r_m)
        self.game.screen = pygame.display.set_mode(self.game.getScreenDims(), 0, 32)
        self.game.clock = pygame.time.Clock()
        self.game.rng = np.random.RandomState(24)

        self.ple = PLE(self.game)
        self.ple.init()
        self.epsilon = self.game.good_creep.radius

        self.terminal = False
        self.reset()

    @staticmethod
    def auxiliary_tasks():
        return list(PuckWorld.AUX_TASKS)

    @staticmethod
    def action_space() -> list:
        return list(PuckWorld.ACTIONS)

    @staticmethod
    def valid_actions_from(state) -> list:
        return PuckWorld.action_space()

    @staticmethod
    def _d(x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    @staticmethod
    def _d_sg(s):
        return PuckWorld._d(s['player_x'], s['player_y'], s['good_creep_x'], s['good_creep_y'])

    def _r_m(self, s):
        return PuckWorld.DELTA_SG if self._d_sg(s) <= self.epsilon else 0

    def _r_corner(self, s, x, y):
        x_s, y_s = s['player_x'], s['player_y']
        return (self._d(x, y, x_s, y_s) ** 2) / (self._d(0, 0, self.width, self.height) ** 2)

    def step(self, action: PuckWorldAction) -> tuple:
        """
        Perform an action on the current environment state
        :param action: The action to be performed
        :return: A two-tuple of (state, reward)
        """
        if self.terminal:
            raise Exception('Cannot perform action on terminal state!')

        main_reward = self.ple.act(self.KEYS[action])
        state = self.game.getGameState()
        self.terminal = self.ple.game_over()
        pygame.display.update()
        return PuckWorldState(state, self.terminal), {PuckWorld.MAIN_TASK: main_reward,
                                                      PuckWorld.NE_TASK: self._r_corner(state, 0, self.height),
                                                      PuckWorld.NW_TASK: self._r_corner(state, self.width, self.height),
                                                      PuckWorld.SE_TASK: self._r_corner(state, 0, 0),
                                                      PuckWorld.SW_TASK: self._r_corner(state, self.width, 0),
                                                      }

    def reset(self):
        """
        Reset the environment state
        :return: A state containing the initial state
        """
        self.ple.reset_game()
        state = self.game.getGameState()
        self.terminal = self.ple.game_over()
        return PuckWorldState(state, self.terminal)

    def valid_actions(self) -> list:
        """
        :return: A list of actions that can be performed on the current environment state
        """
        return self.action_space()

    def sample(self):
        return random.choice(self.ACTIONS)


class ExtPuckWorld(ple.games.PuckWorld):

    def __init__(self, width, height, duration, r_m):
        super().__init__(width, height)
        self.duration = duration
        self.r_m = r_m

    def game_over(self):
        """
            Return bool if the game has 'finished'
        """
        return self.ticks >= self.duration

    def init(self):
        """
            Starts/Resets the game to its initial state
        """

        self.player = Player(
            self.AGENT_RADIUS,
            self.AGENT_COLOR,
            self.AGENT_SPEED,
            self.AGENT_INIT_POS,
            self.width,
            self.height)

        self.good_creep = Creep(
            self.CREEP_GOOD['color'],
            self.CREEP_GOOD['radius'],
            self._rngCreepPos(),
            (1, 1),
            0.0,
            1.0,
            "GOOD",
            self.width,
            self.height,
            0.0  # jitter
        )

        self.bad_creep = PuckCreep(
            (self.width / 2,
             self.height / 2),
            self.CREEP_BAD,
            self.screen_dim[0] * 0.75,
            self.screen_dim[1] * 0.75)

        self.creeps = pygame.sprite.Group()
        self.creeps.add(self.good_creep)
        self.creeps.add(self.bad_creep)

        self.score = 0
        self.ticks = 0
        self.lives = -1

    def step(self, dt):
        """
            Perform one step of game emulation.
        """
        dt /= 1000.0
        self.ticks += 1
        self.screen.fill(self.BG_COLOR)

        self.score += self.rewards["tick"]

        self._handle_player_events()
        self.player.update(self.dx, self.dy, dt)

        dx = self.player.pos.x - self.bad_creep.pos.x
        dy = self.player.pos.y - self.bad_creep.pos.y
        dist_to_bad = math.sqrt(dx * dx + dy * dy)

        if dist_to_bad < self.CREEP_BAD['radius_outer']:
            reward = -1
        else:
            reward = self.r_m(self.getGameState())

        self.score += reward

        if self.ticks % 500 == 0:
            x, y = self._rngCreepPos()
            self.good_creep.pos.x = x
            self.good_creep.pos.y = y

        ndx = 0.0 if dist_to_bad == 0.0 else dx / dist_to_bad
        ndy = 0.0 if dist_to_bad == 0.0 else dy / dist_to_bad

        self.bad_creep.update(ndx, ndy, dt)
        self.good_creep.update(dt)

        self.player.draw(self.screen)
        self.creeps.draw(self.screen)


if __name__ == '__main__':
    import numpy as np
    import time

    _width, _height = _size = 256, 256
    _e = PuckWorld(200, _size)

    _s = _e.reset()
    while not _s.is_terminal():
        _s, _r = _e.step(_e.sample())
        print(_r)
        if _s.is_terminal():
            _s = _e.reset()
        time.sleep(0.1)
