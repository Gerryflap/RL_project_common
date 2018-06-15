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
        :param state: a dictionary containing a state
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

        Built around a wrapper around PLE's PuckWorld
    """

    # Define all possible actions in this environment
    UP = PuckWorldAction('up')
    DOWN = PuckWorldAction('down')
    RIGHT = PuckWorldAction('right')
    LEFT = PuckWorldAction('left')
    ACTIONS = [UP, DOWN, RIGHT, LEFT]

    # Map all possible actions to pygame keys
    KEYS = {UP: pygame.K_w, LEFT: pygame.K_a, RIGHT: pygame.K_d, DOWN: pygame.K_s}

    # Define all auxiliary tasks in this environment
    NE_TASK = Task('ne corner')
    NW_TASK = Task('nw corner')
    SE_TASK = Task('se corner')
    SW_TASK = Task('sw corner')
    GC_TASK = Task('go green')
    #AUX_TASKS = [NE_TASK, NW_TASK, SE_TASK, SW_TASK, GC_TASK]
    AUX_TASKS = [GC_TASK]

    # Reward obtained from epsilon-region around goal state
    DELTA_SG = 1

    def __init__(self, duration, size: tuple = (48, 48)):
        """
        Create a new PuckWorld Environment
        :param size: Game window dimensions
        """
        super().__init__()
        self.width, self.height = size

        self.game = ExtPuckWorld(width=self.width,
                                 height=self.height,
                                 duration=duration,
                                 r_m=self._r_m
                                 )
        self.game.screen = pygame.display.set_mode(self.game.getScreenDims(), 0, 32)
        self.game.clock = pygame.time.Clock()
        self.game.rng = np.random.RandomState(24)

        self.ple = PLE(self.game)
        self.ple.init()
        self.epsilon = 2 * self.game.good_creep.radius  # Size of epsilon-region around goal state

        self.terminal = False
        self.reset()

    @staticmethod
    def auxiliary_tasks():
        """
        :return: A list of all auxiliary tasks
        """
        return list(PuckWorld.AUX_TASKS)

    @staticmethod
    def get_tasks():
        """
        :return: A list of all tasks
        """
        return [PuckWorld.MAIN_TASK] + PuckWorld.auxiliary_tasks()

    @staticmethod
    def action_space() -> list:
        """
        :return: A list of all actions possible in this environment
        """
        return list(PuckWorld.ACTIONS)

    @staticmethod
    def valid_actions_from(state) -> list:
        """
        Get all valid actions that are possible on the given state
        :param state: the state on which actions should be possible
        :return: a list of valid actions
        """
        return PuckWorld.action_space()  # Actions not dependent on state for this environment

    @staticmethod
    def _d(x1, y1, x2, y2):
        """
        Euclidean distance between 1 and 2
        :param x1: x-coordinate of 1
        :param y1: y-coordinate of 1
        :param x2: x-coordinate of 2
        :param y2: y-coordinate of 2
        :return: euclidean distance between the two coordinates
        """
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    @staticmethod
    def _d_sg(s):
        """
        Computes the distance between given state and goal state
        :param s: the state parameter
        :return: the euclidean distance between the state and the goal state (that is, the player coordinates and
                 green creep coordinates)
        """
        return PuckWorld._d(s['player_x'], s['player_y'], s['good_creep_x'], s['good_creep_y'])

    def _r_m(self, s):
        """
        Computes a reward according to the epsilon-region surrounding the goal state
        :param s: the state for which the reward should be computed
        :return: a reward
        """
        return PuckWorld.DELTA_SG if self._d_sg(s) <= self.epsilon else 0

    def _r_corner(self, s, x, y):
        """
        Compute a reward based on the distance to a corner location
        :param s: The state for which a reward should be computed
        :param x: The x coordinate of the corner
        :param y: The y coordinate of the corner
        :return: a reward
        """
        x_s, y_s = s['player_x'], s['player_y']
        d = self._d(x, y, x_s, y_s)
        max_d = self._d(0, 0, self.width, self.height)
        return (max_d - d) / max_d

    def _r_green(self, s):
        """
        Computes a reward based on the distance to the green creep
        :param s: The state for which a reward should be computed
        :return: a reward
        """
        x, y = s['player_x'], s['player_y']
        x_c, y_c = s['good_creep_x'], s['good_creep_y']
        d = self._d(x, y, x_c, y_c)
        max_d = self._d(0, 0, self.width, self.height)
        r = (max_d - d) / max_d
        dx, dy = (x_c - x)/self.width, (y_c - y)/self.height
        vx, vy = s['player_velocity_x'] ,s['player_velocity_y']
        vx, vy = vx/self.width, vy/self.height
        dxp, dyp = (x_c - vx - x) / self.width, (y_c - vy - y) / self.height
        r = (dx**2 + dy**2)**0.5 - (dxp**2 + dyp**2)**0.5
        return r * 1000

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
                                                      PuckWorld.NE_TASK: self._r_corner(state, 0, 0),
                                                      PuckWorld.NW_TASK: self._r_corner(state, self.width, 0),
                                                      PuckWorld.SE_TASK: self._r_corner(state, 0, self.height),
                                                      PuckWorld.SW_TASK: self._r_corner(state, self.width, self.height),
                                                      PuckWorld.GC_TASK: self._r_green(state)
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
        """
        :return: a uniformly sampled action from the currently available actions
        """
        return random.choice(self.ACTIONS)


class ExtPuckWorld(ple.games.PuckWorld):
    """
        Extension on PLE's PuckWorld to modifications to the environment
    """

    def __init__(self, width, height, duration, r_m):
        """
        Create a new ExtPuckWorld
        :param width: The width of the PuckWorld
        :param height: The height of the PuckWorld
        :param duration: The number of ticks an episode lasts
        :param r_m: A function that computes the main task reward given an environment state
        """
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
            (self.width / 2,            # Bad creep starts in the middle
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
            reward = -1                                     # -1 reward for intersection with the bad creep
        else:
            reward = self.r_m(self.getGameState())

        self.score += reward

        # if self.ticks % 500 == 0:
        #     x, y = self._rngCreepPos()
        #     self.good_creep.pos.x = x
        #     self.good_creep.pos.y = y

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
