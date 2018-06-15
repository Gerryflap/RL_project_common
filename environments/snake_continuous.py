import gym
import gym_snake  # This is required to load the snake environment into gym

from core import State, Action, FiniteActionEnvironment

"""
    Environment wrapper for av80r's Snake (Gym-like snake environment)
    Requires installation of https://github.com/av80r/Gym-Snake
"""

class SnakeState(State):
    """
        Snek State class
    """

    def __init__(self, state, terminal: bool):
        """
        Create a new Snake State
        :param state: An state obtained from the OpenAI environment
        :param terminal: A boolean indicating whether the environment state is terminal
        """
        super().__init__(terminal)
        self.state = state

    def __str__(self) -> str:
        return str(self.state)


class SnakeAction(Action):
    """
        Snek Environment Action
    """
    def __init__(self, direction: int):
        """
        Create a new Snake Action
        :param direction: An int indicating the direction 1-3
        """
        self.direction = direction


class Snake(FiniteActionEnvironment):
    """
        Snake environment class
    """

    # There is no Up action as the board is always rotated such that the snake is facing down
    RIGHT = SnakeAction(1)
    DOWN = SnakeAction(2)
    LEFT = SnakeAction(3)
    ACTIONS = [RIGHT, DOWN, LEFT]

    def __init__(self, render=True):
        """
        Create a new Snake Environment
        :param render: A boolean indicating whether the environment should be rendered
        """
        super().__init__()
        self.env = gym.make('snake-rotate-v0')
        self.render = render
        self.terminal = False
        self.reset()

    @staticmethod
    def action_space() -> list:
        return list(Snake.ACTIONS)

    @staticmethod
    def valid_actions_from(state) -> list:
        return Snake.action_space()

    def valid_actions(self) -> list:
        return Snake.action_space()

    def step(self, action: SnakeAction) -> tuple:
        """
        Perform an action on the current environment state
        :param action: The Action to be performed
        :return: A tuple of (state, reward)
        """
        if self.terminal:
            raise Exception('Cannot perform action on terminal state!')
        if self.render:
            self.env.render()
        state, reward, self.terminal, info = self.env.step(action.direction)
        return SnakeState(state, self.terminal), reward

    def reset(self):
        """
        Reset the environment state
        :return: A state containing the initial state
        """
        self.terminal = False
        return SnakeState(self.env.reset(), self.terminal)


if __name__ == '__main__':

    _e = Snake(render=True)
    _s = _e.reset()

    for _ in range(1000):
        while not _s.is_terminal():
            _s, _r = _e.step(_e.sample())
        _s = _e.reset()
