import gym
import numpy as np

from core import State, Action, FiniteActionEnvironment

"""
    Environment wrapper for OpenAI Gym's CartPole
"""


class CartPoleState(State):
    """
        CartPole State
    """

    def __init__(self, state, terminal: bool):
        """
        Create a new CartPole State
        :param state: An state obtained from the OpenAI environment
        :param terminal: A boolean indicating whether the environment state is terminal
        """
        super().__init__(terminal)
        self.state = state

    def __str__(self) -> str:
        return str(self.state)


class CartPoleAction(Action):
    """
        CartPole Environment Action
    """

    def __init__(self, direction: bool):
        """
        Create a new CartPole Action
        :param direction: A boolean indicating the direction of the action (left=False, right=True)
        """
        self.direction = direction


class CartPole(FiniteActionEnvironment):
    """
        CartPole environment class
    """

    LEFT = CartPoleAction(False)
    RIGHT = CartPoleAction(True)
    ACTIONS = [LEFT, RIGHT]

    def __init__(self, render=True):
        """
        Create a new CartPole Environment
        :param render: A boolean indicating whether the environment should be rendered
        """
        super().__init__()
        self.env = gym.make('CartPole-v1')
        self.render = render

        self.terminal = False

        self.reset()

    @staticmethod
    def action_space() -> list:
        return list(CartPole.ACTIONS)

    @staticmethod
    def valid_actions_from(state) -> list:
        return CartPole.action_space()

    def valid_actions(self) -> list:
        return CartPole.action_space()

    def step(self, action: CartPoleAction) -> tuple:
        """
        Perform an action on the current environment state
        :param action: The action to be performed
        :return: A two-tuple of (state, reward)
        """
        if self.terminal:
            raise Exception('Cannot perform action on terminal state!')
        if self.render:
            self.env.render()
        state, reward, self.terminal, info = self.env.step(action.direction)
        return CartPoleState(state, self.terminal), reward

    def reset(self):
        """
        Reset the environment state
        :return: A state containing the initial state
        """
        self.terminal = False
        return CartPoleState(self.env.reset(), self.terminal)


class NoisyCartPole(CartPole):
    """
        CartPole environment that adds Gaussian noise to the state observations
    """

    def __init__(self, std, *args, **kwargs):
        """
        Create a new NoisyCartPole environment
        :param std: Standard deviation of the noise distribution (mean=0)
        :param args: Additional arguments to pass to the super class
        :param kwargs: Additional named arguments to pass to the super class
        """
        self.std = std
        super().__init__(*args, **kwargs)

    def add_noise(self, state: CartPoleState) -> CartPoleState:
        """
        Adds Gaussian noise to the given state
        :param state: The state on which noise should be applied
        :return: the state argument, but with noise added
        """
        state.state[0] += np.random.normal(loc=0, scale=self.std) * 2.4
        state.state[1] += np.random.normal(loc=0, scale=self.std) * 3.6
        state.state[2] += np.random.normal(loc=0, scale=self.std) * 0.26
        state.state[3] += np.random.normal(loc=0, scale=self.std) * 3.5
        return state

    def step(self, action: CartPoleAction) -> tuple:
        """
        Perform a regular CartPole environment step. Add noise to the observation
        :param action: The action to perform
        :return: a two-tuple of a noisy observation and a reward
        """
        s, r = super().step(action)
        return self.add_noise(s), r

    def reset(self):
        """
        Perform a regular CartPole environment reset. Add noise to the observation
        :return: a noisy initial observation
        """
        s = super().reset()
        return self.add_noise(s)


if __name__ == '__main__':

    _e = NoisyCartPole(std=0.1, render=True)
    _s = _e.reset()

    for _ in range(1000):
        while not _s.is_terminal():
            _s, _r = _e.step(_e.sample())
        _s = _e.reset()
