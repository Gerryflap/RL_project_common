import gym

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


if __name__ == '__main__':

    _e = CartPole(render=True)
    _s = _e.reset()

    for _ in range(1000):
        while not _s.is_terminal():
            _s, _r = _e.step(_e.sample())
        _s = _e.reset()
