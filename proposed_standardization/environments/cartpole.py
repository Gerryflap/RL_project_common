import gym

from proposed_standardization.core import DiscreteEnvironment, Observation, Action

"""
    Environment wrapper for OpenAI Gym's CartPole
"""


class CartPoleObservation(Observation):
    """
        CartPole Observation
    """

    def __init__(self, observation, terminal: bool):
        """
        Create a new CartPole Observation
        :param observation: An observation obtained from the OpenAI environment
        :param terminal: A boolean indicating whether the environment state is terminal
        """
        super().__init__(terminal)
        self.observation = observation

    def __str__(self) -> str:
        return str(self.observation)


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


class CartPole(DiscreteEnvironment):
    """
        CartPole environment class
    """

    def __init__(self, render=False):
        """
        Create a new CartPole Environment
        :param render: A boolean indicating whether the environment should be rendered
        """
        super().__init__(CartPoleObservation, CartPoleAction)
        self.env = gym.make('CartPole-v1')
        self.render = render

        left = CartPoleAction(False)
        right = CartPoleAction(True)
        self._actions = [left, right]

        self.terminal = False

        self.reset()

    def step(self, action: CartPoleAction) -> tuple:
        """
        Perform an action on the current environment state
        :param action: The action to be performed
        :return: A two-tuple of (observation, reward)
        """
        if self.terminal:
            raise Exception('Cannot perform action on terminal state!')
        if self.render:
            self.env.render()
        observation, reward, self.terminal, info = self.env.step(action.direction)
        return CartPoleObservation(observation, self.terminal), reward

    def reset(self):
        """
        Reset the environment state
        :return: A state containing the initial observation
        """
        self.terminal = False
        return CartPoleObservation(self.env.reset(), self.terminal)

    def valid_actions(self) -> list:
        """
        :return: A list of actions that can be performed on the current environment state
        """
        return list(self._actions)


if __name__ == '__main__':

    _e = CartPole(render=True)
    _s = _e.reset()

    for _ in range(1000):
        while not _s.is_terminal():
            _s, _r = _e.step(_e.sample())
        _s = _e.reset()
