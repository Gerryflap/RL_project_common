import random

"""
    Core classes of a Reinforcement Learning experiment
"""


class Action:
    """
        Action to be performed on an environment
    """
    pass


class Observation:
    """
        Observation obtained from executing an action in the environment
    """

    def __init__(self, terminal: bool):
        """
        Create a new observation
        :param terminal: A boolean that indicates if the environment state is terminal
        """
        self.terminal = terminal

    def is_terminal(self) -> bool:
        """
        :return: a boolean indicating if the environment state is terminal
        """
        return self.terminal


class Environment:
    """
        Class for describing the environments and how they handle states/actions/rewards/observations for the algorithms
        to learn from
    """

    def __init__(self, observation_space: callable, action_space: callable):
        """
        Create a new Environment
        :param observation_space:  The class of observations that are obtained from this environment
        :param action_space:       The class of actions that can be used on this environment
        """
        self._action_space = action_space
        self._observation_space = observation_space

    @property
    def action_space(self) -> callable:
        """
        :return: The class of actions that can be used on this environment
        """
        return self._action_space

    @property
    def observation_space(self) -> callable:
        """
        :return: The class of observations that are obtained from this environment
        """
        return self._observation_space

    def sample(self):
        """
        Uniformly sample an action that can be performed on the current environment state
        :return: the sampled action
        """
        raise NotImplementedError

    def step(self, action: Action) -> tuple:
        """
        Perform the action on the current model state. Return an observation and a corresponding reward
        :param action: The action to be performed
        :return: A two-tuple of
                        - an observation
                        - reward obtained from performing the action
        """
        raise NotImplementedError

    def reset(self) -> Observation:
        """
        Reset the internal model state
        :return: an initial observation
        """
        raise NotImplementedError


class DiscreteEnvironment(Environment):
    """
        Class of environments that have a finite set of actions
    """

    def valid_actions(self) -> list:
        """
        :return: a list of actions that can be performed on the current environment state
        """
        raise NotImplementedError

    def sample(self):
        """
        Uniformly sample an action from the valid actions
        :return: the sampled action
        """
        actions = self.valid_actions()
        return actions[random.randint(0, len(actions) - 1)]

    def step(self, action: Action) -> tuple:
        raise NotImplementedError

    def reset(self) -> Observation:
        raise NotImplementedError
