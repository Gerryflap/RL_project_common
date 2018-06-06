

class Environment:
    """
        Class for describing the environments and how they handle states/actions/rewards/observations for the algorithms
        to learn from
    """

    def sample_action(self):
        """
        :return: A sampled action from the action space
        """
        raise NotImplementedError

    def step(self, action, update=True) -> tuple:
        """
        Perform the action on the current model state. Return an observation and a corresponding reward
        :param action: The action to be performed
        :param update: An optional parameter indicating whether the internal state model should be updated
        :return: A two-tuple of an observation and reward obtained from this step
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset the internal model state
        :return:
        """
        raise NotImplementedError


class State:
    """
        Stores information about the environment state
    """

    def __init__(self):
        self.terminal = False

    def is_terminal(self) -> bool:
        """
        :return: A boolean indicating whether the state is terminal
        """
        return self.terminal


class DiscreteActionEnvironment(Environment):
    """
        Special case where the action space of the environment is discrete and finite
    """

    def sample_action(self):
        return super().sample_action()

    def step(self, action, update=True) -> tuple:
        return super().step(action, update)

    def reset(self):
        return super().reset()

    def action_space(self, state) -> set:
        """
        Get all the possible actions in the specified state
        :param state: The state from which actions should be possible
        :return: a set of possible actions from the state
        """
        raise NotImplementedError


class EnvironmentWrapper(object):
    def __init__(self, env:DiscreteActionEnvironment, state_transformer=lambda s: s):
        self.state_transformer = state_transformer
        self.env = env
        self.terminated = True
        # Assuming that the action space is constant:
        self.action_space = list(env.action_space(self.env.reset()))

    def reset(self):
        s = self.env.reset()
        self.terminated = False
        return self.state_transformer(s)

    def step(self, action):
        s, r = self.env.step(action)
        self.terminated = s.is_terminal()
        return self.state_transformer(s), r