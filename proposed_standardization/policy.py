import random

from proposed_standardization.q_estimator import QEstimator

"""
    Various Policy classes
"""


class Policy:
    """
        Policy class for sampling actions given a certain state or observation
    """

    def __init__(self, observation_space: callable, valid_actions: callable):
        """
        Create a new Policy
        :param observation_space: The class of observations/states that this policy should dictate actions from
        :param valid_actions: A function that provides the valid actions given an observation
        """
        self.state_space = observation_space
        self.valid_actions = valid_actions

    def sample(self, observation):
        """
        Obtain an action to perform given the observation
        :param observation: The observation the decision should be based on
        :return: the sampled action
        """
        raise NotImplementedError

    def __call__(self, observation, **kwargs):
        return self.sample(observation)


class QPolicy(Policy):
    """
        Policy class that bases its decisions on a Q value
    """

    def __init__(self, observation_space, valid_actions: callable, q_source: QEstimator):
        """
        Create a new QPolicy
        :param observation_space: The observation the decision should be based on
        :param valid_actions: A function that provides the valid actions given an observation
        :param q_source: A source of Q values
        """
        super().__init__(observation_space, valid_actions)
        self.q = q_source

    def sample(self, observation):
        raise NotImplementedError


class Greedy(QPolicy):
    """
        QPolicy class that always picks the action with the highest Q value
    """

    def __init__(self, observation_space, valid_actions: callable, q_source: QEstimator):
        super().__init__(observation_space, valid_actions, q_source)

    def sample(self, observation):
        """
        Get the action with the highest Q value
        :param observation: The observation the decision should be based on
        :return: the sampled action
        """
        if not isinstance(observation, self.state_space):
            raise KeyError
        actions = self.valid_actions()
        qs = self.q.Qs(observation, actions)
        return max(actions, key=qs.get)


class EpsilonGreedy(Greedy):
    """
        QPolicy class that picks a random action with probability epsilon and samples greedily otherwise
    """

    def __init__(self, observation_space, valid_actions: callable, q_source: QEstimator, epsilon):
        super().__init__(observation_space, valid_actions, q_source)
        self.epsilon = epsilon

    def sample(self, observation):
        """
        Get a random action with probability epsilon and sample greedily otherwise
        :param observation: The observation the decision should be based on
        :return: the sampled action
        """
        if not isinstance(observation, self.state_space):
            raise KeyError
        if random.random() < self.epsilon(observation):
            actions = self.valid_actions()
            return actions[random.randint(0, len(actions) - 1)]
        else:
            return super(EpsilonGreedy, self).sample(observation)
