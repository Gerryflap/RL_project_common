import numpy as np


class Policy:
    """
        Policy class. Maps each action that can be taken from a certain state to a float in [0,1] which represents a
        probability of taking that action from the state

        NOTE: Only supports policies over finite action spaces!
    """

    def _actions_from(self, state) -> list:
        """
        Gets all valid actions that can be executed on the state
        :param state: The state on which the actions should be executed
        :return: a list of actions
        """
        raise NotImplementedError

    def _actions_values_from(self, state):
        """
        Get all valid actions that can be executed on this state, as well as values to base the policy on
        :param state: The state on which the actions should be executed
        :return: a dict mapping all valid actions to their corresponding value
        """
        raise NotImplementedError

    def p(self, action, state):
        """
        Get the probability of taking an action from the specified state (as dictated by this policy)
        :param action: Action of which the probability should be returned
        :param state: State from which the action would be taken
        :return: the probability of taking the action as dictated by this policy
        """
        return self.distribution(state)[action]

    def distribution(self, state):
        """
        Get a probability distribution over all actions that can be taken from the state
        :param state: The state from which the actions are taken
        :return: A dict mapping all actions to their probability
        """
        values = self._actions_values_from(state)   # Get a value representing the 'desirability' of each action
        offset = abs(min(values.values()))          # Add an offset to each value so negative numbers are removed
        for a, v in values.items():
            values[a] += offset
        total = sum(values.values())
        if total == 0:
            return {a: 1 / len(values) for a in values.keys()}  # Normalize to uniform probability distribution
        else:
            return {a: v / total for a, v in values.items()}    # Normalize to probability distribution

    def sample(self, state):
        """
        Sample an action to take at the given state from this policy
        :param state: The state at which the action should be taken
        :return: the sampled action
        """
        dist = self.distribution(state)
        actions, probabilities = zip(*dist.items())
        return np.random.choice(actions, p=probabilities)

    def __call__(self, action, state):
        return self.p(action, state)


class GreedyPolicy(Policy):
    """
        Policy class where the action with highest value is always sampled
    """

    def _actions_from(self, state):
        raise NotImplementedError

    def _actions_values_from(self, state):
        raise NotImplementedError

    def distribution(self, state):
        """
        Get a probability distribution over all actions that can be taken from the state
        :param state: The state from which the actions are taken
        :return: A dict mapping all actions to their probability
        """
        values = self._actions_values_from(state)
        a = max(values, key=values.get)
        return {1 if a == a_p else 0 for a_p in values.keys()}  # Since the policy samples greedily, p=1 for one action

    def sample(self, state):
        """
        Sample an action to take at the given state from this policy
        :param state: The state at which the action should be taken
        :return: the sampled action
        """
        values = self._actions_values_from(state)
        return max(values, key=values.get)


class EpsilonGreedyPolicy(Policy):
    """
        Policy class that, with probability epsilon, samples an action uniformly over the action space and samples
        greedily otherwise
    """

    def __init__(self, epsilon: callable):
        """
        Create a new EpsilonGreedyPolicy
        :param epsilon: A function that returns a probability epsilon when given a state
        """
        self.epsilon = epsilon

    def _actions_from(self, state):
        raise NotImplementedError

    def _actions_values_from(self, state):
        raise NotImplementedError

    def distribution(self, state):
        """
        Get a probability distribution over all actions that can be taken from the state
        :param state: The state from which the actions are taken
        :return: A dict mapping all actions to their probability
        """
        epsilon = self.epsilon(state)
        values = self._actions_values_from(state)
        a = max(values, key=values.get)
        dist = {a_p: 1 - epsilon if a == a_p else 0 for a_p in values.keys()}
        return {a: v + epsilon / len(dist) for a, v in dist.items()}

    def sample(self, state):
        """
        Sample an action to take at the given state from this policy
        :param state: The state at which the action should be taken
        :return: the sampled action
        """
        if np.random.random() < self.epsilon(state):
            actions = self._actions_from(state)
            return actions[np.random.randint(0, len(actions))]  # Sample uniformly
        else:
            return super(EpsilonGreedyPolicy, self).sample(state)   # Sample greedily


if __name__ == '__main__':

    class TestPolicy(EpsilonGreedyPolicy):

        def _actions_from(self, state):
            return list(range(5))

        def _actions_values_from(self, state):
            return {a: np.random.randint(0, 4) for a in self._actions_from(state)}

    _p = TestPolicy(epsilon=lambda x: 0.05)
    _dist = _p.distribution(None)
    _a = _p.sample(None)
    _prob = _p(1, None)

    print(_p, _dist, _a, _prob)
