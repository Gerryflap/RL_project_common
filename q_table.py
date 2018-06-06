import collections

from core import Observation, Action, DiscreteEnvironment
from q_estimator import QEstimator


class QTable(collections.MutableMapping, QEstimator):
    """
        Q Table implementation
    """

    def __init__(self, observation_space=Observation, action_space=Action, *args, **kwargs):
        """
        Create a new Q Table
        :param observation_space: The class of observations that are stored in this Q Table
        :param action_space: The class of actions that are stored in this Q Table
        """
        self.store = dict()
        self.update(dict(*args, **kwargs))
        self.observation_space = observation_space
        self.action_space = action_space

    def __getitem__(self, key: tuple) -> float:
        """
        Get the Q-value corresponding to the given key
        :param key: A two-tuple of (observation, action)
        :return: The Q-value corresponding to the (observation, action) pair. Return 0 if the pair is not in this table
        """
        s, a = key
        if not isinstance(s, self.observation_space) or not isinstance(a, self.action_space):
            raise KeyError
        return self.store.get(s, dict()).get(a, 0)

    def __setitem__(self, key: tuple, value: float):
        """
        Set a Q-value for a given (observation, action) pair
        :param key: Two-tuple of (observation, action)
        :param value: Q-value corresponding to the key
        """
        s, a = key
        if not isinstance(s, self.observation_space) or not isinstance(a, self.action_space):
            raise KeyError
        self.store.setdefault(s, dict())[a] = value

    def __delitem__(self, key: tuple):
        """
        Remove an entry from this table
        :param key: The (observation, action) pair of the entry that should be removed
        """
        s, a = key
        if not isinstance(s, self.observation_space) or not isinstance(a, self.action_space):
            raise KeyError
        del self.store[s][a]

    def __iter__(self):
        """
        :return: An iterator that iterates through all (observation, action) pairs stored in this table
        """
        for state, actions in self.store.items():
            for action in iter(actions):
                yield (state, action)

    def __len__(self) -> int:
        """
        :return: The number of entries in this table
        """
        return sum([len(actions) for actions in self.store.values()])

    def __str__(self) -> str:
        """
        :return: A pretty string representation  TODO -- fix
        """
        t = '| Q Table                                    Value    |\n'
        t += '+------------------------------------------+----------+\n'
        t_entry = '| {:<40} | {:8.3f} |\n'
        for s, a in self:
            t += t_entry.format(str(s) + ', ' + str(a), self[s, a])
        t += '+------------------------------------------+----------+\n'
        return t

    def Q(self, observation, action):
        """
        Obtain the Q-value for performing the specified action given the observation
        :param observation: The obtained observation
        :param action: The action to be performed
        :return: the corresponding Q-value
        """
        return self[observation, action]

    def Qs(self, observation, actions):
        """
        Obtain all Q-values for multiple possible actions given the observation
        :param observation: The obtained observation
        :param actions: A list of actions for which the Q-value should be obtained
        :return: A dictionary mapping each action to the corresponding Q-value
        """
        return {a: self[observation, a] for a in actions}


def for_env(env: DiscreteEnvironment):
    """
    Create a Q Table with the environment's observation and action spaces
    :param env: the corresponding environment
    :return: the Q Table
    """
    return QTable(env.observation_space, env.action_space)


if __name__ == '__main__':

    class TestObject(Action, Observation):

        def __init__(self, value):
            super().__init__(terminal=False)
            self.value = value

        def __eq__(self, other):
            return isinstance(other, TestObject) and other.value == self.value

        def __hash__(self):
            return self.value

        def __repr__(self):
            return str(self.value)

    table = QTable()

    zero, one, two = TestObject(0), TestObject(1), TestObject(2)

    table[zero, zero] = 1  # State 0, action 0
    table[zero, one] = 2  # State 0, action 1

    table[one, zero] = 2  # State 1, action 0

    table[one, zero] += 1

    print(table[zero, one])
    print(table[zero, two])
    print(table[one, zero])
    print([v for v in table])
    print(len(table))
    print(table)
