from policy import Policy
from experiment_util import Configurable

class QEstimator(Configurable):
    """
        Describes an object that is used to predict Q-values for performing an action given the observation
    """

    def Q(self, observation, action) -> float:
        """
        Obtain the Q-value for performing the specified action given the observation
        :param observation: The obtained observation
        :param action: The action to be performed
        :return: the corresponding Q-value
        """
        raise NotImplementedError

    def Qs(self, observation, actions) -> dict:
        """
        Obtain all Q-values for multiple possible actions given the observation
        :param observation: The obtained observation
        :param actions: A list of actions for which the Q-value should be obtained
        :return: A dictionary mapping each action to the corresponding Q-value
        """
        raise NotImplementedError

    def derive_policy(self, policy_class: callable, sa_map: callable, **kwargs) -> Policy:
        """
        Obtain a policy following from this QEstimator
        :param policy_class: The policy class that should be derived (by passing its constructor)
        :param sa_map: A function that maps a state to the actions that can be performed on that state
        :param kwargs: Additional named arguments that can be passed to the policy constructor
        :return: a policy that is based on the Q values as dictated by this QEstimator
        """
        p = policy_class(**kwargs)  # Create a new policy, pass additional args
        if isinstance(p, Policy):   # Set missing policy methods
            p._actions_from = sa_map
            p._actions_values_from = lambda s: self.Qs(s, sa_map(s))
            return p
        else:
            raise Exception('policy_class must return Policy object!')
