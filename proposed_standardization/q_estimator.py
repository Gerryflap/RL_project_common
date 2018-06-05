

class QEstimator:
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
