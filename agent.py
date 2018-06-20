from core import Environment
from policy import Policy
from experiment_util import Configurable


class Agent(Configurable):
    """
        Agent class that learns a policy in a certain environment
    """

    def __init__(self, env: Environment):
        """
        Create a new Agent
        :param env: The environment the agent should learn in
        """
        self.env = env

    def learn(self) -> Policy:
        """
        Learn a policy by letting the agent interact with the environment
        :return: the derived policy
        """
        raise NotImplementedError
