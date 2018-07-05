
"""
    Extension on core.py to support auxiliary tasks in environments
"""
from core import Environment, State, Action


class Task:
    """
        Task class for SAC-X type algorithms
    """

    def __init__(self, name: str):
        """
        Create a new task
        :param name: A name for the task
        """
        self.name = name

    def __str__(self):
        return 'Task: ' + self.name

    def __repr__(self):
        return 'Task: ' + self.name


class TaskEnvironment(Environment):
    """
        Environment that supports auxiliary tasks and their corresponding reward functions
    """
    MAIN_TASK = Task('main')

    @staticmethod
    def auxiliary_tasks() -> list:
        """
        :return: a list of all auxiliary tasks defined in this environment
        """
        raise NotImplementedError

    @staticmethod
    def get_tasks():
        """
        :return: a list of all tasks defined in this environment (main + auxiliary)
        """
        raise NotImplementedError

    def step(self, action: Action) -> tuple:
        raise NotImplementedError

    def reset(self) -> State:
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError
