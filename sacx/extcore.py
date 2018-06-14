
"""
    Extension on core.py to support auxiliary tasks in environments
"""
from core import Environment, State, Action


class Task:

    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return 'Task: ' + self.name

    def __repr__(self):
        return 'Task: ' + self.name


class TaskEnvironment(Environment):
    MAIN_TASK = Task('main')

    @staticmethod
    def auxiliary_tasks():
        raise NotImplementedError

    @staticmethod
    def get_tasks():
        return [TaskEnvironment.MAIN_TASK] + TaskEnvironment.auxiliary_tasks()

    def step(self, action: Action) -> tuple:
        raise NotImplementedError

    def reset(self) -> State:
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError
