
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

    def __init__(self):
        self.M = TaskEnvironment.MAIN_TASK
        self.T = list()

    def step(self, action: Action) -> tuple:
        raise NotImplementedError

    def reset(self) -> State:
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def get_tasks(self) -> list:
        return [self.M] + self.T.copy()



