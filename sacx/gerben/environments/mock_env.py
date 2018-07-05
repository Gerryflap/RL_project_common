import numpy as np

from core import FiniteActionEnvironment, State, Action
from sacx.extcore import TaskEnvironment


ACTIONS = [0, 1, 2]

TASKS = [0,1,2]

class MockState(State):
    def __init__(self, terminal: bool, state):
        super().__init__(terminal)
        self.state = state


class MockEnv(FiniteActionEnvironment, TaskEnvironment):
    @staticmethod
    def valid_actions_from(state) -> list:
        return ACTIONS

    @staticmethod
    def action_space() -> list:
        return ACTIONS

    def valid_actions(self) -> list:
        return ACTIONS

    def step(self, action: Action) -> tuple:
        rewards = {i: 1 if action == i else 0 for i in range(3)}

        return MockState(False, np.random.normal(0, 1, (5,))), rewards

    def reset(self) -> State:
        return MockState(False, np.random.normal(0, 1, (5,)))

    @staticmethod
    def auxiliary_tasks() -> list:
        return TASKS[1:]

    @staticmethod
    def get_tasks():
        return TASKS