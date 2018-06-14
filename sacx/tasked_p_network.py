from core import State, Action
from sacx.extcore import Task


class PolicyNetwork:

    def distribution(self, state: State, task: Task) -> dict:
        pass  # TODO

    def train(self):
        pass  # TODO

    def sample(self, state: State, task: Task) -> Action:
        pass  # TODO

    def sample_greedy(self, state: State, task: Task) -> Action:
        pass  # TODO

    def sample_epsilon_greedy(self, state: State, task: Task) -> Action:
        pass  # TODO

    def sample_distribution(self, state: State, task: Task) -> tuple:
        pass  # TODO
