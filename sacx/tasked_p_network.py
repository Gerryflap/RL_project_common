import keras as ks
import numpy as np

from core import State, Action
from sacx.extcore import Task


class PolicyNetwork:

    def __init__(self,
                 model_input: ks.layers.Input,
                 model_outputs: dict,
                 out_map,
                 feature_ex,
                 sa_map,
                 ):

        self.models = {task: ks.Model(inputs=[model_input], outputs=[output]) for task, output in model_outputs.items()}
        self.out_map = out_map
        self.fex = feature_ex
        self.sa_map = sa_map

        for model in self.models.values():
            model.compile()  # TODO

    def distribution(self, state: State, task: Task) -> dict:
        model = self.models[task]
        actions = self.sa_map(state)
        dist = {a: 0 for a in self.out_map}
        values = model.predict([self.fex(state)])[0]
        for a in actions:
            dist[a] = values[self.out_map.index(a)]
        total = sum(dist.values())
        if total == 0:
            return {a: 1 / len(actions) for a in dist.keys()}
        else:
            return {a: v / total for a, v in dist.items()}

    def train(self):
        pass  # TODO

    def sample(self, state: State, task: Task) -> Action:
        return self._sample(self.distribution(state, task))

    @staticmethod
    def _sample(dist):
        actions, probabilities = zip(*dist.items())
        return np.random.choice(actions, p=probabilities)

    # def sample_greedy(self, state: State, task: Task) -> Action:
    #     pass  # TODO?
    #
    # def sample_epsilon_greedy(self, state: State, task: Task) -> Action:
    #     pass  # TODO?

    def sample_distribution(self, state: State, task: Task) -> tuple:
        dist = self.distribution(state, task)
        return self._sample(dist), dist
