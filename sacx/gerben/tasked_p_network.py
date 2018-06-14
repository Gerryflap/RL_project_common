from core import State, Action
from sacx.gerben.extcore import Task
from sacx.gerben.tasked_dual_neural_net import TaskedDualNeuralNet
from sacx.gerben.tasked_q_network import QNetwork
import keras as ks
import numpy as np
import tensorflow as tf


def make_policy_loss(entropy_regularization):
    def policy_loss(q_values, y_pred):
        policy = y_pred
        loss = -tf.reduce_sum(policy * q_values + entropy_regularization * tf.log(policy))
        #loss = tf.reduce_sum(-tf.log(policy))
        return loss
    return policy_loss


class PolicyNetwork:
    def __init__(self,
                 state_shape,
                 action_space,
                 tasks,
                 q_network: QNetwork,
                 shared_layers,
                 task_specific_layers,
                 state_transformer,
                 alpha=0.0001,
                 entropy_regularization=0.1
                 ):
        self.state_shape = state_shape
        self.action_space = action_space
        self.q_network = q_network
        self.tasks = tasks
        self.model = TaskedDualNeuralNet(
            state_shape,
            shared_layers,
            task_specific_layers,
            lambda model: model.compile(optimizer=ks.optimizers.Adam(alpha), loss=make_policy_loss(entropy_regularization)),
            tasks
        )
        self.state_transformer = state_transformer

    def distribution(self, state: State, task: Task) -> dict:
        s = self.state_transformer(state)
        s = np.expand_dims(s, axis=0)
        return self.model.predict(s, task)[0]

    def train(self, trajectories):
        for task in self.tasks:
            xs = []
            q_values = []
            for trajectory in trajectories:
                states = np.array([self.state_transformer(t[0]) for t in trajectory])
                xs.append(states)
                qs = self.q_network.Qp_array(states, task)
                q_values.append(qs)
            xs = np.concatenate(xs, axis=0)
            q_values = np.concatenate(q_values, axis=0)
            self.model.fit(xs, q_values, task)

    def sample(self, state: State, task: Task) -> Action:
        dist = self.distribution(state, task)

        choice = np.random.random()
        p_cumulative = 0
        index = 0
        for i, p in enumerate(dist):
            p_cumulative += p
            if p_cumulative > choice:
                index = i
                break
        return self.action_space[index]

    def sample_greedy(self, state: State, task: Task) -> Action:
        raise NotImplementedError

    def sample_epsilon_greedy(self, state: State, task: Task) -> Action:
        raise NotImplementedError

    def sample_distribution(self, state: State, task: Task) -> tuple:
        raise NotImplementedError


if __name__ == "__main__":
    def shared_net(x):
        return ks.layers.Dense(100, activation='relu')(x)

    def individual_net(x):
        return ks.layers.Dense(3, activation='softmax')(x)

    policy = PolicyNetwork((3,), [0,1,2], [0], QNetwork(), shared_net, individual_net, lambda x: x, entropy_regularization=0.1)

    while True:
        trajectories = []
        for i in range(10):
            trajectory = [(np.random.normal(0, 1, (3,)), None, None, None) for _ in range(100)]
            trajectories.append(trajectory)
        policy.train(trajectories)
        print(policy.distribution(np.random.normal(0, 1, (3,)), 0))


