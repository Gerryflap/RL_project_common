from core import State, Action
from sacx.gerben.extcore import Task
from sacx.gerben.tasked_dual_neural_net import TaskedDualNeuralNet
from sacx.gerben.generic_tasked_q_network import QNetwork
import keras as ks
import numpy as np
import tensorflow as tf


def make_policy_loss(entropy_regularization):
    def policy_loss(q_values, y_pred):
        policy = y_pred
        loss = -tf.reduce_mean(tf.reduce_sum(policy * q_values + entropy_regularization * tf.log(policy),axis=1))
        #loss = tf.reduce_sum(-tf.log(policy))
        return loss
    return policy_loss


class PolicyNetwork:
    def __init__(self,
                 state_shape,
                 action_space,
                 tasks,
                 shared_layers,
                 task_specific_layers,
                 state_transformer,
                 alpha=0.0001,
                 entropy_regularization=0.1,
                 q_network: QNetwork = None,
                 fixed_steps=1000
                 ):
        """
        A Tasked policy network
        :param state_shape: The shape of the state variable WITHOUT the batch dimension
        :param action_space: The Action space
        :param tasks: A list of Tasks
        :param shared_layers: The shared/common layers of the network as a function (using the keras functional API)
        :param task_specific_layers: The task specific layers of the network as a function (using the keras functional API)
        :param state_transformer: A function that takes a state object and transforms it to a network input
        :param alpha: The learning rate
        :param entropy_regularization: The entropy regularization factor for the loss function
        :param q_network: The related Q Network instance.
            This can be left None if it's set later (for instance by the SACU actor)
        :param fixed_steps: The number of training steps that the fixed network is kept fixed.
            After these steps it's updated and the step counter is reset.
        """
        self.fixed_steps = fixed_steps
        self.steps = 0
        self.state_shape = state_shape
        self.action_space = action_space
        self.q_network = q_network
        self.tasks = tasks
        self.model = TaskedDualNeuralNet(
            state_shape,
            shared_layers,
            task_specific_layers,
            lambda model: model.compile(optimizer=ks.optimizers.Adam(alpha, clipnorm=1.0), loss=make_policy_loss(entropy_regularization)),
            tasks
        )
        self.state_transformer = state_transformer

    def distribution(self, state: State, task: Task) -> dict:
        s = self.state_transformer(state)
        s = np.expand_dims(s, axis=0)
        return self.model.predict(s, task)[0]

    def distribution_array(self, states, task: Task, live=True):
        # Expects transformed states
        return self.model.predict(states, task, live=live)

    def train(self, trajectories):
        # Creates a long list of all states and respective Q-values and fits the policy network
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

        # Related to the fixed parameter update
        self.steps += 1
        if self.steps > self.fixed_steps:
            self.steps = 0
            self.sync()

    def sample(self, state: State, task: Task, dist=None) -> Action:
        if dist is None:
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
        dist = self.distribution(state, task)
        a = self.sample(state, task, dist)

        return a, dist

    def sync(self):
        self.model.sync()



if __name__ == "__main__":
    import sacx.gerben.mock_tasked_q_network as mock
    def shared_net(x):
        return ks.layers.Dense(100, activation='relu')(x)

    def individual_net(x):
        return ks.layers.Dense(3, activation='softmax')(x)

    policy = PolicyNetwork((3,), [0,1,2], [0, 1], shared_net, individual_net, lambda x: x, entropy_regularization=0.1, q_network=mock.QNetwork())

    while True:
        trajectories = []
        for i in range(10):
            trajectory = [(np.random.normal(0, 1, (3,)), None, None, None) for _ in range(100)]
            trajectories.append(trajectory)
        policy.train(trajectories)
        print("0:", policy.distribution(np.random.normal(0, 1, (3,)), 0))
        print("1:", policy.distribution(np.random.normal(0, 1, (3,)), 1))


