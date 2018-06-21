"""
    Defines a policy network updated by maximizing π(s) * ( Q(s, θ') - α*log(π(s) )
"""
import keras as ks
import tensorflow as tf
import numpy as np

from core import State, Action
from policy import Policy
from q_network_sarsa_lambda import QNetworkSL


class PNetwork(Policy):
    def _actions_from(self, state) -> list:
        raise NotImplementedError

    def _actions_values_from(self, state):
        raise NotImplementedError

    def __init__(self,
                 model: ks.Model,
                 action_space,
                 state_transformer,
                 alpha=0.0001,
                 entropy_regularization=0.1,
                 q_network: QNetworkSL = None,
                 fixed_steps=1000
                 ):
        """
        A Tasked policy network
        :param action_space: The Action space
        :param model: An uncompiled Keras model
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
        self.action_space = action_space
        self.q_network = q_network
        self.model = model

        self.fixed_model = ks.models.model_from_json(model.to_json())
        self.state_transformer = state_transformer

        self.model.compile(optimizer=ks.optimizers.Adam(alpha), loss=self._make_policy_loss(entropy_regularization))

    def distribution(self, state: State) -> dict:
        s = self.state_transformer(state)
        s = np.expand_dims(s, axis=0)
        return self.model.predict(s)[0]

    def distribution_array(self, states, live=True):
        # Expects transformed states

        if live:
            return self.model.predict(states)
        else:
            return self.fixed_model.predict(states)

    def train(self, trajectories):
        # Creates a list of all "initial" states and respective Q-values and fits the policy network

        xs = []
        for trajectory in trajectories:
            states = np.array([self.state_transformer(t[0]) for t in trajectory[:1]])
            xs.append(states)

        xs = np.concatenate(xs, axis=0)

        # Predict the Q-values  for all actions for each of the states using Q(s, θ')
        q_values = self.q_network.Qp_array(xs)

        # Fit the live model on the Q-values using the policy loss function
        self.model.fit(xs, q_values, verbose=False, batch_size=len(trajectories))

        # Related to the fixed parameter update
        self.steps += 1
        if self.steps > self.fixed_steps:
            self.steps = 0
            self.sync()

    @staticmethod
    def _make_policy_loss(entropy_regularization):
        def policy_loss(q_values, y_pred):
            policy = y_pred
            loss = -tf.reduce_mean(tf.reduce_sum(policy * (q_values - entropy_regularization * tf.log(policy)), axis=1))
            return loss

        return policy_loss

    def sample(self, state: State, dist=None) -> Action:
        if dist is None:
            dist = self.distribution(state)

        choice = np.random.random()
        p_cumulative = 0
        index = 0
        for i, p in enumerate(dist):
            p_cumulative += p
            if p_cumulative > choice:
                index = i
                break
        return self.action_space[index]

    def sample_greedy(self, state: State) -> Action:
        raise NotImplementedError

    def sample_epsilon_greedy(self, state: State) -> Action:
        raise NotImplementedError

    def sample_distribution(self, state: State) -> tuple:
        dist = self.distribution(state)
        a = self.sample(state, dist)

        return a, dist

    def sync(self):
        self.fixed_model.set_weights(self.model.get_weights())



