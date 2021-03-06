"""
    Implements a Tasked Q-Network (Value function approximator, Critic in actor-critic)
"""

import math
import random

import numpy as np
import keras as ks
import sacx.generic_tasked_q_network as generic
from sacx.tasked_dual_neural_net import TaskedDualNeuralNet
from sacx.tasked_p_network import PolicyNetwork
from sacx.extcore import State, Task, Action


class QNetwork(generic.QNetwork):
    def __init__(self,
                 state_shape,
                 action_space,
                 tasks,

                 shared_layers,
                 task_specific_layers,
                 state_transformer,
                 alpha=0.0001,
                 gamma=0.9,
                 p_network: PolicyNetwork = None,
                 fixed_steps=1000,
                 reward_scale=1.0,
                 lambd=0.9,
                 lambd_min=0.0
                 ):
        """
        Initializes a Tasked Q Network. This Q network uses retrace for Q-value calculation
        :param state_shape: The shape of the state variable WITHOUT the batch dimension
        :param action_space: The Action space
        :param tasks: A list of Tasks
        :param shared_layers: The shared/common layers of the network as a function (using the keras functional API)
        :param task_specific_layers: The task specific layers of the network as a function (using the keras functional API)
        :param state_transformer: A function that takes a state object and transforms it to a network input
        :param alpha: The learning rate
        :param gamma: The discount factor
        :param p_network: The Policy Network, this can be None at init as long as it's set later (This is done by the SACU agent)
        :param fixed_steps: The number of training steps that the fixed network is kept fixed.
            After these steps it's updated and the step counter is reset.
        :param reward_scale: A float that is used to scale rewards
        :param lambd: The lambda value used by the retrace algorithm. Similar to its use in SARSA lambda.
        :param lambd_min: A cap on the lambda value similar to the one used in our SARSA lambda implementation.
            It cuts off trajectories when lambda^k goes below this value. Given that SAC-X often operates on
            trajectories with a mean length of ~500 (where lambda^k could easily be 1e-20), using this value
            yields massive speed improvements without large costs.
        """
        self.lambd = lambd
        self.lambda_min = lambd_min
        self.max_trajectory_length = None
        if self.lambda_min != 0:
            self.max_trajectory_length = int(math.log(self.lambda_min, lambd))
            print("Capping trajectories at a max length of ", self.max_trajectory_length)
        self.reward_scale = reward_scale
        self.fixed_steps = fixed_steps
        self.steps = 0
        self.state_transformer = state_transformer
        self.task_specific_layers = task_specific_layers
        self.shared_layers = shared_layers
        self.p_network = p_network
        self.tasks = tasks
        self.action_space = action_space
        self.inverse_action_lookup = dict()
        for i, a in enumerate(action_space):
            self.inverse_action_lookup[a] = i
        self.state_shape = state_shape
        self.gamma = gamma

        self.model = TaskedDualNeuralNet(
            state_shape,
            shared_layers,
            task_specific_layers,
            lambda model: model.compile(optimizer=ks.optimizers.Adam(alpha), loss=ks.losses.mean_squared_error),
            tasks
        )

    def Qs(self, state: State, task: Task):
        return self.model.predict(np.expand_dims(self.state_transformer(state), axis=0), task)[0]

    def Q(self, state: State, action: Action, task: Task):
        pass  # TODO

    def Q_array(self, states, task: Task):
        return self.model.predict(states, task, live=True)

    def Qp_array(self, states, task: Task):
        return self.model.predict(states, task, live=False)

    def train(self, trajectories):
        for task in self.tasks:
            initial_states = np.stack([self.state_transformer(t[0][0]) for t in trajectories], axis=0)

            ys = self.model.predict(initial_states, task)

            for i, trajectory in enumerate(trajectories):
                _, a, _, _ = trajectory[0]
                ys[i, self.inverse_action_lookup[a]] += self.get_q_delta(trajectory, task)

            self.model.fit(initial_states, ys, task)

        self.steps += 1
        if self.steps > self.fixed_steps:
            self.steps = 0
            self.sync()

    def get_q_delta(self, trajectory, task):
        """
        Calculate the Q-delta according to the Retrace algorithm
            (the implementation is based on the Retrace paper, not the SAC-X paper)
        :param trajectory: The trajectory to calculate the delta on
        :param task: The task to calculate the delta for
        :param sess: The TF session
        :return: (The Q delta, the target q-value)
            Here the target q-value is the fixed q-values + the q-delta
        """

        # Initialize the q_delta variable and get all Qsa values for the trajectory
        q_delta = 0
        if self.max_trajectory_length is None:
            states = np.array([self.state_transformer(e[0]) for e in trajectory])
        else:
            states = np.array([self.state_transformer(e[0]) for e in trajectory[:self.max_trajectory_length+1]])

        q_values = self.model.predict(states, task, live=True)
        q_fixed_values = self.model.predict(states, task, live=False)


        # Calculate all values of π(* | state, task_id) for the trajectory for out current task
        #   (using the fixed network)
        policies = self.p_network.distribution_array(states, task, live=False)


        # Pick all π(a_t | state, task_id) from π(* | state, task_id) for every action taken in the trajectory
        all_action_probabilities = np.array([a[self.inverse_action_lookup[i]] for i, a in zip([e[1] for e in trajectory], policies)])


        # Pick all b(a_t | state, B) from the trajectory
        if self.max_trajectory_length is None:
            all_b_action_probabilities = np.array(([experience[3][self.inverse_action_lookup[experience[1]]] for experience in trajectory]))
        else:
            all_b_action_probabilities = np.array(
                (
                    [experience[3][self.inverse_action_lookup[experience[1]]] for experience in trajectory[:self.max_trajectory_length+1]]
                ))

        # Calculate the value of c_k for the whole trajectory
        c = all_action_probabilities / all_b_action_probabilities

        # Make sure that c is capped on 1, so c_k = min(1, c_k)
        c[c > 1] = 1

        c[0] = 1    # According to the retrace paper

        # Keep the product of c_k values in a variable
        c_product = 1

        iterations = len(trajectory)
        if self.max_trajectory_length is not None:
            iterations = min(iterations, self.max_trajectory_length)

        # Iterate over the trajectory to calculate the expected returns
        for j, (s, a, r, _) in enumerate(trajectory[:iterations]):
            # Multiply the c_product with the next c-value,
            #   this makes any move done after a "dumb" move (according to our policy) less significant
            c_product *= c[j] * self.lambd

            a_index = self.inverse_action_lookup[a]

            # Check if we're at the end of the trajectory
            if j != len(trajectory) - 1:
                # If we're not: calculate the next difference and use the Q for j+1 as well

                # The Expected value of the Q(s_j+1, *) under the policy
                expected_q_tp1 = np.sum(policies[j + 1] * q_fixed_values[j + 1])

                # The delta for this lookahead
                delta = self.reward_scale*r[task] + self.gamma * expected_q_tp1 - q_values[j, a_index]
            else:
                # If this is the last entry, we'll assume the Q(s_j+1, *) to be fixed on 0 as the state is terminal
                delta = self.reward_scale*r[task] - q_values[j, a_index]

                # if task == self.tasks[0]:
                #     print("Calculated last delta for main task, old_Q and delta:", q_values[j, a_index], q_delta + c_product * self.gamma ** j * delta)

            # Add this to the sum of q_deltas, where the term is multiplied by gamma and delta
            q_delta += c_product * self.gamma ** j * delta



        return q_delta

    def sync(self):
        self.model.sync()

if __name__ == "__main__":
    def shared_net(x):
        return ks.layers.Dense(100, activation='relu')(x)

    def individual_net(x):
        return ks.layers.Dense(3, activation='softmax')(x)

    def individual_q_net(x):
        return ks.layers.Dense(3, activation='linear')(x)
    q_net = QNetwork((3,), [0,1,2], [0, 1], shared_net, individual_q_net, lambda x: x, alpha=0.01, gamma=0)
    policy = PolicyNetwork((3,), [0,1,2], [0, 1], shared_net, individual_net, lambda x: x, entropy_regularization=100000, q_network=q_net)
    q_net.p_network = policy

    while True:
        trajectories = []
        for i in range(10):
            trajectory = []
            t = random.randint(0, 1)
            for j in range(10):
                s = np.random.normal(0, 1, (3,))
                a, d = policy.sample_distribution(s, t)
                r = [1 if a == 0 else 0, 1 if a == 1 else 0, 0]
                trajectory.append((s,a,r,d))

            trajectories.append(trajectory)
        q_net.train(trajectories)
        policy.train(trajectories)
        q_net.sync()
        policy.sync()
        print("0:", q_net.Qs(np.random.normal(0, 1, (3,)), 0))
        print("1:", q_net.Qs(np.random.normal(0, 1, (3,)), 1))

