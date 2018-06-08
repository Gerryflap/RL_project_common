"""
    This is an implementation of deep SARSA-λ
    It is only roughly based on the slides, as they don't really mention Deep SARSA-λ learning explicitly.
    The implementation below puts trajectories in the replay memory instead of single SARSA tuples.
    By picking a random index in one of these trajectories as our training sample we can calculate the return
    (using forward view) and use that to update our Q-value.


    This agent uses Gerben's implementation of environments. 
    It does not play nicely with the standardized state and environment definition.
    Please use the <insert wrapper here> class to wrap any standardized environment.
    For OpenAI gym environments one can use the relatively light GymEnvWrapper provided in this file.

"""

import tensorflow as tf
import numpy as np
import random
import time
ks = tf.keras

from experiment_util import Configurable
class DeepSARSALambdaAgent(Configurable):
    def __init__(self, lambd, action_space, deep_net: ks.models.Model, state_shape, alpha=0.001, epsilon=0.1, gamma=1.0,

                 state_transformer=lambda s: s, epsilon_step_factor=1.0, epsilon_min=0.0, replay_mem_size=1000,
                 fixed_steps=100, batch_size=32, reward_scale=1.0):
        """
        Initializes the Deep SARSA-λ Agent.
        :param lambd: The value for lambda. The target Q-values are calculated using TD(λ),
            where TD(0) only considers the next "step" in the trajectory and TD(1) uses the entire trajectory.
        :param action_space: The action space: a list (or tuple) of actions. These actions can be any type.
        :param deep_net: A Keras model that accepts a state-shaped input and outputs an |action_shape| shaped output
            Keep in mind that this batch size is not fixed and needs to be flexible for this agent to work.
        :param state_shape: The shape of the state variable (without the batch dimension). A tuple of dimensions.
            A 3-vector as state should, for instance, have a shape of (3,).
        :param alpha: The learning rate used by the Adam optimizer.
            This is both the Sarsa(λ)-learning and the NN optimization learning rate.
        :param epsilon: Epsilon is the chance of the agent performing random moves. Epsilon helps with exploration.
        :param gamma: The discount factor. Lower gamma values make the agent care more about short term rewards.
        :param state_transformer: A function that takes a state and transforms it.
            Useful for changing the state to a different format.
        :param epsilon_step_factor: Used for epsilon decay. Epsilon is multiplied with this factor after every step (not episode!).
        :param epsilon_min: The minimum value of epsilon.
            Once epsilon is lower than this value, it is set to this value instead.
        :param replay_mem_size: The size of the replay memory in number of trajectories (not SARSA tuples!).
        :param fixed_steps: The number of steps the target network parameters are fixed before updating.
            This is done to improve stability.
        :param batch_size: The size of the training batches
        :param reward_scale: A number that is multiplied with the rewards to keep them in a sensible range.
            Neural nets are not good at learning very high (>30) expected rewards.
            Try to keep the network outputs "sort of" around 1.
        :param sarsa: When set to true, the agent will use a' instead of the greedy action for computing the target Q.
        """
        self.lambd = lambd
        self.batch_in_t = tf.placeholder(shape=(None,) + state_shape, dtype=tf.float32)

        # Create the live Q-value neural network in a separate TensorFlow scope
        with tf.variable_scope("q_current"):
            self.Qsa_t = deep_net(self.batch_in_t)

        self.action_space = list(action_space)
        self.alpha_t = tf.placeholder_with_default(alpha, None)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_step_factor = epsilon_step_factor
        self.epsilon_min = epsilon_min
        self.s_transformer = state_transformer
        self.action_indices = tf.placeholder(tf.float32, shape=(None, len(action_space)))
        self.step = 0
        self.reward_scale = reward_scale

        # Create the fixed Q-value neural network in a separate TensorFlow scope
        with tf.variable_scope("q_fixed"):
            self.Qsa_fixed_t = deep_net(self.batch_in_t)

        # The target Q-value tensor, which is used during training
        self.Q_target_t = tf.placeholder(tf.float32, shape=(None, len(action_space)))
        loss_t = (self.Qsa_t * self.action_indices - self.Q_target_t) ** 2

        # Apply the optimizer only on trainable variables in the live scope (not the fixed one):
        self.optimizer_t = tf.train.AdamOptimizer(self.alpha_t).minimize(loss_t, var_list=tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='q_current'))

        self.replay_memory = []
        self.replay_mem_size = replay_mem_size
        self.fixed_steps = fixed_steps
        self.batch_size = batch_size

    def Q(self, s, sess):
        """
        Returns the Q-values of state s for every action as a numpy array.
        This is calculated using the live model parameters.
        :param s: State
        :param sess: TensorFlow session
        :return: Numpy array with a Q-value for every action in the same order as the provided action_space variable
        """
        s = self.s_transformer(s)
        return sess.run(self.Qsa_t, feed_dict={self.batch_in_t: np.array([s])})[0]

    def Q_fixed(self, s, sess):
        """
        Returns the Q-values of state s for every action as a numpy array.
        This is calculated using the fixed model parameters.
        :param s: State
        :param sess: TensorFlow session
        :return: Numpy array with a Q-value for every action in the same order as the provided action_space variable
        """
        if type(s) == str and s == "TERMINAL":
            # Ensure that a terminal state has a fixed 0 reward, which acts like an "anchor" for the value function
            return np.zeros((len(self.action_space, )))
        s = self.s_transformer(s)
        return sess.run(self.Qsa_fixed_t, feed_dict={self.batch_in_t: np.array([s])})[0]

    def get_action(self, s, sess):
        """
        Gets the e-greedy action
        :return: (action, action_index, q-value)
        """
        q_values = self.Q(s, sess)
        if random.random() > self.epsilon:
            # Pick greedy action
            a_i = np.argmax(q_values)
        else:
            # Pick random action
            a_i = random.randint(0, len(self.action_space) - 1)
        a = self.action_space[a_i]
        q_value = q_values[a_i]
        return a, a_i, q_value

    def get_batch(self, sess):
        """
        Generates the batch by sampling randomly from the replay memory.
        :param sess: TF session
        :return: xs, q_targets, action_indices
        """

        xs = []
        q_targets = []
        action_indices = []
        for i in range(self.batch_size):
            trajectory = random.choice(self.replay_memory)
            index = random.randint(0, len(trajectory) - 1)
            trajectory = trajectory[index:]
            total_reward = 0
            q_return = 0

            # Calculate all target Q-values for the trajectory
            if len(trajectory) > 1:
                q_values = sess.run(self.Qsa_fixed_t,
                                    feed_dict={self.batch_in_t: [experience[3] for experience in trajectory[:-1]]})
            else:
                q_values = None
            for j, (s, a_i, r, s_p, a_i_p) in enumerate(trajectory):
                total_reward += r * self.gamma ** j
                if j == len(trajectory) - 1:
                    n_return = total_reward
                else:
                    n_return = total_reward + self.gamma ** (j + 1) * q_values[j, a_i_p]
                q_return += self.lambd ** j * n_return

            s, a_i, r, s_p, a_i_p = trajectory[0]
            q_return *= (1 - self.lambd)
            one_hot_action = np.zeros(len(self.action_space))
            one_hot_action[a_i] = 1
            action_indices.append(one_hot_action)
            q_return_vector = np.zeros(len(self.action_space))
            q_return_vector[a_i] = q_return
            q_targets.append(q_return_vector)
            xs.append(self.s_transformer(s))

        return xs, q_targets, action_indices

    def store_experience(self, trajectory):
        """
        Stores a SARSA trajectory in the replay memory
        """
        self.replay_memory.append(trajectory)
        if len(self.replay_memory) > self.replay_mem_size:
            self.replay_memory.pop(0)

    def train(self, sess, take_training_step=True):
        """
        Trains the agent.
        Picks a random batch from the replay memory using the get_batch method and applies the gradients
        to the network parameters. Also decays learning rate and epsilon if this is enabled.
        :param take_training_step: If set to true: decays the learning rate and epsilon and increments the step variable.
        :param sess: The TF session.
        """
        batch_in, q_targets, action_indices = self.get_batch(sess)
        sess.run(self.optimizer_t, feed_dict={
            self.alpha_t: self.alpha,
            self.batch_in_t: batch_in,
            self.Q_target_t: q_targets,
            self.action_indices: action_indices
        })

        if take_training_step:
            self.alpha *= 1.0
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_step_factor
            elif self.epsilon_min > self.epsilon:
                self.epsilon = self.epsilon_min

            if self.step % self.fixed_steps == 0:
                self.update_fixed_q(sess)
            self.step += 1

    def run_episode(self, env, sess, visual=False):
        """
        Runs an episode on the provided environment.
        Also collects experiences and trains the network (if not running in visual mode).
        :param env: The environment to run on. The environment is reset before the episode by the agent.
        :param sess: The TF session
        :param visual: Enables visual mode if set to True. Visual mode:
            - Renders the screen at every step.
            - Disables training for a smoother visual experience
            - Does not store the experiences in the replay memory
        :return: This method returns the episode score (cumulative reward over the trajectory.)
        """
        s = env.reset()
        a, a_i, _ = self.get_action(s, sess)
        score = 0
        env.render = visual
        trajectory = []

        while not env.terminated:
            s_p, r = env.step(a)
            a_p, a_i_p, _ = self.get_action(s_p, sess)
            if not visual:
                trajectory.append((s, a_i, r * self.reward_scale, s_p, a_i_p))
                if len(self.replay_memory) > 0:
                    self.train(sess)

            s, a, a_i = s_p, a_p, a_i_p
            score += r
        if not visual:
            trajectory.append((s, a_i, 0, "TERMINAL", 0))
            self.store_experience(trajectory)
        return score

    def update_fixed_q(self, sess):
        """
        Copies the "live" parameters to the fixed Q network.
        :param sess: The TF session
        """
        self._copy_from_scope("q_current", "q_fixed", sess)

    def _copy_from_scope(self, from_scope, to_scope, sess):
        """
        Copies all trainable variables from one TensorFlow variable scope to the other.
        Variables do need to have the same name (apart from the prepended scope).
        So copying the variable "x" from scope "current" to "fixed" requires x to be called current/x and fixed/x
            for this to work. Inner scopes (like current/dense/kernel_0) are also allowed.
        :param from_scope: The scope to copy from.
        :param to_scope: The scope to copy to.
        :param sess: The TF scope
        """
        scope_from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=from_scope)
        scope_to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=to_scope)

        from_map = dict()
        values = sess.run(scope_from_vars)
        for var, val in zip(scope_from_vars, values):
            unscoped_name = var.name[len(from_scope) + 1:]
            from_map[unscoped_name] = val

        to_map = dict()
        for var in scope_to_vars:
            unscoped_name = var.name[len(to_scope) + 1:]
            to_map[unscoped_name] = var

        for varname, val in from_map.items():
            to_map[varname].load(val, sess)


class GymEnvWrapper(object):
    """
    A wrapper for OpenAI gym environments that allows them to work with the DeepSarsaLambdaAgent
    """

    def __init__(self, gym_env, state_transformer):
        self.env = gym_env
        self.terminated = True
        self.render = False
        self.state_transformer = state_transformer

    def reset(self):
        self.terminated = False
        return self.state_transformer(self.env.reset())

    def step(self, action):
        s, r, done, _ = self.env.step(action)
        if self.render:
            self.env.render()
            time.sleep(1 / 60)
        self.terminated = done
        return self.state_transformer(s), r

    def set_rendering(self, value: bool):
        self.render = value


if __name__ == "__main__":
    import gym

    env = gym.make("LunarLander-v2")
    env = GymEnvWrapper(env, lambda s: s)


    def network(x):
        ks = tf.keras
        x = ks.layers.Dense(150, activation='relu')(x)
        x = ks.layers.Dense(50, activation='relu')(x)
        return ks.layers.Dense(4, activation='linear')(x)


    agent = DeepSARSALambdaAgent(0.9, [0, 1, 2, 3], network, alpha=0.001, state_shape=(8,), epsilon=1.0,
                                 epsilon_step_factor=0.9999, epsilon_min=0.05, gamma=1.0, fixed_steps=100,
                                 reward_scale=0.01, replay_mem_size=10000)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        s = env.reset()
        while True:
            scores = []
            for i in range(4):
                score = agent.run_episode(env, sess)
                scores.append(score)
            agent.run_episode(env, sess, True)

            print("Score: ", sum(scores) / len(scores))
            print("Eps: ", agent.epsilon)
            print("Q: ", agent.Q(s, sess))
            print("Q fixed: ", agent.Q_fixed(s, sess))
