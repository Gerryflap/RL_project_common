"""
    This is Gerben's implementation of deep Q learning.
    It does not play nicely with the standardized state and environment definition.
    Please use the <insert wrapper here> class to wrap any standardized environment.
    For OpenAI gym environments one can use the relatively light GymEnvWrapper provided in this file.
"""

import tensorflow as tf
import numpy as np
import random
import time
ks = tf.keras


class DeepQAgent(object):
    def __init__(self, action_space, model: ks.models.Model, state_shape, alpha=0.001, epsilon=0.1, gamma=1.0, state_transformer=lambda s: s, epsilon_step_factor=1.0, epsilon_min=0.0, replay_mem_size=1000, fixed_steps=100, batch_size=32, reward_scale=1.0, sarsa=False):
        """
        Initializes the Deep Q Agent
        :param action_space: The action space: a list (or tuple) of actions. These actions can be any type.
        :param model: A Keras model that accepts a state-shaped input and outputs an |action_shape| shaped output
            Keep in mind that this batch size is not fixed and needs to be flexible for this agent to work.
        :param state_shape: The shape of the state variable (without the batch dimension). A tuple of dimensions.
            A 3-vector as state should, for instance, have a shape of (3,).
        :param alpha: The learning rate used by the Adam optimizer.
            This is both the Q-learning and the NN optimization learning rate.
        :param epsilon: Epsilon is the chance of the agent performing random moves. Epsilon helps with exploration.
        :param gamma: The discount factor. Lower gamma values make the agent care more about short term rewards.
        :param state_transformer: A function that takes a state and transforms it.
            Useful for changing the state to a different format.
        :param epsilon_step_factor: Used for epsilon decay. Epsilon is multiplied with this factor after every step (not episode!).
        :param epsilon_min: The minimum value of epsilon.
            Once epsilon is lower than this value, it is set to this value instead.
        :param replay_mem_size: The size of the replay memory in number of SARSA tuples.
        :param fixed_steps: The number of steps the target network parameters are fixed before updating.
            This is done to improve stability.
        :param batch_size: The size of the training batches
        :param reward_scale: A number that is multiplied with the rewards to keep them in a sensible range.
            Neural nets are not good at learning very high (>30) expected rewards.
            Try to keep the network outputs "sort of" around 1.
        :param sarsa: When set to true, the agent will use a' instead of the greedy action for computing the target Q.
        """

        self.batch_in_t = tf.placeholder(shape=(None,) + state_shape, dtype=tf.float32)

        # Create the live Q-value neural network in a separate TensorFlow scope
        self.live_model = model

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
        self.reward_scale=reward_scale

        # This deep Q-learning implementation can also be switched to SARSA if deemed necessary.
        self.sarsa = sarsa

        self.fixed_model = ks.models.model_from_json(self.live_model.to_json())

        self.live_model.compile(ks.optimizers.Adam(alpha), ks.losses.mean_squared_error)
        #self.fixed_model.compile(ks.optimizers.Adam(alpha), ks.losses.mean_squared_error)

        self.replay_memory = []
        self.replay_mem_size = replay_mem_size
        self.fixed_steps = fixed_steps
        self.batch_size = batch_size

    def Q(self, s):
        """
        Returns the Q-values of state s for every action as a numpy array.
        This is calculated using the live model parameters.
        :param s: State
        :param sess: TensorFlow session
        :return: Numpy array with a Q-value for every action in the same order as the provided action_space variable
        """
        s = self.s_transformer(s)
        s = np.expand_dims(s, axis=0)
        return self.live_model.predict(s)[0]

    def Q_fixed(self, s):
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
        s = np.expand_dims(s, axis=0)
        return self.fixed_model.predict([s])[0]

    def get_action(self, s):
        """
        Gets the e-greedy action
        :return: (action, action_index, q-value)
        """
        q_values = self.Q(s)
        if random.random() > self.epsilon:
            # Pick greedy action
            a_i = np.argmax(q_values)
        else:
            # Pick random action
            a_i = random.randint(0, len(self.action_space)-1)
        a = self.action_space[a_i]
        q_value = q_values[a_i]
        return a, a_i, q_value

    def get_batch(self):
        """
        Generates the batch by sampling randomly from the replay memory.
        :param sess: TF session
        :return: xs, q_targets, action_indices
        """
        sarsa_batch = np.array(self.replay_memory)[np.random.choice(np.arange(0, len(self.replay_memory)), self.batch_size)]

        xs = []
        q_targets = np.zeros((self.batch_size, len(self.action_space)))
        action_indices = np.zeros((self.batch_size, len(self.action_space)))
        i = 0
        for s, a_i, r, s_p, a_i_p in sarsa_batch:
            if self.sarsa:
                q_target = r + self.gamma * self.Q_fixed(s_p)[a_i_p]
            else:
                q_target = r + self.gamma * np.max(self.Q_fixed(s_p))
            action_indices[i, a_i] = 1
            q_targets[i] = self.Q(s)
            q_targets[i, a_i] = q_target
            xs.append(self.s_transformer(s))
            i += 1
        return xs, q_targets

    def store_experience(self, s, a_i, r, s_p, a_i_p):
        """
        Stores a SARS'A' tuple in the replay memory
        :param s: state
        :param a_i: action index (not always the action itself)
        :param r: reward
        :param s_p: state' (the next state)
        :param a_i_p: action' index (index for the next action chosen)
        :return:
        """
        self.replay_memory.append((s, a_i, r, s_p, a_i_p))
        if len(self.replay_memory) > self.replay_mem_size:
            self.replay_memory.pop(0)

    def train(self, take_training_step=True):
        """
        Trains the agent.
        Picks a random batch from the replay memory using the get_batch method and applies the gradients
        to the network parameters. Also decays learning rate and epsilon if this is enabled.
        :param take_training_step: If set to true: decays the learning rate and epsilon and increments the step variable.
        :param sess: The TF session.
        """
        batch_in, q_targets = self.get_batch()
        batch_in = np.array(batch_in)
        self.live_model.fit(batch_in, q_targets, verbose=False)

        if take_training_step:
            self.alpha *= 1.0
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_step_factor
            elif self.epsilon_min > self.epsilon:
                self.epsilon = self.epsilon_min

            if self.step%self.fixed_steps == 0:
                self.update_fixed_q()
            self.step += 1

    def run_episode(self, env, visual=False):
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
        a, a_i, _ = self.get_action(s)
        score = 0
        env.render = visual

        while not env.terminated:
            s_p, r = env.step(a)
            a_p, a_i_p, _ = self.get_action(s_p)
            if not visual:
                self.store_experience(s, a_i, r*self.reward_scale, s_p, a_i_p)
                self.train()

            s, a, a_i = s_p, a_p, a_i_p
            score += r
        if not visual:
            self.store_experience(s, a_i, 0, "TERMINAL", 0)
        return score

    def update_fixed_q(self):
        """
        Copies the "live" parameters to the fixed Q network.
        :param sess: The TF session
        """
        self.fixed_model.set_weights(self.live_model.get_weights())


class GymEnvWrapper(object):
    """
    A wrapper for OpenAI gym environments that allows them to work with the DeepQAgent
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
            time.sleep(1/60)
        self.terminated = done
        return self.state_transformer(s), r

    def set_rendering(self, value: bool):
        self.render = value


if __name__ == "__main__":
    import gym
    env = gym.make("LunarLander-v2")
    env = GymEnvWrapper(env, lambda s:s)

    network = ks.models.Sequential()
    network.add(ks.layers.Dense(150, activation='relu', input_shape=(8,)))
    network.add(ks.layers.Dense(50, activation='relu'))
    network.add(ks.layers.Dense(4, activation='linear'))

    agent = DeepQAgent([0,1,2,3], network, alpha=0.001, state_shape=(8,), epsilon=1.0, epsilon_step_factor=0.9999, epsilon_min=0.05, gamma=1.0, fixed_steps=100, reward_scale=0.01, replay_mem_size=10000, sarsa=True)

    s = env.reset()
    while True:
        scores = []
        for i in range(4):
            score = agent.run_episode(env)
            scores.append(score)
        agent.run_episode(env, True)

        print("Score: ", sum(scores)/len(scores))
        print("Eps: ", agent.epsilon)
        print("Q: ", agent.Q(s))
        print("Q fixed: ", agent.Q_fixed(s))
